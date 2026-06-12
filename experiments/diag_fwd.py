#!/usr/bin/env python3
"""FORWARD-FIDELITY CEILING TEST.  Train W1,W2 in numpy (this structure hits ~0.91 on C=5), then
evaluate those SAME weights three ways:
  (a) numpy tanh forward          -> the algorithmic ceiling
  (b) numpy forward, weights clipped to the circuit's representable range -> isolates weight DYN-RANGE
  (c) ngspice analog forward (pc_bp circuit) -> the real analog ceiling (range+offset+neuron shape)
The (a)->(b) drop = weight dynamic-range loss; (b)->(c) drop = analog offset/neuron non-ideality."""
import os, numpy as np
from ng_live import Live
from sklearn.datasets import load_digits
NIN=int(os.environ.get("NIN","16")); N=NIN; C=int(os.environ.get("C","5")); H=int(os.environ.get("H","20")); SEED=int(os.environ.get("SEED","0"))
NTR=int(os.environ.get("NTR","40")); NTE=int(os.environ.get("NTE","40")); EP=int(os.environ.get("EP","200"))
LR=float(os.environ.get("LR","0.02")); LRX=float(os.environ.get("LRX","0.3")); NREL=int(os.environ.get("NREL","20"))
IND=float(os.environ.get("IND","0.30")); KMAP=float(os.environ.get("KMAP","0.30")); VW0=0.5
WL=os.environ.get("WL","1000u"); RCS=os.environ.get("RCS","300k"); RNODE=os.environ.get("RNODE","20k")
VBSYN=os.environ.get("VBSYN","0.45"); VBNEU=os.environ.get("VBNEU","0.35")
SA=float(os.environ.get("SA","-1")); SO=float(os.environ.get("SO","1")); TAG=os.environ.get("RUNTAG",str(os.getpid()))
WRANGE=(0.9-VW0)/KMAP   # circuit-representable |weight| before the wp/wn voltages rail at 0.3/0.9
f=np.tanh; TASK=os.environ.get("TASK","digits")
if TASK=="rings":
    N=2; _rg=np.random.default_rng(123); _n=1500; X=[];y=[]
    for c in range(C):
        r=_rg.uniform(c+0.35,c+0.9,_n//C); th=_rg.uniform(0,2*np.pi,_n//C)
        X+=list(np.c_[r*np.cos(th),r*np.sin(th)]); y+=[c]*(_n//C)
    X=np.array(X); y=np.array(y)
else:
    d=load_digits(); X=(d.images.reshape(-1,64) if NIN==64 else d.images.reshape(-1,4,2,4,2).mean(axis=(2,4)).reshape(-1,16)); y=d.target
m=X.mean(0); s=X.std(0)+1e-6; X=np.clip((X-m)/s,-3,3)/3.0
rng=np.random.default_rng(SEED); CLS=list(range(C)); Xtr=[];ytr=[];Xte=[];yte=[]
for ci,c in enumerate(CLS):
    idx=np.where(y==c)[0]; rng.shuffle(idx)
    for i in idx[:NTR]: Xtr.append(X[i]); ytr.append(ci)
    for i in idx[NTR:NTR+NTE]: Xte.append(X[i]); yte.append(ci)
Xtr=np.array(Xtr); ytr=np.array(ytr); Xte=np.array(Xte); yte=np.array(yte)
# --- train in numpy (PC, the proven ~0.91 recipe) ---
WIH=rng.standard_normal((H,N))*0.3; WHO=rng.standard_normal((C,H))*0.3
def target(lab): t=np.full(C,-1.0); t[lab]=1.0; return t
for ep in range(EP):
    for k in rng.permutation(len(Xtr)):
        x=Xtr[k]; x1=WIH@x
        for _ in range(NREL):
            e1=x1-WIH@x; e2=target(ytr[k])-WHO@f(x1); g=1-f(x1)**2
            x1=x1+LRX*(-e1+g*(WHO.T@e2))
        e1=x1-WIH@x; e2=target(ytr[k])-WHO@f(x1)
        WHO+=LR*np.outer(e2,f(x1)); WIH+=LR*np.outer(e1,x)
def npacc(W1,W2,Xs,ys):
    ok=0
    for x,lab in zip(Xs,ys): ok+=(int(np.argmax(W2@f(W1@x)))==lab)
    return ok/len(Xs)
acc_a=npacc(WIH,WHO,Xte,yte)
WIHc=np.clip(WIH,-WRANGE,WRANGE); WHOc=np.clip(WHO,-WRANGE,WRANGE)
acc_b=npacc(WIHc,WHOc,Xte,yte)
print(f"[numpy] weight stats: |WIH|max={np.abs(WIH).max():.2f} |WHO|max={np.abs(WHO).max():.2f}  circuit range=±{WRANGE:.2f}")
print(f"(a) numpy tanh forward          TEST={acc_a:.2f}")
print(f"(b) numpy, weights clipped ±{WRANGE:.2f}  TEST={acc_b:.2f}   <- (a)->(b) = dynamic-range loss")
# --- (c) evaluate the SAME (clipped) weights in the ngspice circuit ---
SUB=f""".model NNR NMOS (LEVEL=1 VTO=0.2 KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)
.model PNR PMOS (LEVEL=1 VTO=-0.2 KP=40u LAMBDA=0.02 GAMMA=0 PHI=0.7)
.subckt gsyn inp inn wp wn outp outn vdd vbn
Mtail nt vbn 0 0 NNR W=400u L=100u
M5 na wp nt 0 NNR W=200u L=100u
M6 nb wn nt 0 NNR W=200u L=100u
M1 outp inp na 0 NNR W=200u L=100u
M2 outn inn na 0 NNR W=200u L=100u
M3 outn inp nb 0 NNR W=200u L=100u
M4 outp inn nb 0 NNR W=200u L=100u
.ends
.subckt cmld p n vdd
Rcs1 p cmx {RCS}
Rcs2 n cmx {RCS}
MLp p cmx vdd vdd PNR W={WL} L=100u
MLn n cmx vdd vdd PNR W={WL} L=100u
.ends
.subckt dneuron up un ap an vdd vbn
M1 an up tn 0 NNR W=200u L=100u
M2 ap un tn 0 NNR W=200u L=100u
Mt tn vbn 0 0 NNR W=200u L=100u
Xcm ap an vdd cmld
.ends"""
def wgv(v): return float(np.clip(VW0+KMAP*v,0.3,0.9)),float(np.clip(VW0-KMAP*v,0.3,0.9))
def build():
    L=["diag",SUB,"Vdd vdd 0 1.0",f"Vbsyn vbsyn 0 {VBSYN}",f"Vbneu vbneu 0 {VBNEU}"]
    for i in range(N): L+=[f"Vxp{i} a_xp{i} 0 0.5",f"Vxn{i} a_xn{i} 0 0.5"]
    for j in range(H):
        for i in range(N): gp,gn=wgv(WIH[j,i]); L+=[f"Vwp_ih_{i}_{j} wp_ih_{i}_{j} 0 {gp}",f"Vwn_ih_{i}_{j} wn_ih_{i}_{j} 0 {gn}"]
        for c in range(C): gp,gn=wgv(WHO[c,j]); L+=[f"Vwp_ho_{j}_{c} wp_ho_{j}_{c} 0 {gp}",f"Vwn_ho_{j}_{c} wn_ho_{j}_{c} 0 {gn}"]
    for j in range(H):
        xp,xn=f"xhp{j}",f"xhn{j}"; L+=[f"Xcm_xh{j} {xp} {xn} vdd cmld",f"Rnd_xh{j} {xp} {xn} {RNODE}"]
        for i in range(N): L+=[f"X_ihx_{i}_{j} a_xp{i} a_xn{i} wp_ih_{i}_{j} wn_ih_{i}_{j} {xp} {xn} vdd vbsyn gsyn"]
        L+=[f"Xn{j} {xp} {xn} ahp{j} ahn{j} vdd vbneu dneuron"]
    for c in range(C):
        L+=[f"Xcm_mo{c} mop{c} mon{c} vdd cmld",f"Rnd_mo{c} mop{c} mon{c} {RNODE}"]
        for j in range(H): L+=[f"X_ho_{j}_{c} ahp{j} ahn{j} wp_ho_{j}_{c} wn_ho_{j}_{c} mop{c} mon{c} vdd vbsyn gsyn"]
    L+=[".options gmin=1e-10",".end"]; return "\n".join(L)+"\n"
def inp_alters(x):
    A=[]
    for i in range(N): A+=[f"alter Vxp{i} = {0.5+IND*float(x[i]):.4f}",f"alter Vxn{i} = {0.5-IND*float(x[i]):.4f}"]
    return A
RD=sum([[f"mop{c}",f"mon{c}"] for c in range(C)],[])
WIH=WIHc; WHO=WHOc  # use the circuit-representable weights for the analog eval
L=Live(build(),f"diag_{TAG}.cir"); ok=0
for e in range(len(Xte)):
    v=L.step(inp_alters(Xte[e]),"op",RD,fout=f"diag_{TAG}.dat")
    mo=SO*np.array([v[2*c]-v[2*c+1] for c in range(C)]); ok+=(int(np.argmax(mo))==yte[e])
L.close()
print(f"(c) ngspice analog forward      TEST={ok/len(Xte):.2f}   <- (b)->(c) = analog offset/neuron loss")
