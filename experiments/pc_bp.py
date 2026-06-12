#!/usr/bin/env python3
"""Analog FEEDFORWARD inference + clean controller-computed gradient, on sklearn digits.
The lean analog net (input->hidden synapse+neuron->output synapse, NO backward/error nodes) does the
forward pass in ngspice (.op) -- verified faithful (mu_o cos +0.96 vs numpy, and ngspice==Xyce).  The
controller reads a_h and mu_o and computes the 2-layer gradient (this is the HONEST fallback: analog
inference, digital learning -- NOT the local-analog PC, which is gradient-fidelity-limited at ~0.6).
Goal: see if the faithful analog forward is good enough to reach 90% on C>2 digits.
Few NMOS: only the feedforward network (no backward/error circuitry)."""
import os, numpy as np
from ng_live import Live
from sklearn.datasets import load_digits
N=16; C=int(os.environ.get("C","5")); H=int(os.environ.get("H","20")); SEED=int(os.environ.get("SEED","0"))
NTR=int(os.environ.get("NTR","40")); NTE=int(os.environ.get("NTE","40")); EP=int(os.environ.get("EP","30"))
LR=float(os.environ.get("LR","0.1")); IND=float(os.environ.get("IND","0.30"))
KMAP=float(os.environ.get("KMAP","0.30")); VW0=float(os.environ.get("VW0","0.5"))
WL=os.environ.get("WL","1000u"); RCS=os.environ.get("RCS","300k"); RNODE=os.environ.get("RNODE","20k")
VBSYN=os.environ.get("VBSYN","0.45"); VBNEU=os.environ.get("VBNEU","0.35")
MB=int(os.environ.get("MB","20")); WCLIP=float(os.environ.get("WCLIP","3")); ANNEAL=int(os.environ.get("ANNEAL","1"))
SA=float(os.environ.get("SA","-1")); SO=float(os.environ.get("SO","1"))   # sign of a_h, mu_o (empirical)
TAG=os.environ.get("RUNTAG",str(os.getpid()))
d=load_digits(); X=d.images.reshape(-1,4,2,4,2).mean(axis=(2,4)).reshape(-1,16); y=d.target
m=X.mean(0); s=X.std(0)+1e-6; X=np.clip((X-m)/s,-3,3)/3.0
rng=np.random.default_rng(SEED); CLS=list(range(C)); Xtr=[];ytr=[];Xte=[];yte=[]
for ci,c in enumerate(CLS):
    idx=np.where(y==c)[0]; rng.shuffle(idx)
    for i in idx[:NTR]: Xtr.append(X[i]); ytr.append(ci)
    for i in idx[NTR:NTR+NTE]: Xte.append(X[i]); yte.append(ci)
Xtr=np.array(Xtr); ytr=np.array(ytr); Xte=np.array(Xte); yte=np.array(yte)
WIH=rng.standard_normal((H,N))*0.4; WHO=rng.standard_normal((C,H))*0.4
def wgv(v): return float(np.clip(VW0+KMAP*v,0.3,0.9)),float(np.clip(VW0-KMAP*v,0.3,0.9))
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
def build():
    L=["pc feedforward",SUB,"Vdd vdd 0 1.0",f"Vbsyn vbsyn 0 {VBSYN}",f"Vbneu vbneu 0 {VBNEU}"]
    for i in range(N): L+=[f"Vxp{i} a_xp{i} 0 0.5",f"Vxn{i} a_xn{i} 0 0.5"]
    for j in range(H):
        for i in range(N): gp,gn=wgv(WIH[j,i]); L+=[f"Vwp_ih_{i}_{j} wp_ih_{i}_{j} 0 {gp}",f"Vwn_ih_{i}_{j} wn_ih_{i}_{j} 0 {gn}"]
        for c in range(C): gp,gn=wgv(WHO[c,j]); L+=[f"Vwp_ho_{j}_{c} wp_ho_{j}_{c} 0 {gp}",f"Vwn_ho_{j}_{c} wn_ho_{j}_{c} 0 {gn}"]
    for j in range(H):
        xp,xn=f"xhp{j}",f"xhn{j}"
        L+=[f"Xcm_xh{j} {xp} {xn} vdd cmld",f"Rnd_xh{j} {xp} {xn} {RNODE}"]
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
def wt_alters():
    A=[]
    for j in range(H):
        for i in range(N): gp,gn=wgv(WIH[j,i]); A+=[f"alter Vwp_ih_{i}_{j} = {gp:.4f}",f"alter Vwn_ih_{i}_{j} = {gn:.4f}"]
        for c in range(C): gp,gn=wgv(WHO[c,j]); A+=[f"alter Vwp_ho_{j}_{c} = {gp:.4f}",f"alter Vwn_ho_{j}_{c} = {gn:.4f}"]
    return A
RD=sum([[f"ahp{j}",f"ahn{j}"] for j in range(H)],[])+sum([[f"mop{c}",f"mon{c}"] for c in range(C)],[])
def read(L,x):
    v=L.step(inp_alters(x),"op",RD,fout=f"pcbp_{TAG}.dat")
    ah=SA*np.array([v[2*j]-v[2*j+1] for j in range(H)]); mo=SO*np.array([v[2*H+2*c]-v[2*H+2*c+1] for c in range(C)])
    return ah,mo
if __name__=="__main__":
    L=Live(build(),f"pcbp_{TAG}.cir")
    def evaluate():
        ok=0
        for e in range(len(Xte)):
            _,mo=read(L,Xte[e]); ok+=(int(np.argmax(mo))==yte[e])
        return ok/len(Xte)
    print(f"analog-fwd + controller-grad: C={C} H={H} weights={H*N+C*H}")
    print(f"  epoch 0 test={evaluate():.2f} (chance {1/C:.2f})")
    for ep in range(1,EP+1):
        lr=LR*(0.1+0.9*(1-ep/EP)) if ANNEAL else LR
        dIH=np.zeros_like(WIH); dHO=np.zeros_like(WHO); nb=0
        for k in rng.permutation(len(Xtr)):
            x=Xtr[k]; lab=ytr[k]; ah,mo=read(L,x)
            t=np.array([1.0 if c==lab else -1.0 for c in range(C)])
            do=t-mo                              # output error (backprop delta_o)
            dHO+=np.outer(do,ah)
            dh=(1-ah**2)*(WHO.T@do)              # backprop into hidden, gated by f'=1-a_h^2
            dIH+=np.outer(dh,x); nb+=1
            if nb>=MB:
                WHO[:]=np.clip(WHO+lr*dHO/nb,-WCLIP,WCLIP); WIH[:]=np.clip(WIH+lr*dIH/nb,-WCLIP,WCLIP)
                L._burst(wt_alters()); dIH[:]=0; dHO[:]=0; nb=0
        if nb>0:
            WHO[:]=np.clip(WHO+lr*dHO/nb,-WCLIP,WCLIP); WIH[:]=np.clip(WIH+lr*dIH/nb,-WCLIP,WCLIP); L._burst(wt_alters())
        if ep%5==0 or ep==EP: print(f"  epoch {ep:2d}: test={evaluate():.2f}  lr={lr:.3f}")
    L.close()
