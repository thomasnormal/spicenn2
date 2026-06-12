#!/usr/bin/env python3
"""Per-layer fidelity probe: at numpy-optimal weights, read a_h and mu_o from the circuit and cosine
them against the ideal tanh(W1 x) and W2 tanh(W1 x).  Also fit the best linear gain+offset per layer
(so a pure scale error doesn't masquerade as distortion).  Pinpoints WHERE forward fidelity is lost."""
import os, numpy as np
from ng_live import Live
from sklearn.datasets import load_digits
NIN=int(os.environ.get("NIN","16")); N=NIN; C=int(os.environ.get("C","5")); H=int(os.environ.get("H","20")); SEED=int(os.environ.get("SEED","0"))
NTR=int(os.environ.get("NTR","40")); NTE=int(os.environ.get("NTE","30")); EP=int(os.environ.get("EP","200"))
LR=0.02; LRX=0.3; NREL=20; IND=float(os.environ.get("IND","0.30")); KMAP=float(os.environ.get("KMAP","0.30")); VW0=0.5
WL=os.environ.get("WL","1000u"); RCS=os.environ.get("RCS","300k"); RNODE=os.environ.get("RNODE","20k")
VBSYN=os.environ.get("VBSYN","0.45"); VBNEU=os.environ.get("VBNEU","0.35"); TAG=os.environ.get("RUNTAG",str(os.getpid()))
WRANGE=(0.9-VW0)/KMAP
f=np.tanh; d=load_digits(); X=(d.images.reshape(-1,64) if NIN==64 else d.images.reshape(-1,4,2,4,2).mean(axis=(2,4)).reshape(-1,16)); y=d.target
m=X.mean(0); s=X.std(0)+1e-6; X=np.clip((X-m)/s,-3,3)/3.0
rng=np.random.default_rng(SEED); CLS=list(range(C)); Xtr=[];ytr=[];Xte=[];yte=[]
for ci,c in enumerate(CLS):
    idx=np.where(y==c)[0]; rng.shuffle(idx)
    for i in idx[:NTR]: Xtr.append(X[i]); ytr.append(ci)
    for i in idx[NTR:NTR+NTE]: Xte.append(X[i]); yte.append(ci)
Xtr=np.array(Xtr); ytr=np.array(ytr); Xte=np.array(Xte); yte=np.array(yte)
WS=float(os.environ.get("WS","0.3")); RAND=int(os.environ.get("RAND","0"))
WIH=rng.standard_normal((H,N))*WS; WHO=rng.standard_normal((C,H))*WS
def target(lab): t=np.full(C,-1.0); t[lab]=1.0; return t
for ep in range(0 if RAND else EP):
    for k in rng.permutation(len(Xtr)):
        x=Xtr[k]; x1=WIH@x
        for _ in range(NREL):
            e1=x1-WIH@x; e2=target(ytr[k])-WHO@f(x1); g=1-f(x1)**2
            x1=x1+LRX*(-e1+g*(WHO.T@e2))
        e1=x1-WIH@x; e2=target(ytr[k])-WHO@f(x1)
        WHO+=LR*np.outer(e2,f(x1)); WIH+=LR*np.outer(e1,x)
WIH=np.clip(WIH,-WRANGE,WRANGE); WHO=np.clip(WHO,-WRANGE,WRANGE)
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
RDh=sum([[f"xhp{j}",f"xhn{j}"] for j in range(H)],[])
RDa=sum([[f"ahp{j}",f"ahn{j}"] for j in range(H)],[])
RDo=sum([[f"mop{c}",f"mon{c}"] for c in range(C)],[])
RD=RDh+RDa+RDo
L=Live(build(),f"diaglay_{TAG}.cir")
HV=[];AV=[];OV=[];NETV=[];AIDEAL=[];OIDEAL=[];NETIDEAL=[]
for e in range(len(Xte)):
    x=Xte[e]; v=L.step(inp_alters(x),"op",RD,fout=f"diaglay_{TAG}.dat")
    hv=np.array([v[2*j]-v[2*j+1] for j in range(H)])            # pre-neuron node (synapse-sum)
    av=np.array([v[2*H+2*j]-v[2*H+2*j+1] for j in range(H)])    # post-neuron a_h
    ov=np.array([v[4*H+2*c]-v[4*H+2*c+1] for c in range(C)])    # mu_o
    HV.append(hv); AV.append(av); OV.append(ov)
    NETIDEAL.append(WIH@x); AIDEAL.append(f(WIH@x)); OIDEAL.append(WHO@f(WIH@x))
L.close()
def cosfit(meas,ideal):
    M=np.array(meas).ravel(); I=np.array(ideal).ravel()
    cos=float(M@I/(np.linalg.norm(M)*np.linalg.norm(I)+1e-12))
    A=np.vstack([I,np.ones_like(I)]).T; g,b=np.linalg.lstsq(A,M,rcond=None)[0]
    resid=M-(g*I+b); r2=1-resid.var()/(M.var()+1e-12)
    return cos,float(g),float(b),float(r2)
print(f"per-layer fidelity  C={C} H={H} SEED={SEED}  (cos, gain, offset, R^2 vs best linear fit)")
print("  layer1 synapse-sum (h_pre vs WIH@x): cos=%.3f gain=%.4f off=%+.4f R2=%.3f"%cosfit(HV,NETIDEAL))
print("  layer1 neuron     (a_h vs tanh):     cos=%.3f gain=%.4f off=%+.4f R2=%.3f"%cosfit(AV,AIDEAL))
print("  layer2 output     (mu_o vs WHO@a):   cos=%.3f gain=%.4f off=%+.4f R2=%.3f"%cosfit(OV,OIDEAL))
# how much of the output error is just inherited from the hidden distortion vs added at layer2?
av_arr=np.array(AV); o_from_realA=av_arr@WHO.T
print("  layer2 alone (mu_o vs WHO@a_h_REAL): cos=%.3f gain=%.4f off=%+.4f R2=%.3f"%cosfit(OV,o_from_realA))
