#!/usr/bin/env python3
"""SINGLE-LAYER 8x8 forward ceiling: input(64 diff) -> gsyn synapses -> output column (C) -> argmax(mu_o).
No hidden layer => NO cross-layer compounding (the killer for the 2-layer net).  argmax(tanh(z))=argmax(z)
so the neuron is irrelevant for inference -> read mu_o directly.  Train W in numpy (one-vs-rest tanh
target), eval the SAME weights in ngspice.  numpy linear ceiling: C=5~0.96, C=10~0.89 at 8x8."""
import os, numpy as np
from ng_live import Live
from sklearn.datasets import load_digits
C=int(os.environ.get("C","5")); SEED=int(os.environ.get("SEED","0")); EP=int(os.environ.get("EP","300"))
NIN=int(os.environ.get("NIN","64")); LR=float(os.environ.get("LR","0.05"))
IND=float(os.environ.get("IND","0.30")); KMAP=float(os.environ.get("KMAP","0.30")); VW0=0.5
WL=os.environ.get("WL","1000u"); RCS=os.environ.get("RCS","300k"); RNODE=os.environ.get("RNODE","20k")
VBSYN=os.environ.get("VBSYN","0.45"); SO=float(os.environ.get("SO","-1")); TAG=os.environ.get("RUNTAG",str(os.getpid()))
WRANGE=(0.9-VW0)/KMAP
f=np.tanh; d=load_digits()
if NIN==64: X=d.images.reshape(-1,64)
else: X=d.images.reshape(-1,4,2,4,2).mean(axis=(2,4)).reshape(-1,16)
y=d.target; N=X.shape[1]; m=X.mean(0); s=X.std(0)+1e-6; X=np.clip((X-m)/s,-3,3)/3.0
rng=np.random.default_rng(SEED); idx=np.concatenate([np.where(y==c)[0] for c in range(C)])
Xc=X[idx]; yc=y[idx]; p=rng.permutation(len(yc)); Xc,yc=Xc[p],yc[p]
ntr=int(0.6*len(yc)); Xtr,ytr,Xte,yte=Xc[:ntr],yc[:ntr],Xc[ntr:],yc[ntr:]
W=np.zeros((C,N))
for ep in range(EP):
    for k in rng.permutation(len(ytr)):
        t=-np.ones(C); t[ytr[k]]=1; o=f(W@Xtr[k]); W+=LR*np.outer((t-o)*(1-o**2),Xtr[k])
accN=np.mean([np.argmax(W@Xte[i])==yte[i] for i in range(len(yte))])
W=np.clip(W,-WRANGE,WRANGE); accB=np.mean([np.argmax(W@Xte[i])==yte[i] for i in range(len(yte))])
print(f"[1layer] C={C} N={N} range=±{WRANGE:.2f} weights={C*N}  (a)numpy={accN:.3f} (b)clipped={accB:.3f}")
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
.ends"""
def wgv(v): return float(np.clip(VW0+KMAP*v,0.3,0.9)),float(np.clip(VW0-KMAP*v,0.3,0.9))
def build():
    L=["1layer",SUB,"Vdd vdd 0 1.0",f"Vbsyn vbsyn 0 {VBSYN}"]
    for i in range(N): L+=[f"Vxp{i} a_xp{i} 0 0.5",f"Vxn{i} a_xn{i} 0 0.5"]
    for c in range(C):
        for i in range(N): gp,gn=wgv(W[c,i]); L+=[f"Vwp_{i}_{c} wp_{i}_{c} 0 {gp}",f"Vwn_{i}_{c} wn_{i}_{c} 0 {gn}"]
    for c in range(C):
        L+=[f"Xcm_mo{c} mop{c} mon{c} vdd cmld",f"Rnd_mo{c} mop{c} mon{c} {RNODE}"]
        for i in range(N): L+=[f"X_{i}_{c} a_xp{i} a_xn{i} wp_{i}_{c} wn_{i}_{c} mop{c} mon{c} vdd vbsyn gsyn"]
    L+=[".options gmin=1e-10",".end"]; return "\n".join(L)+"\n"
def inp_alters(x):
    A=[]
    for i in range(N): A+=[f"alter Vxp{i} = {0.5+IND*float(x[i]):.4f}",f"alter Vxn{i} = {0.5-IND*float(x[i]):.4f}"]
    return A
RD=sum([[f"mop{c}",f"mon{c}"] for c in range(C)],[])
L=Live(build(),f"f1_{TAG}.cir"); ok=0
for e in range(len(Xte)):
    v=L.step(inp_alters(Xte[e]),"op",RD,fout=f"f1_{TAG}.dat")
    mo=SO*np.array([v[2*c]-v[2*c+1] for c in range(C)]); ok+=(int(np.argmax(mo))==yte[e])
L.close()
print(f"(c) ngspice SINGLE-LAYER forward C={C} N={N}  TEST={ok/len(Xte):.3f}   (2-layer gsyn was ~0.77avg)")
