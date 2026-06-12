#!/usr/bin/env python3
"""CROSSBAR-VMM forward fidelity test (vs the Gilbert gsyn, which caps C=5 at ~0.77 avg / 0.86 best,
highly seed-variable).  Same protocol as diag_fwd: train W in numpy (~0.92 ceiling), eval the SAME
weights in the ngspice analog forward -- but the synapse is now a 4-NMOS TRIODE crossbar cell:
  Ip-In = (G(wp)-G(wn))*(vxp-vxn),  G=triode conductance ~ linear in (Vgate-Vt).
Columns summed by KCL onto a resistive transimpedance to a CM reference; differential col voltage
-> dneuron.  Fewer NMOS (4 vs 7) AND more linear (triode is linear in Vds for small Vds; the diff
structure cancels the Vds^2/2 term) AND no tail-current CM headache.  Goal: forward ceiling > 0.86
and LOW seed variance."""
import os, numpy as np
from ng_live import Live
from sklearn.datasets import load_digits
N=16; C=int(os.environ.get("C","5")); H=int(os.environ.get("H","20")); SEED=int(os.environ.get("SEED","0"))
NTR=int(os.environ.get("NTR","40")); NTE=int(os.environ.get("NTE","40")); EP=int(os.environ.get("EP","200"))
LR=float(os.environ.get("LR","0.02")); LRX=float(os.environ.get("LRX","0.3")); NREL=int(os.environ.get("NREL","20"))
IND=float(os.environ.get("IND","0.30")); KMAP=float(os.environ.get("KMAP","0.075")); WCM=float(os.environ.get("WCM","0.80"))
RCOL=os.environ.get("RCOL","40k"); RNODE=os.environ.get("RNODE","20k"); WL=os.environ.get("WL","1000u"); RCS=os.environ.get("RCS","300k")
VBNEU=os.environ.get("VBNEU","0.35"); VREF=float(os.environ.get("VREF","0.5")); WBND=float(os.environ.get("WBND","0.10"))
SO=float(os.environ.get("SO","1")); SA=float(os.environ.get("SA","-1")); TAG=os.environ.get("RUNTAG",str(os.getpid()))
WRANGE=WBND/KMAP   # circuit-representable |weight| before wp/wn rail at WCM +- WBND
f=np.tanh; d=load_digits(); X=d.images.reshape(-1,4,2,4,2).mean(axis=(2,4)).reshape(-1,16); y=d.target
m=X.mean(0); s=X.std(0)+1e-6; X=np.clip((X-m)/s,-3,3)/3.0
rng=np.random.default_rng(SEED); CLS=list(range(C)); Xtr=[];ytr=[];Xte=[];yte=[]
for ci,c in enumerate(CLS):
    idx=np.where(y==c)[0]; rng.shuffle(idx)
    for i in idx[:NTR]: Xtr.append(X[i]); ytr.append(ci)
    for i in idx[NTR:NTR+NTE]: Xte.append(X[i]); yte.append(ci)
Xtr=np.array(Xtr); ytr=np.array(ytr); Xte=np.array(Xte); yte=np.array(yte)
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
WIHc=np.clip(WIH,-WRANGE,WRANGE); WHOc=np.clip(WHO,-WRANGE,WRANGE)
print(f"[xbar] range=±{WRANGE:.2f}  (a)numpy={npacc(WIH,WHO,Xte,yte):.2f}  (b)clipped={npacc(WIHc,WHOc,Xte,yte):.2f}")
WIH=WIHc; WHO=WHOc
SUB=f""".model NNR NMOS (LEVEL=1 VTO=0.2 KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)
.model PNR PMOS (LEVEL=1 VTO=-0.2 KP=40u LAMBDA=0.02 GAMMA=0 PHI=0.7)
.subckt xsyn vxp vxn wp wn colp coln
M1 colp wp vxp 0 NNR W=200u L=100u
M2 colp wn vxn 0 NNR W=200u L=100u
M3 coln wp vxn 0 NNR W=200u L=100u
M4 coln wn vxp 0 NNR W=200u L=100u
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
def wgv(v): return float(np.clip(WCM+KMAP*v,WCM-WBND,WCM+WBND)),float(np.clip(WCM-KMAP*v,WCM-WBND,WCM+WBND))
def build():
    L=["xbar",SUB,"Vdd vdd 0 1.0",f"Vbneu vbneu 0 {VBNEU}",f"Vref vref 0 {VREF}"]
    for i in range(N): L+=[f"Vxp{i} a_xp{i} 0 0.5",f"Vxn{i} a_xn{i} 0 0.5"]
    for j in range(H):
        for i in range(N): gp,gn=wgv(WIH[j,i]); L+=[f"Vwp_ih_{i}_{j} wp_ih_{i}_{j} 0 {gp}",f"Vwn_ih_{i}_{j} wn_ih_{i}_{j} 0 {gn}"]
        for c in range(C): gp,gn=wgv(WHO[c,j]); L+=[f"Vwp_ho_{j}_{c} wp_ho_{j}_{c} 0 {gp}",f"Vwn_ho_{j}_{c} wn_ho_{j}_{c} 0 {gn}"]
    # hidden columns: KCL sum onto colp/coln, resistive transimpedance to vref, then dneuron
    for j in range(H):
        cp,cn=f"hcp{j}",f"hcn{j}"
        L+=[f"Rtp_h{j} {cp} vref {RCOL}",f"Rtn_h{j} {cn} vref {RCOL}"]
        for i in range(N): L+=[f"X_ihx_{i}_{j} a_xp{i} a_xn{i} wp_ih_{i}_{j} wn_ih_{i}_{j} {cp} {cn} xsyn"]
        L+=[f"Xn{j} {cp} {cn} ahp{j} ahn{j} vdd vbneu dneuron"]
    for c in range(C):
        cp,cn=f"ocp{c}",f"ocn{c}"
        L+=[f"Rtp_o{c} {cp} vref {RCOL}",f"Rtn_o{c} {cn} vref {RCOL}"]
        for j in range(H): L+=[f"X_ho_{j}_{c} ahp{j} ahn{j} wp_ho_{j}_{c} wn_ho_{j}_{c} {cp} {cn} xsyn"]
        L+=[f"Rmo{c} {cp} {cn} {RNODE}"]
    L+=[".options gmin=1e-10",".end"]; return "\n".join(L)+"\n"
def inp_alters(x):
    A=[]
    for i in range(N): A+=[f"alter Vxp{i} = {0.5+IND*float(x[i]):.4f}",f"alter Vxn{i} = {0.5-IND*float(x[i]):.4f}"]
    return A
RD=sum([[f"ocp{c}",f"ocn{c}"] for c in range(C)],[])
L=Live(build(),f"xbar_{TAG}.cir"); ok=0
for e in range(len(Xte)):
    v=L.step(inp_alters(Xte[e]),"op",RD,fout=f"xbar_{TAG}.dat")
    mo=SO*np.array([v[2*c]-v[2*c+1] for c in range(C)]); ok+=(int(np.argmax(mo))==yte[e])
L.close()
print(f"(c) ngspice CROSSBAR forward    TEST={ok/len(Xte):.2f}   (gsyn baseline ~0.77avg/0.86best)")
