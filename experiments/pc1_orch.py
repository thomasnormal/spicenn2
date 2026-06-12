#!/usr/bin/env python3
"""SINGLE-LAYER 8x8 in-circuit PREDICTIVE-CODING learning, the LOCAL delta rule.
Inference is the analog forward (gsyn synapses -> output column mu_o, read faithfully -- single stage,
no compounding, ~lossless vs numpy).  Learning: output error eps_c = t_c - mu_c (one-vs-rest +-1 target,
a local esub error neuron), weight update dW[c,i] = lr * eps_c * x_i -- the product of the TWO TERMINAL
voltages of weight (c,i): output-error node and input node.  Single layer => this local product IS the
exact gradient (no backprop, no hidden delta).  gen_pc1 proved the analog esub+gprod do this physically
on OR/AND; here the controller forms the same local product from the circuit's analog mu_o and the input.
Goal: train C=5 (and C=10) digits to ~90% IN-CIRCUIT with the local rule."""
import os, numpy as np
from ng_live import Live
from sklearn.datasets import load_digits
C=int(os.environ.get("C","5")); SEED=int(os.environ.get("SEED","0")); EP=int(os.environ.get("EP","40"))
NIN=int(os.environ.get("NIN","64")); LR=float(os.environ.get("LR","0.04")); MB=int(os.environ.get("MB","16"))
IND=float(os.environ.get("IND","0.30")); KMAP=float(os.environ.get("KMAP","0.30")); VW0=0.5
WL=os.environ.get("WL","1000u"); RCS=os.environ.get("RCS","300k"); RNODE=os.environ.get("RNODE","20k")
VBSYN=os.environ.get("VBSYN","0.45"); WCLIP=float(os.environ.get("WCLIP","1.33")); ANNEAL=int(os.environ.get("ANNEAL","1"))
SO=float(os.environ.get("SO","-1")); TG=float(os.environ.get("TG","8")); TAG=os.environ.get("RUNTAG",str(os.getpid()))
# SO = sign of the synapse (mu_o read); TG = readout gain mapping mu_o(volts ~mV) -> target scale ~[-1,1]
f=np.tanh; d=load_digits()
X=d.images.reshape(-1,64) if NIN==64 else d.images.reshape(-1,4,2,4,2).mean(axis=(2,4)).reshape(-1,16)
y=d.target; N=X.shape[1]; m=X.mean(0); s=X.std(0)+1e-6; X=np.clip((X-m)/s,-3,3)/3.0
rng=np.random.default_rng(SEED); idx=np.concatenate([np.where(y==c)[0] for c in range(C)])
Xc=X[idx]; yc=y[idx]; p=rng.permutation(len(yc)); Xc,yc=Xc[p],yc[p]
ntr=min(int(0.6*len(yc)), int(os.environ.get("NTR","999999"))); Xtr,ytr,Xte,yte=Xc[:ntr],yc[:ntr],Xc[ntr:],yc[ntr:]
W=rng.standard_normal((C,N))*0.05
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
    L=["pc1",SUB,"Vdd vdd 0 1.0",f"Vbsyn vbsyn 0 {VBSYN}"]
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
def wt_alters():
    A=[]
    for c in range(C):
        for i in range(N): gp,gn=wgv(W[c,i]); A+=[f"alter Vwp_{i}_{c} = {gp:.4f}",f"alter Vwn_{i}_{c} = {gn:.4f}"]
    return A
RD=sum([[f"mop{c}",f"mon{c}"] for c in range(C)],[])
def readmu(L,x):
    v=L.step(inp_alters(x),"op",RD,fout=f"pc1_{TAG}.dat")
    if v is None: return None   # unstable .op solve -> caller skips this example
    return SO*np.array([v[2*c]-v[2*c+1] for c in range(C)])
if __name__=="__main__":
    L=Live(build(),f"pc1_{TAG}.cir")
    NEVAL=int(os.environ.get("NEVAL","0")); EVERY=int(os.environ.get("EVERY","5"))
    ne=len(Xte) if NEVAL<=0 else min(NEVAL,len(Xte))
    def evaluate():
        ok=0; n=0
        for e in range(ne):
            mu=readmu(L,Xte[e])
            if mu is None: continue
            n+=1; ok+=(int(np.argmax(mu))==yte[e])
        return ok/max(1,n), n
    print(f"single-layer in-circuit PC (local delta): C={C} N={N} weights={C*N}")
    _a,_n=evaluate(); print(f"  epoch 0 test={_a:.3f} (chance {1/C:.2f}) [evaluated {_n}/{ne}]")
    for ep in range(1,EP+1):
        lr=LR*(0.1+0.9*(1-ep/EP)) if ANNEAL else LR
        dW=np.zeros_like(W); nb=0
        for k in rng.permutation(len(Xtr)):
            x=Xtr[k]; mu=readmu(L,x)                          # analog forward
            if mu is None: continue                          # skip unstable-solve example
            mu=mu*TG                                          # scaled to target range
            t=np.array([1.0 if c==ytr[k] else -1.0 for c in range(C)])
            eps=t-mu                                          # local output error (esub error neuron)
            dW+=np.outer(eps,x); nb+=1                        # local product eps_c * x_i
            if nb>=MB:
                W[:]=np.clip(W+lr*dW/nb,-WCLIP,WCLIP); L._burst(wt_alters()); dW[:]=0; nb=0
        if nb>0: W[:]=np.clip(W+lr*dW/nb,-WCLIP,WCLIP); L._burst(wt_alters())
        if ep%EVERY==0 or ep==EP:
            _a,_n=evaluate(); print(f"  epoch {ep:2d}: test={_a:.3f} [{_n}/{ne}] lr={lr:.3f}")
    L.close()
