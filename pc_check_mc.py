#!/usr/bin/env python3
"""ALGORITHM SEARCH (numpy proxy, NOT spice): multi-class Predictive Coding on sklearn digits,
matching the circuit's structure (one-vs-rest +-TD targets, tanh neuron, continuous-ish update),
to find WHAT makes C>2 train (the circuit collapses at C=3).  Knobs mirror circuit fixes:
 FP=1 keep f'(x) gating | COMP=1 output lateral inhibition (subtract mean) | DECAY weight decay
 ANNEAL lr decay | BWSCALE backward 1/C scaling | TNEG negative-target level.
Whatever recipe trains C>2 here gets ported to ngspice (gen_pcM)."""
import os, numpy as np
from sklearn.datasets import load_digits
C=int(os.environ.get("C","3")); H=int(os.environ.get("H","12")); NTR=int(os.environ.get("NTR","40"))
NTE=int(os.environ.get("NTE","40")); EP=int(os.environ.get("EP","200")); SEED=int(os.environ.get("SEED","0"))
LR=float(os.environ.get("LR","0.02")); LRX=float(os.environ.get("LRX","0.3")); NREL=int(os.environ.get("NREL","20"))
FP=int(os.environ.get("FP","1")); COMP=int(os.environ.get("COMP","0")); DECAY=float(os.environ.get("DECAY","0"))
ANNEAL=int(os.environ.get("ANNEAL","0")); BWSCALE=float(os.environ.get("BWSCALE","1")); TNEG=float(os.environ.get("TNEG","1"))
f=np.tanh; fp=lambda z:1-np.tanh(z)**2
d=load_digits(); X=d.images.reshape(-1,4,2,4,2).mean(axis=(2,4)).reshape(-1,16); y=d.target
m=X.mean(0); s=X.std(0)+1e-6; X=np.clip((X-m)/s,-3,3)/3.0
rng=np.random.default_rng(SEED); CLS=list(range(C))
def split():
    Xtr=[];ytr=[];Xte=[];yte=[]
    for ci,c in enumerate(CLS):
        idx=np.where(y==c)[0]; rng.shuffle(idx)
        for i in idx[:NTR]: Xtr.append(X[i]); ytr.append(ci)
        for i in idx[NTR:NTR+NTE]: Xte.append(X[i]); yte.append(ci)
    return np.array(Xtr),np.array(ytr),np.array(Xte),np.array(yte)
Xtr,ytr,Xte,yte=split()
W1=rng.standard_normal((H,16))*0.3; W2=rng.standard_normal((C,H))*0.3
def target(lab):
    t=np.full(C,-TNEG,float); t[lab]=1.0; return t
def infer(x,t=None,clamp=True):
    x1=f(W1@x)*0  # start hidden at 0
    x1=W1@x
    for _ in range(NREL):
        mu1=W1@x; e1=x1-mu1
        x2=t if (clamp and t is not None) else (W2@f(x1))
        mu2=W2@f(x1); e2=x2-mu2
        if COMP: e2=e2-e2.mean()      # lateral inhibition / subtractive normalization on output errors
        g=fp(x1) if FP else 1.0
        x1=x1+LRX*(-e1+g*(W2.T@e2)*BWSCALE)
    mu1=W1@x; e1=x1-mu1; mu2=W2@f(x1); e2=(t-mu2) if (clamp and t is not None) else 0
    if COMP and clamp: e2=e2-e2.mean()
    return x1,e1,e2
for ep in range(EP):
    lr=LR*(0.1+0.9*(1-ep/EP)) if ANNEAL else LR
    for k in rng.permutation(len(Xtr)):
        x1,e1,e2=infer(Xtr[k],target(ytr[k]),clamp=True)
        W2+=lr*np.outer(e2,f(x1))-DECAY*W2
        W1+=lr*np.outer(e1,Xtr[k])-DECAY*W1
def acc(Xs,ys):
    ok=0
    for x,lab in zip(Xs,ys):
        o=W2@f(W1@x); ok+=(int(np.argmax(o))==lab)
    return ok/len(Xs)
print(f"C={C} H={H} FP={FP} COMP={COMP} DECAY={DECAY} ANNEAL={ANNEAL} BWSCALE={BWSCALE} TNEG={TNEG}: train={acc(Xtr,ytr):.2f} TEST={acc(Xte,yte):.2f} (chance {1/C:.2f})")
