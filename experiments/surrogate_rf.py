#!/usr/bin/env python3
# NUMPY SCREENING SURROGATE (NOT ngspice) for the 16->9->C receptive-field net. Purpose: triage
# activation x backward-gate combinations fast, then verify the winners in real ngspice.
# Models the analog trainer's structure: batch-1 online updates, linear output keystone, and the
# transpose-read backward delta = W2^T @ (y-t) (which the circuit computes faithfully). The knob
# under test is the HIDDEN-UNIT DERIVATIVE GATE: 'true' = exact act'(z); 'step' = hard ReLU'
# (what the comparator does now); 'act' = gate by the activation magnitude (a buildable node).
import numpy as np, sys, os
INLO,INHI=0.3,1.7
def band(X): return INLO+X*(INHI-INLO)               # z-scored [0,1] -> input voltage band
tr=np.load("digits_train.npz"); te=np.load("digits_test.npz")
Xtr=band(tr["X"]); Ytr=tr["Y"]; Xte=band(te["X"]); Yte=te["Y"]
C=int(Ytr.max())+1; D=Xtr.shape[1]; H=9
fields=[[4*pr+pc, 4*pr+pc+1, 4*pr+pc+4, 4*pr+pc+5] for pr in range(3) for pc in range(3)]
mask=np.zeros((H,D))
for j,f in enumerate(fields): mask[j,f]=1.0

ACT=os.environ.get("ACT","relu2"); GATE=os.environ.get("GATE","step")
def act(z):
    if ACT=="relu2": r=np.maximum(z,0); return r*r
    if ACT=="relu":  return np.maximum(z,0)
    if ACT=="relu6": return np.clip(z,0,6)
    if ACT=="tanh":  return np.tanh(z)
def dact_true(z,h):
    if ACT=="relu2": return 2*np.maximum(z,0)
    if ACT=="relu":  return (z>0).astype(float)
    if ACT=="relu6": return ((z>0)&(z<6)).astype(float)
    if ACT=="tanh":  return 1-h*h
def gate(z,h):
    if GATE=="true": return dact_true(z,h)
    if GATE=="step": return (z>0).astype(float)              # hard ReLU' (current comparator)
    if GATE=="act":  return np.maximum(h,0)                  # gate by activation magnitude
    if GATE=="lin":  return np.maximum(z,0)                  # gate by relu(z) (=exact for relu2 up to 2x)
    if GATE=="ones": return np.ones_like(z)                  # NO derivative gate (circuit's saturated uon)
    if GATE=="stepC":return (z>float(os.environ.get("GTHR","1.0"))).astype(float)  # step at offset thr

rng=np.random.default_rng(int(os.environ.get("SEED","1")))
W1=rng.uniform(-0.8,0.8,(H,D))*mask; b1=np.full(H,0.6)
W2=rng.uniform(-0.05,0.05,(C,H)); b2=np.zeros(C)
lr1=float(os.environ.get("LR1","0.02")); lr2=float(os.environ.get("LR2","0.02"))
EP=int(os.environ.get("EP","40"))
THI=float(os.environ.get("THI","1.0")); TLO=float(os.environ.get("TLO","0.0"))
# circuit non-idealities: YSQUASH compresses the output keystone into [YLO,YHI] (measured ~0.1-0.6);
# SUBMEAN subtracts the common-mode error across classes (the COMP=subtractive competition).
YSQUASH=int(os.environ.get("YSQUASH","0")); YLO=float(os.environ.get("YLO","0.1")); YHI=float(os.environ.get("YHI","0.6"))
SUBMEAN=int(os.environ.get("SUBMEAN","0"))
def outy(raw):
    if YSQUASH: return 0.5*(YLO+YHI)+0.5*(YHI-YLO)*np.tanh(raw)
    return raw
n=len(Xtr); idx=np.arange(n)
for ep in range(EP):
    rng.shuffle(idx)
    for k in idx:
        x=Xtr[k]; t=np.full(C,TLO); t[Ytr[k]]=THI
        z=W1@x+b1; h=act(z); y=outy(W2@h+b2)
        e=y-t                                   # output error
        if SUBMEAN: e=e-e.mean()                # remove common-mode error (competition)
        W2-=lr2*np.outer(e,h); b2-=lr2*e
        delta=(W2.T@e)*gate(z,h)                # transpose-read backward x derivative gate
        W1-=lr1*np.outer(delta,x)*mask; b1-=lr1*delta
def acc(X,Y):
    Z=X@W1.T+b1; Hh=act(Z); Yp=outy(Hh@W2.T+b2); p=Yp.argmax(1)
    perc=[round(100*(p[Y==c]==c).mean()) for c in range(C)]
    return (p==Y).mean(), perc
a,perc=acc(Xte,Yte)
print(f"SURROGATE ACT={ACT:5s} GATE={GATE:4s} acc={100*a:5.1f}% per-class={perc}")
