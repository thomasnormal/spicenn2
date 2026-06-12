#!/usr/bin/env python3
# NUMPY SCREENING SURROGATE (not ngspice) to test the FAN-IN hypothesis and the 16->9->10->10
# sparse-random architecture for C=10. Models the analog KEYSTONE SATURATION: each neuron's output
# y = mid + amp*tanh(g * sum_in w*a). With high fan-in the summed current is large -> tanh saturates
# -> tiny per-class margin AND vanishing gradient (the C=10 collapse). Low fan-in (sparse) keeps the
# sum in the linear region. So sparse/low-fan-in should beat dense/high-fan-in HERE if the hypothesis
# holds. Connectivity per weight-layer: layer0 = rf3x3 (overlapping 2x2); later layers = sparse random
# with fan-in K. Trains all layers by backprop (transpose) with per-layer common-mode subtraction.
import numpy as np, os, sys
INLO,INHI=0.3,1.7
tr=np.load("digits_train.npz"); te=np.load("digits_test.npz")
Xtr=INLO+tr["X"]*(INHI-INLO); Ytr=tr["Y"]; Xte=INLO+te["X"]*(INHI-INLO); Yte=te["Y"]
C=int(Ytr.max())+1; D=16
ARCH=os.environ.get("ARCH","16,9,10")           # layer sizes; e.g. "16,9,10" or "16,9,10,10"
sizes=[int(x) for x in ARCH.split(",")]
K=int(os.environ.get("K","3"))                  # sparse fan-in for layers after layer0
G=float(os.environ.get("G","1.2"))              # keystone gain (sets how fast it saturates)
AMP=float(os.environ.get("AMP","0.45")); MID=float(os.environ.get("MID","0.0"))
CMSUB=int(os.environ.get("CMSUB","1"))
rng=np.random.default_rng(int(os.environ.get("SEED","1")))
# build connectivity masks per weight-layer
def rf3x3():
    m=np.zeros((9,16))
    F=[[4*pr+pc,4*pr+pc+1,4*pr+pc+4,4*pr+pc+5] for pr in range(3) for pc in range(3)]
    for j,f in enumerate(F): m[j,f]=1
    return m
masks=[]
for l in range(len(sizes)-1):
    nin,nout=sizes[l],sizes[l+1]
    if l==0 and nin==16 and nout==9: masks.append(rf3x3())
    else:
        m=np.zeros((nout,nin))
        for o in range(nout):
            sel=rng.choice(nin, size=min(K,nin), replace=False); m[o,sel]=1
        masks.append(m)
Ws=[rng.uniform(-0.8,0.8,m.shape)*m for m in masks]
bs=[rng.uniform(-0.2,0.2,m.shape[0]) for m in masks]
def fwd(X):
    acts=[X]; a=X
    for l,(W,b,m) in enumerate(zip(Ws,bs,masks)):
        z=a@W.T+b
        last=(l==len(Ws)-1)
        a = z if last else (MID+AMP*np.tanh(G*z))   # hidden = saturating keystone; output linear
        acts.append(a)
    return acts
lr=float(os.environ.get("LR","0.02")); EP=int(os.environ.get("EP","50"))
idx=np.arange(len(Xtr))
for ep in range(EP):
    rng.shuffle(idx)
    for k in idx:
        x=Xtr[k]; t=np.full(C,0.0); t[Ytr[k]]=1.0
        acts=fwd(x[None])[:]; acts=[a[0] for a in acts]
        e=acts[-1]-t
        if CMSUB: e=e-e.mean()
        # backprop
        delta=e
        for l in reversed(range(len(Ws))):
            a_prev=acts[l]; z = a_prev@Ws[l].T+bs[l]
            if l==len(Ws)-1: g=delta
            else:
                g=delta*(1-np.tanh(G*z)**2)*(AMP*G)         # tanh' of the keystone
                if CMSUB: g=g-g.mean()
            Ws[l]-=lr*np.outer(g,a_prev)*masks[l]; bs[l]-=lr*g
            delta=Ws[l].T@g
def acc(X,Y):
    p=fwd(X)[-1].argmax(1); perc=[round(100*(p[Y==c]==c).mean()) for c in range(C)]
    return (p==Y).mean(),perc
a,perc=acc(Xte,Yte)
fanins=[int(m.sum(1).max()) for m in masks]
print(f"ARCH={ARCH:>12s} K={K} fan-ins={fanins}  acc={100*a:5.1f}%  per-class={perc}")
