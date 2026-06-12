#!/usr/bin/env python3
# REFINEMENT 2 (robustness): tanh depth-net is a seed lottery (min->chance). Test principled
# stabilizers: orthogonal init at the tanh edge-of-chaos gain (dynamical isometry) and residual
# skips (just wires in analog). Goal: lift the min off chance + shrink std, keeping it simple.
import numpy as np, time
from mc_experiment import scale_fit, scale_apply, onehot
from neurons import make_checker

def ortho(a,b,gain,rng):
    M=rng.standard_normal((max(a,b),max(a,b))); Q,_=np.linalg.qr(M); return gain*Q[:a,:b]

def train(nh,Xtr,Ytr,Xte,Yte,seed,ep=900,lr=5e-3,init='gauss',resid=False,gain=1.1):
    sizes=[6]+[8]*nh+[2]; rng=np.random.default_rng(seed); C=2; T=onehot(Ytr,C); B=len(Xtr)
    Ws=[];
    for a,b in zip(sizes[:-1],sizes[1:]):
        Ws.append(ortho(a,b,gain,rng) if init=='ortho' else rng.normal(0,np.sqrt(1.0/a),(a,b)))
    bs=[np.zeros(b) for b in sizes[1:]]
    mW=[np.zeros_like(w) for w in Ws]; vW=[np.zeros_like(w) for w in Ws]
    mb=[np.zeros_like(b) for b in bs]; vb=[np.zeros_like(b) for b in bs]
    def fwd(X):
        acts=[X]; zs=[]; A=X
        for l,(W,b) in enumerate(zip(Ws,bs)):
            Z=A@W+b; zs.append(Z); last=(l==len(Ws)-1)
            if last: A=Z
            else:
                h=np.tanh(Z); A=h+A if (resid and A.shape[1]==h.shape[1]) else h
            acts.append(A)
        return acts,zs
    def smax(Z): e=np.exp(Z-Z.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
    t=0
    for _ in range(ep):
        acts,zs=fwd(Xtr); d=(smax(acts[-1])-T)/B; t+=1; gW=[None]*len(Ws); gb=[None]*len(Ws)
        for l in reversed(range(len(Ws))):
            gW[l]=acts[l].T@d; gb[l]=d.sum(0)
            if l>0:
                da=d@Ws[l].T
                dz=da*(1-np.tanh(zs[l-1])**2)
                d=dz+da if (resid and acts[l].shape[1]==acts[l].shape[1] and l<len(Ws)-1 and l>0) else dz
                # residual backward: gradient also flows through the skip (identity)
                if resid and l<len(Ws)-1: d=dz+da
                else: d=dz
        for l in range(len(Ws)):
            for (m,v,g,P_) in [(mW[l],vW[l],gW[l],Ws[l]),(mb[l],vb[l],gb[l],bs[l])]:
                m[...]=0.9*m+0.1*g; v[...]=0.999*v+0.001*g*g
                P_-=lr*(m/(1-0.9**t))/(np.sqrt(v/(1-0.999**t))+1e-8)
    pr=lambda X: np.argmax(fwd(X)[0][-1],1)
    return (pr(Xte)==Yte).mean()

if __name__=="__main__":
    t0=time.time()
    Xtr,Ytr=make_checker(900,1,4); Xte,Yte=make_checker(500,2,4)
    p=scale_fit(Xtr); Xtr=scale_apply(Xtr,p); Xte=scale_apply(Xte,p)
    print("K=4, deep tanh (5 hidden, width 8), 10 seeds — robustness of stabilizers:")
    for tag,kw in [("baseline (gauss init)",{}),
                   ("orthogonal init",{'init':'ortho'}),
                   ("residual skips",{'resid':True}),
                   ("ortho + residual",{'init':'ortho','resid':True})]:
        a=np.array([train(5,Xtr,Ytr,Xte,Yte,s,**kw) for s in range(10)])
        print(f"  {tag:22s} mean {a.mean()*100:4.0f}  std {a.std()*100:4.1f}  min {a.min()*100:3.0f}  max {a.max()*100:3.0f}  [{time.time()-t0:.0f}s]",flush=True)
