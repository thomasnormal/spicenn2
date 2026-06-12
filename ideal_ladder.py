#!/usr/bin/env python3
# Localize the depth wall. Ladder of forwards, all trained with their own EXACT gradient
# (real backprop) so we isolate REAL depth (composition), not a per-layer shortcut.
#   (1) ideal MLP   : standard ReLU, linear layers           -> what depth CAN do at this capacity
#   (2) analog-fwd  : keystone + ReLU^2 + dead-zone + clamp  -> cost of the analog forward neuron
# Both use exact backprop of their OWN forward (finite-diff-checked autograd by hand).
# Task: KxK checkerboard, capped width 8. Shallow vs deep. Does deep beat shallow?
import numpy as np, time
from mc_experiment import scale_fit, scale_apply, onehot
from surrogate import iDS, diode, VC, Gs, VREFH, RTH, VREFO, RTO, HI, VG0, VT_REL, BETA_REL, RH
clamp=2.6

def make_checker(n,seed,K,D=6):
    rng=np.random.default_rng(seed); uv=rng.uniform(0,1,(n,2))
    lab=((np.floor(K*uv[:,0])+np.floor(K*uv[:,1]))%2).astype(int)
    return np.hstack([uv,rng.uniform(0,1,(n,D-2))]),lab

# ---------- (1) ideal MLP with exact backprop ----------
def mlp_train(sizes,X,Y,Xte,Yte,seed,ep=600,lr=0.1):
    rng=np.random.default_rng(seed); C=sizes[-1]; T=onehot(Y,C)
    Ws=[rng.normal(0,np.sqrt(2/a),(a,b)) for a,b in zip(sizes[:-1],sizes[1:])]
    bs=[np.zeros(b) for b in sizes[1:]]
    def fwd(X):
        acts=[X]; A=X
        for l,(W,b) in enumerate(zip(Ws,bs)):
            Z=A@W+b; A=Z if l==len(Ws)-1 else np.maximum(Z,0); acts.append(A)
        return acts
    def softmax(Z): e=np.exp(Z-Z.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
    for _ in range(ep):
        acts=fwd(X); P=softmax(acts[-1]); d=(P-T)/len(X)
        for l in reversed(range(len(Ws))):
            gW=acts[l].T@d; gb=d.sum(0)
            if l>0: d=(d@Ws[l].T)*(acts[l]>0)
            Ws[l]-=lr*gW; bs[l]-=lr*gb
    pr=lambda X: np.argmax(fwd(X)[-1],1)
    return (pr(X)==Y).mean(),(pr(Xte)==Yte).mean()

if __name__=="__main__":
    t0=time.time()
    print("IDEAL MLP (standard ReLU, exact backprop), width 8 — does REAL depth beat shallow?")
    for K in (2,3,4):
        Xtr,Ytr=make_checker(400,1,K); Xte,Yte=make_checker(300,2,K)
        p=scale_fit(Xtr); Xtr=scale_apply(Xtr,p); Xte=scale_apply(Xte,p)
        sh=[mlp_train([6,8,2],Xtr,Ytr,Xte,Yte,s)[1] for s in range(4)]
        d2=[mlp_train([6,8,8,2],Xtr,Ytr,Xte,Yte,s)[1] for s in range(4)]
        d3=[mlp_train([6,8,8,8,2],Xtr,Ytr,Xte,Yte,s)[1] for s in range(4)]
        print(f"  {K}x{K}: shallow[8] {np.mean(sh)*100:3.0f}  deep[8,8] {np.mean(d2)*100:3.0f}  "
              f"deep[8,8,8] {np.mean(d3)*100:3.0f}   (best {max(d3)*100:3.0f})  [{time.time()-t0:.0f}s]",flush=True)
