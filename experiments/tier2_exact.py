#!/usr/bin/env python3
# Does a SIGNAL-PRESERVING neuron (tanh) + an EXACT backward give analog depth-gain?
# If deep >> shallow here (unlike with the approximate backward), it confirms: with a good
# neuron the approximate transpose-read backward becomes the binding constraint for depth.
import numpy as np, time
from mc_experiment import onehot, scale_fit, scale_apply
from surrogate import iDS, diode, VC, Gs, VREFH, RTH, VREFO, RTO, HI
from deep_super import make_checker
LO,HIb=0.4,1.6
def solve_col(gp,X,it=22):
    B=X.shape[0];No=gp.shape[0];lo=np.full((B,No),0.5+1e-9);hi=np.full((B,No),3.0)
    for _ in range(it):
        m=0.5*(lo+hi);f=iDS(gp[None],X[:,None],m[:,:,None]).sum(2)-diode(m);hi=np.where(f<0,m,hi);lo=np.where(f>=0,m,lo)
    return diode(0.5*(lo+hi))
def keystone(gp,X,vref,R): return vref+0.64*R*(solve_col(gp,X)-solve_col(VC*np.ones((1,X.shape[1])),X))
s =lambda z: 0.5+0.5*np.tanh((z-0.8)/0.4)
ds=lambda z: 0.5*(1-np.tanh((z-0.8)/0.4)**2)/0.4
def divis(Y): r=np.clip(Y,0,None); return r/(r.sum(1,keepdims=True)+1e-6)
def fwd(Ws,X):
    acts=[X];A=X;L=len(Ws)
    for l,Wl in enumerate(Ws):
        AB=np.concatenate([A,np.full((len(A),1),HI)],1);last=(l==L-1)
        Z=keystone(VC+Gs*Wl,AB,(VREFO if last else VREFH),(RTO if last else RTH))
        A=Z if last else LO+(HIb-LO)*s(Z); acts.append(A)
    return acts
def approx_grad(Ws,X,e):
    acts=[X];zs=[];A=X;L=len(Ws)
    for l,Wl in enumerate(Ws):
        AB=np.concatenate([A,np.full((len(A),1),HI)],1);last=(l==L-1)
        Z=keystone(VC+Gs*Wl,AB,(VREFO if last else VREFH),(RTO if last else RTH));zs.append(Z)
        A=Z if last else LO+(HIb-LO)*s(Z);acts.append(A)
    gr=[None]*L;d=e;B=len(e)
    for l in reversed(range(L)):
        AB=np.concatenate([acts[l],np.full((B,1),HI)],1)
        dz=d if l==L-1 else d*((HIb-LO)*ds(zs[l]))
        gr[l]=(dz.T@AB)/B
        if l>0: d=dz@Ws[l][:,:-1]
    return gr
def exact_grad(Ws,X,e,eps=5e-3):
    def sof(W_): return (e*fwd(W_,X)[-1]).sum(1).mean()
    gr=[np.zeros_like(W) for W in Ws]
    for l in range(len(Ws)):
        for i in range(Ws[l].shape[0]):
            for j in range(Ws[l].shape[1]):
                Wp=[w.copy() for w in Ws];Wm=[w.copy() for w in Ws];Wp[l][i,j]+=eps;Wm[l][i,j]-=eps
                gr[l][i,j]=(sof(Wp)-sof(Wm))/(2*eps)
    return gr
def acc(Ws,X,Y): return (np.argmax(fwd(Ws,X)[-1],1)==Y).mean()
def train(sizes,X,Y,seed,which,ep):
    rng=np.random.default_rng(seed);T=onehot(Y,2)
    Ws=[np.clip(rng.uniform(-0.3,0.3,(b,a+1)),-1.5,1.0) for a,b in zip(sizes[:-1],sizes[1:])]
    V=[np.zeros_like(W) for W in Ws]
    for _ in range(ep):
        e=divis(fwd(Ws,X)[-1])-T
        g=exact_grad(Ws,X,e) if which=='exact' else approx_grad(Ws,X,e)
        for l in range(len(Ws)): V[l]=0.9*V[l]-0.4*g[l];Ws[l]=np.clip(Ws[l]+np.clip(V[l],-0.1,0.1),-1.5,1.0)
    return Ws

if __name__=="__main__":
    t0=time.time();K=3
    Xtr,Ytr=make_checker(120,1,K);Xte,Yte=make_checker(200,2,K)
    p=scale_fit(Xtr);Xtr=scale_apply(Xtr,p);Xte=scale_apply(Xte,p)
    print(f"ANALOG tanh forward, K={K} (chance 50): shallow vs deep, APPROX vs EXACT backward")
    for which,ep in [('approx',400),('exact',120)]:
        for seed in (0,1):
            sh=acc(train([6,6,2],Xtr,Ytr,seed,which,ep),Xte,Yte)
            dp=acc(train([6,6,6,6,2],Xtr,Ytr,seed,which,ep),Xte,Yte)
            print(f"  {which:6s} seed{seed}: shallow {sh*100:3.0f}  deep {dp*100:3.0f}  depth-gain {(dp-sh)*100:+3.0f}  [{time.time()-t0:.0f}s]",flush=True)
