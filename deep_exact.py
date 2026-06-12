#!/usr/bin/env python3
# Decisive diagnosis: is the ~0.5-cosine gradient the depth bottleneck, or the lossy forward?
# Train deep analog net on 2x2 checkerboard with the APPROX (transpose-read) gradient to
# convergence, then continue with the EXACT (finite-diff) gradient of the same forward.
# If accuracy jumps -> gradient FIDELITY is the bottleneck (path: better backward / EqProp).
# If it stays stuck -> the lossy FORWARD is the bottleneck (path: better neuron).
import numpy as np, time
from mc_experiment import onehot, scale_fit, scale_apply
from surrogate import iDS, diode, VC, Gs, VREFH, RTH, VREFO, RTO, HI, VG0, VT_REL, BETA_REL, RH
from deep_super import make_checker
clamp=2.6
def solve_col(gp,X,iters=40):
    B=X.shape[0]; No=gp.shape[0]; lo=np.full((B,No),0.5+1e-9); hi=np.full((B,No),3.0)
    for _ in range(iters):
        mid=0.5*(lo+hi); f=iDS(gp[None,:,:],X[:,None,:],mid[:,:,None]).sum(2)-diode(mid)
        hi=np.where(f<0,mid,hi); lo=np.where(f>=0,mid,lo)
    return diode(0.5*(lo+hi))
def keystone(gp,X,vref,R): return vref+0.64*R*(solve_col(gp,X)-solve_col(VC*np.ones((1,X.shape[1])),X))
def relu_g(z): return np.clip(VG0+RH*0.5*BETA_REL*np.clip(z-VT_REL,0,None)**2,VG0,clamp)
def divis(Y): r=np.clip(Y,0,None); return r/(r.sum(1,keepdims=True)+1e-6)
def forward(Ws,X):
    acts=[X]; zs=[]; A=X; L=len(Ws)
    for l,Wl in enumerate(Ws):
        AB=np.concatenate([A,np.full((len(A),1),HI)],1); last=(l==L-1)
        Z=keystone(VC+Gs*Wl,AB,(VREFO if last else VREFH),(RTO if last else RTH))
        zs.append(Z); A=Z if last else relu_g(Z); acts.append(A)
    return acts,zs
def approx_grad(Ws,acts,zs,e):
    L=len(Ws); gr=[None]*L; d=e; B=len(e)
    for l in reversed(range(L)):
        AB=np.concatenate([acts[l],np.full((B,1),HI)],1)
        dz=d if l==L-1 else d*(zs[l]>VT_REL).astype(float)
        gr[l]=(dz.T@AB)/B
        if l>0: d=dz@Ws[l][:,:-1]
    return gr
def exact_grad(Ws,X,e,eps=5e-3):
    def s_of(W_): return (e*forward(W_,X)[0][-1]).sum(1).mean()
    gr=[np.zeros_like(W) for W in Ws]
    for l in range(len(Ws)):
        for i in range(Ws[l].shape[0]):
            for j in range(Ws[l].shape[1]):
                Wp=[w.copy() for w in Ws]; Wm=[w.copy() for w in Ws]; Wp[l][i,j]+=eps; Wm[l][i,j]-=eps
                gr[l][i,j]=(s_of(Wp)-s_of(Wm))/(2*eps)
    return gr
def acc(Ws,X,Y): return (np.argmax(forward(Ws,X)[0][-1],1)==Y).mean()

if __name__=="__main__":
    t0=time.time(); C=2
    Xtr,Ytr=make_checker(120,1,2); Xte,Yte=make_checker(200,2,2)
    p=scale_fit(Xtr); Xtr=scale_apply(Xtr,p); Xte=scale_apply(Xte,p); T=onehot(Ytr,C)
    sizes=[6,6,6,2]
    for seed in (1,2):
        rng=np.random.default_rng(seed)
        Ws=[np.clip(rng.uniform(-0.3,0.3,(b,a+1)),-1.5,1.0) for a,b in zip(sizes[:-1],sizes[1:])]
        V=[np.zeros_like(W) for W in Ws]
        # phase 1: APPROX gradient to convergence
        for _ in range(300):
            acts,zs=forward(Ws,Xtr); e=divis(acts[-1])-T; g=approx_grad(Ws,acts,zs,e)
            for l in range(len(Ws)): V[l]=0.9*V[l]-0.4*g[l]; Ws[l]=np.clip(Ws[l]+np.clip(V[l],-0.1,0.1),-1.5,1.0)
        a_approx=acc(Ws,Xte,Yte)
        # phase 2: EXACT gradient from the SAME point
        V=[np.zeros_like(W) for W in Ws]
        for _ in range(40):
            acts,zs=forward(Ws,Xtr); e=divis(acts[-1])-T; g=exact_grad(Ws,Xtr,e)
            for l in range(len(Ws)): V[l]=0.9*V[l]-0.4*g[l]; Ws[l]=np.clip(Ws[l]+np.clip(V[l],-0.1,0.1),-1.5,1.0)
        a_exact=acc(Ws,Xte,Yte)
        print(f"  seed {seed}: approx-converged test {a_approx*100:3.0f}%  -> +60 EXACT-grad steps {a_exact*100:3.0f}%  [{time.time()-t0:.0f}s]",flush=True)
