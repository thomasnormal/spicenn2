#!/usr/bin/env python3
# Gradient-fidelity diagnostic, precision-hardened + sanity-controlled.
#  - high-precision column solve (55 bisection iters) so finite-diff isn't quantization noise
#  - eps swept to confirm FD is converged
#  - LINEAR-network control: transpose-read is provably exact -> cosine MUST be ~1 (validates code)
import numpy as np
from mc_experiment import onehot, scale_fit, scale_apply
from surrogate import iDS, diode, VC, Gs, VREFH, RTH, VREFO, RTO, HI, VG0, VT_REL, BETA_REL, RH
from deep_super import make_checker
clamp=2.6

def solve_col(gp, X, iters=55):
    B=X.shape[0]; No=gp.shape[0]
    lo=np.full((B,No),0.5+1e-9); hi=np.full((B,No),3.0)
    for _ in range(iters):
        mid=0.5*(lo+hi); I=iDS(gp[None,:,:],X[:,None,:],mid[:,:,None]).sum(2)
        f=I-diode(mid); hi=np.where(f<0,mid,hi); lo=np.where(f>=0,mid,lo)
    return diode(0.5*(lo+hi))
def keystone(gp,X,vref,R):
    Icolp=solve_col(gp,X); Icoln=solve_col(VC*np.ones((1,X.shape[1])),X)
    return vref+0.64*R*(Icolp-Icoln)
def relu_g(z):
    I=0.5*BETA_REL*np.clip(z-VT_REL,0,None)**2; return np.clip(VG0+RH*I,VG0,clamp)
def divis(Y): r=np.clip(Y,0,None); return r/(r.sum(1,keepdims=True)+1e-6)

def forward(Ws,X,linear=False):
    acts=[X]; zs=[]; A=X; L=len(Ws)
    for l,Wl in enumerate(Ws):
        AB=np.concatenate([A,np.full((len(A),1),HI)],1)
        last=(l==L-1)
        if linear:                       # pure linear net: transpose-read is EXACT
            Z=AB@Wl.T
        else:
            gp=VC+Gs*Wl; vref,R=(VREFO,RTO) if last else (VREFH,RTH); Z=keystone(gp,AB,vref,R)
        zs.append(Z); A=Z if last else (AB@Wl.T if False else relu_g(Z));
        if linear and not last: A=np.maximum(Z,0)
        acts.append(A)
    return acts,zs

def approx_grad(Ws,acts,zs,e,linear=False):
    L=len(Ws); gr=[None]*L; d=e; B=len(e)
    for l in reversed(range(L)):
        AB=np.concatenate([acts[l],np.full((B,1),HI)],1); last=(l==L-1)
        dz=d if last else d*(zs[l]>(0 if linear else VT_REL)).astype(float)
        gr[l]=(dz.T@AB)/B
        if l>0: d=dz@Ws[l][:,:-1]
    return gr

def exact_grad(Ws,X,e,linear=False,eps=5e-3):
    def s_of(Ws_): return (e*forward(Ws_,X,linear)[0][-1]).sum(1).mean()
    gr=[np.zeros_like(W) for W in Ws]
    for l in range(len(Ws)):
        for i in range(Ws[l].shape[0]):
            for j in range(Ws[l].shape[1]):
                Wp=[w.copy() for w in Ws]; Wm=[w.copy() for w in Ws]
                Wp[l][i,j]+=eps; Wm[l][i,j]-=eps
                gr[l][i,j]=(s_of(Wp)-s_of(Wm))/(2*eps)
    return gr
def cos(a,b): a=a.flatten();b=b.flatten(); return float(a@b/((np.linalg.norm(a)*np.linalg.norm(b))+1e-12))

if __name__=="__main__":
    C=2; Xtr,Ytr=make_checker(150,1,2); p=scale_fit(Xtr); Xtr=scale_apply(Xtr,p); T=onehot(Ytr,C)
    rng=np.random.default_rng(0)
    print("CONTROL: pure LINEAR net (transpose-read is exact) — cosine MUST be ~1.0")
    for sizes in [[6,8,2],[6,8,8,2]]:
        Ws=[rng.uniform(-0.5,0.5,(b,a+1)) for a,b in zip(sizes[:-1],sizes[1:])]
        acts,zs=forward(Ws,Xtr,linear=True); e=divis(acts[-1])-T
        ga=approx_grad(Ws,acts,zs,e,linear=True); ge=exact_grad(Ws,Xtr,e,linear=True)
        print(f"   {sizes}: {[round(cos(ga[l],ge[l]),3) for l in range(len(Ws))]}")
    print("\nANALOG net (keystone+ReLU^2), high-precision FD, eps converged:")
    for sizes in [[6,8,2],[6,8,8,2],[6,8,8,8,2]]:
        Ws=[np.clip(rng.uniform(-0.3,0.3,(b,a+1)),-1.5,1.0) for a,b in zip(sizes[:-1],sizes[1:])]
        acts,zs=forward(Ws,Xtr); e=divis(acts[-1])-T
        ga=approx_grad(Ws,acts,zs,e)
        ge1=exact_grad(Ws,Xtr,e,eps=5e-3); ge2=exact_grad(Ws,Xtr,e,eps=1e-2)
        c1=[round(cos(ga[l],ge1[l]),3) for l in range(len(Ws))]
        conv=[round(cos(ge1[l],ge2[l]),3) for l in range(len(Ws))]   # FD self-consistency across eps
        print(f"   {sizes}: cos(approx,exact)={c1}   FD-converged(eps5e-3 vs 1e-2)={conv}")
