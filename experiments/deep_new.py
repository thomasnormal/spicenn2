#!/usr/bin/env python3
# New depth ideas tested in the surrogate on the checkerboard (a depth-NEEDING task).
# Extends deep_fixes with: (A) analog layer-norm (divisive activity normalization between
# layers, = the buildable competition primitive), and (C) stochastic weight noise (SGLD).
# Baseline to beat: deep [8,8,8] gets train 58% / test 54% (chance 50%, can't even fit).
import numpy as np, time
import deep_fixes as df
from mc_experiment import scale_fit, scale_apply, onehot
from surrogate import relu_act, VC, Gs, VREFH, RTH, VREFO, RTO, HI, VG0, VT_REL, BETA_REL, RH
kg,clamp=np.load('cal.npy')

def make_checker(n,seed,K=4,D=6):
    rng=np.random.default_rng(seed); uv=rng.uniform(0,1,(n,2))
    lab=((np.floor(K*uv[:,0])+np.floor(K*uv[:,1]))%2).astype(int)
    noise=rng.uniform(0,1,(n,D-2)); X=np.hstack([uv,noise]); return X,lab
Xtr,Ytr=make_checker(360,1); Xte,Yte=make_checker(240,2)
p=scale_fit(Xtr); Xtr=scale_apply(Xtr,p); Xte=scale_apply(Xte,p)
C=2; T=onehot(Ytr,C); B=len(Xtr); fn=df.NORMS['divisive']

def relu_g(z,gain=1.0,clmp=None):
    clmp=clamp if clmp is None else clmp
    I=0.5*BETA_REL*np.clip(z-VT_REL,0,None)**2
    return np.clip(VG0+gain*RH*I, VG0, clmp)

def forward(Ws,X,residual,lnorm=False,gain=1.0):
    acts=[X]; zs=[]; A=X; L=len(Ws)
    for l,Wl in enumerate(Ws):
        AB=np.concatenate([A,np.full((A.shape[0],1),HI)],1); gp=VC+Gs*Wl
        last=(l==L-1); vref,R=(VREFO,RTO) if last else (VREFH,RTH)
        Z=df.keystone(gp,AB,vref,R); zs.append(Z)
        if last: A=Z
        else:
            h=relu_g(Z,gain)
            if lnorm:   # analog layer-norm: divisive activity normalization (competition primitive)
                h=VG0+(h-VG0)/ (np.abs(h-VG0).mean(1,keepdims=True)+0.05) * 0.4
            A=np.clip(h+(A-VG0),VG0,clamp) if (residual and A.shape[1]==h.shape[1]) else h
        acts.append(A)
    return acts,zs

def backward(Ws,acts,zs,residual,dfa_B,leak=0.0):
    L=len(Ws); gr=[None]*L; e=fn(acts[-1])-T; d=e
    for l in reversed(range(L)):
        AB=np.concatenate([acts[l],np.full((B,1),HI)],1); last=(l==L-1)
        if last: dz=d
        else:
            gpa=np.where(zs[l]>VT_REL,1.0,leak)
            dz=((e@dfa_B[l])*gpa) if dfa_B is not None else d*gpa
        gr[l]=(dz.T@AB)/B
        if l>0 and dfa_B is None:
            d_in=dz@Ws[l][:,:-1]
            if residual and (not last) and acts[l].shape[1]==acts[l+1].shape[1]: d_in=d_in+d
            d=d_in
    return gr

def train(sizes,seed,ep=400,lr=0.4,mom=0.9,residual=False,dfa=False,ota_clip=0.1,
          wclamp=(-1.5,1.0),orth=False,leak=0.0,lnorm=False,noise=0.0):
    Ws=df.init_W(sizes,seed,orth,0.0)
    if wclamp: Ws=[np.clip(W,*wclamp) for W in Ws]
    V=[np.zeros_like(W) for W in Ws]; rng=np.random.default_rng(seed+5)
    dfa_B={l:rng.normal(0,1.0/np.sqrt(C),(C,sizes[l+1])) for l in range(len(Ws)-1)} if dfa else None
    for ep_i in range(ep):
        acts,zs=forward(Ws,Xtr,residual,lnorm); gr=backward(Ws,acts,zs,residual,dfa_B,leak)
        for l in range(len(Ws)):
            V[l]=mom*V[l]-lr*gr[l]; step=V[l] if ota_clip is None else np.clip(V[l],-ota_clip,ota_clip)
            Ws[l]+=step
            if noise>0: Ws[l]+=rng.normal(0,noise,Ws[l].shape)
            if wclamp: Ws[l]=np.clip(Ws[l],*wclamp)
    gn=[round(float(np.sqrt((g**2).mean())),3) for g in gr]
    te=(np.argmax(forward(Ws,Xte,residual,lnorm)[0][-1],1)==Yte).mean()
    tr=(np.argmax(forward(Ws,Xtr,residual,lnorm)[0][-1],1)==Ytr).mean()
    return tr,te,gn

if __name__=="__main__":
    t0=time.time()
    print("CHECKERBOARD 4x4 (chance 50%) — deep [6,8,8,8,2]. Baseline best: train 58 / test 54")
    cfgs=[
      ("residual (baseline)",       dict(residual=True)),
      ("+ layer-norm",              dict(residual=True, lnorm=True)),
      ("+ noise",                   dict(residual=True, noise=0.02)),
      ("+ layer-norm + noise",      dict(residual=True, lnorm=True, noise=0.02)),
      ("DFA + layer-norm + noise",  dict(dfa=True, lnorm=True, noise=0.02, lr=0.8)),
    ]
    for tag,kw in cfgs:
        best=None
        for s in (0,1,2):
            tr,te,gn=train([6,8,8,8,2],s,**kw)
            if best is None or te>best[1]: best=(tr,te,gn)
        print(f"  {tag:26s} train {best[0]*100:3.0f} test {best[1]*100:3.0f}  grad {best[2]}  [{time.time()-t0:.0f}s]",flush=True)
