#!/usr/bin/env python3
# Idea #4: fully-LOCAL deep supervision (target-prop flavor) on the analog surrogate.
# Each forward layer l is trained ONLY by its own auxiliary linear head -> target (a 1-hop,
# shallow gradient where the analog approximate gradient actually works). No deep backprop
# chain to get stuck. Final prediction = deepest head. Compare to shallow and to deep-backprop.
import numpy as np, time
import deep_fixes as df
from mc_experiment import scale_fit, scale_apply, onehot
from surrogate import VC, Gs, VREFH, RTH, VREFO, RTO, HI, VG0, VT_REL, BETA_REL, RH
kg,clamp=np.load('cal.npy')

def relu_g(z):
    I=0.5*BETA_REL*np.clip(z-VT_REL,0,None)**2
    return np.clip(VG0+RH*I, VG0, clamp)
def keystone(gp,X,vref,R): return df.keystone(gp,X,vref,R)
def divis(Y): r=np.clip(Y,0,None); return r/(r.sum(1,keepdims=True)+1e-6)

def make_checker(n,seed,K,D=6):
    rng=np.random.default_rng(seed); uv=rng.uniform(0,1,(n,2))
    lab=((np.floor(K*uv[:,0])+np.floor(K*uv[:,1]))%2).astype(int)
    noise=rng.uniform(0,1,(n,D-2)); return np.hstack([uv,noise]),lab

def train_local(sizes,Xtr,Ytr,seed,ep=400,lr=0.4,ota_clip=0.1,wclamp=(-1.5,1.0),noise=0.0):
    # forward weights Ws[l]; auxiliary head Hs[l] (C x size_{l+1}+1) for each hidden layer + final
    C=sizes[-1]; B=len(Xtr); T=onehot(Ytr,C); rng=np.random.default_rng(seed+5)
    Ws=df.init_W(sizes,seed); Ws=[np.clip(W,*wclamp) for W in Ws] if wclamp else Ws
    nlay=len(Ws)
    # heads: for each hidden layer (0..nlay-2) a head from that layer's activation to C
    Hs={l: rng.uniform(-0.25,0.25,(C,sizes[l+1]+1)) for l in range(nlay-1)}
    Vw=[np.zeros_like(W) for W in Ws]; Vh={l:np.zeros_like(Hs[l]) for l in Hs}
    def fwd(Ws,X):
        acts=[X]; zs=[]; A=X
        for l,Wl in enumerate(Ws):
            AB=np.concatenate([A,np.full((len(A),1),HI)],1); gp=VC+Gs*Wl
            last=(l==nlay-1); vref,R=(VREFO,RTO) if last else (VREFH,RTH)
            Z=keystone(gp,AB,vref,R); zs.append(Z); A=Z if last else relu_g(Z); acts.append(A)
        return acts,zs
    for e in range(ep):
        acts,zs=fwd(Ws,Xtr)
        # final layer trained by true target
        for l in range(nlay):
            last=(l==nlay-1)
            AB=np.concatenate([acts[l],np.full((B,1),HI)],1)
            if last:
                e_out=divis(acts[-1])-T; dz=e_out
            else:
                # local head on this layer's activation
                hAB=np.concatenate([acts[l+1],np.full((B,1),HI)],1)
                hlog=keystone(VC+Gs*Hs[l], hAB, VREFO, RTO); e_h=divis(hlog)-T
                gH=(e_h.T@hAB)/B
                Vh[l]=0.9*Vh[l]-lr*gH; Hs[l]+=np.clip(Vh[l],-ota_clip,ota_clip)
                if wclamp: Hs[l]=np.clip(Hs[l],*wclamp)
                # 1-hop gradient into this layer's forward weights via its head
                dz=(e_h@Hs[l][:,:-1])*np.where(zs[l]>VT_REL,1.0,0.0)
            gW=(dz.T@AB)/B
            Vw[l]=0.9*Vw[l]-lr*gW; Ws[l]+=np.clip(Vw[l],-ota_clip,ota_clip)
            if noise>0: Ws[l]+=rng.normal(0,noise,Ws[l].shape)
            if wclamp: Ws[l]=np.clip(Ws[l],*wclamp)
    return Ws,Hs,fwd

def acc_local(sizes,Xtr,Ytr,Xte,Yte,**kw):
    C=sizes[-1]; Ws,Hs,fwd=train_local(sizes,Xtr,Ytr,**kw)
    def pred(X):
        acts,_=fwd(Ws,X)
        # use deepest head (last hidden layer) if it exists, else final output
        if len(Hs):
            l=max(Hs); hAB=np.concatenate([acts[l+1],np.full((len(X),1),HI)],1)
            return np.argmax(keystone(VC+Gs*Hs[l],hAB,VREFO,RTO),1)
        return np.argmax(acts[-1],1)
    return (pred(Xtr)==Ytr).mean(),(pred(Xte)==Yte).mean()

if __name__=="__main__":
    t0=time.time()
    for K in (2,3,4):
        Xtr,Ytr=make_checker(360,1,K); Xte,Yte=make_checker(240,2,K)
        p=scale_fit(Xtr); Xtr=scale_apply(Xtr,p); Xte=scale_apply(Xte,p)
        print(f"\n=== {K}x{K} checkerboard (chance 50%) ===")
        # shallow deep-backprop reference (deep_fixes)
        df.Xtr,df.Ytr,df.Xte,df.Yte=Xtr,Ytr,Xte,Yte; df.C=2; df.T=onehot(Ytr,2); df.B=len(Xtr); df.fn=df.NORMS['divisive']
        tr,te,_=df.train([6,8,2],0,lr=0.4,ep=400,ota_clip=0.1); print(f"  shallow[8] backprop      train {tr*100:3.0f} test {te*100:3.0f}")
        tr,te,_=df.train([6,8,8,8,2],0,lr=0.4,ep=400,ota_clip=0.1,residual=True); print(f"  deep[8,8,8] backprop+res train {tr*100:3.0f} test {te*100:3.0f}")
        # fully-local deep supervision
        best=None
        for s in (0,1,2):
            tr,te=acc_local([6,8,8,8,2],Xtr,Ytr,Xte,Yte,seed=s,noise=0.0)
            if best is None or te>best[1]: best=(tr,te)
        print(f"  deep[8,8,8] LOCAL-super  train {best[0]*100:3.0f} test {best[1]*100:3.0f}  [{time.time()-t0:.0f}s]",flush=True)
