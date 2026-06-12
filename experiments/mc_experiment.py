#!/usr/bin/env python3
# Multi-class scaling experiment on the circuit-faithful surrogate (batch-vectorized).
import numpy as np
from surrogate import iDS, diode, relu_act, VC, Gs, VREFH, RTH, VREFO, RTO, HI, LO, VT_REL
kg,clamp = np.load('cal.npy')

def solve_col(gp, X, iters=12):
    # gp:(No,Nin), X:(B,Nin) -> Icol:(B,No)
    B=X.shape[0]; No=gp.shape[0]
    lo=np.full((B,No),0.5+1e-4); hi=np.full((B,No),3.0)
    for _ in range(iters):
        mid=0.5*(lo+hi)
        I=iDS(gp[None,:,:], X[:,None,:], mid[:,:,None]).sum(2)
        f=I-diode(mid); hi=np.where(f<0,mid,hi); lo=np.where(f>=0,mid,lo)
    return diode(0.5*(lo+hi))

def keystone(gp, X, vref, R):
    Icolp=solve_col(gp, X)
    Icoln=solve_col(VC*np.ones((1,X.shape[1])), X)   # shared neg ref (B,1)
    return vref + kg*R*(Icolp-Icoln)

# normalizations on Y:(B,C) per row; error always delta=p-t
def n_none(Y):       return Y
def n_divisive(Y):   r=np.clip(Y,0,None); return r/(r.sum(1,keepdims=True)+1e-6)
def n_subtractive(Y):return Y - Y.mean(1,keepdims=True) + 1.0/Y.shape[1]
def n_shiftmax(Y):   return Y - Y.max(1,keepdims=True) + 1.0
def n_softmax(Y):    e=np.exp((Y-Y.max(1,keepdims=True))/0.3); return e/e.sum(1,keepdims=True)
NORMS={'none':n_none,'divisive':n_divisive,'subtractive':n_subtractive,'shiftmax':n_shiftmax,'softmax':n_softmax}

class Net:
    def __init__(self, sizes, seed=0):
        self.sizes=sizes; rng=np.random.default_rng(seed)
        self.W=[rng.uniform(-0.25,0.25,size=(b,a+1)) for a,b in zip(sizes[:-1],sizes[1:])]
    def forward(self, X):
        acts=[X]; zs=[]; A=X
        for l,Wl in enumerate(self.W):
            AB=np.concatenate([A,np.full((A.shape[0],1),HI)],1); gp=VC+Gs*Wl
            last=(l==len(self.W)-1); vref,R=((VREFO,RTO) if last else (VREFH,RTH))
            Z=keystone(gp,AB,vref,R); zs.append(Z)
            acts.append(Z if last else relu_act(Z,clamp)); A=acts[-1]
        return acts,zs
    def grads(self, acts, zs, delta):
        B=delta.shape[0]; gr=[None]*len(self.W); D=delta
        for l in reversed(range(len(self.W))):
            AB=np.concatenate([acts[l],np.full((B,1),HI)],1); gr[l]=(D.T@AB)/B
            if l>0: D=(zs[l-1]>VT_REL).astype(float)*(D@self.W[l][:,:-1])
        return gr

IN_LO,IN_HI=0.3,1.7   # input voltage band (wider than [LO,HI] so interior points carry signal)
def scale_fit(X): return (X.min(0),X.max(0))
def scale_apply(X,p): mn,mx=p; return IN_LO+(X-mn)/(mx-mn+1e-9)*(IN_HI-IN_LO)
def make_blobs(C,D,n,seed,spread):
    cen=np.random.default_rng(2024).normal(0,1.5,(C,D))      # FIXED centers (shared train/test)
    rng=np.random.default_rng(seed); X=[];Y=[]
    for c in range(C): X.append(cen[c]+rng.normal(0,spread,(n,D))); Y+=[c]*n
    return np.vstack(X), np.array(Y)
def onehot(Y,C): T=np.zeros((len(Y),C)); T[np.arange(len(Y)),Y]=1; return T

def train_eval(sizes,norm,Xtr,Ytr,Xte,Yte,epochs,lr,seed,noise=0.0,mom=0.0):
    C=sizes[-1]; net=Net(sizes,seed); fn=NORMS[norm]; T=onehot(Ytr,C); rng=np.random.default_rng(seed+7)
    V=[np.zeros_like(w) for w in net.W]
    for ep in range(epochs):
        acts,zs=net.forward(Xtr); P=fn(acts[-1]); gr=net.grads(acts,zs,P-T)
        for l in range(len(net.W)):
            V[l]=mom*V[l]-lr*gr[l]; net.W[l]+=V[l]
            if noise>0: net.W[l]+=rng.normal(0,noise,net.W[l].shape)
    acts,_=net.forward(Xte); acc=(np.argmax(acts[-1],1)==Yte).mean()
    accs_tr=(np.argmax(net.forward(Xtr)[0][-1],1)==Ytr).mean()
    return acc, accs_tr, net

if __name__=="__main__":
    C,D,spread=4,6,0.9
    Xtr,Ytr=make_blobs(C,D,80,1,spread); Xte,Yte=make_blobs(C,D,40,2,spread)
    p=scale_fit(Xtr); Xtr=scale_apply(Xtr,p); Xte=scale_apply(Xte,p)   # fit on train, apply to both
    sizes=[D,16,C]
    print(f"multi-class A/B: C={C} D={D} hidden=16 spread={spread} ({len(Xtr)} train/{len(Xte)} test)")
    for norm in ['none','divisive','subtractive','softmax']:   # shiftmax dropped (fails: hard-max kills gradient)
        r=[train_eval(sizes,norm,Xtr,Ytr,Xte,Yte,400,0.1,s) for s in (0,1,2)]
        te=np.mean([a for a,_,_ in r]); tr=np.mean([b for _,b,_ in r])
        print(f"  {norm:12s} train {tr*100:5.1f}%  test {te*100:5.1f}%  (test seeds {[round(a*100) for a,_,_ in r]})")
