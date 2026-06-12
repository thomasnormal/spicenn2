#!/usr/bin/env python3
# Ten analog-plausible neurons, each = forward activation f(z) + backward derivative f'(z).
# Tier-1 test: which activation lets a DEEP net solve high-frequency compositional tasks
# (checkerboard at rising K) where ReLU^2 collapses. Ideal linear layers + Adam isolate the
# activation; exact backprop uses each neuron's f' (the backward the analog cell must supply).
import numpy as np, time
from mc_experiment import scale_fit, scale_apply, onehot

# ---- activations: return (a, cache) forward; and grad given upstream da -> dz ----
SQ=lambda z: np.where(z>0,z*z,0.0)
def act(name):
    if name=='relu2':   return (lambda z: np.where(z>0,z*z,0.0)),      (lambda z,a: np.where(z>0,2*z,0.0))
    if name=='tanh':    return (lambda z: np.tanh(z)),                  (lambda z,a: 1-a*a)
    if name=='hardtanh': return (lambda z: np.clip(z,-1,1)),            (lambda z,a: (np.abs(z)<1).astype(float))  # simplest bounded: a limiter
    if name=='lecuntanh':return (lambda z: 1.7159*np.tanh(2*z/3)),      (lambda z,a: 1.7159*(2/3)*(1-np.tanh(2*z/3)**2))  # isometry-tuned (f'(0)=1.14)
    if name=='selu':
        lam,al=1.0507,1.6733
        return (lambda z: lam*np.where(z>0,z,al*(np.exp(np.minimum(z,0))-1))), \
               (lambda z,a: lam*np.where(z>0,1.0,al*np.exp(np.minimum(z,0))))
    if name=='abs':     return (lambda z: np.abs(z)),                   (lambda z,a: np.sign(z))
    if name=='sine':    w=3.0; return (lambda z: np.sin(w*z)),          (lambda z,a: w*np.cos(w*z))
    if name=='gauss':   return (lambda z: np.exp(-0.5*z*z)),            (lambda z,a: -z*a)
    if name=='softplus':return (lambda z: np.logaddexp(0,z)),           (lambda z,a: 1/(1+np.exp(-z)))
    if name=='logcomp': return (lambda z: np.sign(z)*np.log1p(np.abs(z))), (lambda z,a: 1/(1+np.abs(z)))
    raise ValueError(name)

# RMSNorm+tanh and quadratic are layer-level; handled specially in forward/backward.
PERELEM=['relu2','tanh','selu','abs','sine','gauss','softplus','logcomp']

def mlp(sizes, name, Xtr,Ytr,Xte,Yte, seed, ep=500, lr=5e-3):
    rng=np.random.default_rng(seed); C=sizes[-1]; T=onehot(Ytr,C); B=len(Xtr)
    sc = 1.0 if name in ('tanh','sine','selu','softplus','logcomp','rmstanh') else np.sqrt(2)
    Ws=[rng.normal(0,np.sqrt(sc/a),(a,b)) for a,b in zip(sizes[:-1],sizes[1:])]
    bs=[np.zeros(b) for b in sizes[1:]]
    mW=[np.zeros_like(w) for w in Ws]; vW=[np.zeros_like(w) for w in Ws]
    mb=[np.zeros_like(b) for b in bs]; vb=[np.zeros_like(b) for b in bs]
    if name not in ('rmstanh','quad'): f,df=act(name)
    def fwd(X):
        acts=[X]; zs=[]; A=X
        for l,(W,b) in enumerate(zip(Ws,bs)):
            Z=A@W+b; zs.append(Z)
            if l==len(Ws)-1: A=Z
            elif name=='rmstanh':
                r=np.sqrt((Z*Z).mean(1,keepdims=True)+1e-4); A=np.tanh(Z/r)
            elif name=='quad':   # multiplicative sigma-pi: a_i = tanh(z_i * z_{i-1}) (width-preserving)
                A=np.tanh(Z*np.roll(Z,1,axis=1))
            else: A=f(Z)
            acts.append(A)
        return acts,zs
    def smax(Z): e=np.exp(Z-Z.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
    t=0
    for _ in range(ep):
        acts,zs=fwd(Xtr); P=smax(acts[-1]); d=(P-T)/B; t+=1
        gW=[None]*len(Ws); gb=[None]*len(Ws)
        for l in reversed(range(len(Ws))):
            gW[l]=acts[l].T@d; gb[l]=d.sum(0)
            if l>0:
                da=d@Ws[l].T; Z=zs[l-1]
                if name=='rmstanh':
                    r=np.sqrt((Z*Z).mean(1,keepdims=True)+1e-4); a=np.tanh(Z/r); d=da*(1-a*a)/r   # approx (stop-grad through r)
                elif name=='quad':
                    a=np.tanh(Z*np.roll(Z,1,axis=1)); g=da*(1-a*a)
                    d=g*np.roll(Z,1,axis=1)+np.roll(g*Z,-1,axis=1)
                else:
                    a=acts[l]; d=da*df(Z,a)
        for l in range(len(Ws)):
            for (m,v,g,P_) in [(mW[l],vW[l],gW[l],Ws[l]),(mb[l],vb[l],gb[l],bs[l])]:
                m[...] =0.9*m+0.1*g; v[...]=0.999*v+0.001*g*g
                P_-=lr*(m/(1-0.9**t))/(np.sqrt(v/(1-0.999**t))+1e-8)
    pr=lambda X: np.argmax(fwd(X)[0][-1],1)
    return (pr(Xtr)==Ytr).mean(),(pr(Xte)==Yte).mean()

def make_checker(n,seed,K,D=6):
    rng=np.random.default_rng(seed); uv=rng.uniform(0,1,(n,2))
    lab=((np.floor(K*uv[:,0])+np.floor(K*uv[:,1]))%2).astype(int)
    return np.hstack([uv,rng.uniform(0,1,(n,D-2))]),lab

if __name__=="__main__":
    t0=time.time(); names=PERELEM+['rmstanh','quad']
    Ks=[2,3,4,5]
    data={}
    for K in Ks:
        Xtr,Ytr=make_checker(500,1,K); Xte,Yte=make_checker(400,2,K)
        p=scale_fit(Xtr); data[K]=(scale_apply(Xtr,p),Ytr,scale_apply(Xte,p),Yte)
    print(f"DEEP [6,8,8,8,2], Adam, exact backprop. Test acc (best of 3 seeds) by neuron x checkerboard-K (chance 50):")
    print("  neuron     " + "  ".join(f"K={K}" for K in Ks))
    for nm in names:
        row=[]
        for K in Ks:
            Xtr,Ytr,Xte,Yte=data[K]
            best=max(mlp([6,8,8,8,2],nm,Xtr,Ytr,Xte,Yte,s)[1] for s in range(3))
            row.append(best)
        print(f"  {nm:9s}  " + "  ".join(f"{v*100:3.0f}" for v in row) + f"   [{time.time()-t0:.0f}s]",flush=True)
