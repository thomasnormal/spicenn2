#!/usr/bin/env python3
# TIER 2: do the winning neurons (tanh, gauss) still beat ReLU^2 with the ANALOG front-end
# (real keystone weighted-sum + the chip's APPROXIMATE transpose-read backward)?
# All activations mapped to a common output band [0.4,1.6] (the synapse input range) so the
# comparison is about the activation SHAPE, not its range. Backward uses each neuron's f'.
import numpy as np, time
from mc_experiment import onehot, scale_fit, scale_apply
from surrogate import iDS, diode, VC, Gs, VREFH, RTH, VREFO, RTO, HI
from deep_super import make_checker
LO,HIb=0.4,1.6   # activation output band

def solve_col(gp,X,it=25):
    B=X.shape[0]; No=gp.shape[0]; lo=np.full((B,No),0.5+1e-9); hi=np.full((B,No),3.0)
    for _ in range(it):
        m=0.5*(lo+hi); f=iDS(gp[None],X[:,None],m[:,:,None]).sum(2)-diode(m)
        hi=np.where(f<0,m,hi); lo=np.where(f>=0,m,lo)
    return diode(0.5*(lo+hi))
def keystone(gp,X,vref,R): return vref+0.64*R*(solve_col(gp,X)-solve_col(VC*np.ones((1,X.shape[1])),X))

# activation: squash(z) in [0,1] and its derivative; a = LO+(HIb-LO)*squash
def actfns(name):
    if name=='relu2':
        s =lambda z: np.clip(((np.clip(z-0.5,0,None))/1.5)**2,0,1)
        ds=lambda z: np.where((z>0.5)&(z<2.0), 2*(z-0.5)/1.5**2, 0.0)
    elif name=='tanh':
        s =lambda z: 0.5+0.5*np.tanh((z-0.8)/0.4)
        ds=lambda z: 0.5*(1-np.tanh((z-0.8)/0.4)**2)/0.4
    elif name=='gauss':
        s =lambda z: np.exp(-((z-0.9)/0.35)**2)
        ds=lambda z: np.exp(-((z-0.9)/0.35)**2)*(-2*(z-0.9)/0.35**2)
    return s,ds

def divis(Y): r=np.clip(Y,0,None); return r/(r.sum(1,keepdims=True)+1e-6)

def run(sizes,name,Xtr,Ytr,Xte,Yte,seed,ep=400,lr=0.4):
    rng=np.random.default_rng(seed); C=sizes[-1]; T=onehot(Ytr,C); B=len(Xtr)
    s,ds=actfns(name)
    Ws=[np.clip(rng.uniform(-0.3,0.3,(b,a+1)),-1.5,1.0) for a,b in zip(sizes[:-1],sizes[1:])]
    V=[np.zeros_like(W) for W in Ws]; L=len(Ws)
    def fwd(X):
        acts=[X]; zs=[]; A=X
        for l,Wl in enumerate(Ws):
            AB=np.concatenate([A,np.full((len(A),1),HI)],1); last=(l==L-1)
            Z=keystone(VC+Gs*Wl,AB,(VREFO if last else VREFH),(RTO if last else RTH)); zs.append(Z)
            A=Z if last else (LO+(HIb-LO)*s(Z)); acts.append(A)
        return acts,zs
    def grad(acts,zs,e):
        gr=[None]*L; d=e
        for l in reversed(range(L)):
            AB=np.concatenate([acts[l],np.full((B,1),HI)],1); last=(l==L-1)
            dz=d if last else d*((HIb-LO)*ds(zs[l]))
            gr[l]=(dz.T@AB)/B
            if l>0: d=dz@Ws[l][:,:-1]
        return gr
    for _ in range(ep):
        acts,zs=fwd(Xtr); e=divis(acts[-1])-T; g=grad(acts,zs,e)
        for l in range(L): V[l]=0.9*V[l]-lr*g[l]; Ws[l]=np.clip(Ws[l]+np.clip(V[l],-0.1,0.1),-1.5,1.0)
    pr=lambda X: np.argmax(fwd(X)[0][-1],1)
    return (pr(Xtr)==Ytr).mean(),(pr(Xte)==Yte).mean()

if __name__=="__main__":
    t0=time.time(); K=3
    Xtr,Ytr=make_checker(400,1,K); Xte,Yte=make_checker(300,2,K)
    p=scale_fit(Xtr); Xtr=scale_apply(Xtr,p); Xte=scale_apply(Xte,p)
    print(f"ANALOG front-end (keystone + approx transpose-read backward), K={K} checkerboard (chance 50)")
    print("  neuron   shallow[8]  deep[8,8,8]  depth-gain   (best of 3 seeds)")
    for nm in ['relu2','tanh','gauss']:
        sh=max(run([6,8,2],nm,Xtr,Ytr,Xte,Yte,s)[1] for s in range(3))
        dp=max(run([6,8,8,8,2],nm,Xtr,Ytr,Xte,Yte,s)[1] for s in range(3))
        print(f"  {nm:7s}    {sh*100:3.0f}        {dp*100:3.0f}        {(dp-sh)*100:+3.0f}     [{time.time()-t0:.0f}s]",flush=True)
