import numpy as np
from mc_experiment import keystone, NORMS, onehot, make_blobs, scale_fit, scale_apply
from surrogate import relu_act, VC, Gs, VREFH, RTH, VREFO, RTO, HI, VG0, VT_REL, RH, BETA_REL
kg,clamp=np.load('cal.npy'); ORTH_SCALE=1.2
def relu_g(z,gain,clmp):
    I=0.5*BETA_REL*np.clip(z-VT_REL,0,None)**2
    return np.clip(VG0+gain*RH*I, VG0, clmp)
C,D,spread=8,8,1.2
Xtr,Ytr=make_blobs(C,D,50,1,spread); Xte,Yte=make_blobs(C,D,30,2,spread)
p=scale_fit(Xtr); Xtr=scale_apply(Xtr,p); Xte=scale_apply(Xte,p)
fn=NORMS['divisive']; T=onehot(Ytr,C); B=len(Xtr)

def spec_norm(W,iters=4):
    v=np.random.default_rng(0).standard_normal(W.shape[1]); v/=np.linalg.norm(v)+1e-9
    for _ in range(iters):
        u=W@v; u/=np.linalg.norm(u)+1e-9; v=W.T@u; v/=np.linalg.norm(v)+1e-9
    return float(u@(W@v))+1e-9
def init_W(sizes,seed,orth=False,bias_on=0.0):
    rng=np.random.default_rng(seed); Ws=[]
    for a,b in zip(sizes[:-1],sizes[1:]):
        W=rng.uniform(-0.25,0.25,(b,a+1))
        if orth and a>=b:
            M=rng.normal(0,1,(b,a)); U,_,Vt=np.linalg.svd(M,full_matrices=False); W[:,:-1]=ORTH_SCALE*(U@Vt)
        if bias_on: W[:,-1]=bias_on
        Ws.append(W)
    return Ws

def forward(Ws,X,residual,gain=1.0,clmp=None):
    clmp=clamp if clmp is None else clmp
    acts=[X]; zs=[]; A=X; L=len(Ws)
    for l,Wl in enumerate(Ws):
        AB=np.concatenate([A,np.full((A.shape[0],1),HI)],1); gp=VC+Gs*Wl
        last=(l==L-1); vref,R=(VREFO,RTO) if last else (VREFH,RTH)
        Z=keystone(gp,AB,vref,R); zs.append(Z)
        if last: A=Z
        else:
            h=relu_g(Z,gain,clmp)
            A=np.clip(h+(A-VG0),VG0,clmp) if (residual and A.shape[1]==h.shape[1]) else h
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

def train(sizes,seed,ep=200,lr=0.4,mom=0.9,residual=False,dfa=False,ota_clip=None,wclamp=(-1.5,1.0),orth=False,gain=1.0,clmp=None,leak=0.0,bias_on=0.0,wnorm=None,wnorm_row=None):
    Ws=init_W(sizes,seed,orth,bias_on)
    if wclamp: Ws=[np.clip(W,*wclamp) for W in Ws]
    V=[np.zeros_like(W) for W in Ws]; rng=np.random.default_rng(seed+5)
    dfa_B={l:rng.normal(0,1.0/np.sqrt(C),(C,sizes[l+1])) for l in range(len(Ws)-1)} if dfa else None
    gn=None
    for ep_i in range(ep):
        acts,zs=forward(Ws,Xtr,residual,gain,clmp); gr=backward(Ws,acts,zs,residual,dfa_B,leak)
        if ep_i==ep-1: gn=[round(float(np.sqrt((g**2).mean())),3) for g in gr]
        for l in range(len(Ws)):
            V[l]=mom*V[l]-lr*gr[l]; step=V[l] if ota_clip is None else np.clip(V[l],-ota_clip,ota_clip)
            Ws[l]+=step
            if wclamp: Ws[l]=np.clip(Ws[l],*wclamp)
            if wnorm is not None and l<len(Ws)-1:
                sn=spec_norm(Ws[l][:,:-1])
                if sn>wnorm: Ws[l][:,:-1]*=wnorm/sn
            if wnorm_row is not None and l<len(Ws)-1:      # per-neuron fan-in L2 norm (buildable divisive feedback)
                rn=np.linalg.norm(Ws[l][:,:-1],axis=1,keepdims=True)+1e-9
                Ws[l][:,:-1]*=np.minimum(1.0, wnorm_row/rn)
    te=(np.argmax(forward(Ws,Xte,residual,gain,clmp)[0][-1],1)==Yte).mean()
    tr=(np.argmax(forward(Ws,Xtr,residual,gain,clmp)[0][-1],1)==Ytr).mean()
    return tr,te,gn

if __name__=="__main__":
    import time; t0=time.time()
    print("=== 2 hidden layers [20,20] ===  (1-layer ref ~68%)")
    for tag,kw in [("honest (wclamp+otaclip)",{'ota_clip':0.1}),
                   ("+ residual",{'ota_clip':0.1,'residual':True}),
                   ("+ DFA",{'ota_clip':0.1,'dfa':True}),
                   ("+ residual + orth-init",{'ota_clip':0.1,'residual':True,'orth':True})]:
        tr,te,gn=train([D,20,20,C],0,**kw)
        print(f"  {tag:26s} train {tr*100:3.0f} test {te*100:3.0f}  grad-rms {gn}  [{time.time()-t0:.0f}s]",flush=True)
