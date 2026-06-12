import numpy as np, time
import deep_fixes as df
from mc_experiment import scale_fit, scale_apply, onehot
# checkerboard: 2 informative dims, K x K cells, label parity-of-cell -> needs depth at capped width
def make_checker(n,seed,K=4,D=6):
    rng=np.random.default_rng(seed); uv=rng.uniform(0,1,(n,2))
    lab=((np.floor(K*uv[:,0])+np.floor(K*uv[:,1]))%2).astype(int)
    noise=rng.uniform(0,1,(n,D-2)); X=np.hstack([uv,noise]); return X,lab
Xtr,Ytr=make_checker(360,1); Xte,Yte=make_checker(240,2)
p=scale_fit(Xtr); Xtr=scale_apply(Xtr,p); Xte=scale_apply(Xte,p)
# install into deep_fixes globals
df.Xtr,df.Ytr,df.Xte,df.Yte=Xtr,Ytr,Xte,Yte
df.C=2; df.T=onehot(Ytr,2); df.B=len(Xtr); df.fn=df.NORMS['divisive']
W=8; t0=time.time()
print(f"CHECKERBOARD 4x4 (chance 50%), capped width {W}")
print("-- shallow [.,8,.] (1 hidden) --")
for tag,kw in [('transpose wclamp+clip',dict(lr=0.4,ep=400,ota_clip=0.1)),
               ('no-clamp transpose',dict(lr=0.4,ep=400,wclamp=None))]:
    tr,te,gn=df.train([6,W,2],0,**kw); print(f"  {tag:24s} train {tr*100:3.0f} test {te*100:3.0f}  [{time.time()-t0:.0f}s]",flush=True)
print("-- deep [.,8,8,8,.] (3 hidden) --")
for tag,kw in [('transpose wclamp+clip',dict(lr=0.4,ep=400,ota_clip=0.1)),
               ('residual',dict(lr=0.4,ep=400,ota_clip=0.1,residual=True)),
               ('DFA',dict(lr=0.8,ep=400,dfa=True,ota_clip=0.1)),
               ('residual+DFA',dict(lr=0.8,ep=400,dfa=True,residual=True,ota_clip=0.1))]:
    tr,te,gn=df.train([6,W,W,W,2],0,**kw); print(f"  {tag:24s} train {tr*100:3.0f} test {te*100:3.0f}  grad {gn}  [{time.time()-t0:.0f}s]",flush=True)
