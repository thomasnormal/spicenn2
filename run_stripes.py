import numpy as np, time
import deep_fixes as df
from mc_experiment import scale_fit, scale_apply, onehot
# 1D high-frequency stripes: one informative coord -> 8 alternating stripes. Capped narrow width => depth wins.
def make_stripes(n,seed,K=8,D=2):
    rng=np.random.default_rng(seed); u=rng.uniform(0,1,(n,1))
    lab=(np.floor(K*u[:,0])%2).astype(int)
    extra=rng.uniform(0,1,(n,D-1)); X=np.hstack([u,extra]); return X,lab
Xtr,Ytr=make_stripes(600,1); Xte,Yte=make_stripes(400,2)
p=scale_fit(Xtr); Xtr=scale_apply(Xtr,p); Xte=scale_apply(Xte,p)
df.Xtr,df.Ytr,df.Xte,df.Yte=Xtr,Ytr,Xte,Yte
df.C=2; df.T=onehot(Ytr,2); df.B=len(Xtr); df.fn=df.NORMS['divisive']
t0=time.time()
print("8-STRIPE 1D task (chance 50%)")
print("-- reference: WIDE shallow [2,32,2] (width can solve it) --")
tr,te,_=df.train([2,32,2],0,lr=0.4,ep=400,ota_clip=0.15); print(f"   wide shallow      train {tr*100:3.0f} test {te*100:3.0f}  [{time.time()-t0:.0f}s]",flush=True)
print("-- NARROW shallow [2,4,2] (1 hidden, width 4) --")
tr,te,_=df.train([2,4,2],0,lr=0.4,ep=400,ota_clip=0.15); print(f"   narrow shallow    train {tr*100:3.0f} test {te*100:3.0f}  [{time.time()-t0:.0f}s]",flush=True)
print("-- NARROW deep [2,4,4,4,2] (3 hidden, width 4) --")
for tag,kw in [('transpose',dict(lr=0.4,ep=400,ota_clip=0.15)),
               ('residual',dict(lr=0.4,ep=400,ota_clip=0.15,residual=True)),
               ('DFA',dict(lr=0.8,ep=400,ota_clip=0.15,dfa=True))]:
    tr,te,gn=df.train([2,4,4,4,2],0,**kw); print(f"   deep {tag:10s}   train {tr*100:3.0f} test {te*100:3.0f}  grad {gn}  [{time.time()-t0:.0f}s]",flush=True)
