#!/usr/bin/env python3
# Prepare sklearn digits for the in-circuit trainer: 8x8 -> 4x4 (2x2 avg-pool), pick 2-3 labels,
# preprocess, map to [0,1] (the trainer maps [0,1] -> input voltage band), split train/test.
# NORM (env): ""(raw) | perimage (per-sample min-max) | standard (StandardScaler, fit on TRAIN)
#             | minmax (MinMaxScaler per-feature, fit on TRAIN)
import numpy as np, sys, os
from sklearn.datasets import load_digits
from sklearn import preprocessing
labs=[int(x) for x in (sys.argv[1] if len(sys.argv)>1 else "0,1").split(",")]
ntr=int(sys.argv[2]) if len(sys.argv)>2 else 16
nte=int(sys.argv[3]) if len(sys.argv)>3 else 40
NORM=os.environ.get("NORM","")
d=load_digits()
X=d.images.reshape(-1,4,2,4,2).mean(axis=(2,4)).reshape(-1,16)/16.0   # 4x4 in [0,1]
y=d.target
rng=np.random.default_rng(0)
tr_i=[]; te_i=[]; lab_of={}
for ci,L in enumerate(labs):
    idx=np.where(y==L)[0]; rng.shuffle(idx)
    tr_i+=list(idx[:ntr]); te_i+=list(idx[ntr:ntr+nte])
    for i in idx[:ntr+nte]: lab_of[i]=ci
tr_i=np.array(tr_i); te_i=np.array(te_i)
Xtr=X[tr_i]; Xte=X[te_i]; Ytr=np.array([lab_of[i] for i in tr_i]); Yte=np.array([lab_of[i] for i in te_i])

if NORM=="perimage":                      # per-SAMPLE min-max (each image fills full range)
    for A in (Xtr,Xte):
        lo=A.min(1,keepdims=True); hi=A.max(1,keepdims=True); A[:]=(A-lo)/(hi-lo+1e-6)
elif NORM=="standard":                    # per-FEATURE StandardScaler, FIT ON TRAIN ONLY
    sc=preprocessing.StandardScaler().fit(Xtr)
    Xtr=sc.transform(Xtr); Xte=sc.transform(Xte)
    # standardized ~N(0,1); map to [0,1] for the input band (clip +-3 sigma -> [0,1])
    Xtr=np.clip(Xtr/6+0.5,0,1); Xte=np.clip(Xte/6+0.5,0,1)
elif NORM=="minmax":                      # per-FEATURE MinMaxScaler, FIT ON TRAIN ONLY
    sc=preprocessing.MinMaxScaler().fit(Xtr); Xtr=sc.transform(Xtr); Xte=np.clip(sc.transform(Xte),0,1)

np.savez("digits_train.npz", X=Xtr, Y=Ytr); np.savez("digits_test.npz", X=Xte, Y=Yte)
print(f"labels {labs} C={len(labs)} D=16 NORM={NORM or 'raw'}  train {Xtr.shape} test {Xte.shape}")
