#!/usr/bin/env python3
# Prepare real MNIST (cached in mnist_raw.npz, already 28x28 -> 4x4) for the in-circuit trainer.
# Pick >=3 labels, per-image contrast-normalize (the reliability fix), balanced train/val/test.
# Saves digits_train.npz / digits_val.npz / digits_test.npz (val is for restart model-selection,
# so the restart loop never peeks at test).
import numpy as np, sys, os
d=np.load("mnist_raw.npz"); X4=d["X4"]; y=d["y"]
labs=[int(x) for x in (sys.argv[1] if len(sys.argv)>1 else "0,1,7").split(",")]
ntr=int(sys.argv[2]) if len(sys.argv)>2 else 24    # per class, train (presented in SPICE)
nte=int(sys.argv[3]) if len(sys.argv)>3 else 50    # per class, test
nval=int(sys.argv[4]) if len(sys.argv)>4 else 30   # per class, validation (model selection)
rng=np.random.default_rng(0)
Xtr=[];Ytr=[];Xte=[];Yte=[];Xva=[];Yva=[]
for ci,L in enumerate(labs):
    idx=np.where(y==L)[0]; rng.shuffle(idx)
    Xtr.append(X4[idx[:ntr]]); Ytr+=[ci]*ntr
    Xva.append(X4[idx[ntr:ntr+nval]]); Yva+=[ci]*nval
    Xte.append(X4[idx[ntr+nval:ntr+nval+nte]]); Yte+=[ci]*nte
Xtr=np.vstack(Xtr).astype(float); Xte=np.vstack(Xte).astype(float); Xva=np.vstack(Xva).astype(float)
Ytr=np.array(Ytr); Yte=np.array(Yte); Yva=np.array(Yva)
NORM=os.environ.get("NORM","perimage")
if NORM=="perimage":   # per-image min-max (each image fills full range)
    for A in (Xtr,Xte,Xva):
        lo=A.min(1,keepdims=True); hi=A.max(1,keepdims=True); A[:]=(A-lo)/(hi-lo+1e-6)
elif NORM=="zscore":   # per-image z-score -> every image has EQUAL energy (mean 0, var 1), then
    for A in (Xtr,Xte,Xva):   # map to [0,1] band. Removes the input-energy asymmetry that makes
        m=A.mean(1,keepdims=True); s=A.std(1,keepdims=True)+1e-6  # sparse digits (1) lose to dense (0).
        A[:]=np.clip((A-m)/s*0.25+0.5, 0, 1)
np.savez("digits_train.npz", X=Xtr, Y=Ytr); np.savez("digits_test.npz", X=Xte, Y=Yte)
np.savez("digits_val.npz", X=Xva, Y=Yva)
print(f"MNIST labels {labs} -> C={len(labs)} D=16, train {Xtr.shape} val {Xva.shape} test {Xte.shape}, NORM={os.environ.get('NORM','perimage')}")
