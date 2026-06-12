#!/usr/bin/env python3
# Prepare a 2D task (circles/rings/spirals, cached in /tmp/insp/) for the in-circuit backprop
# trainer (gen_mc.py): balanced subsample -> digits_train.npz / digits_val.npz / digits_test.npz
# with X (N x 2, already minmax [0,1]) and Y (N,) in {0..C-1}. Same file contract as prep_mnist.py.
import numpy as np, sys, os
task=sys.argv[1] if len(sys.argv)>1 else "circles"
ntr=int(sys.argv[2]) if len(sys.argv)>2 else 50    # per class, train (presented in SPICE)
nte=int(sys.argv[3]) if len(sys.argv)>3 else 60    # per class, test
nval=int(sys.argv[4]) if len(sys.argv)>4 else 30   # per class, validation
tr=np.load(f"/tmp/insp/{task}_train.npz"); te=np.load(f"/tmp/insp/{task}_test.npz")
Xa,ya=tr[tr.files[0]].astype(float),tr[tr.files[1]].astype(int)
Xb,yb=te[te.files[0]].astype(float),te[te.files[1]].astype(int)
rng=np.random.default_rng(0)
Xtr=[];Ytr=[];Xva=[];Yva=[];Xte=[];Yte=[]
for c in sorted(set(ya.tolist())):
    ia=np.where(ya==c)[0]; rng.shuffle(ia)
    Xtr.append(Xa[ia[:ntr]]); Ytr+=[c]*ntr
    Xva.append(Xa[ia[ntr:ntr+nval]]); Yva+=[c]*min(nval,len(ia)-ntr)
    ib=np.where(yb==c)[0]; rng.shuffle(ib)
    Xte.append(Xb[ib[:nte]]); Yte+=[c]*min(nte,len(ib))
Xtr=np.vstack(Xtr); Xva=np.vstack(Xva); Xte=np.vstack(Xte)
np.savez("digits_train.npz",X=Xtr,Y=np.array(Ytr))
np.savez("digits_val.npz",X=Xva,Y=np.array(Yva))
np.savez("digits_test.npz",X=Xte,Y=np.array(Yte))
print(f"{task}: D=2 C={len(set(Ytr))} train {Xtr.shape} val {Xva.shape} test {Xte.shape}")
