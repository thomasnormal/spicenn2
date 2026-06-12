#!/bin/bash
# Stable in-circuit 10-class MNIST training (the zero-sum-target breakthrough, IDEAS2 #7).
# Fixed-random nonlinear expansion (8x8 zscore -> N tanh features, fan-in 4) + a single trained
# DIRECT linear readout, trained fully in ngspice with ZERO-SUM TARGETS that stop the multi-class
# drift-collapse: TGHI=0.8, TGLO = vrefo - (TGHI-vrefo)/(C-1) = 0.467 for C=10.
# Verified: N=96 -> 64.8% all-10-classes stable (was collapsing to 14% with one-hot targets).
# RULES (learned the hard way): run ONE at a time, do NOT pkill (it races your own run).
cd ~/spicenn2
N=${N:-96}; NTR=${NTR:-25}; NTE=${NTE:-40}; SLOTS=${SLOTS:-9000}; CG=${CG:-1.5e-3}
python3 - "$N" "$NTR" "$NTE" <<'PY'
import numpy as np, sys
N,ntr,nte=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])
d=np.load('mnist_8x8.npz'); X=d['X'].astype(float); y=d['y']
rng=np.random.default_rng(0); idx=rng.permutation(len(y)); X,y=X[idx],y[idx]
X=(X-X.mean(1,keepdims=True))/(X.std(1,keepdims=True)+1e-6)
k=4; r=np.random.default_rng(1); D=64
P=[(r.choice(D,k,replace=False), r.normal(0,1,k), r.normal(0,0.5)) for _ in range(N)]
ex=lambda A: np.stack([np.tanh(A[:,i]@w+b) for i,w,b in P],1)
def pick(n,off):
    Xs=[];Ys=[]
    for L in range(10):
        ix=np.where(y==L)[0][off:off+n]; Xs.append(ex(X[ix])); Ys+=[L]*len(ix)
    return np.vstack(Xs),np.array(Ys)
Xtr,Ytr=pick(ntr,0); Xte,Yte=pick(nte,ntr)
np.savez('digits_train.npz',X=(Xtr+1)/2,Y=Ytr); np.savez('digits_test.npz',X=(Xte+1)/2,Y=Yte)
from sklearn.linear_model import LogisticRegression
print('N=%d ideal linear=%.1f%%'%(N,100*LogisticRegression(max_iter=400).fit((Xtr+1)/2,Ytr).score((Xte+1)/2,Yte)))
PY
SYNW=$(python3 -c "print(round(4*96/$N,2))")   # scale synapse current down as fan-in N grows
B="D=$N C=10 DIRECT=1 INIT=0.1 INLO=0.3 INHI=1.7 VREFH=0.5 SYNW=$SYNW RTO=7e3 OWTW=20 OCMSUB=1 TGHI=0.8 TGLO=0.467"
env $B DATAFILE=digits_train.npz SEED=1 python3 gen_mc.py $SLOTS 0.3 $CG >/dev/null
echo "MOS=$(grep -c '^M' mc.cir)  training $SLOTS slots..."
T0=$(date +%s); ngspice -b mc.cir >/dev/null 2>&1; echo "train $(($(date +%s)-T0))s"
env $B DATAFILE_TE=digits_test.npz AVGW=20 python3 gen_mc_infer.py >/dev/null
ngspice -b mc_infer.cir >/dev/null 2>&1
printf "N=%s C=10 zero-sum: " "$N"; C=10 python3 score_mc.py ""
