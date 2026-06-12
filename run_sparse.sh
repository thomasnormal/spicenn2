#!/bin/bash
# Isolated SPARSE-readout C=10 trainer: N features (fan-in-16 fixed expansion), each class reads its
# top-K L1-selected features -> small deck even at large N. zero-sum targets + tuned per-class auto-zero.
# Usage: bash run_sparse.sh <DIR> <N> <K> <NTR> <SLOTS>
DIR=$1; N=$2; K=$3; NTR=${4:-150}; SLOTS=${5:-10000}
rm -rf "$DIR" && mkdir -p "$DIR" && cd "$DIR" || exit 1
cp ~/spicenn2/gen_mc.py ~/spicenn2/gen_mc_infer.py ~/spicenn2/score_mc.py ~/spicenn2/mnist_8x8.npz .
python3 - "$N" "$NTR" <<'PY'
import numpy as np, sys
N,ntr=int(sys.argv[1]),int(sys.argv[2])
d=np.load('mnist_8x8.npz'); X=d['X'].astype(float); y=d['y']
rng=np.random.default_rng(0); idx=rng.permutation(len(y)); X,y=X[idx],y[idx]
X=(X-X.mean(1,keepdims=True))/(X.std(1,keepdims=True)+1e-6)
k=16;r=np.random.default_rng(1);D=64
P=[(r.choice(D,k,replace=False),r.normal(0,1,k),r.normal(0,0.5)) for _ in range(N)]
ex=lambda A: np.stack([np.tanh(A[:,i]@w+b) for i,w,b in P],1)
def pick(n,off):
    Xs=[];Ys=[]
    for L in range(10):
        ix=np.where(y==L)[0][off:off+n];Xs.append(ex(X[ix]));Ys+=[L]*len(ix)
    return np.vstack(Xs),np.array(Ys)
Xtr,Ytr=pick(ntr,0);Xte,Yte=pick(120,ntr+20)
np.savez('digits_train.npz',X=(Xtr+1)/2,Y=Ytr);np.savez('digits_test.npz',X=(Xte+1)/2,Y=Yte)
print('N=%d ntr=%d ready'%(N,ntr))
PY
B="D=$N C=10 DIRECT=1 INIT=0.1 INLO=0.3 INHI=1.7 VREFH=0.5 SYNW=8 RTO=7e3 OWTW=20 OCMSUB=1 TGHI=0.8 TGLO=0.467 PERAZ=1 RAZ=6e3 CAZ=3e-3 ROUTSP=$K"
env $B DATAFILE=digits_train.npz SEED=1 python3 gen_mc.py $SLOTS 0.3 1.5e-3 >/dev/null 2>&1
echo "MOS=$(grep -c '^M' mc.cir) weights=$(($K*10+10))"
T0=$(date +%s); ngspice -b mc.cir >/dev/null 2>&1; echo "train $(($(date +%s)-T0))s"
env $B DATAFILE_TE=digits_test.npz AVGW=20 python3 gen_mc_infer.py >/dev/null 2>&1; ngspice -b mc_infer.cir >/dev/null 2>&1
printf "SPARSE N=$N K=$K: "; C=10 python3 score_mc.py "" 2>&1
