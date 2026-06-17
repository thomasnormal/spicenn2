#!/bin/bash
# Keystone C=10 on SPECTRE (provenance match for the paper). Args: DIR N NTR NTE SLOTS
DIR=$1; N=$2; NTR=${3:-40}; NTE=${4:-60}; SLOTS=${5:-4000}
rm -rf "$DIR" && mkdir -p "$DIR" && cd "$DIR" || exit 1
cp ~/spicenn2/gen_mc.py ~/spicenn2/gen_mc_infer.py ~/spicenn2/score_mc.py ~/spicenn2/spectre_mc.py ~/spicenn2/mnist_8x8.npz .
python3 - "$N" "$NTR" "$NTE" <<'PY'
import numpy as np, sys
N,ntr,nte=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])
d=np.load('mnist_8x8.npz'); X=d['X'].astype(float); y=d['y']
rng=np.random.default_rng(0); idx=rng.permutation(len(y)); X,y=X[idx],y[idx]
X=(X-X.mean(1,keepdims=True))/(X.std(1,keepdims=True)+1e-6)
k=4;r=np.random.default_rng(1);D=64
P=[(r.choice(D,k,replace=False),r.normal(0,1,k),r.normal(0,0.5)) for _ in range(N)]
ex=lambda A: np.stack([np.tanh(A[:,i]@w+b) for i,w,b in P],1)
def pick(n,off):
    Xs=[];Ys=[]
    for L in range(10):
        ix=np.where(y==L)[0][off:off+n];Xs.append(ex(X[ix]));Ys+=[L]*n
    return np.vstack(Xs),np.array(Ys)
Xtr,Ytr=pick(ntr,0);Xte,Yte=pick(nte,ntr)
np.savez('digits_train.npz',X=(Xtr+1)/2,Y=Ytr);np.savez('digits_test.npz',X=(Xte+1)/2,Y=Yte)
PY
SYNW=$(python3 -c "print(round(4*96/$N,3))")
B="D=$N C=10 DIRECT=1 INIT=0.1 INLO=0.3 INHI=1.7 VREFH=0.5 SYNW=$SYNW RTO=7e3 OWTW=20 OCMSUB=1 TGHI=0.8 TGLO=0.467 PERAZ=1 RAZ=6e3 CAZ=3e-3"
env $B DATAFILE=digits_train.npz SEED=1 python3 gen_mc.py $SLOTS 0.3 1.5e-3 >/dev/null
echo "deck $(ls -la mc.cir|awk '{print $5}')B; training on SPECTRE..."
T0=$(date +%s); MT=4 python3 spectre_mc.py train >/dev/null 2>&1; echo "train $(($(date +%s)-T0))s"
env $B DATAFILE_TE=digits_test.npz AVGW=20 python3 gen_mc_infer.py >/dev/null; MT=4 python3 spectre_mc.py infer >/dev/null 2>&1
printf "SPECTRE N=%s: " "$N"; C=10 python3 score_mc.py ""
