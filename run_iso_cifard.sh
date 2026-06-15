#!/bin/bash
# Keystone C=10 on CIFAR-10 color-8 (192-dim) — random features + zero-sum + PERAZ, ngspice (token-free).
# Tests whether the PERAZ anti-collapse mechanism generalizes beyond digits. Args: DIR N NTR NTE SLOTS SEED
DIR=$1; N=$2; NTR=${3:-40}; NTE=${4:-60}; SLOTS=${5:-5000}; SD=${6:-1}
rm -rf "$DIR" && mkdir -p "$DIR" && cd "$DIR" || exit 1
cp ~/spicenn2/gen_mc.py ~/spicenn2/gen_mc_infer.py ~/spicenn2/score_mc.py . ; cp ~/spicenn2/data/cifar_color_8.npz .
python3 - "$N" "$NTR" "$NTE" <<'PY'
import numpy as np, sys
N,ntr,nte=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])
d=np.load('cifar_color_8.npz'); X=np.concatenate([d['Xtr'],d['Xte']]).astype(float); y=np.concatenate([d['ytr'],d['yte']])
rng=np.random.default_rng(0); idx=rng.permutation(len(y)); X,y=X[idx],y[idx]
X=(X-X.mean(1,keepdims=True))/(X.std(1,keepdims=True)+1e-6)
k=4;r=np.random.default_rng(1);D=192
P=[(r.choice(D,k,replace=False),r.normal(0,1,k),r.normal(0,0.5)) for _ in range(N)]
ex=lambda A: A  # identity: use raw color features directly (CIFAR is linearly separable; random feats hurt)
def pick(n,off):
    Xs=[];Ys=[]
    for L in range(10):
        ix=np.where(y==L)[0][off:off+n];Xs.append(ex(X[ix]));Ys+=[L]*n
    return np.vstack(Xs),np.array(Ys)
Xtr,Ytr=pick(ntr,0);Xte,Yte=pick(nte,ntr)
np.savez('digits_train.npz',X=(Xtr+1)/2,Y=Ytr);np.savez('digits_test.npz',X=(Xte+1)/2,Y=Yte)
from sklearn.linear_model import LogisticRegression
print('CIFAR N=%d ideal=%.1f%%'%(N,100*LogisticRegression(max_iter=400).fit((Xtr+1)/2,Ytr).score((Xte+1)/2,Yte)))
PY
SYNW=$(python3 -c "print(round(4*96/$N,3))")
B="D=$N C=10 DIRECT=1 INIT=0.1 INLO=0.3 INHI=1.7 VREFH=0.5 SYNW=$SYNW RTO=7e3 OWTW=20 OCMSUB=1 TGHI=0.8 TGLO=0.467 PERAZ=1 RAZ=6e3 CAZ=3e-3"
env $B DATAFILE=digits_train.npz SEED=$SD python3 gen_mc.py $SLOTS 0.3 1.5e-3 >/dev/null
ngspice -b mc.cir >/dev/null 2>&1
env $B DATAFILE_TE=digits_test.npz AVGW=20 python3 gen_mc_infer.py >/dev/null; ngspice -b mc_infer.cir >/dev/null 2>&1
printf "CIFAR keystone N=%s: " "$N"; C=10 python3 score_mc.py ""
