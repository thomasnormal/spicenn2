#!/bin/bash
SPICENN_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
# Self-resuming SPARSE checkpoint trainer in an ISOLATED dir. Short chunks (PER slots) each complete
# inside the ~33-min disruption window and checkpoint weights to ckpt.txt; on restart it resumes.
# Feature selection (mc_sel.txt) computed once, reused (gen_mc reuses it if present). Launch ONCE.
# Usage: bash run_sparse_ckpt.sh <DIR> <N> <K> <NTR> <PER>
DIR=$1; N=$2; K=$3; NTR=${4:-40}; PER=${5:-1500}
mkdir -p "$DIR" && cd "$DIR" || exit 1
# SINGLE-INSTANCE lock: if the harness respawns this command into a duplicate, the duplicate exits
# instead of clobbering the shared deck/checkpoint. The one holding the lock keeps training/resuming.
exec 9>"$DIR/.lock"; flock -n 9 || { echo "another instance holds the lock; exiting" >> "$DIR/ckpt_log.txt"; exit 0; }
if [ ! -f digits_train.npz ]; then
  cp "$SPICENN_ROOT/experiments/gen_mc.py" "$SPICENN_ROOT/experiments/gen_mc_infer.py" "$SPICENN_ROOT/experiments/score_mc.py" "$SPICENN_ROOT/mnist_8x8.npz" .
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
print('built N=%d ntr=%d'%(N,ntr))
PY
fi
B="D=$N C=10 DIRECT=1 INIT=0.1 INLO=0.3 INHI=1.7 VREFH=0.5 SYNW=8 RTO=7e3 OWTW=20 OCMSUB=1 TGHI=0.8 TGLO=0.467 PERAZ=1 RAZ=6e3 CAZ=3e-3 ROUTSP=$K"
for it in $(seq 1 80); do
  T=$$   # per-instance tag: respawn-duplicates use separate deck/weight files, never clobber
  if [ -s ckpt.txt ]; then IC="ICW=ckpt.txt"; else IC=""; fi
  env $B TAG=$T $IC DATAFILE=digits_train.npz SEED=1 python3 "$SPICENN_ROOT/experiments/gen_mc.py" $PER 0.3 1.5e-3 >/dev/null 2>&1 || continue
  ngspice -b mc$T.cir >/dev/null 2>&1
  if [ -s mc_weights$T.txt ] && [ "$(wc -l <mc_weights$T.txt)" -gt 5 ]; then cp mc_weights$T.txt ckpt_$T.tmp && mv -f ckpt_$T.tmp ckpt.txt; fi  # atomic checkpoint
  cum=$(cat cum.txt 2>/dev/null||echo 0); cum=$((cum+PER)); echo $cum>cum.txt
  if [ $((it % 3)) -eq 0 ] && [ -s ckpt.txt ]; then
    env $B TAG=$T WFILE=ckpt.txt ICW=ckpt.txt DATAFILE_TE=digits_test.npz AVGW=20 python3 "$SPICENN_ROOT/experiments/gen_mc_infer.py" >/dev/null 2>&1
    ngspice -b mc_infer$T.cir >/dev/null 2>&1
    printf "@%s slots: " "$cum">>ckpt_log.txt; TAG=$T C=10 python3 "$SPICENN_ROOT/experiments/score_mc.py" "">>ckpt_log.txt 2>&1
  fi
done
echo "DONE $(cat cum.txt) slots" >> ckpt_log.txt
