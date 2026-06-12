#!/bin/bash
# Reliable MNIST 4x4 multi-class training via restart-on-collapse with VALIDATION model-selection.
# The trnoise that un-sticks the multi-class collapse is a genuine run-to-run lottery (ngspice
# trnoise is not deterministically seeded), so a single run is ~1-in-few. We train K times, pick
# the attempt with the best BALANCED validation accuracy (mean per-class recall -> collapsed runs
# score ~0 and are rejected), then report that selected model on the held-out TEST set.
# Same idea as restart_xor.sh, but selection is on a real val split (never peeks at test).
cd ~/spicenn2
LABELS=${LABELS:-0,1,7}
NC=$(echo $LABELS | tr ',' '\n' | wc -l)
BCFG="CONN=patch2x2 D=16 C=$NC H=${H:-8} INLO=0.3 INHI=1.7 VREFH=0.5"
TR="BHID=0.6 INITH=0.8 VULK=0.6 OWTW=1 WNOISE=${WNOISE:-1e-4}"
SLOTS=${SLOTS:-1000}; K=${K:-6}
NORM=perimage python3 prep_mnist.py $LABELS ${NTR:-24} ${NTE:-50} ${NVAL:-30}
best=-1; bestatt=0
for a in $(seq 1 $K); do
  env $BCFG $TR SEED=$a NSEED=$a python3 gen_mc.py $SLOTS 0.3 5e-4 >/dev/null 2>&1
  timeout 250 ngspice -b mc.cir >/dev/null 2>&1
  env $BCFG DATAFILE_TE=digits_val.npz AVGW=20 python3 gen_mc_infer.py >/dev/null 2>&1
  timeout 120 ngspice -b mc_infer.cir >/dev/null 2>&1
  v=$(C=$NC python3 valscore.py mc_test_labels.npy)
  printf "  attempt %d: val-balanced=%s\n" "$a" "$v"
  if awk "BEGIN{exit !($v>$best)}"; then best=$v; bestatt=$a; cp mc_weights.txt mc_weights_best.txt; fi
done
echo "=== best attempt $bestatt (val-balanced=$best) -> evaluating on TEST ==="
env $BCFG DATAFILE_TE=digits_test.npz AVGW=20 WFILE=mc_weights_best.txt python3 gen_mc_infer.py >/dev/null 2>&1
timeout 120 ngspice -b mc_infer.cir >/dev/null 2>&1
C=$NC python3 score_mc.py "MNIST $LABELS"
