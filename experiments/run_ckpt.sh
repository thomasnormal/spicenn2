#!/bin/bash
SPICENN_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
# Self-resuming checkpointed in-circuit training (survives the harness's periodic background-command
# restarts). Each iteration: load weights from ckpt.txt (if present), train a SHORT chunk that fits
# inside one restart interval, save weights back to ckpt.txt. On restart the script resumes from the
# checkpoint -> training accumulates across restarts. (IDEAS2 #11/#23: accumulate + transfer.)
# Progress is appended to ckpt_log.txt so it survives restarts too. Run ONE at a time.
cd "$SPICENN_ROOT"
N=${N:-96}; PER=${PER:-1200}; CG=${CG:-1.5e-3}
B="D=$N C=10 DIRECT=1 INIT=0.1 INLO=0.3 INHI=1.7 VREFH=0.5 SYNW=4 RTO=7e3 OWTW=20 OCMSUB=1 TGHI=0.8 TGLO=0.467 PERAZ=1"
for it in $(seq 1 40); do
  if [ -s ckpt.txt ]; then IC="ICW=ckpt.txt"; else IC=""; fi
  env $B $IC DATAFILE=digits_train.npz SEED=1 python3 "$SPICENN_ROOT/experiments/gen_mc.py" $PER 0.3 $CG >/dev/null 2>&1 || continue
  ngspice -b mc.cir >/dev/null 2>&1
  # only advance the checkpoint if this chunk produced a full-size weights file (not a killed/partial run)
  if [ -s mc_weights.txt ]; then
    rows=$(wc -l < mc_weights.txt)
    [ "$rows" -gt 5 ] && cp mc_weights.txt ckpt.txt
  fi
  cum=$(cat cum.txt 2>/dev/null || echo 0); cum=$((cum+PER)); echo $cum > cum.txt
  echo "iter $it: ~$cum cumulative slots ($(date +%H:%M:%S))" >> ckpt_log.txt
  # score every 4 iterations so progress is visible even across restarts
  if [ $((it % 4)) -eq 0 ]; then
    env $B ICW=ckpt.txt DATAFILE_TE=digits_test.npz AVGW=20 python3 "$SPICENN_ROOT/experiments/gen_mc_infer.py" >/dev/null 2>&1
    ngspice -b mc_infer.cir >/dev/null 2>&1
    printf "  @%s slots: " "$cum" >> ckpt_log.txt; C=10 python3 "$SPICENN_ROOT/experiments/score_mc.py" "" >> ckpt_log.txt 2>&1
  fi
done
