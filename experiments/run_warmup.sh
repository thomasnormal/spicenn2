#!/bin/bash
# C=4 warm-up head-to-head: bootstrap with none, then branch -> none-continue vs competition.
# Tests whether the competition stage improves a bootstrapped multi-class net.
cd ~/spicenn2
export D=4 H=8 C=4 VREFH=0.5 BHID=0.6 INITH=0.6
infer_score () { TESTSEED=2 NTE=20 AVGW=15 python3 gen_mc_infer.py >/dev/null 2>&1; timeout 80 ngspice -b mc_infer.cir >/dev/null 2>&1; C=4 python3 score_mc.py "" 2>&1; }
for SEED in 1 7 13; do
  export SEED
  COMP=none python3 gen_mc.py 600 0.3 5e-4 >/dev/null 2>&1; timeout 200 ngspice -b mc.cir >/dev/null 2>&1; cp mc_weights.txt w1_$SEED.txt
  printf "C=4 s=%-2s boot         " "$SEED"; infer_score
  COMP=none ICW=w1_$SEED.txt python3 gen_mc.py 600 0.3 5e-4 >/dev/null 2>&1; timeout 200 ngspice -b mc.cir >/dev/null 2>&1
  printf "C=4 s=%-2s +none-cont   " "$SEED"; infer_score
  COMP=subtractive ICW=w1_$SEED.txt python3 gen_mc.py 600 0.3 5e-4 >/dev/null 2>&1; timeout 250 ngspice -b mc.cir >/dev/null 2>&1
  printf "C=4 s=%-2s +competition " "$SEED"; infer_score
done
echo "=== DONE ==="
