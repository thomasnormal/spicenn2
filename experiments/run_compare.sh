#!/bin/bash
# From-scratch comparison: none vs subtractive competition, C=3 and C=6, multiple seeds.
# Trains in ngspice, scores on the real SPICE inference deck (AVGW=20 late-weight average).
cd ~/spicenn2
export D=4 H=8 VREFH=0.5 BHID=0.6 INITH=0.6
for C in 3 6; do
  for COMP in none subtractive; do
    for SEED in 1 7 13; do
      export C COMP SEED
      python3 gen_mc.py 900 0.3 5e-4 >/dev/null 2>&1
      timeout 250 ngspice -b mc.cir > /dev/null 2>&1
      TESTSEED=2 NTE=20 AVGW=20 python3 gen_mc_infer.py >/dev/null 2>&1
      timeout 80 ngspice -b mc_infer.cir > /dev/null 2>&1
      printf "C=%s %-12s seed=%-2s " "$C" "$COMP" "$SEED"
      python3 score_mc.py "" 2>&1
    done
  done
done
echo "=== DONE ==="
