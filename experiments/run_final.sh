#!/bin/bash
# Definitive comparison with the slow-output bootstrap fix (OWTW=1).
# none vs subtractive competition across C and seeds; scored on the real SPICE inference deck.
cd ~/spicenn2
export D=4 H=8 BHID=0.6 INITH=0.6 VREFH=0.5 VULK=0.6 OWTW=1
score () { export C; TESTSEED=2 NTE=20 AVGW=15 python3 gen_mc_infer.py >/dev/null 2>&1; timeout 90 ngspice -b mc_infer.cir >/dev/null 2>&1; C=$C python3 score_mc.py "" 2>&1; }
for C in 3 4 6; do
  for COMP in none subtractive; do
    for SEED in 1 7; do
      export C COMP SEED
      python3 gen_mc.py 1400 0.3 5e-4 >/dev/null 2>&1
      timeout 400 ngspice -b mc.cir >/dev/null 2>&1
      printf "C=%s %-12s s=%-2s " "$C" "$COMP" "$SEED"; score
    done
  done
done
echo "=== DONE ==="
