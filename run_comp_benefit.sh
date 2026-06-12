#!/bin/bash
# Competition benefit in its stable regime: harder C=3 (spread=1.4, none won't saturate).
# Bootstrap with none, then branch: none-continue (control) vs warm-start competition.
cd ~/spicenn2
export D=4 H=8 C=3 SPREAD=1.4 VREFH=0.5 BHID=0.6 INITH=0.6 VULK=0.6 OWTW=1
score () { TESTSEED=2 NTE=25 AVGW=15 python3 gen_mc_infer.py >/dev/null 2>&1; timeout 80 ngspice -b mc_infer.cir >/dev/null 2>&1; C=3 python3 score_mc.py "" 2>&1; }
for SEED in 1 7 13; do
  export SEED
  COMP=none python3 gen_mc.py 1200 0.3 5e-4 >/dev/null 2>&1; timeout 350 ngspice -b mc.cir >/dev/null 2>&1; cp mc_weights.txt wb_$SEED.txt
  printf "C3h s=%-2s boot         " "$SEED"; score
  COMP=none ICW=wb_$SEED.txt python3 gen_mc.py 800 0.3 5e-4 >/dev/null 2>&1; timeout 300 ngspice -b mc.cir >/dev/null 2>&1
  printf "C3h s=%-2s +none-cont   " "$SEED"; score
  COMP=subtractive RCM=2300 ICW=wb_$SEED.txt python3 gen_mc.py 800 0.3 5e-4 >/dev/null 2>&1; timeout 350 ngspice -b mc.cir >/dev/null 2>&1
  printf "C3h s=%-2s +competition " "$SEED"; score
done
echo "=== DONE ==="
