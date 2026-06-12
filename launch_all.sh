#!/bin/bash
cd /home/thomas-ahle/spicenn2
# C=5 robustness: 5 seeds, full eval (the headline result's robustness)
for S in 0 1 2 3 4; do
  C=5 SEED=$S NIN=64 EP=15 EVERY=5 LR=0.04 MB=16 RUNTAG=r5s$S python3 -u pc1_orch.py > /tmp/r5s$S.log 2>&1 &
done
# C=10 (all ten digits): 2 seeds, capped eval to go faster
for S in 0 1; do
  C=10 SEED=$S NIN=64 EP=12 EVERY=4 NEVAL=250 LR=0.04 MB=16 RUNTAG=r10s$S python3 -u pc1_orch.py > /tmp/r10s$S.log 2>&1 &
done
wait
echo "ALL DONE"
