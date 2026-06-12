#!/bin/bash
cd /home/thomas-ahle/spicenn2
# genuine 2-layer analog PC (2-solve injection). Compare frozen-hidden (LRH=0) vs hidden-learning.
for LRH in 0 30 100 300; do
  C=5 H=16 N=16 SEED=0 INJ=1 VBBK=0.25 GMO=400 EINJ=1 LRH=$LRH LR=0.05 EP=25 MB=20 ANNEAL=1 \
    RUNTAG=inj_l$LRH python3 -u pc_orch.py > /tmp/inj_l$LRH.log 2>&1 &
done
wait
echo ALLDONE
