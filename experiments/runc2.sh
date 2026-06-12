#!/bin/bash
# like runc.sh but also reports attempts used. args: cfg width seed [ENV=val ...]
cfg=$1 nh=$2 s=$3; shift 3
raw=$(env NEU=base RUNTAG=${cfg}_${nh}_${s} NHID=$nh VW0=0.40 WLOAD=3000u RB=2e5 TD=0.10 \
  USIGN=-1 ETA=0.006 MOM=0.3 ITERS=220 SEED=$s UPDATE=sqdiff "$@" \
  timeout 180 python3 ep_dtcm.py train 2>&1)
line=$(echo "$raw" | grep -oE "FINAL_acc=[0-9.]+ \(stable\) best_acc=[0-9.]+ attempts=[0-9]+")
fin=$(echo "$line" | grep -oE "FINAL_acc=[0-9.]+" | cut -d= -f2)
att=$(echo "$line" | grep -oE "attempts=[0-9]+" | cut -d= -f2)
echo "$cfg $nh $s fin=${fin:-NA} att=${att:-NA}"
