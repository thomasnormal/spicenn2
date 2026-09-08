#!/bin/bash
SPICENN_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
# args: cfgname width seed [EXTRA_ENV=val ...]
cfg=$1 nh=$2 s=$3; shift 3
raw=$(env NEU=base RUNTAG=${cfg}_${nh}_${s} NHID=$nh VW0=0.40 WLOAD=3000u RB=2e5 TD=0.10 \
  USIGN=-1 ETA=0.006 MOM=0.3 ITERS=160 SEED=$s UPDATE=sqdiff "$@" \
  timeout 120 python3 "$SPICENN_ROOT/experiments/ep_dtcm.py" train 2>&1)
line=$(echo "$raw" | grep -oE "FINAL_acc=[0-9.]+ \(stable\) best_acc=[0-9.]+")
fin=$(echo "$line" | grep -oE "FINAL_acc=[0-9.]+" | cut -d= -f2)
bst=$(echo "$line" | grep -oE "best_acc=[0-9.]+"  | cut -d= -f2)
echo "$cfg $nh $s fin=${fin:-NA} best=${bst:-NA}"
