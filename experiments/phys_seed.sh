#!/bin/bash
SPICENN_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
# args: seed [EXTRA_ENV=val ...].  Trains XOR with the PHYSICAL update cell (no numpy prim).
s=$1; shift
r=$(env NEU=base RUNTAG=ph$s NHID=${NHID:-6} VW0=0.40 WLOAD=3000u RB=2e5 TD=0.10 USIGN=-1 \
    PHYSUPD=1 PHISC=${PHISC:-0.5} CWU=${CWU:-80p} GBNU=${GBNU:-0.9} \
    ETA=${ETA:-15} MOM=${MOM:-0.6} ITERS=${ITERS:-150} SEED=$s \
    ANNEAL=${ANNEAL:-cos} AFLOOR=${AFLOOR:-0.15} FREEZE=${FREEZE:-1} MARGIN=0.15 HOLD=2 \
    NRESTART=${NRESTART:-4} STALLW=${STALLW:-70} "$@" \
    timeout 600 python3 "$SPICENN_ROOT/experiments/ep_dtcm.py" train 2>&1 | grep -oE "FINAL_acc=[0-9.]+ \(stable\) best_acc=[0-9.]+ attempts=[0-9]+")
fin=$(echo "$r" | grep -oE "FINAL_acc=[0-9.]+" | cut -d= -f2)
bst=$(echo "$r" | grep -oE "best_acc=[0-9.]+" | cut -d= -f2)
att=$(echo "$r" | grep -oE "attempts=[0-9]+" | cut -d= -f2)
echo "seed $s: final=${fin:-NA} best=${bst:-NA} attempts=${att:-NA}"
