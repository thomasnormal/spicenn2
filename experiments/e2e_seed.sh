#!/bin/bash
SPICENN_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
# args: seed [EXTRA_ENV=val ...].  Trains one continuous-time XOR run and prints accuracy.
s=$1; shift
H=${H:-6}; NEPOCH=${NEPOCH:-140}; TH=${TH:-150}
env H=$H UPD=1 NEPOCH=$NEPOCH KCYC=1 TH=$TH SEED=$s SO=${SO:--1} SH=${SH:--1} CELL=${CELL:-gupd} \
    GBN=${GBN:-0.8} GBN1=${GBN1:-0.45} GBNH=${GBNH:-0.8} CW=${CW:-8p} CMW=${CMW:-700u} NDIV=${NDIV:-0.08} \
    RUNTAG=_e$s "$@" python3 "$SPICENN_ROOT/experiments/gen_e2e.py" >/dev/null 2>&1
timeout 700 ngspice -b e2e_e$s.cir >/dev/null 2>&1
NEPOCH=$NEPOCH TH=$TH python3 "$SPICENN_ROOT/experiments/e2e_acc.py" _e$s 2>/dev/null || echo "seed $s: no output (timeout/err)"
