#!/bin/bash
SPICENN_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
# Reliable, DETERMINISTIC MNIST-4x4 multi-class training fully in ngspice.
# Recipe = DIRECT (hidden-free) linear multi-class delta-rule classifier + z-score (energy-equalized)
# inputs. The output-weight update is an EXACT in-circuit gradient (error x input), so no
# approximate transpose-read backward and no 2-vs-1 collapse. z-score per image removes the
# input-energy asymmetry (sparse digits summing less current) that otherwise starves one class.
# No noise, no restart, seed-independent.  Usage: LABELS=0,1,7 bash run_mnist.sh
cd "$SPICENN_ROOT"
LABELS=${LABELS:-0,1,7}; NC=$(echo $LABELS | tr ',' '\n' | wc -l)
CFG="D=16 C=$NC INLO=0.3 INHI=1.7 VREFH=0.5 DIRECT=1 H=8 INIT=0.05"
NORM=zscore python3 "$SPICENN_ROOT/experiments/prep_mnist.py" $LABELS ${NTR:-24} ${NTE:-50} ${NVAL:-30}
env $CFG DATAFILE=digits_train.npz OWTW=20 SEED=${SEED:-1} python3 "$SPICENN_ROOT/experiments/gen_mc.py" ${SLOTS:-1400} 0.3 5e-4 >/dev/null
timeout 300 ngspice -b mc.cir >/dev/null 2>&1
env $CFG DATAFILE_TE=digits_test.npz AVGW=20 python3 "$SPICENN_ROOT/experiments/gen_mc_infer.py" >/dev/null
timeout 150 ngspice -b mc_infer.cir >/dev/null 2>&1
C=$NC python3 "$SPICENN_ROOT/experiments/score_mc.py" "MNIST $LABELS"
