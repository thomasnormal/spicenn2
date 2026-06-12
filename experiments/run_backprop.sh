#!/bin/bash
# In-circuit BACKPROP through a hidden layer: 16->9->C, fully-transistor, trained in one ngspice
# .tran (forward + transpose-read backward + weight-cap updates all physical). Requested topology:
# 4x4 inputs -> 3x3 hidden units each with an OVERLAPPING 2x2 receptive field -> C classes.
#
# The four things that make hidden-layer backprop train (each diagnosed, see BACKPROP_DEPTH.md):
#   1. ACT=tanh  : bounded center-biased neuron (units swing both sides -> real nonlinearity;
#                  ReLU2 sits saturated-on and degenerates). Differential pair: VMID/VTB/TW.
#   2. NORM=zscore: per-image energy equalization (removes the input-energy class asymmetry).
#   3. CMSUB=1   : subtract the backward COMMON-MODE. The transpose-read d1pre_j is dominated by a
#                  shared offset (compressed outputs sit above one-hot targets -> sum_c err != 0),
#                  which drifts every hidden cap the same way -> collapse. d1bar = mean_j(d1pre_j),
#                  used as the hidden-update reference, removes it. RCMB=10e3 = tight averaging.
#   4. exact output delta rule (always faithful) + faithful transpose backward (measured cos 0.92).
#
# Result: 8/8 seeds train 0,1,7 to 72-85% (mean ~79%), all classes alive. Usage: LABELS=0,1,7 bash run_backprop.sh
cd ~/spicenn2
LABELS=${LABELS:-0,1,7}; NC=$(echo $LABELS | tr ',' '\n' | wc -l)
CFG="D=16 C=$NC H=9 CONN=rf3x3 INLO=0.3 INHI=1.7 VREFH=0.5 INITH=0.8 INIT=0.3 \
ACT=tanh TW=32 VTB=1.4 VMID=0.5 BHID=0.3 OWTW=20 CMSUB=1 RCMB=10e3"
NORM=zscore python3 prep_mnist.py $LABELS ${NTR:-24} ${NTE:-50} ${NVAL:-30}
env $CFG DATAFILE=digits_train.npz SEED=${SEED:-5} python3 gen_mc.py ${SLOTS:-1400} 0.3 5e-4 >/dev/null
timeout 400 ngspice -b mc.cir >/dev/null 2>&1
env $CFG DATAFILE_TE=digits_test.npz AVGW=20 python3 gen_mc_infer.py >/dev/null
timeout 150 ngspice -b mc_infer.cir >/dev/null 2>&1
C=$NC python3 score_mc.py "BACKPROP $LABELS (16->9->$NC)"
