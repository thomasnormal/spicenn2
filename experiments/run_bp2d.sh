#!/bin/bash
# In-circuit BACKPROP on a 2D task (circles/rings/spirals): 2 -> H -> 2, fully-transistor,
# trained in one ngspice .tran (forward + transpose-read backward + weight-cap updates all
# physical; controller only streams x/targets as PWL). Recipe = the proven MNIST backprop
# recipe (BACKPROP_DEPTH.md): ACT=tanh + CMSUB=1 + dense connectivity.
# Usage: TASK=circles H=16 SLOTS=1400 SEED=5 bash run_bp2d.sh
cd ${WDIR:-~/spicenn2}
TASK=${TASK:-circles}
CFG="D=2 C=2 H=${H:-16} CONN=dense INLO=0.3 INHI=1.7 VREFH=0.5 INITH=0.8 INIT=0.3 \
ACT=tanh TW=32 VTB=1.4 VMID=0.5 BHID=0.3 OWTW=20 CMSUB=1 RCMB=10e3"
python3 prep_2d.py $TASK ${NTR:-50} ${NTE:-60} ${NVAL:-30}
env $CFG DATAFILE=digits_train.npz SEED=${SEED:-5} python3 gen_mc.py ${SLOTS:-1400} 0.3 5e-4 >/dev/null
if [ "${SIM:-ngspice}" = "spectre" ]; then
  ln -sf ~/spicenn2/spectre_mc.py spectre_mc.py 2>/dev/null
  AVGW=20 python3 spectre_mc.py train
else
  timeout ${TMO:-1200} ngspice -b mc.cir >/dev/null 2>&1
fi
env $CFG DATAFILE_TE=digits_test.npz AVGW=20 python3 gen_mc_infer.py >/dev/null
if [ "${SIM:-ngspice}" = "spectre" ]; then
  python3 spectre_mc.py infer
else
  timeout 400 ngspice -b mc_infer.cir >/dev/null 2>&1
fi
C=2 python3 score_mc.py "BACKPROP-2D $TASK (2->${H:-16}->2) seed=${SEED:-5} ${SIM:-ngspice}"
