#!/bin/bash
# Monte-Carlo mismatch queue: fire next chip when <=2 of our spectre jobs run.
# Usage: bash mm_queue.sh <MMVT> <MMSEED>  e.g. bash mm_queue.sh 0.010 1
BASE="AFLOOR=0.4 ANNS=2560 BKSIGN=1 C=4 CHL=0 CWW=300p EVK=8 FANIN=4 GBLH=0.6 GBLO=0.6 GDEG=12k HINGE=0 IND=0.3 LAYERS=64,32,16,8 LQ=30000 MT=4 NEP=64 NTE=25 NTR=40 RCLAMP=120k RDEG=12k RWL=2g SCALE=1.0 SEED=1 SGNH=-1 SGNO=-1 SIM=spectre SINIT=0 SOFTC=1 SQ=0 STEP=2n STO=42000 TASK=digits TD=0.1 TFG=0 TH=400 VBBK=0.6 WINIT=1.5 WINITO=0.3 WSEED=1"
s=$1; cs=$2; t=mm$(python3 -c "print(int($s*1000))")s$cs
env $BASE RUNTAG=$t MMVT=$s MMSEED=$cs nohup timeout 42100 python3 -u pc_deep.py > $t.log 2>&1 &
echo "launched $t pid $!"
