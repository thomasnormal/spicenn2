#!/bin/bash
# Reliable XOR training via restart-on-failure: for each "task", train and retry with a fresh
# init until 4/4 corners (or MAXTRY). With a ~7/8 single-run rate, P(fail K times) ~ (1/8)^K,
# so a couple of restarts make XOR training effectively certain. Works even at H=6.
cd ~/spicenn2
base="TS=0.3 DTFRAC=0.34 VTOSYN=0.2 VC=1.8 GS=1.2 RTH=8e3 RTO=7e3 VG0=0.7 LO=0.7 HI=1.3 VREFH=0.5 VREFO=0.5 BWD=trans2 RTBK=8e3 VCBIAS=1.0 ACT=tanh VMID=0.5 TW=32 VTB=1.4 OWTW=20 RH=8e3 H=${H:-6}"
MAXTRY=${MAXTRY:-5}
ok=0
for task in 1 2 3 4 5 6; do
  done=0
  for a in $(seq 1 $MAXTRY); do
    S=$((task*100+a))
    env $base SEED=$S python3 gen_xor_full.py 1000 0.3 5e-4 >/dev/null 2>&1
    timeout 200 ngspice -b xor_full.cir >/dev/null 2>&1
    env $base SEED=$S AVGW=40 python3 gen_infer.py >/dev/null 2>&1
    timeout 60 ngspice -b xor_infer.cir >/dev/null 2>&1
    r=$(python3 score.py "")
    if echo "$r" | grep -q SOLVED; then
      echo "task $task: SOLVED on attempt $a   $r"; done=1; ok=$((ok+1)); break
    fi
  done
  [ $done -eq 0 ] && echo "task $task: FAILED after $MAXTRY attempts"
done
echo "=== restart reliability (H=${H:-6}): $ok/6 tasks reached 4/4 within $MAXTRY restarts ==="
