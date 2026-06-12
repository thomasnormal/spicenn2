#!/bin/bash
# args: H seed   -> trains 2-layer continuous PC XOR, prints "H seed N/4"
h=$1 s=$2
r=$(H=$h SEED=$s SGNH=1 SGNO=-1 NEP=160 RUNTAG=cf${h}_${s} timeout 300 python3 gen_pcL.py 2>&1 | grep "epoch 159" | grep -oE "[0-4]/4" | head -1)
echo "$h $s ${r:-NA}"
