import os, numpy as np, torch
os.environ.update(C="10",NTR="40")
exec(open("experiments/train_faithful.py").read().split('print("=== HIDDEN')[0])  # reuse tf/relax/avg defs
print("=== how much BACKWARD linearization recovers hidden training? (frozen ~73, ideal 78.5) 3-seed ===")
for g in [0.0,2.0,3.0,4.0,6.0]:
    mu,sd=avg(g=g,bsign="graded",kind="tanh")
    print(f"  backward synapse g={g:.1f} (graded): {mu:4.1f}% +/-{sd:.1f}  ({'>frozen TRAINS' if mu>73.5 else '<frozen HURTS'})")
print("=> RDEG=24k ~ g3-4; find the g where training flips above frozen 73.5")
