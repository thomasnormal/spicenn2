#!/usr/bin/env python3
# Honest learning curve: freeze the weights at points along the training trajectory
# and evaluate the full 4-corner XOR statically (forward-only deck) at each checkpoint.
import numpy as np, subprocess, os, re

W=np.loadtxt('weights_full.txt')           # gate-voltage trajectory, one row per tran point
NCP=40
idx=np.linspace(60, len(W)-1, NCP).astype(int)
env=dict(os.environ); env["AVGW"]="1"; env["WFILE"]="wcp.txt"
targets=np.array([0.,1.,1.,0.])

def frozen_mse(row):
    np.savetxt('wcp.txt', W[row:row+1])    # single checkpoint (2D, 1 row)
    subprocess.run(["python3","gen_infer.py"],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    subprocess.run(["ngspice","-b","xor_infer.cir"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=60)
    inf=np.loadtxt('xor_infer_trace.txt'); ti=inf[:,0]
    y=np.array([np.interp((k+0.9)*0.2,ti,inf[:,1]) for k in range(4)])
    return float(np.mean((y-targets)**2)), y

prog=[]; 
for r in idx:
    mse,y=frozen_mse(r); prog.append((r/len(W),mse))
    print(f"checkpoint {r:5d} ({r/len(W)*100:4.0f}%)  frozen MSE={mse:.3f}  y={np.round(y,2)}")
np.savetxt('val_curve.txt', np.array(prog))
print("saved val_curve.txt")
