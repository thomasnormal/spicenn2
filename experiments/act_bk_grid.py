import os, numpy as np, torch
os.environ.update(C="10",NTR="40")
exec(open("experiments/train_faithful.py").read().split('print("=== HIDDEN')[0])
print("=== activation x backward-linearity (frozen 73.5); cleaner-derivative neuron + backward-lin fix? 3-seed ===")
for kind in ["tanh","relu","lrelu"]:
    try:
        lo=avg(g=0.0,bsign="graded",kind=kind)
        hi=avg(g=6.0,bsign="graded",kind=kind)
        print(f"  {kind:6s}: linear-bk(g0)={lo[0]:4.1f}+/-{lo[1]:.1f}   sat-bk(g6)={hi[0]:4.1f}+/-{hi[1]:.1f}   (lin gain={lo[0]-hi[0]:+.1f})")
    except Exception as e:
        import traceback; print(f"  {kind:6s}: ERR {e}")
