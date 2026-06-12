#!/usr/bin/env python3
# Numeric balanced-accuracy (mean per-class recall) for restart model-selection.
# Reads mc_infer_trace.txt + a labels file (default mc_test_labels.npy). Prints one float.
# Balanced accuracy is 0 when any class is fully collapsed -> restart loop rejects collapse.
import numpy as np, os, sys
C=int(os.environ.get("C","3")); Ts=0.2; hold=Ts-0.02
lab=sys.argv[1] if len(sys.argv)>1 else "mc_test_labels.npy"
try:
    raw=np.loadtxt("mc_infer_trace.txt"); Y=np.load(lab)
except Exception:
    print("-1.0"); sys.exit()
if raw.ndim<2 or raw.shape[0]<C: print("-1.0"); sys.exit()
t=raw[:,0]; vals=raw[:,1::2]; Npat=len(Y)
slot=np.floor(t/Ts+1e-9).astype(int); frac=t-slot*Ts
pred=[]
for k in range(Npat):
    m=(slot==k)&(frac<hold-0.005); idx=np.where(m)[0]
    if len(idx)==0: pred.append(-1); continue
    pred.append(int(np.argmax(vals[idx[-1]])))
pred=np.array(pred)
perc=[ (pred[Y==c]==c).mean() if (Y==c).sum() else 0.0 for c in range(C)]
print(f"{np.mean(perc):.4f}")
