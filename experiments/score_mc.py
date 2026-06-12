#!/usr/bin/env python3
# Score multi-class frozen inference: argmax(y_c) per settled slot vs the test labels.
import numpy as np, os as _os, sys
name=sys.argv[1] if len(sys.argv)>1 else "mc"
TAG=_os.environ.get("TAG","")
C=int(_os.environ.get("C","3")); Ts=0.2; hold=Ts-0.02
try: raw=np.loadtxt(f"mc_infer_trace{TAG}.txt")
except Exception: print(f"{name:14s} NO-DATA (sim failed/timeout)"); sys.exit()
if raw.ndim<2 or raw.shape[0]<C: print(f"{name:14s} NO-DATA"); sys.exit()
Y=np.load("mc_test_labels.npy")
t=raw[:,0]; vals=raw[:,1::2]   # columns: y_1..y_C
Npat=len(Y)
slot=np.floor(t/Ts+1e-9).astype(int); frac=t-slot*Ts
pred=[]; conf=[]
for k in range(Npat):
    m=(slot==k)&(frac<hold-0.005); idx=np.where(m)[0]
    if len(idx)==0: pred.append(-1); conf.append(0); continue
    yk=vals[idx[-1]]                 # settled outputs for slot k
    pred.append(int(np.argmax(yk))); conf.append(float(np.ptp(yk)))
pred=np.array(pred); acc=(pred==Y).mean(); chance=1.0/C
# per-class recall
perc=[ (Y==c).sum() and (pred[Y==c]==c).mean() for c in range(C)]
print(f"{name:14s} acc={acc*100:5.1f}%  chance={chance*100:4.1f}%  n={Npat}  "
      f"per-class={[round(100*p) for p in perc]}  mean-margin={np.mean(conf):.2f}"
      f"{'  *** ABOVE CHANCE' if acc>chance*1.5 else ''}")
