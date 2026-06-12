#!/usr/bin/env python3
# Measure the FIDELITY of the circuit's transpose-read backward against the ideal gradient signal.
# Run gen_mc.py with FREEZE=1 PROBE=1 first (frozen random weights, logs d1pre/z1/hv/uon).
# ideal delta_j = sum_c W2_cj*(y_c - t_c) ; circuit delta_j = d1pre_j - vrefd(1.2).
# Reports per-slot cosine similarity (how aligned the backward direction is) over hidden units.
import numpy as np, os
H=int(os.environ.get("H","9")); C=int(os.environ.get("C","3")); D=int(os.environ.get("D","16"))
VC,Gs=1.8,1.2; vrefd=1.2; Ts=float(os.environ.get("TS","0.3")); hold=Ts*0.6
CONN=os.environ.get("CONN","rf3x3")
# hidden weight count (to locate W2 in allgp)
if CONN=="rf3x3": nhw=H*(4+1)
elif CONN=="patch2x2": nhw=H*(4+1)
else: nhw=H*(D+1)
W=np.loadtxt("mc_weights.txt"); wv=(W[-1,1::2]-VC)/Gs        # frozen -> last row, decode to weight
W2=np.zeros((C,H))
for c in range(C):
    for j in range(H): W2[c,j]=wv[nhw+c*(H+1)+j]
tr=np.loadtxt("mc_trace.txt"); pr=np.loadtxt("mc_probe.txt")
t=tr[:,0]; y=tr[:,1:1+2*C:2]; tg=tr[:,1+2*(C+D):1+2*(C+D)+2*C:2]
d1=pr[:,1:1+2*H:2]
slot=np.floor(t/Ts+1e-9).astype(int); frac=t-slot*Ts
coss=[]; mags=[]
for k in range(2,slot.max()):                # skip first slots (settling)
    idx=np.where((slot==k)&(frac>hold))[0]
    if len(idx)==0: continue
    i=idx[-1]
    e=y[i]-tg[i]
    ideal=W2.T@e                              # (H,)
    circ=d1[i]-vrefd                          # (H,)
    if np.linalg.norm(ideal)<1e-6 or np.linalg.norm(circ)<1e-6: continue
    coss.append(float(ideal@circ/(np.linalg.norm(ideal)*np.linalg.norm(circ))))
    mags.append((np.linalg.norm(ideal),np.linalg.norm(circ)))
coss=np.array(coss)
print(f"backward fidelity over {len(coss)} slots: mean cos={coss.mean():.3f}  median={np.median(coss):.3f}  "
      f"frac>0.7={np.mean(coss>0.7):.2f}  frac<0={np.mean(coss<0):.2f}")
print(f"  W2 range [{W2.min():.2f},{W2.max():.2f}]  |ideal| med={np.median([m[0] for m in mags]):.3f}  |circ| med={np.median([m[1] for m in mags]):.3f}")
