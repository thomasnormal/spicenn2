#!/usr/bin/env python3
# Hungarian (bijective) anti-lock accuracy on a CHOSEN eval block of a (possibly PARTIAL) raw.
# Mirrors pc_deep's slot schedule: EVK interval eval after epoch 8 (block 0) + final (block 1).
# Usage: python3 hung_eval.py TAG C NTR NTE NEP EVK [block_idx=0]
import numpy as np, glob, sys, re, os
from scipy.optimize import linear_sum_assignment
TAG=sys.argv[1]; C=int(sys.argv[2]); NTR=int(sys.argv[3]); NTE=int(sys.argv[4])
NEP=int(sys.argv[5]); EVK=int(sys.argv[6]); BLK=int(sys.argv[7]) if len(sys.argv)>7 else 0
TH=float(os.environ.get("TH","400"))  # per-slot horizon in ns; MUST match the run's TH (was hardcoded 400 -> wrong sampling for TH!=400 runs)
# rebuild eval_block slot starts (CHL -> 2 train slots/item)
ntr_items=NTR*C; nte_items=NTE*C; SLOTS=0; eval_blocks=[]
for ep in range(NEP):
    SLOTS+=ntr_items*2
    if EVK and (ep+1)%EVK==0 and ep<NEP-1:
        eval_blocks.append(SLOTS); SLOTS+=nte_items
eval_blocks.append(SLOTS); SLOTS+=nte_items
blk_start=eval_blocks[BLK]
print(f"block {BLK} starts slot {blk_start} = {blk_start*TH/1e6:.3f}ms ; eval_blocks={eval_blocks} ; total slots={SLOTS} ({SLOTS*TH/1e6:.3f}ms)")
RAW=sorted(glob.glob(f"raw_{TAG}/*.tran*"))[0]
# find output layer index
hdr=set(); fh=open(RAW)
for l in fh:
    if l.strip()=="VALUE": break
    m=re.match(r'"m(\d+)p_0"',l.strip())
    if m: hdr.add(int(m.group(1)))
fh.close(); Lout=max(hdr)
want=set([f"m{Lout}p_{c}" for c in range(C)]+[f"m{Lout}n_{c}" for c in range(C)])
t=[]; D={n:[] for n in want}; fh=open(RAW)
for l in fh:
    if l.strip()=="VALUE": break
for l in fh:
    s=l.strip()
    if s=="END": break
    if not s or s[0]!='"': continue
    q=s.find('"',1)
    if q<0: continue   # partial/mid-write line (raw still being appended) -> skip
    nm=s[1:q]
    if nm=="time": t.append(float(s[q+1:]))
    elif nm in want: D[nm].append(float(s[q+1:]))
fh.close()
n=min([len(t)]+[len(D[k]) for k in want]); t=np.array(t[:n])
tmax=t[-1]; print(f"raw reaches {tmax*1e3:.3f}ms ({tmax/(SLOTS*TH*1e-9)*100:.0f}% of {SLOTS} slots)")
need_t=(blk_start+nte_items-1)*TH*1e-9
if tmax < need_t:
    print(f"NOT YET: need {need_t*1e3:.3f}ms for full block {BLK}, have {tmax*1e3:.3f}ms"); sys.exit(0)
Marg=np.array([np.array(D[f"m{Lout}p_{c}"][:n])-np.array(D[f"m{Lout}n_{c}"][:n]) for c in range(C)])
yte=np.array([c for c in range(C) for _ in range(NTE)])
preds=[]
for k in range(nte_items):
    g=blk_start+k; tt=(g*TH+TH-3)*1e-9
    j=int(np.argmin(np.abs(t-tt))); col=Marg[:,j]-Marg[:,j].mean(); preds.append(int(np.argmax(col)))
preds=np.array(preds)
# Hungarian bijective: confusion matrix (true x pred), maximize assignment
conf=np.zeros((C,C))
for yt,pp in zip(yte,preds): conf[yt,pp]+=1
ri,ci=linear_sum_assignment(-conf)
hung=conf[ri,ci].sum()/len(yte)
raw=(preds==yte).mean()
print(f"{TAG} blk{BLK}: HUNGARIAN={hung:.3f}  raw={raw:.3f}  chance={1/C:.3f}  hist={np.bincount(preds,minlength=C).tolist()}")
