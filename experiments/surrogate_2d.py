#!/usr/bin/env python3
# NUMPY screening surrogate for the gen_mc 2->H->C DENSE in-circuit backprop net on 2D tasks.
# Models the analog trainer faithfully: tanh diff-pair hidden, COMPRESSED output keystone (YSQUASH,
# y in [YLO,YHI]), transpose-read backward delta=W2^T(y-t) with a derivative gate, and CMSUB
# (subtract backward common-mode). Goal: find a recipe that clears the bar BEFORE spending Spectre.
import numpy as np, sys, os
task=os.environ.get("TASK","spirals")
tr=np.load(f"/tmp/insp/{task}_train.npz"); te=np.load(f"/tmp/insp/{task}_test.npz")
Xa,ya=tr[tr.files[0]],tr[tr.files[1]]; Xb,yb=te[te.files[0]],te[te.files[1]]
INLO,INHI=0.3,1.7
def band(X): return INLO+X*(INHI-INLO)            # [0,1] -> input voltage band (same as gen_mc scale_fwd)
ntr=int(os.environ.get("NTR","160"))
rng0=np.random.default_rng(0); idx=rng0.permutation(len(Xa))[:ntr]
Xtr=band(Xa[idx]); Ytr=ya[idx]; Xte=band(Xb); Yte=yb
C=int(max(Ytr.max(),Yte.max()))+1; D=2; H=int(os.environ.get("H","96"))
ACT=os.environ.get("ACT","tanh"); GATE=os.environ.get("GATE","true")
VMID=float(os.environ.get("VMID","1.0"))          # tanh center (input band centers ~1.0)
GAIN=float(os.environ.get("GAIN","1.0"))          # tanh gain (VTB/TW analog)
def act(z): return np.tanh(GAIN*(z-VMID)) if ACT=="tanh" else np.maximum(z,0)
def dact(z,h): return GAIN*(1-h*h) if ACT=="tanh" else (z>0).astype(float)
def gate(z,h):
    if GATE=="true": return dact(z,h)
    if GATE=="step": return (z>VMID).astype(float)
    if GATE=="ones": return np.ones_like(z)
YSQUASH=int(os.environ.get("YSQUASH","1")); YLO=float(os.environ.get("YLO","0.1")); YHI=float(os.environ.get("YHI","0.6"))
def outy(raw): return (0.5*(YLO+YHI)+0.5*(YHI-YLO)*np.tanh(raw)) if YSQUASH else raw
def douty(raw,y):
    if YSQUASH: return 0.5*(YHI-YLO)*(1-np.tanh(raw)**2)
    return np.ones_like(raw)
CMSUB=int(os.environ.get("CMSUB","1"))
def run(seed):
    rng=np.random.default_rng(seed)
    iw=float(os.environ.get("INITH","2.0")); bw=float(os.environ.get("BHID","2.0")); ow=float(os.environ.get("INITO","0.1"))
    W1=rng.standard_normal((H,D))*iw; b1=rng.standard_normal(H)*bw+VMID
    W2=rng.standard_normal((C,H))*ow; b2=np.zeros(C)
    lr1=float(os.environ.get("LR1","0.05")); lr2=float(os.environ.get("LR2","0.05"))
    EP=int(os.environ.get("EP","100")); THI=float(os.environ.get("THI","1.0")); TLO=float(os.environ.get("TLO","0.0"))
    HINGE=int(os.environ.get("HINGE","0"))
    n=len(Xtr); ix=np.arange(n)
    for ep in range(EP):
        rng.shuffle(ix)
        for k in ix:
            x=Xtr[k]; t=np.full(C,TLO); t[Ytr[k]]=THI
            z=W1@x+b1; h=act(z); raw=W2@h+b2; y=outy(raw)
            if HINGE and np.argmax(y)==Ytr[k] and y[Ytr[k]]-np.sort(y)[-2]>float(os.environ.get("MARG","0.1")): continue
            e=(y-t)*douty(raw,y)
            if CMSUB: ec=e-e.mean()
            else: ec=e
            W2-=lr2*np.outer(e,h); b2-=lr2*e
            delta=(W2.T@ec)*gate(z,h)
            W1-=lr1*np.outer(delta,x); b1-=lr1*delta
    def acc(X,Y):
        z=X@W1.T+b1; h=act(z); y=outy(h@W2.T+b2); return np.mean(np.argmax(y,1)==Y)
    return acc(Xtr,Ytr), acc(Xte,Yte)
seeds=[int(s) for s in os.environ.get("SEEDS","0,1,2").split(",")]
res=[run(s) for s in seeds]
print(f"{task} H={H} ACT={ACT} GATE={GATE} INITH={os.environ.get('INITH','2.0')} BHID={os.environ.get('BHID','2.0')} "
      f"GAIN={GAIN} YSQUASH={YSQUASH} CMSUB={CMSUB} LR1={os.environ.get('LR1','0.05')} EP={os.environ.get('EP','100')} "
      f"HINGE={os.environ.get('HINGE','0')}: train={[round(r[0],3) for r in res]} TEST={[round(r[1],3) for r in res]}")
# --- circuit-faithful variant: clipped OTA update + railed weights ---
