#!/usr/bin/env python3
# REFINEMENT 1 (the elegant law): does forward signal-rank preservation predict depth-gain?
# For each neuron measure (a) rank-retention through stacked IDEAL layers (one number), and
# (b) depth-gain = deep[8,8,8] minus matched-param wide-shallow[24] on K=3 checkerboard.
# Tight correlation => "a neuron enables depth iff it preserves signal rank." Also tests the
# SIMPLEST bounded neuron (hardtanh, a limiter) — is elegance free?
import numpy as np, neurons as N
from mc_experiment import scale_fit, scale_apply

def rank_retention(name, L=6, w=8, B=400, seed=0):
    # stacked ideal random layers + activation; effective rank of representation at depth L
    rng=np.random.default_rng(seed); A=rng.standard_normal((B,w))
    f,_=N.act(name) if name in N.PERELEM+['hardtanh','lecuntanh'] else (None,None)
    for _ in range(L):
        W=rng.standard_normal((w,w))/np.sqrt(w); Z=A@W
        if name=='rmstanh': r=np.sqrt((Z*Z).mean(1,keepdims=True)+1e-4); A=np.tanh(Z/r)
        elif name=='quad': A=np.tanh(Z*np.roll(Z,1,1))
        else: A=f(Z)
    A=A-A.mean(0); s=np.linalg.svd(A,compute_uv=False); s=s[s>1e-9]; p=s/s.sum()
    return float(np.exp(-(p*np.log(p)).sum()))   # participation ratio (max=w)

names=N.PERELEM+['hardtanh','lecuntanh','rmstanh','quad']
Xtr,Ytr=N.make_checker(600,1,3); Xte,Yte=N.make_checker(400,2,3)
p=scale_fit(Xtr); Xtr=scale_apply(Xtr,p); Xte=scale_apply(Xte,p)
print("neuron      rank-retain(L6/8)   wide-shallow[24]  deep[8,8,8]  depth-gain")
rows=[]
for nm in names:
    rr=np.mean([rank_retention(nm,seed=s) for s in range(3)])
    sh=np.mean([N.mlp([6,24,2],nm,Xtr,Ytr,Xte,Yte,s,ep=600)[1] for s in range(3)])
    dp=np.mean([N.mlp([6,8,8,8,2],nm,Xtr,Ytr,Xte,Yte,s,ep=600)[1] for s in range(3)])
    rows.append((nm,rr,sh,dp,dp-sh))
    print(f"  {nm:10s}    {rr:4.2f}              {sh*100:4.0f}            {dp*100:4.0f}        {(dp-sh)*100:+4.0f}",flush=True)
rr=np.array([r[1] for r in rows]); dg=np.array([r[4] for r in rows])
print(f"\ncorrelation(rank-retention, depth-gain) = {np.corrcoef(rr,dg)[0,1]:.2f}")
