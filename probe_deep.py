#!/usr/bin/env python3
# Per-layer backward-fidelity probe for gen_deep depth-2 (4x4->2x2x4->1x1x16->3). Run gen_deep with
# FREEZE=1 PROBE=1 first. Measures cosine of the circuit's propagated delta vs the IDEAL gradient at
# each stage, and specifically whether propagating the UNGATED dpre (vs tanh'-gated) is the fidelity loss.
import numpy as np, os
VC,Gs,vrefd=1.8,1.2,1.2; C=3; Ts=float(os.environ.get("TS","0.3")); ROUTK=int(os.environ.get("ROUTK","4"))
# --- reconstruct connectivity exactly as gen_deep ---
def build_layers(RES,ARCH):
    layers=[]; H=W=RES; Cin=1
    for li,part in enumerate(ARCH.split(",")):
        Cout,ks,st,fi=[int(x) for x in part.split(":")]
        Hout=(H-ks)//st+1; Wout=(W-ks)//st+1; nin=H*W*Cin; nout=Hout*Wout*Cout
        conn=[[] for _ in range(nout)]; r=np.random.default_rng(li+1)
        for io in range(Hout):
          for jo in range(Wout):
            rf=np.array([(io*st+di)*W*Cin+(jo*st+dj)*Cin+c for di in range(ks) for dj in range(ks) for c in range(Cin)])
            fo=np.zeros(len(rf))
            for co in range(Cout):
                o=(io*Wout+jo)*Cout+co
                cand=sorted(range(len(rf)),key=lambda x:(fo[x],r.random()))[:fi]
                for ci in cand: conn[o].append(int(rf[ci])); fo[ci]+=1
        layers.append((conn,nin,nout)); H,W,Cin=Hout,Wout,Cout
    return layers,H*W*Cin
hl,nfeat=build_layers(4,"4:2:2:4,16:2:2:4")
connL2=hl[1][0]; nL1=hl[0][2]; nL2=hl[1][2]
r=np.random.default_rng(99); out_conn=[list(r.choice(nfeat,ROUTK,replace=False)) for _ in range(C)]
# --- weights (frozen) ---
wv=(np.loadtxt("deep_weights.txt")[-1,1::2]-VC)/Gs
WL1=wv[:nL1*4]; WL2f=wv[nL1*4:nL1*4+nL2*4]; WOf=wv[nL1*4+nL2*4:]
W2=np.zeros((C,nfeat))
for c in range(C):
    for k in range(ROUTK): W2[c,out_conn[c][k]]=WOf[c*ROUTK+k]
WL2=np.zeros((nL2,nfeat))
for n in range(nL2):
    for k in range(4): WL2[n,connL2[n][k]]=WL2f[n*4+k]
# --- probe data ---
tr=np.loadtxt("deep_trace.txt"); pr=np.loadtxt("deep_probe.txt")
t=tr[:,0]; y=tr[:,1:1+2*C:2]
nh=nL1+nL2
tg=pr[:,1:1+2*C:2]; dp=pr[:,1+2*C:1+2*C+2*nh:2]; z=pr[:,1+2*C+2*nh:1+2*C+4*nh:2]
frac=t-np.floor(t/Ts+1e-9)*Ts; sett=frac>Ts*0.6
dpL1=dp[:,:nL1]-vrefd; dpL2=dp[:,nL1:]-vrefd; zL2=z[:,nL1:]
# tanh' of z_L2: NREL square-law neuron; approximate derivative as bump peaked where z near vmid.
# Use a generic saturating-deriv proxy: tanh'(a*(z-zc)) with zc=median, just to test gating effect.
zc=np.median(zL2); tpL2=1-np.tanh(1.5*(zL2-zc))**2
cos=lambda a,b: float((a@b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
c2=[]; c1u=[]; c1g=[]
for i in np.where(sett)[0]:
    e=y[i]-tg[i]
    ideal_L2=W2.T@e
    if np.linalg.norm(ideal_L2)>1e-6 and np.linalg.norm(dpL2[i])>1e-6: c2.append(cos(ideal_L2,dpL2[i]))
    ideal_L1_ungated=WL2.T@dpL2[i]
    ideal_L1_gated  =WL2.T@(dpL2[i]*tpL2[i])
    if np.linalg.norm(dpL1[i])>1e-6:
        if np.linalg.norm(ideal_L1_ungated)>1e-6: c1u.append(cos(ideal_L1_ungated,dpL1[i]))
        if np.linalg.norm(ideal_L1_gated)>1e-6:   c1g.append(cos(ideal_L1_gated,dpL1[i]))
print(f"stage1 (out->L2) cosine: mean={np.mean(c2):.3f}  (depth-1 was ~0.92)")
print(f"stage2 (L2->L1) cosine vs UNGATED ideal: mean={np.mean(c1u):.3f}  (this is what the circuit computes)")
print(f"stage2 (L2->L1) cosine vs tanh'-GATED ideal: mean={np.mean(c1g):.3f}  (what backprop SHOULD do)")
print(f"  n_slots={len(c2)}")
