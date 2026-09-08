#!/usr/bin/env python3
# Fast eval of frozen SPICE weights via the calibrated surrogate forward (predicts SPICE
# inference; matched the collapse exactly). Reports hidden liveness + test argmax accuracy.
from pathlib import Path
import numpy as np, os, sys
from surrogate import keystone, relu_act, VREFH as _vh, RTH, VREFO, RTO, VT_REL
def ef(k,d): return float(os.environ.get(k,d))
def ei(k,d): return int(os.environ.get(k,d))
D=ei("D",4); H=ei("H",8); C=ei("C",3)
VREFH=ef("VREFH","0.5"); RTHe=ef("RTH","8e3"); RTOe=ef("RTO","7e3")
INLO=ef("INLO","0.5"); INHI=ef("INHI","1.5")
kg,clamp=np.load(Path(__file__).resolve().parent / "fixtures" / "cal.npy")
Wm=np.atleast_2d(np.loadtxt(os.environ.get("WFILE","mc_weights.txt")))[:,1::2]; AVG=ei("AVGW","1")
W=Wm[-AVG:].mean(0)
dat=np.load('mc_data.npz'); mn,mx=dat['mn'],dat['mx']; SPREAD=float(dat['spread'])
def mb(C,D,n,seed,spread):
    cen=np.random.default_rng(2024).normal(0,1.5,(C,D)); rng=np.random.default_rng(seed); X=[];Y=[]
    for c in range(C): X.append(cen[c]+rng.normal(0,spread,(n,D))); Y+=[c]*n
    return np.vstack(X),np.array(Y)
def fwd(W,Xf):
    gp_h=W[:H*(D+1)].reshape(H,D+1); gp_o=W[H*(D+1):].reshape(C,H+1)
    HV=[];Y=[];Z=[]
    for x in Xf:
        z1=keystone(gp_h,np.concatenate([x,[INHI]]),kg,VREFH,RTHe); hv=relu_act(z1,clamp)
        y=keystone(gp_o,np.concatenate([hv,[INHI]]),kg,VREFO,RTOe)
        Z.append(z1);HV.append(hv);Y.append(y)
    return np.array(Z),np.array(HV),np.array(Y)
Xte,Yte=mb(C,D,ei("NTE",20),2,SPREAD); Xf=INLO+(Xte-mn)/(mx-mn+1e-9)*(INHI-INLO)
Z,HV,Yp=fwd(W,Xf); acc=(np.argmax(Yp,1)==Yte).mean()
alive=(Z>VT_REL).mean(); hvstd=HV.std(0).mean()
tag=sys.argv[1] if len(sys.argv)>1 else "eval"
print(f"{tag:16s} acc={acc*100:5.1f}% chance={100/C:4.1f}% alive={alive:.2f} hv-std={hvstd:.3f} ymargin={np.ptp(Yp,1).mean():.2f}")
