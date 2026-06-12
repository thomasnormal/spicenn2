#!/usr/bin/env python3
# Decoupling test: train the surrogate Net to good weights, write them as frozen gate
# voltages in the SPICE deck's weight order, so we can load them into the real SPICE
# inference deck and check the SPICE forward can separate classes (representation test).
import numpy as np, os
import mc_experiment as M
import os
D=4;H=8;C=int(os.environ.get("C","3"));SPREAD=0.9; VC=1.8; Gs=1.2
norm=os.environ.get("NORM","subtractive")
Xtr,Ytr=M.make_blobs(C,D,8,1,SPREAD)
mn=Xtr.min(0); mx=Xtr.max(0)
# train surrogate net with SAME input scaling as SPICE deck ([0.5,1.5] forward)
def scale(X): return 0.5+(X-mn)/(mx-mn+1e-9)*(1.5-0.5)
Xs=scale(Xtr)
acc,tr,net=M.train_eval([D,H,C],norm,Xs,Ytr,Xs,Ytr,800,0.1,0)
print(f"surrogate net trained: norm={norm} train-acc={tr*100:.1f}%")
# flatten in SPICE order: hidden W[0] (H,D+1) rows j, cols [1..D,b]; output W[1] (C,H+1)
gh=(VC+Gs*net.W[0]).flatten()
go=(VC+Gs*net.W[1]).flatten()
gp=np.concatenate([gh,go])
# write as mc_weights.txt-style row: interleave (t,val) so [1::2] recovers val
row=np.empty(2*len(gp)); row[0::2]=np.arange(len(gp)); row[1::2]=gp
np.savetxt("mc_weights.txt", row[None,:])
# also save data params so infer regenerates matching blobs/scaling
np.savez("mc_data.npz", mn=mn, mx=mx, spread=SPREAD, C=C, D=D, INLO=0.5, INHI=1.5)
print(f"wrote {len(gp)} gate voltages, range {gp.min():.2f}..{gp.max():.2f}")
