#!/usr/bin/env python3
# Pre-validate the multi-class SPICE target config on the circuit-faithful surrogate.
# Few inputs (transistor economy), one hidden layer (shallow-and-wide), sweep C and norm.
# Goal: find a small config where (a) competition >> none (necessity), (b) it trains well.
import numpy as np, sys
import mc_experiment as M

def run(C, D, H, spread, epochs=400, lr=0.1, seeds=(0,1,2), ntr=60, nte=40):
    Xtr,Ytr=M.make_blobs(C,D,ntr,1,spread); Xte,Yte=M.make_blobs(C,D,nte,2,spread)
    p=M.scale_fit(Xtr); Xtr=M.scale_apply(Xtr,p); Xte=M.scale_apply(Xte,p)
    out={}
    for norm in ['none','divisive','subtractive','softmax']:
        r=[M.train_eval([D,H,C],norm,Xtr,Ytr,Xte,Yte,epochs,lr,s) for s in seeds]
        te=np.mean([a for a,_,_ in r]); tr=np.mean([b for _,b,_ in r])
        out[norm]=(tr,te,[round(a*100) for a,_,_ in r])
    return out

if __name__=="__main__":
    print(f"=== prevalidation: shallow-and-wide, D=4 inputs, H=8 hidden ===")
    for C in (3,4,6):
        for spread in (0.9, 1.4):
            print(f"\n--- C={C} D=4 H=8 spread={spread} chance={100/C:.0f}% ---")
            res=run(C,4,8,spread)
            for norm,(tr,te,seeds) in res.items():
                print(f"  {norm:12s} train {tr*100:5.1f}%  test {te*100:5.1f}%  seeds {seeds}")
    sys.stdout.flush()
