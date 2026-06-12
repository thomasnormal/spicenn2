#!/usr/bin/env python3
"""Read e2e_o<TAG>.dat and report continuous-time XOR accuracy over epochs."""
import numpy as np, sys, os
TAG=sys.argv[1] if len(sys.argv)>1 else ""
TH=float(os.environ.get("TH","150")); NEPOCH=int(os.environ.get("NEPOCH","150"))
d=np.loadtxt(f"e2e_o{TAG}.dat"); t=d[:,0]*1e9; o=d[:,1]-d[:,3]; vt=d[:,9]
def acc(e):
    ok=0; outs=[]
    for m in range(4):
        g=e*4+m; i=np.argmin(np.abs(t-(g*2*TH+TH-2)))
        tg=1 if vt[i]>0.5 else -1; outs.append(o[i]); ok+=((o[i]>0)==(tg>0))
    return ok,outs
def acc2(e):  # accuracy requiring NON-degenerate outputs (spread across patterns)
    ok,outs=acc(e); spread=max(outs)-min(outs)
    return (ok if spread>0.005 else min(ok, 2-(ok>2)*0), spread)  # if all ~equal, it's not real XOR
accs=[acc(e)[0] for e in range(NEPOCH)]
best=max(accs); bestep=accs.index(best)
fa,fouts=acc(NEPOCH-1); ftg=[(1 if False else s) for s in range(4)]
spread=(max(fouts)-min(fouts))*1e3
degen = spread<5.0   # outputs differ by <5mV across patterns => input-independent (degenerate)
realbest=max((acc(e)[0] if (max(acc(e)[1])-min(acc(e)[1]))>0.005 else 0) for e in range(NEPOCH))
print(f"TAG={TAG} final={fa}/4 best={best}/4@ep{bestep} realbest(non-degen)={realbest}/4 "
      f"spread={spread:.1f}mV{' DEGENERATE' if degen else ''}  o_final(mV)={np.round(np.array(fouts)*1e3,1)}")
