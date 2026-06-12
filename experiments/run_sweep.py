import os,numpy as np,time
from mc_experiment import train_eval, make_blobs, scale_fit, scale_apply
C,D,hid,spread,ntr,nte,ep,mom=8,8,20,1.2,50,30,150,0.9
Xtr,Ytr=make_blobs(C,D,ntr,1,spread); Xte,Yte=make_blobs(C,D,nte,2,spread)
p=scale_fit(Xtr); Xtr=scale_apply(Xtr,p); Xte=scale_apply(Xte,p)
done=set()
if os.path.exists('results_hard.txt'):
    for ln in open('results_hard.txt'):
        a=ln.split();  done.add((a[0],a[1],a[2]))
combos=[(n,lr,s) for n in ['none','divisive','subtractive','softmax'] for lr in [0.2,0.6,1.8,5.0] for s in [0,1]]
t0=time.time()
for n,lr,s in combos:
    if (n,str(float(lr)),str(s)) in done: continue
    acc,tr,_=train_eval([D,hid,C],n,Xtr,Ytr,Xte,Yte,ep,float(lr),s,mom=mom)
    open('results_hard.txt','a').write(f"{n} {float(lr)} {s} {tr:.3f} {acc:.3f}\n")
    print(f"{n:11s} lr{lr:<4} s{s}: tr{tr*100:3.0f} te{acc*100:3.0f}  [{time.time()-t0:.0f}s]",flush=True)
    if time.time()-t0>235: print("...timeout guard, re-run to continue"); break
print("done" if len([1 for _ in combos])==len(done)+0 else "partial")
