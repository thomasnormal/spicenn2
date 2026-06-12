import os,numpy as np,time
from mc_experiment import train_eval, make_blobs, scale_fit, scale_apply
D,spread,ep,mom=10,1.0,150,0.9
LR={'none':0.15,'divisive':0.4,'softmax':0.4}     # per-norm reasonable lr from earlier sweep
done=set()
if os.path.exists('results_classes.txt'):
    for ln in open('results_classes.txt'): f=ln.split(); done.add((f[0],f[1],f[2]))
t0=time.time()
for C in [4,8,12,16]:
    Xtr,Ytr=make_blobs(C,D,35,1,spread); Xte,Yte=make_blobs(C,D,25,2,spread)
    p=scale_fit(Xtr); Xtr=scale_apply(Xtr,p); Xte=scale_apply(Xte,p)
    for norm in ['none','divisive','softmax']:
        for s in [0,1]:
            if (str(C),norm,str(s)) in done: continue
            acc,tr,_=train_eval([D,24,C],norm,Xtr,Ytr,Xte,Yte,ep,LR[norm],s,mom=mom)
            open('results_classes.txt','a').write(f"{C} {norm} {s} {tr:.3f} {acc:.3f}\n")
            print(f"C={C:2d} {norm:9s} s{s}: train {tr*100:3.0f} test {acc*100:3.0f}  (chance {100/C:.0f}%) [{time.time()-t0:.0f}s]",flush=True)
            if time.time()-t0>235: print("timeout guard"); raise SystemExit
print("CLASSES SWEEP COMPLETE")
