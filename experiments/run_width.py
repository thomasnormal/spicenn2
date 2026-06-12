import os,numpy as np,time
from mc_experiment import train_eval, make_blobs, scale_fit, scale_apply
C,D,spread,ntr,nte,ep,mom,lr,norm=8,8,1.2,50,30,150,0.9,0.4,'divisive'
Xtr,Ytr=make_blobs(C,D,ntr,1,spread); Xte,Yte=make_blobs(C,D,nte,2,spread)
p=scale_fit(Xtr); Xtr=scale_apply(Xtr,p); Xte=scale_apply(Xte,p)
SIG=[0.05,0.1,0.2,0.4]   # gaussian weight (gate-voltage) perturbation, represented-weight units
def perturb_acc(net,sigma,reps,seed):
    rng=np.random.default_rng(seed); a=[]
    for _ in range(reps):
        W0=[w.copy() for w in net.W]
        for w in net.W: w+=rng.normal(0,sigma,w.shape)
        ac,_=net.forward(Xte); a.append((np.argmax(ac[-1],1)==Yte).mean())
        for l in range(len(net.W)): net.W[l]=W0[l]
    return float(np.mean(a))
done=set()
if os.path.exists('results_width.txt'):
    for ln in open('results_width.txt'):
        f=ln.split();  done.add((f[0],f[1]))
t0=time.time()
for hid in [8,16,32,48,72]:
    for s in [0,1,2]:
        if (str(hid),str(s)) in done: continue
        acc,tr,net=train_eval([D,hid,C],norm,Xtr,Ytr,Xte,Yte,ep,lr,s,mom=mom)
        robs=[perturb_acc(net,sig,10,s+100) for sig in SIG]
        open('results_width.txt','a').write(f"{hid} {s} {acc:.3f} "+" ".join(f"{r:.3f}" for r in robs)+"\n")
        print(f"hid{hid:3d} s{s}: clean {acc*100:3.0f}  perturbed σ{SIG}={[round(r*100) for r in robs]} [{time.time()-t0:.0f}s]",flush=True)
        if time.time()-t0>235: print("timeout guard — re-run to continue"); raise SystemExit
print("WIDTH SWEEP COMPLETE")
