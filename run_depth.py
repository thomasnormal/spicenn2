import os,numpy as np,time
from mc_experiment import Net,NORMS,onehot,make_blobs,scale_fit,scale_apply
C,D,spread=8,8,1.2
Xtr,Ytr=make_blobs(C,D,50,1,spread); Xte,Yte=make_blobs(C,D,30,2,spread)
p=scale_fit(Xtr); Xtr=scale_apply(Xtr,p); Xte=scale_apply(Xte,p)
fn=NORMS['divisive']; T=onehot(Ytr,C)
def run(hiddens,lr,seed,ep=200,mom=0.9):
    sizes=[D]+hiddens+[C]; net=Net(sizes,seed); V=[np.zeros_like(w) for w in net.W]; gn=None
    for e in range(ep):
        acts,zs=net.forward(Xtr); P=fn(acts[-1]); gr=net.grads(acts,zs,P-T)
        if e==ep-1: gn=[round(float(np.sqrt((g**2).mean())),4) for g in gr]   # rms grad per layer
        for l in range(len(net.W)): V[l]=mom*V[l]-lr*gr[l]; net.W[l]+=V[l]
    tr=(np.argmax(net.forward(Xtr)[0][-1],1)==Ytr).mean()
    te=(np.argmax(net.forward(Xte)[0][-1],1)==Yte).mean()
    return tr,te,gn
done=set()
if os.path.exists('results_depth.txt'):
    for ln in open('results_depth.txt'): f=ln.split('|'); done.add((f[0].strip(),f[1].strip()))
t0=time.time()
configs=[('[20]',[20]),('[20,20]',[20,20]),('[20,20,20]',[20,20,20])]
for name,h in configs:
    for s in [0,1,2]:
        if (name,str(s)) in done: continue
        tr,te,gn=run(h,0.4,s)
        open('results_depth.txt','a').write(f"{name} | {s} | {tr:.3f} | {te:.3f} | {gn}\n")
        print(f"{name:12s} s{s}: train {tr*100:3.0f} test {te*100:3.0f}  grad-rms/layer {gn}  [{time.time()-t0:.0f}s]",flush=True)
        if time.time()-t0>235: print("timeout guard"); raise SystemExit
print("DEPTH SWEEP COMPLETE")
