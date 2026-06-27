import os, numpy as np, torch
os.environ.update(C="10",NTR="40")
src=open("experiments/pc_surrogate.py").read().split('if __name__=="__main__":')[0]
ns={}; exec(src,ns)
dev=ns["dev"]; C=ns["C"]; Xtr,ytr,Xte,yte=ns["Xtr"],ns["ytr"],ns["Xte"],ns["yte"]
build,init_params,accuracy=ns["build"],ns["init_params"],ns["accuracy"]
# Hypothesis: readout output = sum_k w_k*a_k + noise where noise grows with #features summed.
# Test: frozen feats + dense readout, inject per-feature readout noise. Does it flatten the feature-count gain?
def relax_ro(Ws,bs,A0,t,beta,gamma,Tst,fnoise,rng):
    L=len(Ws); x=[A0]; a=[A0]
    for l in range(L):
        m=a[-1]@Ws[l].t()+bs[l]; x.append(m.clone()); a.append(torch.tanh(m) if l<L-1 else m)
    for _ in range(Tst):
        a=[A0]+[(torch.tanh(x[l]) if l<L else x[l]) for l in range(1,L+1)]
        # inject readout-input noise on the LAST hidden layer's activation feeding the readout
        if fnoise>0:
            nz=torch.tensor(rng.standard_normal(a[L-1].shape),dtype=a[L-1].dtype,device=dev)*fnoise
            a[L-1]=a[L-1]+nz
        m=[None]+[a[l-1]@Ws[l-1].t()+bs[l-1] for l in range(1,L+1)]
        eps=[None]+[x[l]-m[l] for l in range(1,L+1)]
        for l in range(1,L):
            tanhp=1-torch.tanh(x[l])**2; x[l]=x[l]-gamma*(eps[l]-tanhp*(eps[l+1]@Ws[l]))
        x[L]=x[L]-gamma*(eps[L]+beta*(x[L]-t))
    a=[A0]+[(torch.tanh(x[l]) if l<L else x[l]) for l in range(1,L+1)]
    m=[None]+[a[l-1]@Ws[l-1].t()+bs[l-1] for l in range(1,L+1)]
    eps=[None]+[x[l]-m[l] for l in range(1,L+1)]; return eps,a
def tf(SIZES,wseed,fnoise,EP=80,lr=0.05,beta=0.3,gamma=0.3,Tst=30):
    rng=np.random.default_rng(wseed+5)
    masks,_=build(SIZES,"local",2,4,wseed); masks[-1]=torch.ones_like(masks[-1])
    Ws=[w.detach().clone() for w in init_params(SIZES,masks,wseed)[0]]
    bs=[torch.zeros(s,device=dev) for s in SIZES[1:]]; L=len(Ws)
    T1=torch.full((len(ytr),C),-1.0,device=dev); T1[torch.arange(len(ytr)),ytr]=1.0; idx=np.arange(len(ytr))
    for ep in range(EP):
        np.random.default_rng(ep).shuffle(idx)
        for s in range(0,len(idx),100):
            b=idx[s:s+100]; A0=Xtr[b]; tg=T1[b]
            eP,aP=relax_ro(Ws,bs,A0,tg,+beta,gamma,Tst,fnoise,rng); eN,aN=relax_ro(Ws,bs,A0,tg,-beta,gamma,Tst,fnoise,rng)
            l=L-1; gW=((eP[l+1].t()@aP[l])-(eN[l+1].t()@aN[l]))/(2*beta)
            Ws[l]=Ws[l]+lr*gW/len(b); bs[l]=bs[l]+lr*((eP[l+1].sum(0)-eN[l+1].sum(0))/(2*beta))/len(b)
    return accuracy(Ws,bs,Xte,yte)
def avg(SZ,fn): a=[tf(SZ,10+s,fn) for s in range(3)]; return np.mean(a)*100,np.std(a)*100
print("=== readout-input noise flattens feature-count gain? (clean: 36=83.9, 49=86, 64=87.9) 3-seed C=10 ===")
for fn in [0.0,0.3,0.6]:
    r36=avg([64,49,36,10],fn); r49=avg([64,64,49,10],fn); r64=avg([64,64,64,10],fn)
    print(f"  fnoise={fn}: 36feat={r36[0]:.1f}  49feat={r49[0]:.1f}  64feat={r64[0]:.1f}  (gain 36->64 = {r64[0]-r36[0]:+.1f})")
print("=> if gain 36->64 SHRINKS as noise rises -> confirms readout-noise caps feature-scaling in-circuit")
