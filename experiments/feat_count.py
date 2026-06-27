import os, numpy as np, torch
os.environ.update(C="10",NTR="40")
src=open("experiments/pc_surrogate.py").read().split('if __name__=="__main__":')[0]
ns={}; exec(src,ns)
dev=ns["dev"]; C=ns["C"]; Xtr,ytr,Xte,yte=ns["Xtr"],ns["ytr"],ns["Xte"],ns["yte"]
build,init_params,accuracy=ns["build"],ns["init_params"],ns["accuracy"]
def relax_ro(Ws,bs,A0,t,beta,gamma,Tst):
    L=len(Ws); x=[A0]
    for l in range(L):
        m=([A0]+[None]*L)[l] if False else None
    x=[A0]; a=[A0]
    for l in range(L):
        m=a[-1]@Ws[l].t()+bs[l]; x.append(m.clone()); a.append(torch.tanh(m) if l<L-1 else m)
    for _ in range(Tst):
        a=[A0]+[(torch.tanh(x[l]) if l<L else x[l]) for l in range(1,L+1)]
        m=[None]+[a[l-1]@Ws[l-1].t()+bs[l-1] for l in range(1,L+1)]
        eps=[None]+[x[l]-m[l] for l in range(1,L+1)]
        for l in range(1,L):
            tanhp=1-torch.tanh(x[l])**2; x[l]=x[l]-gamma*(eps[l]-tanhp*(eps[l+1]@Ws[l]))
        x[L]=x[L]-gamma*(eps[L]+beta*(x[L]-t))
    a=[A0]+[(torch.tanh(x[l]) if l<L else x[l]) for l in range(1,L+1)]
    m=[None]+[a[l-1]@Ws[l-1].t()+bs[l-1] for l in range(1,L+1)]
    eps=[None]+[x[l]-m[l] for l in range(1,L+1)]; return eps,a
def tf(SIZES,wseed,EP=80,lr=0.05,beta=0.3,gamma=0.3,Tst=30):
    masks,_=build(SIZES,"local",2,4,wseed); masks[-1]=torch.ones_like(masks[-1])
    Ws=[w.detach().clone() for w in init_params(SIZES,masks,wseed)[0]]
    bs=[torch.zeros(s,device=dev) for s in SIZES[1:]]; L=len(Ws)
    T1=torch.full((len(ytr),C),-1.0,device=dev); T1[torch.arange(len(ytr)),ytr]=1.0; idx=np.arange(len(ytr))
    for ep in range(EP):
        np.random.default_rng(ep).shuffle(idx)
        for s in range(0,len(idx),100):
            b=idx[s:s+100]; A0=Xtr[b]; tg=T1[b]
            eP,aP=relax_ro(Ws,bs,A0,tg,+beta,gamma,Tst); eN,aN=relax_ro(Ws,bs,A0,tg,-beta,gamma,Tst)
            l=L-1; gW=((eP[l+1].t()@aP[l])-(eN[l+1].t()@aN[l]))/(2*beta)
            Ws[l]=Ws[l]+lr*gW/len(b); bs[l]=bs[l]+lr*((eP[l+1].sum(0)-eN[l+1].sum(0))/(2*beta))/len(b)
    return accuracy(Ws,bs,Xte,yte)
def avg(SZ): a=[tf(SZ,10+s) for s in range(3)]; return np.mean(a)*100,np.std(a)*100
print("=== frozen+dense, feature-count scaling (read last layer) 3-seed C=10 ===")
for tag,SZ in [("64,49,36 (36 feat)",[64,49,36,10]),("64,49,49 (49 feat)",[64,49,49,10]),
               ("64,64,49 (49 feat)",[64,64,49,10]),("64,64,64 (64 feat)",[64,64,64,10])]:
    mu,sd=avg(SZ); print(f"  {tag:22s}: {mu:4.1f}% +/-{sd:.1f}")
