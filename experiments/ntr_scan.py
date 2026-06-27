import os, numpy as np, torch
src=open("experiments/pc_surrogate.py").read().split('if __name__=="__main__":')[0]
def run_ntr(ntr):
    os.environ.update(C="10",NTR=str(ntr)); ns={}; exec(src,ns)
    dev=ns["dev"];C=ns["C"];Xtr,ytr,Xte,yte=ns["Xtr"],ns["ytr"],ns["Xte"],ns["yte"]
    build,init_params,accuracy=ns["build"],ns["init_params"],ns["accuracy"]
    SIZES=[64,49,36,10]
    def relax(Ws,bs,A0,t,beta,gamma,Tst):
        L=len(Ws);x=[A0];a=[A0]
        for l in range(L):
            m=a[-1]@Ws[l].t()+bs[l];x.append(m.clone());a.append(torch.tanh(m) if l<L-1 else m)
        for _ in range(Tst):
            a=[A0]+[(torch.tanh(x[l]) if l<L else x[l]) for l in range(1,L+1)]
            m=[None]+[a[l-1]@Ws[l-1].t()+bs[l-1] for l in range(1,L+1)]
            eps=[None]+[x[l]-m[l] for l in range(1,L+1)]
            for l in range(1,L):
                tp=1-torch.tanh(x[l])**2;x[l]=x[l]-gamma*(eps[l]-tp*(eps[l+1]@Ws[l]))
            x[L]=x[L]-gamma*(eps[L]+beta*(x[L]-t))
        a=[A0]+[(torch.tanh(x[l]) if l<L else x[l]) for l in range(1,L+1)]
        m=[None]+[a[l-1]@Ws[l-1].t()+bs[l-1] for l in range(1,L+1)]
        eps=[None]+[x[l]-m[l] for l in range(1,L+1)];return eps,a
    def tf(ws,EP):
        m,_=build(SIZES,"local",2,4,ws);m[-1]=torch.ones_like(m[-1])
        Ws=[w.detach().clone() for w in init_params(SIZES,m,ws)[0]];bs=[torch.zeros(s,device=dev) for s in SIZES[1:]];L=len(Ws)
        T1=torch.full((len(ytr),C),-1.0,device=dev);T1[torch.arange(len(ytr)),ytr]=1.0;idx=np.arange(len(ytr))
        for ep in range(EP):
            np.random.default_rng(ep).shuffle(idx)
            for s in range(0,len(idx),100):
                b=idx[s:s+100];A0=Xtr[b];tg=T1[b]
                eP,aP=relax(Ws,bs,A0,tg,0.3,0.3,30);eN,aN=relax(Ws,bs,A0,tg,-0.3,0.3,30)
                gW=((eP[L].t()@aP[L-1])-(eN[L].t()@aN[L-1]))/0.6;Ws[L-1]=Ws[L-1]+0.05*gW/len(b)
                bs[L-1]=bs[L-1]+0.05*((eP[L].sum(0)-eN[L].sum(0))/0.6)/len(b)
        return accuracy(Ws,bs,Xte,yte)
    EP=max(2, 480//ntr)  # constant ~ NTR*EP train slots (matches in-circuit cost swap)
    a=[tf(10+s,EP) for s in range(3)];return np.mean(a)*100,np.std(a)*100,EP
print("=== frozen-dense 64,49,36: does more DATA (NTR) help readout? (const NTR*EP) 3-seed C=10 ===")
for ntr in [40,80,120]:
    mu,sd,ep=run_ntr(ntr);print(f"  NTR={ntr:3d} (EP={ep:2d}): {mu:4.1f}% +/-{sd:.1f}")
