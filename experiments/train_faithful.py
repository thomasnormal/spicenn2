# Fast model of DEEP-PC hidden TRAINING with realistic in-circuit backward non-idealities.
# Q: does the analog backward (synapse-saturation on gradient + f'-gate variants) break hidden learning,
# and does a cleaner-derivative neuron (ReLU) rescue it? Full PC, hidden TRAINS (no freezing).
import os, numpy as np, torch
os.environ.update(C="10",NTR="40")
src=open("experiments/pc_surrogate.py").read().split('if __name__=="__main__":')[0]
ns={}; exec(src,ns)
dev=ns["dev"];C=ns["C"];Xtr,ytr,Xte,yte=ns["Xtr"],ns["ytr"],ns["Xte"],ns["yte"]
build,init_params,accuracy=ns["build"],ns["init_params"],ns["accuracy"]
SIZES=[64,36,16,10]
def sat(a,g): return a if g<=0 else torch.tanh(g*a)/np.tanh(g)*min(1.0,g)   # synapse saturation
def act(m,kind):     # neuron activation
    if kind=="tanh": return torch.tanh(m)
    if kind=="relu": return torch.clamp(m,-1,1).clamp(min=0)*2-1   # shifted relu in [-1,1]-ish
    if kind=="lrelu": return torch.where(m>0,torch.clamp(m,max=1),0.1*m)   # leaky: nonzero deriv everywhere
def dact(x,kind):    # f'-gate
    if kind=="tanh": return 1-torch.tanh(x)**2
    if kind=="relu": return (x>0).float()
    if kind=="lrelu": return torch.where(x>0,torch.ones_like(x),0.1*torch.ones_like(x))
def relax(Ws,bs,A0,t,beta,gamma,Tst,g,bsign,kind):
    L=len(Ws);x=[A0];a=[A0]
    for l in range(L):
        m=sat(a[-1],g)@Ws[l].t()+bs[l];x.append(m.clone());a.append(act(m,kind) if l<L-1 else m)
    for _ in range(Tst):
        a=[A0]+[(act(x[l],kind) if l<L else x[l]) for l in range(1,L+1)]
        m=[None]+[sat(a[l-1],g)@Ws[l-1].t()+bs[l-1] for l in range(1,L+1)]
        eps=[None]+[x[l]-m[l] for l in range(1,L+1)]
        for l in range(1,L):
            fp=dact(x[l],kind)
            top=sat(eps[l+1],g)@Ws[l]            # backward transported through SATURATING synapse
            if bsign=="hard": top=torch.sign(top)*top.abs().mean()  # comparator-regen (hard sign, BKSIGN-like)
            x[l]=x[l]-gamma*(eps[l]-fp*top)
        x[L]=x[L]-gamma*(eps[L]+beta*(x[L]-t))
    a=[A0]+[(act(x[l],kind) if l<L else x[l]) for l in range(1,L+1)]
    m=[None]+[sat(a[l-1],g)@Ws[l-1].t()+bs[l-1] for l in range(1,L+1)]
    eps=[None]+[x[l]-m[l] for l in range(1,L+1)];return eps,a
def tf(ws,g,bsign,kind,EP=80,lr=0.05):
    m,_=build(SIZES,"local",2,4,ws); m[-1]=torch.ones_like(m[-1])
    Ws=[w.detach().clone() for w in init_params(SIZES,m,ws)[0]];bs=[torch.zeros(s,device=dev) for s in SIZES[1:]];L=len(Ws)
    T1=torch.full((len(ytr),C),-1.0,device=dev);T1[torch.arange(len(ytr)),ytr]=1.0;idx=np.arange(len(ytr))
    for ep in range(EP):
        np.random.default_rng(ep).shuffle(idx)
        for s in range(0,len(idx),100):
            b=idx[s:s+100];A0=Xtr[b];tg=T1[b]
            eP,aP=relax(Ws,bs,A0,tg,0.3,0.3,30,g,bsign,kind);eN,aN=relax(Ws,bs,A0,tg,-0.3,0.3,30,g,bsign,kind)
            for l in range(L):
                gW=((eP[l+1].t()@aP[l])-(eN[l+1].t()@aN[l]))/0.6*m[l]
                Ws[l]=(Ws[l]+lr*gW/len(b))*m[l]; bs[l]=bs[l]+lr*((eP[l+1].sum(0)-eN[l+1].sum(0))/0.6)/len(b)
            for l in range(L): Ws[l]=Ws[l]*(1-1e-3)*m[l]
    def acc():
        x=[Xte]
        for l in range(L): x.append((act(sat(x[-1],g)@Ws[l].t()+bs[l],kind) if l<L-1 else sat(x[-1],g)@Ws[l].t()+bs[l]))
        return (x[-1].argmax(1)==yte).float().mean().item()
    return acc()
def avg(**k): a=[tf(10+s,**k) for s in range(3)];return np.mean(a)*100,np.std(a)*100
print("=== HIDDEN-TRAINING under in-circuit backward (full PC, hidden trains, 3-seed C=10) ===")
print("  (frozen ref ~73-74; ideal trained ~88)")
for tag,k in [("ideal: tanh, linear-syn, graded-bk", dict(g=0,bsign="graded",kind="tanh")),
              ("+ synapse-sat backward (g=4)",        dict(g=4,bsign="graded",kind="tanh")),
              ("+ hard-sign backward (BKSIGN-like)",  dict(g=4,bsign="hard",kind="tanh")),
              ("ReLU neuron + sat + graded-bk",       dict(g=4,bsign="graded",kind="relu")),
              ("ReLU neuron + sat + hard-sign",       dict(g=4,bsign="hard",kind="relu"))]:
    mu,sd=avg(**k); print(f"  {tag:40s}: {mu:4.1f}% +/-{sd:.1f}")
