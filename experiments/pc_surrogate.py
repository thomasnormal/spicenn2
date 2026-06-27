#!/usr/bin/env python3
"""PyTorch surrogate of the deep-sparse analog PC net (pc_deep.py).

Goal: separate the ARCHITECTURE ceiling from the analog TRAINER. Same topology, tanh activations,
OWN weight per edge (no weight sharing -- faithful to one-cap-per-synapse), but full numeric precision.

Two switches:
  CONN = local | random   -- local receptive fields (conv-like windows, valid stride) vs random fan-in
  RULE = bp | pc          -- ideal backprop (architecture ceiling) vs Predictive Coding + symmetric nudge

PC rule mirrors pc_deep: m_l=W_l a_{l-1}, x_l=value node, a_l=tanh(x_l), eps_l=x_l-m_l,
inference relaxes hidden x to equilibrium, output SOFT-nudged +/-beta (SYMNUDGE) toward/away the
one-hot label, weight update = (1/2beta)[ (eps_l a_{l-1}^T)^{+b} - (..)^{-b} ] (masked).
"""
import os, numpy as np, torch
torch.set_num_threads(int(os.environ.get("THREADS","4")))
dev="cpu"
g=lambda k,d: type(d)(os.environ.get(k,d))

# ---------- data: sklearn 8x8 digits, same prep as pc_deep (z-score, NTR/NTE per class) ----------
from sklearn.datasets import load_digits
d=load_digits(); X=d.data.astype(np.float64); y=d.target
C=int(g("C",10)); NTR=int(g("NTR",40)); NTE=int(g("NTE",25))
X=(X-X.mean(0))/(X.std(0)+1e-6)*float(g("SCALE",1.0))
drng=np.random.default_rng(int(g("DSEED",1)))
Xtr=[];ytr=[];Xte=[];yte=[]
for c in range(C):
    i=np.where(y==c)[0]; drng.shuffle(i)
    Xtr+=[X[k] for k in i[:NTR]]; ytr+=[c]*NTR
    Xte+=[X[k] for k in i[NTR:NTR+NTE]]; yte+=[c]*NTE
Xtr=torch.tensor(np.array(Xtr),device=dev); ytr=torch.tensor(np.array(ytr),device=dev)
Xte=torch.tensor(np.array(Xte),device=dev); yte=torch.tensor(np.array(yte),device=dev)

# ---------- connectivity masks ----------
def issq(n):
    r=int(round(n**0.5)); return r if r*r==n else 0
def local_mask(nprev,nl,rfk):
    """valid-conv local window: square parent WpxWp, square child WcxWc=(Wp-rfk+1)^2; each child pools
    an rfk x rfk window. Returns (nl,nprev) 0/1 mask. Falls back to None if not square-compatible."""
    Wp=issq(nprev); Wc=issq(nl)
    if not (Wp and Wc and Wc==Wp-rfk+1): return None
    M=np.zeros((nl,nprev))
    for j in range(nl):
        rc,cc=divmod(j,Wc)
        for dr in range(rfk):
            for dc in range(rfk):
                M[j,(rc+dr)*Wp+(cc+dc)]=1
    return M
def random_mask(nprev,nl,k,seed):
    r=np.random.default_rng(seed); M=np.zeros((nl,nprev)); k=min(k,nprev)
    for j in range(nl): M[j,r.choice(nprev,k,replace=False)]=1
    return M

def build(sizes,conn,rfk,kfan,seed):
    masks=[]; info=[]
    for l in range(1,len(sizes)):
        nprev,nl=sizes[l-1],sizes[l]
        M=None
        if conn=="local" and l<len(sizes)-1:        # local RF for hidden transitions
            M=local_mask(nprev,nl,rfk)
        if M is None:                                 # readout layer, or non-square -> random fan-in (dense if small)
            kk=kfan if l<len(sizes)-1 else min(nprev, int(g("KOUT",kfan)))
            M=random_mask(nprev,nl,kk,seed+l) if conn=="random" or l==len(sizes)-1 else random_mask(nprev,nl,kk,seed+l)
        masks.append(torch.tensor(M,device=dev)); info.append(int(M.sum()))
    return masks,info

# ---------- model params ----------
def init_params(sizes,masks,wseed):
    r=torch.Generator(device=dev); r.manual_seed(wseed)
    Ws=[]; bs=[]
    for l in range(1,len(sizes)):
        fan=masks[l-1].sum(1,keepdim=True).clamp(min=1)
        W=(torch.randn(sizes[l],sizes[l-1],generator=r,device=dev)/fan.sqrt())*float(g("WINIT",1.0))
        Ws.append((W*masks[l-1]).clone().requires_grad_(True))
        bs.append(torch.zeros(sizes[l],device=dev,requires_grad=True))
    return Ws,bs

def forward(Ws,bs,A0):
    a=A0; acts=[a]
    for l,(W,b) in enumerate(zip(Ws,bs)):
        m=a@W.t()+b
        a=torch.tanh(m) if l<len(Ws)-1 else m   # linear output
        acts.append(a)
    return acts   # acts[-1] = output prediction

def accuracy(Ws,bs,Xs,ys):
    with torch.no_grad():
        o=forward(Ws,bs,Xs)[-1]
        o=o-o.mean(1,keepdim=True)             # mirror analog readout (subtract cross-class mean)
        return (o.argmax(1)==ys).float().mean().item()

# ---------- trainer: backprop (architecture ceiling) ----------
def train_bp(sizes,masks,EP,lr,wseed):
    Ws,bs=init_params(sizes,masks,wseed)
    opt=torch.optim.Adam(Ws+bs,lr=lr)
    T=torch.full((len(ytr),C),-1.0,device=dev); T[torch.arange(len(ytr)),ytr]=1.0
    for ep in range(EP):
        opt.zero_grad()
        o=forward(Ws,bs,Xtr)[-1]
        loss=((o-T)**2).mean()
        loss.backward()
        with torch.no_grad():
            for l in range(len(Ws)): Ws[l].grad*=masks[l]   # keep updates on existing edges only
        opt.step()
        with torch.no_grad():
            for l in range(len(Ws)): Ws[l]*=masks[l]
    return accuracy(Ws,bs,Xte,yte),accuracy(Ws,bs,Xtr,ytr)

# ---------- trainer: Predictive Coding + symmetric nudge (algorithm-faithful) ----------
def relax(Ws,bs,A0,target,beta,gamma,Tsteps):
    """relax value nodes x to equilibrium. output soft-nudged by beta toward(+)/away(-) target.
    returns eps per layer and a per layer at equilibrium."""
    L=len(Ws)
    # init x = feedforward prediction
    x=[A0]; a=[A0]
    for l in range(L):
        m=a[-1]@Ws[l].t()+bs[l]
        x.append(m.clone()); a.append(torch.tanh(m) if l<L-1 else m)
    for _ in range(Tsteps):
        a=[A0]+[ (torch.tanh(x[l]) if l<L else x[l]) for l in range(1,L+1) ]
        m=[None]+[ a[l-1]@Ws[l-1].t()+bs[l-1] for l in range(1,L+1) ]
        eps=[None]+[ x[l]-m[l] for l in range(1,L+1) ]
        for l in range(1,L):   # hidden value nodes
            tanhp=1-torch.tanh(x[l])**2
            top=eps[l+1]@Ws[l]   # (batch,size_l)
            x[l]=x[l]-gamma*(eps[l]-tanhp*top)
        # output node L
        go=eps[L]+beta*(x[L]-target)
        x[L]=x[L]-gamma*go
    a=[A0]+[ (torch.tanh(x[l]) if l<L else x[l]) for l in range(1,L+1) ]
    m=[None]+[ a[l-1]@Ws[l-1].t()+bs[l-1] for l in range(1,L+1) ]
    eps=[None]+[ x[l]-m[l] for l in range(1,L+1) ]
    return eps,a

def train_pc(sizes,masks,EP,lr,wseed,sym=True):
    Ws=[w.detach().clone() for w in init_params(sizes,masks,wseed)[0]]
    bs=[b.detach().clone() for b in [torch.zeros(s,device=dev) for s in sizes[1:]]]
    L=len(Ws)
    beta=float(g("BETA",0.3)); gamma=float(g("GAMMA",0.3)); Tst=int(g("TSTEPS",30))
    bsz=int(g("BSZ",100))
    T1=torch.full((len(ytr),C),-1.0,device=dev); T1[torch.arange(len(ytr)),ytr]=1.0
    idx=np.arange(len(ytr))
    for ep in range(EP):
        np.random.default_rng(ep).shuffle(idx)
        for s in range(0,len(idx),bsz):
            b=idx[s:s+bsz]; A0=Xtr[b]; tg=T1[b]
            epsP,aP=relax(Ws,bs,A0,tg,+beta,gamma,Tst)
            if sym:
                epsN,aN=relax(Ws,bs,A0,tg,-beta,gamma,Tst)
            for l in range(L):
                gP=epsP[l+1].t()@aP[l]
                if sym:
                    gN=epsN[l+1].t()@aN[l]; gW=(gP-gN)/(2*beta)
                else:
                    gW=gP/beta
                gW=gW*masks[l]
                Ws[l]=Ws[l]+lr*gW/len(b)
                Ws[l]=Ws[l]*masks[l]
                bs[l]=bs[l]+lr*( (epsP[l+1].sum(0)-(epsN[l+1].sum(0) if sym else 0))/( (2*beta) if sym else beta) )/len(b)
            # weight decay (RWL analog)
            wd=float(g("WD",1e-3))
            for l in range(L): Ws[l]=Ws[l]*(1-wd)*masks[l]
    return accuracy(Ws,bs,Xte,yte),accuracy(Ws,bs,Xtr,ytr)

# ---------- driver ----------
if __name__=="__main__":
    SIZES_ANALOG=[64,32,16,10,10]      # exact analog topology
    SIZES_LOCAL=[64,36,16,10]          # square pyramid for valid local 3x3 conv: 8x8 ->6x6 ->4x4 -> 10 out
    rfk=int(g("RFK",3)); kfan=int(g("FANIN",4)); EP=int(g("EP",200)); seed=int(g("SEED",1))
    rule=g("RULE","both"); conn=g("CONN","both")
    print(f"data: C={C} NTR={NTR} NTE={NTE}  (test n={len(yte)})")
    def run(tag,sizes,conn_,rule_,lr):
        masks,info=build(sizes,conn_,rfk,kfan,seed)
        nw=sum(info)
        if rule_=="bp": te,tr=train_bp(sizes,masks,EP,lr,seed+777)
        else:           te,tr=train_pc(sizes,masks,int(g("EPPC",80)),lr,seed+777,sym=(rule_!="pc1"))
        print(f"{tag:34s} sizes={sizes} edges={nw:4d} rule={rule_:4s} conn={conn_:6s} -> TEST={te*100:5.1f}%  train={tr*100:5.1f}%")
        return te
    # architecture ceiling (backprop): analog-random vs local-RF
    run("BP analog-topology RANDOM", SIZES_ANALOG,"random","bp",float(g("LRBP",0.01)))
    run("BP local RF (3x3 pyramid)", SIZES_LOCAL,"local","bp",float(g("LRBP",0.01)))
    run("BP local-sizes RANDOM ctl", SIZES_LOCAL,"random","bp",float(g("LRBP",0.01)))
    # algorithm-faithful (PC + symmetric nudge)
    run("PC+sym analog-topo RANDOM", SIZES_ANALOG,"random","pc",float(g("LRPC",0.05)))
    run("PC+sym local RF pyramid  ", SIZES_LOCAL,"local","pc",float(g("LRPC",0.05)))
    print("DONE")
