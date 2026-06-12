#!/usr/bin/env python3
# Search for a SPARSE, LOCALLY-CONNECTED (non-weight-shared) MNIST net with fan-in<=4 and fan-out<=4
# on every neuron, NO conv (each synapse its own weight), NO pooling (reduce via 2x2 receptive-field
# layers instead). PyTorch + Adam to find the achievable accuracy ceiling for a topology; the analog
# in-circuit training comes after. Reports accuracy AND the actual fan-in/fan-out degree distribution.
import numpy as np, os, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(16)
RES=int(os.environ.get("RES","16"))
d=np.load(f"mnist_{RES}x{RES}.npz"); X=d["X"].astype(np.float32); y=d["y"].astype(np.int64)
rng=np.random.default_rng(0); idx=rng.permutation(len(y)); X,y=X[idx],y[idx]
# per-image z-score (the proven analog input norm)
X=(X-X.mean(1,keepdims=True))/(X.std(1,keepdims=True)+1e-6)
ntr=60000
Xtr=torch.tensor(X[:ntr]); ytr=torch.tensor(y[:ntr]); Xte=torch.tensor(X[ntr:]); yte=torch.tensor(y[ntr:])

def four_regular(nL,nR,k,seed):  # bipartite: each left node k edges, each right node ~k edges
    r=np.random.default_rng(seed); edges=set()
    for _ in range(k):
        perm=r.permutation(nR)
        for L in range(nL): edges.add((L, perm[L%nR]))
    return edges

class SparseLayer(nn.Module):
    # input grid Hin*Win*Cin (location-major: idx=(i*W+j)*C+c) -> output (Hin/st)*(Win/st)*Cout
    def __init__(self, Hin,Win,Cin,Cout,ksize,stride,fanin,seed):
        super().__init__()
        Hout=(Hin-ksize)//stride+1; Wout=(Win-ksize)//stride+1
        self.Hout,self.Wout,self.Cout=Hout,Wout,Cout
        nin=Hin*Win*Cin; nout=Hout*Wout*Cout
        self.nin,self.nout=nin,nout
        mask=torch.zeros(nout,nin)
        fanout=np.zeros(nin)
        r=np.random.default_rng(seed)
        for io in range(Hout):
            for jo in range(Wout):
                # receptive-field input neuron indices (ksize x ksize spatial x Cin channels)
                rf=[]
                for di in range(ksize):
                    for dj in range(ksize):
                        ii=io*stride+di; jj=jo*stride+dj
                        for c in range(Cin): rf.append((ii*Win+jj)*Cin+c)
                rf=np.array(rf)
                for co in range(Cout):
                    o=(io*Wout+jo)*Cout+co
                    # pick fanin inputs from rf, preferring low current fan-out (balance fanout)
                    order=sorted(rf, key=lambda x:(fanout[x], r.random()))
                    sel=order[:fanin]
                    for s in sel: mask[o,s]=1; fanout[s]+=1
        self.register_buffer("mask",mask)
        self.weight=nn.Parameter((torch.randn(nout,nin)*0.5)*mask)
        self.bias=nn.Parameter(torch.zeros(nout))
    def forward(self,x): return F.linear(x, self.weight*self.mask, self.bias)

class Net(nn.Module):
    def __init__(self, layers, readout):
        super().__init__(); self.layers=nn.ModuleList(layers); self.readout=readout
    def forward(self,x):
        for l in self.layers: x=torch.tanh(l(x))
        return self.readout(x)

# ---- build pyramid from TOPO spec: list of (Cout,ksize,stride,fanin); then readout spec ----
# default: 16x16x1 -2x2s2-> 8x8x4 -> 4x4x16 -> 2x2x64 -> 1x1x256 ; readout 256->10 (fanin from env)
import json
TOPO=os.environ.get("TOPO","4:2:2:4,16:2:2:4,64:2:2:4,256:2:2:4")
ROUT_FANIN=int(os.environ.get("ROUTK","4"))
H=W=RES; Cin=1; layers=[]
for li,part in enumerate(TOPO.split(",")):
    Cout,ks,st,fi=[int(x) for x in part.split(":")]
    L=SparseLayer(H,W,Cin,Cout,ks,st,fi,seed=li+1); layers.append(L)
    H,W,Cin=L.Hout,L.Wout,L.Cout
nfeat=H*W*Cin
# readout: single sparse layer nfeat->10 with fan-in ROUT_FANIN (or dense if ROUTK<=0)
class Readout(nn.Module):
    def __init__(self,nin,seed):
        super().__init__(); mask=torch.zeros(10,nin); r=np.random.default_rng(seed)
        if ROUT_FANIN<=0: mask[:]=1
        else:
            for o in range(10):
                for s in r.choice(nin,size=min(ROUT_FANIN,nin),replace=False): mask[o,s]=1
        self.register_buffer("mask",mask); self.weight=nn.Parameter(torch.randn(10,nin)*0.3*mask); self.bias=nn.Parameter(torch.zeros(10))
    def forward(self,x): return F.linear(x,self.weight*self.mask,self.bias)
readout=Readout(nfeat,seed=99)
net=Net(layers,readout)

# degree stats
def degrees():
    fi_max=0; fo_max=0; nw=0
    for l in layers+[readout]:
        m=l.mask; nw+=int(m.sum().item())
        fi_max=max(fi_max,int(m.sum(1).max().item()))
        fo=m.sum(0); fo_max=max(fo_max,int(fo.max().item()))
    return fi_max,fo_max,nw
fi,fo,nw=degrees()
nneur=sum(l.nout for l in layers)+10                      # hidden + output neurons (input not a neuron)
mos=11*nw+21*nneur                                         # ~11 MOS/trained-synapse + ~21 MOS/neuron
print(f"RES={RES} TOPO={TOPO} ROUTK={ROUT_FANIN}  weights={nw} neurons={nneur}  max fan-in={fi} max fan-out={fo}  est_MOS={mos//1000}k")

OPT=os.environ.get("OPT","adam"); NOISE=float(os.environ.get("NOISE","0")); WCLIP=float(os.environ.get("WCLIP","0"))
LR=float(os.environ.get("LR","2e-3"))
if   OPT=="sgd":     opt=torch.optim.SGD(net.parameters(),lr=LR,momentum=0.0)      # plain integrator
elif OPT=="sgdm":    opt=torch.optim.SGD(net.parameters(),lr=LR,momentum=0.9)      # +leaky-integrator momentum
elif OPT=="rmsprop": opt=torch.optim.RMSprop(net.parameters(),lr=LR,alpha=0.95)   # per-weight /sqrt(v)
else:                opt=torch.optim.Adam(net.parameters(),lr=2e-3,weight_decay=1e-5)
EP=int(os.environ.get("EP","30")); bs=int(os.environ.get("BS","256"))
for ep in range(EP):
    perm=torch.randperm(ntr)
    for i in range(0,ntr,bs):
        b=perm[i:i+bs]; opt.zero_grad()
        loss=F.cross_entropy(net(Xtr[b]),ytr[b]); loss.backward(); opt.step()
        with torch.no_grad():
            for l in list(net.layers)+[net.readout]:
                if NOISE>0: l.weight.add_(torch.randn_like(l.weight)*NOISE*l.mask)   # weight-cap noise (SGLD)
                if WCLIP>0: l.weight.clamp_(-WCLIP,WCLIP)                            # cap voltage rails
                l.weight.mul_(l.mask)
    if ep%5==0 or ep==EP-1:
        with torch.no_grad(): acc=(net(Xte).argmax(1)==yte).float().mean().item()
        print(f"  ep{ep} test={acc*100:.2f}%")

EXPORT=os.environ.get("EXPORT","")
if EXPORT:
    # export connectivity (per-layer: for each output neuron, its input indices + weights) + biases,
    # the activation params, and a test batch, so the ngspice N-layer deck generator can build the
    # exact circuit. Each "layer" maps a flat input vector to a flat output vector.
    import numpy as np
    exp={"AMP":np.float32(AMP),"G":np.float32(G),"nlayers":len(net.layers)+1}
    allL=list(net.layers)+[net.readout]
    kinds_all=["h"]*len(net.layers)+["o"]
    for li,l in enumerate(allL):
        m=l.mask.numpy(); W=(l.weight.detach().numpy()*m); b=l.bias.detach().numpy()
        nout,nin=m.shape
        inidx=[np.where(m[o]>0)[0].astype(np.int32) for o in range(nout)]
        # pad to rectangular (fan-in<=4): store idx and weight arrays padded with -1 / 0
        K=max(len(x) for x in inidx)
        IDX=-np.ones((nout,K),np.int32); WT=np.zeros((nout,K),np.float32)
        for o in range(nout):
            IDX[o,:len(inidx[o])]=inidx[o]; WT[o,:len(inidx[o])]=W[o,inidx[o]]
        exp[f"L{li}_idx"]=IDX; exp[f"L{li}_w"]=WT; exp[f"L{li}_b"]=b.astype(np.float32)
        exp[f"L{li}_nin"]=np.int32(nin); exp[f"L{li}_nout"]=np.int32(nout); exp[f"L{li}_kind"]=kinds_all[li]
    nte=int(os.environ.get("EXPTE","300"))
    exp["Xte"]=Xte[:nte].numpy().astype(np.float32); exp["yte"]=yte[:nte].numpy().astype(np.int32); exp["D"]=np.int32(Xtr.shape[1])
    np.savez(EXPORT,**exp); print(f"exported {EXPORT}: {len(allL)} layers, test {nte}")
