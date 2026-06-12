#!/usr/bin/env python3
# Analog-FAITHFUL training simulator (Python), calibrated to the ngspice cells, used to design the
# sparse multi-layer topology that reaches 90% on 10-class MNIST. Models the non-idealities that the
# ngspice runs actually showed:
#   - tanh keystone neuron (bounded diff-pair), output = linear keystone with finite range
#   - in-circuit SGD: online batch-1 updates integrated on weight caps, with weight RAIL clipping
#   - CMSUB (subtract backward common-mode) + OCMSUB (subtract output-error common-mode) -- the two
#     fixes derived in silicon; without them multi-class collapses (verified in ngspice)
#   - weight-cap NOISE (SGLD-style) -- the "noise" the goal asks for
#   - sparse connectivity (local receptive fields) to keep analog fan-in low
# It is a DESIGN tool; ngspice remains the proof of training. Reports test accuracy on full MNIST.
import numpy as np, os, sys

S=int(os.environ.get("RES","8"))                 # input resolution SxS
d=np.load(f"mnist_{S}x{S}.npz"); X=d["X"].astype(np.float32); y=d["y"]
rng=np.random.default_rng(0)
idx=rng.permutation(len(y)); X,y=X[idx],y[idx]
NTR=int(os.environ.get("NTR","20000")); NTE=8000
Xtr,ytr=X[:NTR],y[:NTR]; Xte,yte=X[NTR:NTR+NTE],y[NTR:NTR+NTE]
# per-image z-score (energy equalization, the proven analog-input normalization) -> input band
def norm(A):
    m=A.mean(1,keepdims=True); s=A.std(1,keepdims=True)+1e-6
    return np.clip((A-m)/s*0.25+0.5,0,1)
Xtr=norm(Xtr); Xte=norm(Xte); D=S*S; C=10

# ---- sparse connectivity: conv-like local receptive fields on the SxS grid ----
def conv_mask(S, ksize, stride, nfilt):
    pos=[(r,c) for r in range(0,S-ksize+1,stride) for c in range(0,S-ksize+1,stride)]
    nout=len(pos)*nfilt; m=np.zeros((nout,S*S),np.float32); o=0
    for (r,c) in pos:
        rf=[ (r+i)*S+(c+j) for i in range(ksize) for j in range(ksize) ]
        for f in range(nfilt): m[o,rf]=1; o+=1
    return m
def rand_sparse(nin,nout,K,seed):
    r=np.random.default_rng(seed); m=np.zeros((nout,nin),np.float32)
    for o in range(nout): m[o, r.choice(nin,size=min(K,nin),replace=False)]=1
    return m

# topology spec from env: e.g. LAYERS="conv:3,1,4 ; rand:64 ; out"
spec=os.environ.get("TOPO","conv:3,1,6|rand:64,8|out")
layers=[]; nin=D
for part in spec.split("|"):
    if part.startswith("conv:"):
        k,st,nf=[int(x) for x in part[5:].split(",")]
        m=conv_mask(S,k,st,nf); layers.append(("h",m)); nin=m.shape[0]
    elif part.startswith("rand:"):
        a=part[5:].split(","); nout=int(a[0]); K=int(a[1]) if len(a)>1 else 8
        m=rand_sparse(nin,nout,K,len(layers)); layers.append(("h",m)); nin=m.shape[0]
    elif part=="out":
        m=rand_sparse(nin,C,int(os.environ.get("OUTK",str(min(nin,12)))),99) if os.environ.get("OUTSPARSE") else np.ones((C,nin),np.float32)
        layers.append(("o",m)); nin=C
G=float(os.environ.get("G","1.0")); AMP=0.45
NOISE=float(os.environ.get("NOISE","0.0")); WCLIP=float(os.environ.get("WCLIP","1.5"))
CMSUB=int(os.environ.get("CMSUB","1")); LR=float(os.environ.get("LR","0.03")); EP=int(os.environ.get("EP","12"))
Ws=[ (rng.uniform(-0.7,0.7,m.shape)*m).astype(np.float32) for _,m in layers]
bs=[ np.zeros(m.shape[0],np.float32) for _,m in layers]
masks=[m for _,m in layers]; kinds=[k for k,_ in layers]
def fwd(A):
    acts=[A]
    for l,(W,b,m,k) in enumerate(zip(Ws,bs,masks,kinds)):
        z=A@W.T+b
        A = z if k=="o" else AMP*np.tanh(G*z)
        acts.append(A)
    return acts
def evaluate(X,Y):
    p=fwd(X)[-1].argmax(1); return (p==Y).mean()
nb=len(Xtr)
for ep in range(EP):
    order=rng.permutation(nb)
    for k in order:
        x=Xtr[k][None]; t=np.full((1,C),0.0,np.float32); t[0,ytr[k]]=1.0
        acts=fwd(x)
        e=acts[-1]-t; e=e-e.mean(1,keepdims=True)          # OCMSUB
        delta=e
        for l in reversed(range(len(Ws))):
            a_prev=acts[l]; z=a_prev@Ws[l].T+bs[l]
            g = delta if kinds[l]=="o" else delta*(AMP*G)*(1-np.tanh(G*z)**2)
            if CMSUB and kinds[l]!="o": g=g-g.mean(1,keepdims=True)
            gW=(g.T@a_prev)*masks[l]
            Ws[l]-=LR*gW; bs[l]-=LR*g[0]
            if NOISE>0: Ws[l]+=rng.normal(0,NOISE,Ws[l].shape).astype(np.float32)*masks[l]
            Ws[l]=np.clip(Ws[l],-WCLIP,WCLIP)
            delta=g@Ws[l]
    if ep%3==0 or ep==EP-1: print(f"  ep{ep} test={evaluate(Xte,yte)*100:.1f}%")
nw=sum(int(m.sum()) for m in masks); fanins=[int(m.sum(1).max()) for m in masks]
print(f"RES={S} TOPO={spec}  weights={nw} fanins={fanins}  FINAL test={evaluate(Xte,yte)*100:.2f}%")
