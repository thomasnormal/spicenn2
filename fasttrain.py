#!/usr/bin/env python3
"""FAST device-calibrated trainer-simulator of pc_deep's deep-narrow tanh PC (spirals recipe).
Cell models fitted from Spectre DC characterization (fastsim/char.npz):
  gsyn:    out_diff = -G * tanh(kx*in_diff) * tanh(kw*w_diff)   [measured: INVERTS; G~0.0699@|w|=.3 -> fit kw]
  dneuron: a = 0.537*tanh(15.7*x)
  esub:    eps = 0.631*tanh(3.91*(x-m))
  gprod:   dW/slot = ETA * tanh(ke*e) * tanh(ka*y)   [ETA calibrated vs a known Spectre run]
Topology/init/splits reproduce pc_deep EXACTLY (same rng draw order, same SEED/WSEED semantics).
Settle-per-slot fixed point (timescale separation), Euler weight step per slot, early-stop eval blocks.
VALIDATION GATE: must reproduce hard-clamp [s0 .963 s2 .950 s3 .944 s1 .912] / soft-clamp [.969 .969 .956 ~.88]
orderings before being trusted for exploration."""
import os, numpy as np

# ---------------- fitted cell constants ----------------
G_SYN  = float(os.environ.get("G_SYN","0.105"))   # full-sat synapse swing (fit: 0.0699 at tanh(kw*0.3); kw~2.6 -> G=0.0699/tanh(.78)~0.107)
KX     = 2.25
KW     = float(os.environ.get("KW","2.6"))
A_NEU  = 0.537; K_NEU = 15.7
A_EPS  = 0.631; K_EPS = 3.91
ETA    = float(os.environ.get("ETA","0.0035"))    # per-slot weight step scale (calibrate)
KE, KA = 3.0, 3.0                                  # gprod input gains (Gilbert-like)
GBK    = float(os.environ.get("GBK","0.5"))       # backward injection strength (vbbk-dependent)
def syn(in_d, w):   return -G_SYN*np.tanh(KX*in_d)*np.tanh(KW*w)
def neu(x):         return A_NEU*np.tanh(K_NEU*x)
def epsf(x,m):      return A_EPS*np.tanh(K_EPS*(x-m))
def gprod(e,y):     return np.tanh(KE*e)*np.tanh(KA*y)

# ---------------- exact pc_deep data/init/topology ----------------
SEED=int(os.environ.get("SEED","0")); rng=np.random.default_rng(SEED)
WSEED=int(os.environ.get("WSEED",str(SEED))); wrng=np.random.default_rng(WSEED+1000)
TASK=os.environ.get("TASK","spirals"); C=2
LAYERS=[int(x) for x in os.environ.get("LAYERS","2,4,4,4,4,4").split(",")]+[C]
K=int(os.environ.get("FANIN","4"))
NTR=int(os.environ.get("NTR","60")); NTE=int(os.environ.get("NTE","80")); NEP=int(os.environ.get("NEP","24"))
SCALE=float(os.environ.get("SCALE","1.0")); IND=float(os.environ.get("IND","0.30")); TD=float(os.environ.get("TD","0.1"))
KMAP=0.30; VW0=0.5
SGNH=float(os.environ.get("SGNH","-1")); SGNO=float(os.environ.get("SGNO","-1"))
WINIT=float(os.environ.get("WINIT","1.5")); WINITO=float(os.environ.get("WINITO","0.3"))
GBLH=float(os.environ.get("GBLH","0.6")); GBLO=float(os.environ.get("GBLO","0.6")); AFL=float(os.environ.get("AFLOOR","0.4"))
SOFTC=int(os.environ.get("SOFTC","1")); BETA=float(os.environ.get("BETA","0.17"))  # clamp mix weight ~ RNODE/RCLAMP
EVK=int(os.environ.get("EVK","4"))
NL=len(LAYERS)
def gen():
    rg=np.random.default_rng(7); X=[];y=[]
    if TASK=="spirals":
        for c in range(C):
            t=np.linspace(0.2,1,1500); r=t*2.5; th=t*1.8*np.pi+c*2*np.pi/C+rg.normal(0,0.08,1500)
            X+=list(np.c_[r*np.cos(th),r*np.sin(th)]); y+=[c]*1500
    elif TASK=="circles":
        for c in range(2):
            r=(rg.uniform(0.2,0.6,1500) if c==0 else rg.uniform(1.0,1.5,1500)); th=rg.uniform(0,2*np.pi,1500)
            X+=list(np.c_[r*np.cos(th),r*np.sin(th)]); y+=[c]*1500
    return np.array(X,float),np.array(y)
X,y=gen(); m_=X.mean(0); s_=X.std(0)+1e-6; X=(X-m_)/s_*SCALE
Xtr=[];ytr=[];Xte=[];yte=[]
for c in range(C):
    i=np.where(y==c)[0]; rng.shuffle(i)
    for k in i[:NTR]: Xtr.append(X[k]); ytr.append(c)
    for k in i[NTR:NTR+NTE]: Xte.append(X[k]); yte.append(c)
Xtr=np.array(Xtr); ytr=np.array(ytr); Xte=np.array(Xte); yte=np.array(yte)
NIN=LAYERS[0]
parents={}
for l in range(1,NL):
    nprev=LAYERS[l-1]; nl=LAYERS[l]; kk=min(K,nprev)
    for j in range(nl):
        ps=list(rng.choice(nprev,size=kk,replace=False)) if kk<nprev else list(range(nprev))
        parents[(l,j)]=ps
W={}
for l in range(1,NL):
    for j in range(LAYERS[l]):
        for p in parents[(l,j)]:
            v = WINITO*float(wrng.standard_normal()) if l==NL-1 else WINIT*float(wrng.standard_normal())
            gp=float(np.clip(VW0+KMAP*v,0.3,0.9)); gn=float(np.clip(VW0-KMAP*v,0.3,0.9))
            W[(l,j,p)] = gp-gn   # EXACT circuit wgv mapping: differential in [-0.6,0.6]
# ---------------- network settle + train ----------------
def encode(xv):
    out=np.zeros(NIN)
    for d in range(NIN):
        v = xv[d] if d<len(xv) else 1.0
        out[d] = 2*IND*np.clip(v,-3,3)/3.0*2   # differential input = 2*(0.5+IND*clip/3*2 - 0.5)
    return out
def settle(a0, yclamp=None, train=False, iters=12):
    """a0: input activations (differential). Returns m_out, internal acts."""
    a={0:a0}
    x={}; m={}; eps={l:np.zeros(LAYERS[l]) for l in range(1,NL)}
    for it in range(iters):
        for l in range(1,NL):
            mm=np.zeros(LAYERS[l])
            for j in range(LAYERS[l]):
                for p in parents[(l,j)]:
                    mm[j]+=syn(a[l-1][p], W[(l,j,p)])
            m[l]=mm
            if l<NL-1:
                bk=np.zeros(LAYERS[l])
                if train:
                    for j2 in range(LAYERS[l+1]):
                        for p2 in parents[(l+1,j2)]:
                            bk[p2]+= syn(SGNH*eps[l+1][j2], W[(l+1,j2,p2)])
                x[l]=m[l]+GBK*bk
                a[l]=neu(x[l])
            else:
                if train and yclamp is not None:
                    if SOFTC: x[l]=(m[l]+BETA*yclamp)/(1+BETA)
                    else:     x[l]=yclamp
                else: x[l]=m[l]
            eps[l]=epsf(x[l],m[l])
        # one more pass refines via updated eps (loop)
    return m[NL-1], a, eps
def evaluate():
    correct=0
    M=[]
    for i in range(len(Xte)):
        mo,_,_=settle(encode(Xte[i]),train=False,iters=4)
        M.append(mo)
    M=np.array(M); M=M-M.mean(0)
    return float((np.argmax(M,1)==yte).mean())
total=NEP*len(Xtr); cnt=0; curve=[]
for ep in range(NEP):
    idx=list(range(len(Xtr))); rng.shuffle(idx)
    for k in idx:
        f=cnt/total; cnt+=1
        lrh=(GBLH-(GBLH-AFL)*f)/GBLH; lro=(GBLO-(GBLO-AFL)*f)/GBLO
        yc=np.full(C,-TD); yc[ytr[k]]=TD; yc*=2  # differential clamp
        _,a,eps=settle(encode(Xtr[k]),yclamp=yc,train=True)
        for l in range(1,NL):
            sg = SGNO if l==NL-1 else SGNH
            lr = ETA*(lro if l==NL-1 else lrh)
            for j in range(LAYERS[l]):
                for p in parents[(l,j)]:
                    W[(l,j,p)] = float(np.clip(W[(l,j,p)] + lr*gprod(sg*eps[l][j], a[l-1][p]), -0.6, 0.6))
    if (ep+1)%EVK==0 or ep==NEP-1:
        curve.append(round(evaluate(),3))
print(f"[fasttrain {TASK} s{SEED} SOFTC={SOFTC}] BEST={max(curve):.3f} curve={curve}")
