#!/usr/bin/env python3
"""FAST numpy ODE prototype of the pc_spice IN-CIRCUIT readout learning, to find a STABLE converging rule
+ component values before paying for 20-min ngspice .tran runs.  Models the same nonlinear cells:
  - readout mu_c = sum_j Gilbert(a_h_j, w_jc)   [tanh.tanh product, the gsyn]
  - error eps_c (saturating diff-pair = esub)
  - weight caps integrate the gprod product current (saturating); leak to 0
  - MARGIN GATE (the hinge): freeze the per-class update once that example is correct-with-margin; gate has a
    finite time-constant (the conf filter cap) -> models the comparator feedback.
Goal: a config that trains circles >0.95 from zero, stably.  Then port w-scale/rates/gate to pc_spice.
"""
import numpy as np, os, sys
rng=np.random.default_rng(0)
TASK=os.environ.get("TASK","circles"); C=int(os.environ.get("C","2")); H=int(os.environ.get("H","48"))
# ---- data + FIXED random analog-like hidden features (tanh of random projection; separable, like the real a_h) ----
def gen():
    rg=np.random.default_rng(7); X=[];y=[]
    if TASK=="circles":
        for c in range(2):
            r=(rg.uniform(0.2,0.6,1500) if c==0 else rg.uniform(1.0,1.5,1500)); th=rg.uniform(0,2*np.pi,1500)
            X+=list(np.c_[r*np.cos(th),r*np.sin(th)]); y+=[c]*1500
    elif TASK=="rings":
        for c in range(C):
            r=rg.uniform(1.2*c+0.25,1.2*c+0.55,1500); th=rg.uniform(0,2*np.pi,1500); X+=list(np.c_[r*np.cos(th),r*np.sin(th)]); y+=[c]*1500
    elif TASK=="spirals":
        for c in range(C):
            t=np.linspace(0.2,1,1500); r=t*2.5; th=t*1.8*np.pi+c*2*np.pi/C+rng.normal(0,0.08,1500); X+=list(np.c_[r*np.cos(th),r*np.sin(th)]); y+=[c]*1500
    return np.array(X),np.array(y)
X,y=gen(); X=(X-X.mean(0))/(X.std(0)+1e-6)*1.5; X=np.c_[X,np.ones(len(X))]; N=X.shape[1]
WIH=rng.standard_normal((H,N))*1.5
AH=np.tanh(X@WIH.T)*0.35   # analog a_h scale ~0.35 like the real circuit
AH=np.c_[AH,np.full(len(AH),0.30)]   # + constant bias feature
NTR=int(os.environ.get("NTR","120")); NTE=120
idx=rng.permutation(len(y));
def split():
    tr_i=[];te_i=[]
    for c in range(C):
        ii=idx[y[idx]==c]; tr_i+=list(ii[:NTR]); te_i+=list(ii[NTR:NTR+NTE])
    return np.array(tr_i),np.array(te_i)
tri,tei=split(); Atr=AH[tri]; Ytr=y[tri]; Ate=AH[tei]; Yte=y[tei]; Hf=AH.shape[1]
if os.environ.get("LOADAH"):   # use REAL ngspice-extracted a_h instead of idealized features
    Z=np.load(os.environ["LOADAH"]); Atr=Z["Atr"]; Ytr=Z["Ytr"]; Ate=Z["Ate"]; Yte=Z["Yte"]
    Atr=np.c_[Atr,np.full(len(Atr),0.30)]; Ate=np.c_[Ate,np.full(len(Ate),0.30)]   # +bias feature
    Hf=Atr.shape[1]; C=int(Yte.max())+1
    print(f"  [LOADAH] real a_h Atr={Atr.shape} Ate={Ate.shape} |a_h|mean={np.abs(Atr).mean():.4f}")
# ---- in-circuit cell models ----
kw=float(os.environ.get("KW","3.0"))      # weight tanh sharpness (Gilbert weight nonlinearity)
ka=float(os.environ.get("KA","3.0"))      # input tanh sharpness
def readout(W,A):    # mu_c = sum_j tanh(ka*a)*tanh(kw*w)   (W = differential weight, ~[-0.4,0.4])
    return (np.tanh(ka*A)[:,None,:]*np.tanh(kw*W)[None,:,:]).sum(-1)   # (B,C)
def esub(e):         # saturating error (diff pair)
    return np.tanh(float(os.environ.get("ESG","8"))*e)                 # eps saturates -> perceptron-ish magnitude
# ---- training (Euler ODE over example presentations) ----
RULE=os.environ.get("RULE","hinge"); MAR=float(os.environ.get("MAR","0.04")); TD=float(os.environ.get("TD","0.05"))
ETA=float(os.environ.get("ETA","0.02")); LEAK=float(os.environ.get("LEAK","2e-3")); WCL=float(os.environ.get("WCL","0.45"))
EP=int(os.environ.get("EP","60")); NORM=int(os.environ.get("NORM","1"))   # NORM=subtract output common-mode (CE-like)
# separability of the prototype features (sanity)
Ab=np.c_[Atr.reshape(len(Atr),-1),np.ones(len(Atr))]; T=np.full((len(Ytr),C),-1.);T[np.arange(len(Ytr)),Ytr]=1
Wr=np.linalg.solve(Ab.T@Ab+1e-1*np.eye(Ab.shape[1]),Ab.T@T)
sep=float(((np.c_[Ate,np.ones(len(Ate))]@Wr).argmax(1)==Yte).mean())
W=np.zeros((C,Hf))
def evaluate(W):
    M=readout(W,Ate); M=M-M.mean(0)
    return float((M.argmax(1)==Yte).mean())
best=0; Wbest=W.copy(); ANNEAL=int(os.environ.get("ANNEAL","1"))
for ep in range(EP):
    eta=ETA*(((1-ep/EP)**2) if ANNEAL else 1.0)
    o=rng.permutation(len(Ytr))
    for k in o:
        a=Atr[k]; lab=Ytr[k]; mu=readout(W,a[None])[0]           # (C,)
        mn=mu-mu.mean() if NORM else mu                          # normalize across classes (CE-like / GLOBALAZ)
        tg=np.full(C,-TD); tg[lab]=TD
        eps=esub(tg-mn)                                          # (C,) saturating error
        gate=np.ones(C)
        GG=float(os.environ.get("GAING","0"))   # >0 = SOFT gate (sigmoid), models a low-gain non-stiff comparator
        if RULE=="hinge":   # PER-CLASS hinge: freeze a class's update once it's on the right side w/ margin
            for c in range(C):
                if int(os.environ.get("SIMPLEGATE","0")): sat=mn[c] if c==lab else -mn[c]   # use centered output directly (no WTA)
                else: sat=(mn[c]-np.max(np.delete(mn,lab))) if c==lab else (mn[lab]-mn[c])   # how far past the margin
                if GG>0: gate[c]=0.5*(1-np.tanh(GG*(sat-MAR)))                          # soft
                else: gate[c]=0.0 if sat>MAR else 1.0                                   # hard
        dW=gate[:,None]*eta*(eps[:,None]*np.tanh(ka*a)[None,:]) - LEAK*W
        W=np.clip(W+dW,-WCL,WCL)
    if ep%3==2 or ep==EP-1:
        a=evaluate(W)
        if a>best: best=a; Wbest=W.copy()
print(f"[PROTO {TASK} C={C} RULE={RULE} NORM={NORM} H={H}] sep={sep:.3f} FINAL test={evaluate(W):.3f} best={best:.3f}")
