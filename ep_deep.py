#!/usr/bin/env python3
"""
Layer-based current-mode EP (scales to arbitrary DEPTH and width).
Same validated method as ep_cm.py (current-mode synapses, correlation update, cos~0.97),
generalized to LAYERS=[2, H1, H2, ..., 1]. Demonstrates depth scalability of the method.

Couplings between adjacent layers are bidirectional + weight-shared (symmetric -> valid energy).
Inputs (layer 0) are clamped; their coupling to layer 1 is forward-only.
"""
import numpy as np, subprocess, os
LAYERS=[int(x) for x in os.environ.get("LAYERS","2,6,1").split(",")]
GAIN=float(os.environ.get("GAIN","12")); RL=os.environ.get("RL","1")
RB=float(os.environ.get("RB","5")); BETA=1.0/RB
ALO,AHI=0.1,0.9; TLO,THI=float(os.environ.get("TLO","0.15")),float(os.environ.get("THI","0.85"))
PATS=[(0,0,0),(0,1,1),(1,0,1),(1,1,0)]
def xv(b): return AHI if b else ALO
def tv(b): return THI if b else TLO

# node name for (layer l, index i): inputs are x; others are L{l}_{i}
def node(l,i): return f"a_x{i+1}" if l==0 else f"L{l}_{i}"
def unode(l,i): return f"u_L{l}_{i}"
NONIN=[(l,i) for l in range(1,len(LAYERS)) for i in range(LAYERS[l])]
OUTL=len(LAYERS)-1
def wkey(l,i,j): return f"w{l}_{i}_{j}"      # layer l index i -> layer l+1 index j
def bkey(l,i): return f"b{l}_{i}"
WEIGHTS=[wkey(l,i,j) for l in range(len(LAYERS)-1) for i in range(LAYERS[l]) for j in range(LAYERS[l+1])]
WEIGHTS+=[bkey(l,i) for (l,i) in NONIN]
READ=[unode(l,i) for (l,i) in NONIN]+[node(l,i) for (l,i) in NONIN if l!=OUTL]
idx={n:k for k,n in enumerate(READ)}
OI=idx[unode(OUTL,0)]

def aexpr(l,i): return node(l,i) if l==0 else (unode(l,i) if l==OUTL else node(l,i))
def deck(w):
    L=["deep current-mode EP","Vmid nmid 0 0.5","Vt vt 0 0.5"]
    for i in range(LAYERS[0]): L.append(f"Va_x{i+1} a_x{i+1} 0 0.5")
    for (l,i) in NONIN:
        u=unode(l,i); terms=[]
        # forward from previous layer
        for p in range(LAYERS[l-1]):
            terms.append(f"({w[wkey(l-1,p,i)]:.5f})*(V({aexpr(l-1,p)})-0.5)")
        # backward from next layer (shared weight wkey(l,i,q))
        if l<OUTL:
            for q in range(LAYERS[l+1]):
                terms.append(f"({w[wkey(l,i,q)]:.5f})*(V({aexpr(l+1,q)})-0.5)")
        terms.append(f"({w[bkey(l,i)]:.5f})*0.4")
        L.append(f"B_{l}_{i} 0 {u} I="+"+".join(terms))
        L.append(f"R_{l}_{i} {u} nmid {RL}")
        if l<OUTL:  # hidden neuron (gain); output stays linear (a=u)
            L.append(f"Ba_{l}_{i} {node(l,i)} 0 V=0.5+0.5*tanh({GAIN}*(V({u})-0.5))")
    L.append("Rb "+unode(OUTL,0)+" vt 1e9")
    L.append(".control"); cols=" ".join(f"v({n})" for n in READ)
    for k,(p,q,_) in enumerate(PATS):
        L+=[f"alter Va_x1 {xv(p)}",f"alter Va_x2 {xv(q)}","alter Rb 1e9","op",f"wrdata df{k}.dat {cols}"]
        L+=[f"alter Vt {tv(PATS[k][2])}",f"alter Rb {RB}","op",f"wrdata dn{k}.dat {cols}"]
    L+=[".endc",".end"]; return "\n".join(L)+"\n"

def rd(fn): return np.loadtxt(fn).reshape(-1)[1::2]
def run(w):
    open("ep_deep.cir","w").write(deck(w))
    subprocess.run(["ngspice","-b","ep_deep.cir"],capture_output=True,text=True,timeout=120)
    return (np.array([rd(f"df{k}.dat") for k in range(4)]),
            np.array([rd(f"dn{k}.dat") for k in range(4)]))

def phi(l,i,arr):
    if l==0: return 0.0  # set per pattern below
    return arr[idx[aexpr(l,i)]]-0.5

def ep_grad(w):
    F,N=run(w); g={k:0.0 for k in WEIGHTS}
    for k,pat in enumerate(PATS):
        xp={0:xv(pat[0])-0.5,1:xv(pat[1])-0.5}
        def ph(l,i,arr): return xp[i] if l==0 else arr[idx[aexpr(l,i)]]-0.5
        for l in range(len(LAYERS)-1):
            for i in range(LAYERS[l]):
                for j in range(LAYERS[l+1]):
                    kk=wkey(l,i,j)
                    f=ph(l,i,F[k])*ph(l+1,j,F[k]); n=ph(l,i,N[k])*ph(l+1,j,N[k])
                    g[kk]+=((n-f)/BETA)/4.0
        for (l,i) in NONIN:
            f=0.4*ph(l,i,F[k]); n=0.4*ph(l,i,N[k]); g[bkey(l,i)]+=((n-f)/BETA)/4.0
    return g,F

def main():
    import sys; mode=sys.argv[1] if len(sys.argv)>1 else "train"
    rng=np.random.default_rng(int(os.environ.get("SEED","0")))
    tgt=np.array([tv(p[2]) for p in PATS])
    if mode=="gradcheck":
        d=0.01; coss=[]
        for t in range(int(os.environ.get("T","4"))):
            w={k:float(0.3*rng.standard_normal()) for k in WEIGHTS}; gfd={}
            for k in WEIGHTS:
                wp=dict(w);wp[k]+=d; wm=dict(w);wm[k]-=d
                Fp,_=run(wp); Fm,_=run(wm)
                gfd[k]=(np.mean((Fp[:,OI]-tgt)**2)-np.mean((Fm[:,OI]-tgt)**2))/(2*d)
            gep,_=ep_grad(w); a=np.array([gep[k] for k in WEIGHTS]); b=np.array([gfd[k] for k in WEIGHTS])
            coss.append(float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-18)))
        print(f"LAYERS={LAYERS} MEANCOS {np.mean(coss):+.3f} (min {min(coss):+.3f})"); return
    eta=float(os.environ.get("ETA","0.04")); iters=int(os.environ.get("ITERS","250"))
    mom=float(os.environ.get("MOM","0.8")); sign=float(os.environ.get("USIGN","1"))
    w={k:float(0.3*rng.standard_normal()) for k in WEIGHTS}; vel={k:0.0 for k in WEIGHTS}; best=9
    print(f"DEEP-TRAIN LAYERS={LAYERS} eta={eta} seed={os.environ.get('SEED','0')}")
    for it in range(iters):
        g,F=ep_grad(w); o=F[:,OI]; loss=float(np.mean((o-tgt)**2)); best=min(best,loss)
        for k in WEIGHTS: vel[k]=mom*vel[k]+sign*eta*g[k]; w[k]=float(np.clip(w[k]+vel[k],-3,3))
        if it%25==0 or it==iters-1:
            acc=float(np.mean((o>0.5)==(tgt>0.5)))
            print(f"it{it:3d} loss={loss:.5f} best={best:.5f} acc={acc:.2f} u_o={np.round(o,3)}")
    print(f"FINAL LAYERS={LAYERS} best={round(best,5)} u_o={np.round(o,3)} tgt={tgt}")

if __name__=="__main__": main()
