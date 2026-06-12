#!/usr/bin/env python3
"""
RISK 3 (stretch): two-phase EP on XOR -- requires an IN-RANGE nonlinearity.

2 inputs -> 2 hidden -> 1 output, all reciprocal triode conductances.
Signed weights via differential pairs. Hidden units are PHYSICALLY differential:
each hidden unit j has nodes hj and hjb wired with swapped gates, so hjb carries the
complement response (no behavioral sources). Tight diode clamps (~[0.35,0.65]) give the
hidden/output nodes an in-range saturating nonlinearity.

Tension under test: in-range nonlinearity needs swing, but reciprocity needs small Vds.
"""
import numpy as np, subprocess, os

VT=0.2; LO=float(os.environ.get("LO","0.30")); HI=float(os.environ.get("HI","0.70"))
CL=float(os.environ.get("CL","0.35")); CH=float(os.environ.get("CH","0.65"))  # clamp rails
RB=float(os.environ.get("RB","33e3")); RLEAK=float(os.environ.get("RLEAK","80e3"))
NDIODE=os.environ.get("ND","0.2")

PATS=[(0,0,0),(0,1,1),(1,0,1),(1,1,0)]   # XOR
def vl(b): return HI if b else LO

# gates (trainable). input->hidden: g{in}{hid}{p/n}; hidden->output: o{hid}{p/n}
GATES=['g11p','g11n','g21p','g21n','g12p','g12n','g22p','g22n','o1p','o1n','o2p','o2n']
# conductances: (nodeA,nodeB,gate). nodeB current sign irrelevant (we use |Δv|^2).
def conductances():
    C=[]
    # unit1: h1 (normal), h1b (swapped gates)
    C+=[('x1','h1','g11p'),('x1b','h1','g11n'),('x2','h1','g21p'),('x2b','h1','g21n')]
    C+=[('x1b','h1b','g11p'),('x1','h1b','g11n'),('x2b','h1b','g21p'),('x2','h1b','g21n')]
    # unit2
    C+=[('x1','h2','g12p'),('x1b','h2','g12n'),('x2','h2','g22p'),('x2b','h2','g22n')]
    C+=[('x1b','h2b','g12p'),('x1','h2b','g12n'),('x2b','h2b','g22p'),('x2','h2b','g22n')]
    # hidden->output (single-ended output)
    C+=[('h1','o','o1p'),('h1b','o','o1n'),('h2','o','o2p'),('h2b','o','o2n')]
    return C
CONDS=conductances()
FREE_NODES=['h1','h1b','h2','h2b','o']

def deck(vg):
    L=["EP XOR step",
       ".model NSYN NMOS (LEVEL=1 VTO=0.2 KP=50u LAMBDA=0 GAMMA=0 PHI=0.7)",
       f".model DCL D (IS=1e-9 N={NDIODE})"]
    for s in ['x1','x2','x1b','x2b']: L.append(f"V{s} {s} 0 0.5")
    L.append("Vt vt 0 0.5")
    L.append(f"Vcl ncl 0 {CL}"); L.append(f"Vch nch 0 {CH}"); L.append("Vmid nmid 0 0.5")
    n=0
    for (a,b,g) in CONDS:
        L.append(f"M{n} {a} {g} {b} 0 NSYN W=200u L=100u"); n+=1
    for g in GATES: L.append(f"V{g} {g} 0 {vg[g]:.5f}")
    # neurons: leak to mid + clamp diodes to [CL,CH] on each free node
    for nd in FREE_NODES:
        L.append(f"Rl{nd} {nd} nmid {RLEAK}")
        L.append(f"Dlo{nd} ncl {nd} DCL")   # conducts if node<CL -> pulls up
        L.append(f"Dhi{nd} {nd} nch DCL")   # conducts if node>CH -> pulls down
    L.append("Rb o vt 1e12")
    L.append(".control")
    for i,(a,b,_) in enumerate(PATS):
        x1,x2=vl(a),vl(b)
        L+=[f"alter Vx1 {x1}",f"alter Vx2 {x2}",
            f"alter Vx1b {round(1-x1,3)}",f"alter Vx2b {round(1-x2,3)}",
            "alter Rb 1e12","op",f"wrdata xf{i}.dat "+" ".join(f"v({nd})" for nd in FREE_NODES)]
        t=vl(PATS[i][2])
        L+=[f"alter Vt {t}",f"alter Rb {RB}","op",
            f"wrdata xn{i}.dat "+" ".join(f"v({nd})" for nd in FREE_NODES)]
    L+=[".endc",".end"]
    return "\n".join(L)+"\n"

def readnodes(fn):
    d=np.loadtxt(fn).reshape(-1)  # pairs (t,v) x5 -> values at odd idx
    return d[1::2]                # [h1,h1b,h2,h2b,o]

def run(vg):
    open("ep_xor.cir","w").write(deck(vg))
    subprocess.run(["ngspice","-b","ep_xor.cir"],capture_output=True,text=True,timeout=120)
    F=np.array([readnodes(f"xf{i}.dat") for i in range(4)])
    N=np.array([readnodes(f"xn{i}.dat") for i in range(4)])
    return F,N   # shape (4 patterns, 5 nodes)

def main():
    eta=float(os.environ.get("ETA","0.4")); iters=int(os.environ.get("ITERS","40"))
    sign=float(os.environ.get("USIGN","-1")); seed=int(os.environ.get("SEED","0"))
    rng=np.random.default_rng(seed)
    vg={g:1.30+0.15*rng.standard_normal() for g in GATES}
    idx={nd:k for k,nd in enumerate(FREE_NODES)}
    tgt=np.array([vl(p[2]) for p in PATS]); beta=1.0/RB
    print(f"XOR eta={eta} iters={iters} sign={sign} seed={seed} swing[{LO},{HI}] clamp[{CL},{CH}]")
    best=9
    for it in range(iters):
        F,N=run(vg); o_free=F[:,idx['o']]
        loss=float(np.mean((o_free-tgt)**2)); best=min(best,loss)
        dvg={g:0.0 for g in GATES}
        for i,(a,b,_) in enumerate(PATS):
            x1,x2=vl(a),vl(b)
            val={'x1':x1,'x2':x2,'x1b':round(1-x1,3),'x2b':round(1-x2,3)}
            def nv(node,arr): return val[node] if node in val else arr[idx[node]]
            for (na,nb,g) in CONDS:
                df=(nv(na,F[i])-nv(nb,F[i]))**2
                dn=(nv(na,N[i])-nv(nb,N[i]))**2
                dvg[g]+=sign*eta*((dn-df)/beta)
        for g in GATES: vg[g]=float(np.clip(vg[g]+dvg[g]/4,0.9,2.2))
        if it%5==0 or it==iters-1:
            acc=float(np.mean((o_free>0.5)==(tgt>0.5)))
            print(f"it{it:3d} loss={loss:.5f} best={best:.5f} acc={acc:.2f} o*={np.round(o_free,3)}")
    print("XOR final loss",round(loss,5),"best",round(best,5),"o*",np.round(o_free,3),"tgt",tgt)

if __name__=="__main__": main()
