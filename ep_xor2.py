#!/usr/bin/env python3
"""
RISK 3 redux: XOR with the two unblocks from this session.
  - synapses = CMOS transmission gates (reciprocal at FULL swing; rec3.cir result)
  - neurons  = one-port nonlinear elements (diode clamp to rails -> reciprocal for free)
  - full signal swing -> the neuron nonlinearity actually engages

Topology 2->2->1, signed weights via differential inputs + physically mirrored hidden
nodes (h, hb). Each conductance is a T-gate driven by one stored control vc:
    NMOS gate = 1.0 + vc ,  PMOS gate = -vc   (=> G ~ 60 + 200*vc uS, reciprocal; rec3b.cir)
Each hidden/output node has a weak leak to 0.5 + sharp diode clamp to [CLO,CHI] (the
nonlinearity) + a programmable bias T-gate pair to rails 0/1 (so drive can saturate it).

Two-phase EP, equilibrium by .op. Local update per weight (sum over its T-gates):
    dG ~ (1/beta)[ (dv^nudge)^2 - (dv^free)^2 ] ,  dv = v(A)-v(B) across the T-gate.
Optional centered nudging (+beta and -beta via a reflected target) for an unbiased estimate.
"""
import numpy as np, subprocess, os

VT=0.2
LO=float(os.environ.get("LO","0.15")); HI=float(os.environ.get("HI","0.85"))
CLO=float(os.environ.get("CLO","0.08")); CHI=float(os.environ.get("CHI","0.92"))   # output clamp (wide)
HCLO=float(os.environ.get("HCLO","0.40")); HCHI=float(os.environ.get("HCHI","0.60"))  # hidden clamp (tight = in-band nonlinearity)
ND=os.environ.get("ND","0.25")
RB=float(os.environ.get("RB","20e3")); RLEAK=float(os.environ.get("RLEAK","300e3"))
VCMIN,VCMAX=-0.30,0.55
PATS=[(0,0,0),(0,1,1),(1,0,1),(1,1,0)]
def vl(b): return HI if b else LO

# weights: input->hidden w_i_j_s ; hidden->output o_j_s ; bias bh_j_s, bo_s
WEIGHTS=[]
for i in (1,2):
    for j in (1,2):
        for s in ('p','n'): WEIGHTS.append(f"w{i}{j}{s}")
for j in (1,2):
    for s in ('p','n'): WEIGHTS.append(f"o{j}{s}")
for j in (1,2):
    for s in ('p','n'): WEIGHTS.append(f"bh{j}{s}")
for s in ('p','n'): WEIGHTS.append(f"bo{s}")

def conductances():
    """list of (nodeA,nodeB,weight_key). T-gate built for each."""
    C=[]
    for i in (1,2):
        xi,xib=f"x{i}",f"x{i}b"
        for j in (1,2):
            # hidden hj: +weight from xi, -weight from xib ; mirror hjb: swapped
            C+=[(xi,f"h{j}",f"w{i}{j}p"),(xib,f"h{j}",f"w{i}{j}n")]
            C+=[(xib,f"h{j}b",f"w{i}{j}p"),(xi,f"h{j}b",f"w{i}{j}n")]
    for j in (1,2):
        C+=[(f"h{j}","o",f"o{j}p"),(f"h{j}b","o",f"o{j}n")]
        # hidden bias: hj toward vhi(+)/vlo(-), mirror opposite
        C+=[("vhi",f"h{j}",f"bh{j}p"),("vlo",f"h{j}",f"bh{j}n")]
        C+=[("vlo",f"h{j}b",f"bh{j}p"),("vhi",f"h{j}b",f"bh{j}n")]
    C+=[("vhi","o","bop"),("vlo","o","bon")]
    return C
CONDS=conductances()
FREE=['h1','h1b','h2','h2b','o']

def deck(vc):
    L=["EP XOR2 t-gate + one-port neuron",
       ".model NSYN NMOS (LEVEL=1 VTO=0.2  KP=50u LAMBDA=0 GAMMA=0 PHI=0.7)",
       ".model PSYN PMOS (LEVEL=1 VTO=-0.2 KP=40u LAMBDA=0 GAMMA=0 PHI=0.7)",
       f".model DCL D (IS=1e-9 N={ND})",
       "Vdd vdd 0 1.5","Vhi vhi 0 1.0","Vlo vlo 0 0.0",
       "Vmid nmid 0 0.5",f"Vclo nclo 0 {CLO}",f"Vchi nchi 0 {CHI}",
       f"Vhclo nhclo 0 {HCLO}",f"Vhchi nhchi 0 {HCHI}","Vt vt 0 0.5"]
    for s in ('x1','x2','x1b','x2b'): L.append(f"V{s} {s} 0 0.5")
    # gate sources per weight
    for w in WEIGHTS:
        v=vc[w]; L.append(f"Vgn_{w} gn_{w} 0 {1.0+v:.5f}"); L.append(f"Vgp_{w} gp_{w} 0 {-v:.5f}")
    # T-gates
    for k,(a,b,w) in enumerate(CONDS):
        L.append(f"Mn{k} {a} gn_{w} {b} 0   NSYN W=200u L=100u")
        L.append(f"Mp{k} {a} gp_{w} {b} vdd PSYN W=250u L=100u")
    # neurons
    for nd in FREE:
        L.append(f"Rl{nd} {nd} nmid {RLEAK}")
        lo,hi=("nhclo","nhchi") if nd!="o" else ("nclo","nchi")  # tight clamp on hidden, wide on output
        L.append(f"Dlo{nd} {lo} {nd} DCL")
        L.append(f"Dhi{nd} {nd} {hi} DCL")
    L.append("Rb o vt 1e12")
    L.append(".control")
    cols=" ".join(f"v({nd})" for nd in FREE)
    for i,(a,b,_) in enumerate(PATS):
        x1,x2=vl(a),vl(b)
        L+=[f"alter Vx1 {x1}",f"alter Vx2 {x2}",
            f"alter Vx1b {round(1-x1,3)}",f"alter Vx2b {round(1-x2,3)}",
            "alter Rb 1e12","op",f"wrdata x2f{i}.dat {cols}"]
        t=vl(PATS[i][2])
        L+=[f"alter Vt {t}",f"alter Rb {RB}","op",f"wrdata x2n{i}.dat {cols}"]
    L+=[".endc",".end"]
    return "\n".join(L)+"\n"

def rd(fn):
    d=np.loadtxt(fn).reshape(-1); return d[1::2]

def run(vc):
    open("ep_xor2.cir","w").write(deck(vc))
    subprocess.run(["ngspice","-b","ep_xor2.cir"],capture_output=True,text=True,timeout=180)
    F=np.array([rd(f"x2f{i}.dat") for i in range(4)])
    N=np.array([rd(f"x2n{i}.dat") for i in range(4)])
    return F,N

def main():
    eta=float(os.environ.get("ETA","0.3")); iters=int(os.environ.get("ITERS","60"))
    sign=float(os.environ.get("USIGN","-1")); seed=int(os.environ.get("SEED","1"))
    diag=os.environ.get("DIAG","0")=="1"
    rng=np.random.default_rng(seed)
    vc={w:0.10+0.25*rng.standard_normal() for w in WEIGHTS}   # wider spread breaks hidden symmetry
    idx={nd:k for k,nd in enumerate(FREE)}
    tgt=np.array([vl(p[2]) for p in PATS]); beta=1.0/RB
    print(f"XOR2 eta={eta} iters={iters} sign={sign} seed={seed} swing[{LO},{HI}] clamp[{CLO},{CHI}] RB={RB}")
    best=9; bestv=None
    for it in range(iters):
        F,N=run(vc); o=F[:,idx['o']]
        loss=float(np.mean((o-tgt)**2))
        if loss<best: best=loss; bestv=dict(vc)
        dvc={w:0.0 for w in WEIGHTS}
        for i,(a,b,_) in enumerate(PATS):
            x1,x2=vl(a),vl(b); val={'x1':x1,'x2':x2,'x1b':round(1-x1,3),'x2b':round(1-x2,3),
                                     'vhi':1.0,'vlo':0.0}
            def nv(n,arr): return val[n] if n in val else arr[idx[n]]
            for (na,nb,w) in CONDS:
                df=(nv(na,F[i])-nv(nb,F[i]))**2; dn=(nv(na,N[i])-nv(nb,N[i]))**2
                dvc[w]+=sign*eta*((dn-df)/beta)
        for w in WEIGHTS: vc[w]=float(np.clip(vc[w]+dvc[w]/4,VCMIN,VCMAX))
        if it%5==0 or it==iters-1:
            acc=float(np.mean((o>0.5)==(tgt>0.5)))
            msg=f"it{it:3d} loss={loss:.5f} best={best:.5f} acc={acc:.2f} o*={np.round(o,3)}"
            if diag:
                h=F[:,[idx['h1'],idx['h2']]]
                msg+=f"  h1={np.round(F[:,idx['h1']],2)} h2={np.round(F[:,idx['h2']],2)}"
            print(msg)
    print("FINAL best_loss",round(best,5),"o*",np.round(o,3),"tgt",tgt)

if __name__=="__main__": main()
