#!/usr/bin/env python3
"""
FULL transistor EP training of XOR in ngspice.
Recipe validated this session: reciprocal T-gate synapses + active high-gain neurons.

Neurons (active gain): u -> a=inv(u) -> ac=inv(a)   (low-Vt inverters, gain~14, drive weak synapses)
Synapses: CMOS transmission gates (reciprocal at full swing), weak (~few uS), signed via
          differential activations (a / ac). Hidden<->output coupling is BIDIRECTIONAL and
          weight-shared (symmetric) so the output nudge propagates back to hidden = the EP signal.
Phases: free (.op) and nudged (a_o tied to target via weak beta). Local update per weight:
        dVc ~ sign*eta*(1/beta)*sum_over_its_Tgates[ (dv^nudge)^2 - (dv^free)^2 ].
Read a_o as the output.
"""
import numpy as np, subprocess, os

NHID=int(os.environ.get("NHID","2"))        # hidden width (width buys trainability)
NEURONS=[f"h{j}" for j in range(1,NHID+1)]+['o']
INP=['a_x1','a_x1c','a_x2','a_x2c']
def build_syn():
    S=[]
    for j in range(1,NHID+1):
        # input -> hidden hj (signed via differential inputs)
        for i in (1,2):
            S+=[(f"a_x{i}",f"u_h{j}",f"i{i}{j}p"),(f"a_x{i}c",f"u_h{j}",f"i{i}{j}n")]
        # hidden hj -> output forward, and output -> hj backward (SHARED weight)
        S+=[(f"a_h{j}","u_o",f"o{j}p"),(f"ac_h{j}","u_o",f"o{j}n")]
        S+=[(f"a_o",f"u_h{j}",f"o{j}p"),(f"ac_o",f"u_h{j}",f"o{j}n")]
        # hidden bias
        S+=[("vhi",f"u_h{j}",f"bh{j}p"),("vlo",f"u_h{j}",f"bh{j}n")]
    S+=[("vhi","u_o","bop"),("vlo","u_o","bon")]
    return S
SYN=build_syn()
WEIGHTS=sorted(set(w for _,_,w in SYN))
READ=sum(([f"u_h{j}",f"a_h{j}",f"ac_h{j}"] for j in range(1,NHID+1)),[])+['u_o','a_o','ac_o']
VCMIN,VCMAX=-0.30,0.55
PATS=[(0,0,0),(0,1,1),(1,0,1),(1,1,0)]
LO,HI=0.15,0.85
def vl(b): return HI if b else LO          # INPUT activation levels
OUT='a_o'                                   # readout + nudge = output activation (full swing, finite Zout)
UTLO,UTHI=float(os.environ.get("UTLO","0.20")),float(os.environ.get("UTHI","0.80"))
def tvl(b): return UTHI if b else UTLO      # TARGET levels (a_o space, full swing)
RB=float(os.environ.get("RB","1e6"))        # nudge resistor (beta=1/RB)
BEFF=1.0/RB
ROUT=os.environ.get("ROUT","30k")           # output-neuron finite output impedance (nudgeable)
OGAIN=float(os.environ.get("OGAIN","4"))    # output-neuron gain
WN=os.environ.get("WN","10u"); WP=os.environ.get("WP","12.5u")   # weak T-gate

GAIN=float(os.environ.get("GAIN","6"))
IDEAL=os.environ.get("IDEAL","0")=="1"
RLK=os.environ.get("RLK","200k")            # neuron leak (stronger leak -> more contractive/stable)
def neuron(n):
    L=[]
    u,a,ac=f"u_{n}",f"a_{n}",f"ac_{n}"
    if IDEAL:
        # ideal monotonic-INCREASING saturating Hopfield amplifier (diagnostic scaffold; g'>0).
        L.append(f"Rlk_{n} {u} nmid {RLK}")
        if n=="o":
            # output neuron: finite output impedance ROUT so a conductance nudge can move a_o.
            L.append(f"Baoi_{n} aoi 0 V=0.5+0.5*tanh({OGAIN}*(V({u})-0.5))")
            L.append(f"Rout_{n} aoi {a} {ROUT}")
            L.append(f"Bac_{n} {ac} 0 V=1-V({a})")
        else:
            L.append(f"Ba_{n} {a} 0 V=0.5+0.5*tanh({GAIN}*(V({u})-0.5))")
            L.append(f"Bac_{n} {ac} 0 V=0.5-0.5*tanh({GAIN}*(V({u})-0.5))")
        return L
    # transistor neuron: a=inv(u) (decreasing!), ac=inv(a)
    L.append(f"Mn1_{n} {a} {u} 0 0 NNR W=100u L=100u")
    L.append(f"Mp1_{n} {a} {u} vdd vdd PNR W=300u L=100u")
    L.append(f"Mn2_{n} {ac} {a} 0 0 NNR W=100u L=100u")
    L.append(f"Mp2_{n} {ac} {a} vdd vdd PNR W=300u L=100u")
    L.append(f"Rlk_{n} {u} nmid 200k")           # leak / operating point
    nload=os.environ.get("NLOAD","")              # optional activation load -> lowers neuron gain
    if nload:
        L.append(f"Rla_{n} {a} nmid {nload}")
        L.append(f"Rlac_{n} {ac} nmid {nload}")
    return L

def deck(vc):
    L=["FULL transistor EP XOR",
       ".model NSYN NMOS (LEVEL=1 VTO=0.2  KP=50u  LAMBDA=0    GAMMA=0 PHI=0.7)",
       ".model PSYN PMOS (LEVEL=1 VTO=-0.2 KP=40u  LAMBDA=0    GAMMA=0 PHI=0.7)",
       ".model NNR  NMOS (LEVEL=1 VTO=0.2  KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)",
       ".model PNR  PMOS (LEVEL=1 VTO=-0.2 KP=40u  LAMBDA=0.02 GAMMA=0 PHI=0.7)",
       "Vdd vdd 0 1.0","Vmid nmid 0 0.5","Vhi vhi 0 1.0","Vlo vlo 0 0.0","Vt vt 0 0.5"]
    # clamped input activations
    for s in INP: L.append(f"V{s} {s} 0 0.5")
    # neurons
    for n in NEURONS: L+=neuron(n)
    # gate sources per weight
    for w in WEIGHTS:
        v=vc[w]; L.append(f"Vgn_{w} gn_{w} 0 {1.0+v:.5f}"); L.append(f"Vgp_{w} gp_{w} 0 {-v:.5f}")
    # synapse T-gates
    for k,(a,b,w) in enumerate(SYN):
        L.append(f"Mn{k} {a} gn_{w} {b} 0   NSYN W={WN} L=100u")
        L.append(f"Mp{k} {a} gp_{w} {b} vdd PSYN W={WP} L=100u")
    # EP nudge: conductance from output state u_o toward u-space target (standard EP).
    L.append(f"Rb {OUT} vt 1e12")
    # centered nudging support: a second nudge resistor to a reflected target (idea 5)
    L.append(f"Rb2 {OUT} vt2 1e12")
    L.append("Vt2 vt2 0 0.5")
    L.append(".control")
    cols=" ".join(f"v({n})" for n in READ)
    for i,(p,q,_) in enumerate(PATS):
        x1,x2=vl(p),vl(q)
        L+=[f"alter Va_x1 {x1}",f"alter Va_x1c {round(1-x1,3)}",
            f"alter Va_x2 {x2}",f"alter Va_x2c {round(1-x2,3)}",
            "alter Rb 1e12","alter Rb2 1e12","op",f"wrdata tf{i}.dat {cols}"]
        t=tvl(PATS[i][2])
        L+=[f"alter Vt {t}",f"alter Rb {RB}","op",f"wrdata tn{i}.dat {cols}"]
    L+=[".endc",".end"]
    return "\n".join(L)+"\n"

def rd(fn): return np.loadtxt(fn).reshape(-1)[1::2]
def run(vc):
    open("ep_xor_train.cir","w").write(deck(vc))
    subprocess.run(["ngspice","-b","ep_xor_train.cir"],capture_output=True,text=True,timeout=180)
    F=np.array([rd(f"tf{i}.dat") for i in range(4)])
    N=np.array([rd(f"tn{i}.dat") for i in range(4)])
    return F,N

def main():
    import sys
    mode=sys.argv[1] if len(sys.argv)>1 else "train"
    idx={n:k for k,n in enumerate(READ)}
    oi=idx[OUT]
    tgt=np.array([tvl(p[2]) for p in PATS])
    rng=np.random.default_rng(int(os.environ.get("SEED","0")))
    if mode=="search":
        best=0
        for k in range(int(os.environ.get("N","200"))):
            vc={w:float(np.clip(rng.uniform(VCMIN,VCMAX),VCMIN,VCMAX)) for w in WEIGHTS}
            F,_=run(vc); o=F[:,oi]
            sep=abs(o[[0,3]].mean()-o[[1,2]].mean())
            if sep>best:
                best=sep; print(f"[{k}] XOR-sep={sep:.3f} a_o={np.round(o,3)}")
        print("MAX XOR-sep:",round(best,3)); return
    # train
    eta=float(os.environ.get("ETA","0.2")); iters=int(os.environ.get("ITERS","60"))
    sign=float(os.environ.get("USIGN","-1"))
    mom=float(os.environ.get("MOM","0.0")); avgk=int(os.environ.get("AVGK","20"))
    vc={w:0.10+0.20*rng.standard_normal() for w in WEIGHTS}
    vel={w:0.0 for w in WEIGHTS}
    beta=BEFF; best=9; hist=[]; avgbuf=[]
    print(f"TRAIN eta={eta} iters={iters} sign={sign} BEFF={BEFF} mom={mom} seed={os.environ.get('SEED','0')}")
    for it in range(iters):
        F,N=run(vc); o=F[:,oi]
        loss=float(np.mean((o-tgt)**2)); best=min(best,loss); hist.append(loss)
        dvc={w:0.0 for w in WEIGHTS}
        for i,(p,q,_) in enumerate(PATS):
            x1,x2=vl(p),vl(q)
            val={'a_x1':x1,'a_x1c':round(1-x1,3),'a_x2':x2,'a_x2c':round(1-x2,3),'vhi':1.0,'vlo':0.0}
            def nv(node,arr): return val[node] if node in val else arr[idx[node]]
            for (na,nb,w) in SYN:
                df=(nv(na,F[i])-nv(nb,F[i]))**2; dn=(nv(na,N[i])-nv(nb,N[i]))**2
                dvc[w]+=sign*eta*((dn-df)/beta)
        for w in WEIGHTS:
            vel[w]=mom*vel[w]+dvc[w]/4
            vc[w]=float(np.clip(vc[w]+vel[w],VCMIN,VCMAX))
        if it>=iters-avgk: avgbuf.append(dict(vc))
        if it%5==0 or it==iters-1:
            acc=float(np.mean((o>0.5)==(tgt>0.5)))
            print(f"it{it:3d} loss={loss:.5f} best={best:.5f} acc={acc:.2f} a_o={np.round(o,3)} tgt={tgt}")
    # late-weight average evaluation (fixes limit-cycle oscillation)
    vavg={w:float(np.mean([d[w] for d in avgbuf])) for w in WEIGHTS}
    Fa,_=run(vavg); oa=Fa[:,oi]
    la=float(np.mean((oa-tgt)**2)); acca=float(np.mean((oa>0.5)==(tgt>0.5)))
    print(f"FINAL best_single={round(best,5)}  LATE-AVG loss={la:.5f} acc={acca:.2f} a_o={np.round(oa,3)} tgt={tgt}")
    np.savetxt("ep_xor_loss.dat",np.array(hist))
    import json; json.dump(vavg,open("ep_xor_w.json","w"))

if __name__=="__main__": main()
