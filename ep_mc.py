#!/usr/bin/env python3
"""
MULTI-CLASS fully-differential current-mode EP, flexible in N (inputs), C (classes), width.
Task: label = (sum of input bits) mod C   [C=3 -> sum mod 3, the multi-class cousin of parity].

Reuses the validated differential cells (crossed-diff-pair synapse dsyn, diff-pair neuron,
diode-PMOS loads).  Generalizations vs ep_dtcm.py:
  - N differential inputs (a_xp_i, a_xn_i), i=0..N-1
  - C one-hot differential OUTPUT node-pairs (up_o_c, un_o_c); prediction = argmax_c (up-un)
  - per-output nudge toward a one-hot target (+TD for the true class, -TD otherwise)
  - per-node diode load sized to FAN-IN (W ~ WPER*fanin) so the common mode holds as N/width grow
Everything in the circuit is MOSFETs/resistors (0 behavioral). Only the local correlation
weight-update runs on the controller.
"""
import numpy as np, subprocess, os, itertools

N    = int(os.environ.get("N","3"))           # input bits
C    = int(os.environ.get("C","3"))           # classes
NHID = int(os.environ.get("NHID","8"))        # hidden width
VBN  = os.environ.get("VBN","0.30")
RB   = float(os.environ.get("RB","2e5")); BETA=1.0/RB
VW0  = float(os.environ.get("VW0","0.42")); KMAP=float(os.environ.get("KMAP","0.10"))
WPER = float(os.environ.get("WPER","300"))    # diode-load width (um) per unit fan-in
RLNEU= os.environ.get("RLNEU","800k")
IND  = float(os.environ.get("IND","0.3"))     # input differential half-swing
TD   = float(os.environ.get("TD","0.10"))     # one-hot target differential half-swing
MAXP = int(os.environ.get("MAXP","64"))       # cap patterns (subsample big truth tables)
TAG  = os.environ.get("RUNTAG",str(os.getpid()))  # per-process file tag (no concurrent collisions)

# ---------- dataset: label = sum(bits) mod C ----------
def make_data():
    pats=list(itertools.product([0,1],repeat=N))
    if len(pats)>MAXP:
        idx=np.random.default_rng(0).choice(len(pats),MAXP,replace=False)
        pats=[pats[i] for i in idx]
    labels=[sum(p)%C for p in pats]
    return pats,labels
PATS,LABELS=make_data(); P=len(PATS)

# ---------- weights ----------
WEIGHTS=[]
for j in range(NHID):
    for i in range(N): WEIGHTS.append(f"wih_{i}_{j}")
    for c in range(C): WEIGHTS.append(f"who_{j}_{c}")
    WEIGHTS.append(f"bh_{j}")
for c in range(C): WEIGHTS.append(f"bo_{c}")
READ=[]
for j in range(NHID): READ+=[f"ap_h{j}",f"an_h{j}"]
for c in range(C):    READ+=[f"up_o{c}",f"un_o{c}"]
idx={n:k for k,n in enumerate(READ)}
HID_FANIN=N+C+1          # inputs + backward-from-C-outputs + bias
OUT_FANIN=NHID+1         # forward-from-hidden + bias

SUB=""".subckt dsyn inp inn outp outn wp wn vdd
M1 outp inp tp 0 NNR W=200u L=100u
M2 outn inn tp 0 NNR W=200u L=100u
Mtp tp wp 0 0 NNR W=200u L=100u
M3 outn inp tn 0 NNR W=200u L=100u
M4 outp inn tn 0 NNR W=200u L=100u
Mtn tn wn 0 0 NNR W=200u L=100u
.ends
.subckt dneuron up un ap an vdd vbn
M1 an up tn 0 NNR W=200u L=100u
M2 ap un tn 0 NNR W=200u L=100u
Mt tn vbn 0 0 NNR W=200u L=100u
RLa vdd ap {RLNEU}
RLb vdd an {RLNEU}
.ends
""".replace("{RLNEU}",RLNEU)

def wg(w): return f"{np.clip(VW0+KMAP*w,0.3,0.9):.4f}",f"{np.clip(VW0-KMAP*w,0.3,0.9):.4f}"
def syn(tag,ip,inn,op,on,key,w):
    gp,gn=wg(w[key])
    return [f"Vwp_{tag} wp_{tag} 0 {gp}",f"Vwn_{tag} wn_{tag} 0 {gn}",
            f"X_{tag} {ip} {inn} {op} {on} wp_{tag} wn_{tag} vdd dsyn"]

def deck(w):
    Wh=f"{WPER*HID_FANIN:.0f}u"; Wo=f"{WPER*OUT_FANIN:.0f}u"
    L=["multi-class differential current-mode EP",
       ".model NNR NMOS (LEVEL=1 VTO=0.2  KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)",
       ".model PNR PMOS (LEVEL=1 VTO=-0.2 KP=40u  LAMBDA=0.02 GAMMA=0 PHI=0.7)",
       SUB,"Vdd vdd 0 1.0","Vbn vbn 0 "+VBN,"Vbp_hi bp_hi 0 0.8","Vbp_lo bp_lo 0 0.2"]
    for i in range(N): L+=[f"Vaxp{i} a_xp{i} 0 0.5",f"Vaxn{i} a_xn{i} 0 0.5"]
    for c in range(C): L+=[f"Vtp{c} vtp{c} 0 0.5",f"Vtn{c} vtn{c} 0 0.5"]
    # hidden layer
    for j in range(NHID):
        up,un=f"up_h{j}",f"un_h{j}"
        L+=[f"Mlp_h{j} {up} {up} vdd vdd PNR W={Wh} L=100u",
            f"Mln_h{j} {un} {un} vdd vdd PNR W={Wh} L=100u",
            f"Xn_h{j} {up} {un} ap_h{j} an_h{j} vdd vbn dneuron"]
        for i in range(N): L+=syn(f"ih_{i}_{j}",f"a_xp{i}",f"a_xn{i}",up,un,f"wih_{i}_{j}",w)
        for c in range(C): L+=syn(f"bk_{j}_{c}",f"up_o{c}",f"un_o{c}",up,un,f"who_{j}_{c}",w)
        L+=syn(f"bh_{j}","bp_hi","bp_lo",up,un,f"bh_{j}",w)
    # output layer (linear, one-hot)
    for c in range(C):
        up,un=f"up_o{c}",f"un_o{c}"
        L+=[f"Mlp_o{c} {up} {up} vdd vdd PNR W={Wo} L=100u",
            f"Mln_o{c} {un} {un} vdd vdd PNR W={Wo} L=100u"]
        for j in range(NHID): L+=syn(f"fo_{j}_{c}",f"ap_h{j}",f"an_h{j}",up,un,f"who_{j}_{c}",w)
        L+=syn(f"bo_{c}","bp_hi","bp_lo",up,un,f"bo_{c}",w)
        L+=[f"Rbp{c} {up} vtp{c} 1e10",f"Rbn{c} {un} vtn{c} 1e10"]
    # nodeset: state/activation nodes sit near 0.5 -> fast .op convergence (skip gmin stepping)
    ns=[]
    for j in range(NHID): ns+=[f"v(up_h{j})=0.5",f"v(un_h{j})=0.5",f"v(ap_h{j})=0.5",f"v(an_h{j})=0.5"]
    for c in range(C): ns+=[f"v(up_o{c})=0.5",f"v(un_o{c})=0.5"]
    L.append(".nodeset "+" ".join(ns))
    L.append(".control"); cols=" ".join(f"v({n})" for n in READ)
    for k,(pat,lab) in enumerate(zip(PATS,LABELS)):
        A=[]
        for i in range(N):
            s=1 if pat[i] else -1
            A+=[f"alter Vaxp{i} {0.5+IND*s}",f"alter Vaxn{i} {0.5-IND*s}"]
        A+=[f"alter Rbp{c} 1e10" for c in range(C)]+[f"alter Rbn{c} 1e10" for c in range(C)]
        A+=["op",f"wrdata mf{TAG}_{k}.dat {cols}"]
        for c in range(C):
            td=TD*(1 if c==lab else -1)
            A+=[f"alter Vtp{c} {0.5+td}",f"alter Vtn{c} {0.5-td}"]
        A+=[f"alter Rbp{c} {RB}" for c in range(C)]+[f"alter Rbn{c} {RB}" for c in range(C)]
        A+=["op",f"wrdata mn{TAG}_{k}.dat {cols}"]
        L+=A
    L+=[".endc",".end"]; return "\n".join(L)+"\n"

def rd(fn): return np.loadtxt(fn).reshape(-1)[1::2]
def run(w):
    open(f"ep_mc_{TAG}.cir","w").write(deck(w))
    subprocess.run(["ngspice","-b",f"ep_mc_{TAG}.cir"],capture_output=True,text=True,timeout=300)
    return (np.array([rd(f"mf{TAG}_{k}.dat") for k in range(P)]),
            np.array([rd(f"mn{TAG}_{k}.dat") for k in range(P)]))
def douts(arr): return np.array([arr[idx[f'up_o{c}']]-arr[idx[f'un_o{c}']] for c in range(C)])

def phi(tok,arr,pat):
    t,a=tok
    if t=="x": return 2*IND*(1 if pat[a] else -1)
    if t=="B": return 0.8-0.2
    if t=="h": return arr[idx[f"ap_h{a}"]]-arr[idx[f"an_h{a}"]]
    if t=="o": return arr[idx[f"up_o{a}"]]-arr[idx[f"un_o{a}"]]
def corr_pairs():
    Pd={k:[] for k in WEIGHTS}
    for j in range(NHID):
        for i in range(N): Pd[f"wih_{i}_{j}"].append((("x",i),("h",j)))
        for c in range(C): Pd[f"who_{j}_{c}"].append((("h",j),("o",c)))
        Pd[f"bh_{j}"].append((("B",0),("h",j)))
    for c in range(C): Pd[f"bo_{c}"].append((("B",0),("o",c)))
    return Pd
CP=corr_pairs()
def ep_grad(w):
    F,Nu=run(w); g={k:0.0 for k in WEIGHTS}
    for k in range(P):
        for key in WEIGHTS:
            for (s,d) in CP[key]:
                f=phi(s,F[k],PATS[k])*phi(d,F[k],PATS[k])
                n=phi(s,Nu[k],PATS[k])*phi(d,Nu[k],PATS[k])
                g[key]+=((n-f)/BETA)/P
    return g,F

def onehot(lab): return np.array([TD*(1 if c==lab else -1) for c in range(C)])

def main():
    import sys; mode=sys.argv[1] if len(sys.argv)>1 else "train"
    rng=np.random.default_rng(int(os.environ.get("SEED","0")))
    T=np.array([onehot(l) for l in LABELS])      # (P,C) one-hot targets
    print(f"# N={N} C={C} NHID={NHID} P={P} class-counts={[LABELS.count(c) for c in range(C)]} "
          f"hid_fanin={HID_FANIN} out_fanin={OUT_FANIN}")
    if mode=="fwd":
        w={k:float(0.5*rng.standard_normal()) for k in WEIGHTS}
        F,_=run(w); D=np.array([douts(F[k]) for k in range(P)])
        acc=np.mean(np.argmax(D,1)==np.array(LABELS))
        print("acc(random)=",round(float(acc),3),"  example douts(mV):",np.round(D[0]*1e3,1)); return
    if mode=="gradcheck":
        d=0.05; coss=[]
        for t in range(int(os.environ.get("GT","4"))):
            w={k:float(0.5*rng.standard_normal()) for k in WEIGHTS}; gfd={}
            for k in WEIGHTS:
                wp=dict(w);wp[k]+=d; wm=dict(w);wm[k]-=d
                Fp,_=run(wp); Fm,_=run(wm)
                lp=np.mean([(douts(Fp[i])-T[i])**2 for i in range(P)])
                lm=np.mean([(douts(Fm[i])-T[i])**2 for i in range(P)])
                gfd[k]=(lp-lm)/(2*d)
            gep,_=ep_grad(w); a=np.array([gep[k] for k in WEIGHTS]); b=np.array([gfd[k] for k in WEIGHTS])
            coss.append(float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-18)))
        print(f"MEANCOS {np.mean(coss):+.3f} (min {min(coss):+.3f})"); return
    # train
    eta=float(os.environ.get("ETA","0.006")); iters=int(os.environ.get("ITERS","250"))
    mom=float(os.environ.get("MOM","0.3")); sign=float(os.environ.get("USIGN","-1"))
    w={k:float(0.4*rng.standard_normal()) for k in WEIGHTS}; vel={k:0.0 for k in WEIGHTS}
    best=0; hist=[]
    for it in range(iters):
        g,F=ep_grad(w); D=np.array([douts(F[k]) for k in range(P)])
        acc=float(np.mean(np.argmax(D,1)==np.array(LABELS)))
        loss=float(np.mean((D-T)**2)); hist.append(acc); best=max(best,acc)
        for k in WEIGHTS: vel[k]=mom*vel[k]+sign*eta*g[k]; w[k]=float(np.clip(w[k]+vel[k],-3,3))
        if it%25==0 or it==iters-1:
            print(f"it{it:3d} loss={loss:.4f} acc={acc:.3f} best={best:.3f}")
    print(f"FINAL acc={acc:.3f} best={best:.3f} (chance={1.0/C:.2f})")
    np.savetxt("ep_mc_acc.dat",np.array(hist))

if __name__=="__main__": main()
