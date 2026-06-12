#!/usr/bin/env python3
"""
CURRENT-MODE Equilibrium Propagation (the standard, scalable Hopfield form) in ngspice.
Idea 1: synapses inject CURRENT w_ij*phi_j into node u_i (not conductance) -> the node
integrates a true weighted SUM (no averaging/contraction), and the leak sets the gain.
Symmetric coupling (forward w_ij*phi_j into u_i AND backward w_ij*phi_i into u_j, shared w)
=> valid Hopfield energy WITH gain.  phi = a-0.5 (centered activation).

Neurons here are ideal behavioral amplifiers (scaffold) to validate the METHOD; the synapse
is an ideal VCCS (behavioral). Both have direct transistor realizations (transconductor +
inverter-amp) — built next once the method trains. Signed weights are plain scalars (a VCCS
sources or sinks), so NO differential complements are needed (simpler + scales).

Phases: free (.op) and nudged (output pulled toward target). Local update per weight:
   correlation:  dW ~ (1/beta)[ phi_i^b phi_j^b - phi_i* phi_j* ]   (Hopfield/EP standard)
"""
import numpy as np, subprocess, os
NHID=int(os.environ.get("NHID","2"))
GAIN=float(os.environ.get("GAIN","12"))         # hidden neuron gain
RL=os.environ.get("RL","1")                     # leak resistance (gain ~ 1/G_L)
RB=float(os.environ.get("RB","5"))              # nudge resistor (beta=1/RB)
BETA=1.0/RB
ALO,AHI=0.1,0.9                                 # input activation levels
TLO,THI=float(os.environ.get("TLO","0.15")),float(os.environ.get("THI","0.85"))
PATS=[(0,0,0),(0,1,1),(1,0,1),(1,1,0)]
def xv(b): return AHI if b else ALO
def tv(b): return THI if b else TLO
HID=[f"h{j}" for j in range(1,NHID+1)]

# weights: input i -> hidden j : w_i_j ; hidden j -> output : o_j ; biases bh_j, bo
def wkeys():
    K=[]
    for j in range(1,NHID+1):
        for i in (1,2): K.append(f"w{i}{j}")
        K.append(f"o{j}"); K.append(f"bh{j}")
    K.append("bo"); return K
WEIGHTS=wkeys()
READ=sum(([f"u_h{j}",f"a_h{j}"] for j in range(1,NHID+1)),[])+["u_o"]

def deck(w):
    L=["current-mode EP XOR","Vmid nmid 0 0.5","Vt vt 0 0.5"]
    for s in ("a_x1","a_x2"): L.append(f"V{s} {s} 0 0.5")
    # hidden nodes: leak + current-mode synapses (input fwd + output backward + bias)
    for j in range(1,NHID+1):
        u=f"u_h{j}"
        terms=[f"({w[f'w{i}{j}']:.5f})*(V(a_x{i})-0.5)" for i in (1,2)]
        terms.append(f"({w[f'o{j}']:.5f})*(V(u_o)-0.5)")     # backward from output (shared o_j)
        terms.append(f"({w[f'bh{j}']:.5f})*0.4")             # bias
        L.append(f"Blk_{j} 0 {u} I="+"+".join(terms))
        L.append(f"Rl_{j} {u} nmid {RL}")
        L.append(f"Ba_{j} a_h{j} 0 V=0.5+0.5*tanh({GAIN}*(V({u})-0.5))")
    # output node: linear (a_o=u_o); leak + forward from hidden + bias
    terms=[f"({w[f'o{j}']:.5f})*(V(a_h{j})-0.5)" for j in range(1,NHID+1)]
    terms.append(f"({w['bo']:.5f})*0.4")
    L.append("Blk_o 0 u_o I="+"+".join(terms))
    L.append(f"Rl_o u_o nmid {RL}")
    # nudge: conductance from u_o to target (free: off)
    L.append("Rb u_o vt 1e9")
    L.append(".control")
    cols=" ".join(f"v({n})" for n in READ)
    for i,(p,q,_) in enumerate(PATS):
        L+=[f"alter Va_x1 {xv(p)}",f"alter Va_x2 {xv(q)}",
            "alter Rb 1e9","op",f"wrdata cf{i}.dat {cols}"]
        L+=[f"alter Vt {tv(PATS[i][2])}",f"alter Rb {RB}","op",f"wrdata cn{i}.dat {cols}"]
    L+=[".endc",".end"]
    return "\n".join(L)+"\n"

def rd(fn): return np.loadtxt(fn).reshape(-1)[1::2]
def run(w):
    open("ep_cm.cir","w").write(deck(w))
    subprocess.run(["ngspice","-b","ep_cm.cir"],capture_output=True,text=True,timeout=120)
    F=np.array([rd(f"cf{i}.dat") for i in range(4)])
    N=np.array([rd(f"cn{i}.dat") for i in range(4)])
    return F,N
idx={n:k for k,n in enumerate(READ)}; oi=idx['u_o']

def synpairs():
    """(src_phi_node, dst_neuron) pairs per weight for the correlation update."""
    P={k:[] for k in WEIGHTS}
    for j in range(1,NHID+1):
        for i in (1,2): P[f"w{i}{j}"].append((f"a_x{i}",f"h{j}"))
        P[f"o{j}"].append((f"a_h{j}","o")); P[f"o{j}"].append((f"u_o",f"h{j}"))  # fwd+backward
        P[f"bh{j}"].append(("BIAS",f"h{j}"))
    P["bo"].append(("BIAS","o"))
    return P
SP=synpairs()

def phi(node,arr,pat):
    if node=="BIAS": return 0.4
    if node=="a_x1": return xv(pat[0])-0.5
    if node=="a_x2": return xv(pat[1])-0.5
    if node=="u_o": return arr[idx['u_o']]-0.5
    if node.startswith("a_h"): return arr[idx[node]]-0.5
    return arr[idx[node]]-0.5

def ep_grad(w):
    F,N=run(w); g={k:0.0 for k in WEIGHTS}
    for i,pat in enumerate(PATS):
        for k in WEIGHTS:
            for (src,dst) in SP[k]:
                da=f"a_{dst}" if dst!="o" else "u_o"
                f=phi(src,F[i],pat)*phi(da,F[i],pat); n=phi(src,N[i],pat)*phi(da,N[i],pat)
                g[k]+=((n-f)/BETA)/4.0
    return g,F

def main():
    import sys; mode=sys.argv[1] if len(sys.argv)>1 else "train"
    rng=np.random.default_rng(int(os.environ.get("SEED","0")))
    tgt=np.array([tv(p[2]) for p in PATS])
    if mode=="gradcheck":
        delta=0.01; coss=[]
        for t in range(int(os.environ.get("T","5"))):
            w={k:float(0.3*rng.standard_normal()) for k in WEIGHTS}
            gfd={}
            for k in WEIGHTS:
                wp=dict(w); wp[k]+=delta; wm=dict(w); wm[k]-=delta
                Fp,_=run(wp); Fm,_=run(wm)
                lp=np.mean((Fp[:,oi]-tgt)**2); lm=np.mean((Fm[:,oi]-tgt)**2)
                gfd[k]=(lp-lm)/(2*delta)
            gep,_=ep_grad(w)
            a=np.array([gep[k] for k in WEIGHTS]); b=np.array([gfd[k] for k in WEIGHTS])
            cos=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-18)); coss.append(cos)
            print(f"trial{t}: cos(corr,FD)={cos:+.3f}")
        print(f"MEANCOS {np.mean(coss):+.3f} (min {min(coss):+.3f})"); return
    if mode=="search":
        best=0
        for k in range(int(os.environ.get("N","200"))):
            w={kk:float(rng.uniform(-2,2)) for kk in WEIGHTS}
            F,_=run(w); o=F[:,oi]
            sep=abs(o[[0,3]].mean()-o[[1,2]].mean())
            if sep>best: best=sep; print(f"[{k}] sep={sep:.3f} u_o={np.round(o,3)}")
        print("MAX sep:",round(best,3)); return
    # train
    eta=float(os.environ.get("ETA","0.05")); iters=int(os.environ.get("ITERS","200"))
    mom=float(os.environ.get("MOM","0.8")); sign=float(os.environ.get("USIGN","-1"))
    w={k:float(0.3*rng.standard_normal()) for k in WEIGHTS}; vel={k:0.0 for k in WEIGHTS}
    best=9; hist=[]
    print(f"CM-TRAIN NHID={NHID} GAIN={GAIN} eta={eta} mom={mom} beta={BETA} seed={os.environ.get('SEED','0')}")
    for it in range(iters):
        g,F=ep_grad(w); o=F[:,oi]
        loss=float(np.mean((o-tgt)**2)); best=min(best,loss); hist.append(loss)
        for k in WEIGHTS:
            vel[k]=mom*vel[k]+sign*eta*g[k]; w[k]=float(np.clip(w[k]+vel[k],-3,3))
        if it%10==0 or it==iters-1:
            acc=float(np.mean((o>0.5)==(tgt>0.5)))
            print(f"it{it:3d} loss={loss:.5f} best={best:.5f} acc={acc:.2f} u_o={np.round(o,3)} tgt={tgt}")
    print(f"FINAL best={round(best,5)} u_o={np.round(o,3)} tgt={tgt}")
    np.savetxt("ep_cm_loss.dat",np.array(hist)); import json; json.dump(w,open("ep_cm_w.json","w"))

if __name__=="__main__": main()
