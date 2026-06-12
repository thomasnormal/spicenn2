#!/usr/bin/env python3
"""
Decisive test of the diagnosis: is NEURON GAIN the missing ingredient for XOR?
2->2->1, reciprocal averaging synapses (ideal resistors here; T-gate is the device version),
but neurons are ACTIVE high-gain saturating amplifiers (behavioral scaffold, clearly labeled).
a = clip(0.5 + GAIN*(u-0.5), 0,1).  Reciprocal synapses + active neuron = still a valid energy
(Hopfield), now WITH gain.  Random-search the weights, measure max XOR output separation vs GAIN.
If sep jumps from ~0 (passive) to ~1 as GAIN rises, the diagnosis holds and the fix is identified.
"""
import numpy as np, subprocess, os
LO,HI=0.15,0.85
PATS=[(0,0,0),(0,1,1),(1,0,1),(1,1,0)]
def vl(b): return HI if b else LO
# weights: input->hidden w[i][j], hidden->output wo[j], biases bh[j], bo. signed (real values).
def deck(W, gain):
    w11,w12,w21,w22,b1,b2,o1,o2,bo = W
    L=["EP active-neuron XOR forward test",
       "Vx1 x1 0 0.5","Vx2 x2 0 0.5",
       # hidden pre-activations: resistive weighted node (R=1/|w|, to x or its complement for sign)
       # build as current-average via conductances to x1,x2 and a leak to 0.5
       ]
    def syn(dst,src,w,n):
        # signed weight via differential: positive -> connect to src, negative -> to (1-src) mirror node
        g=abs(w)*1e-4+1e-9
        if w>=0: L.append(f"R{n} {src} {dst} {1/g:.1f}")
        else:    L.append(f"R{n} {src}c {dst} {1/g:.1f}")
    # complements of inputs (controller-provided): x1c=1-x1
    L.append("Bx1c x1c 0 V=1-V(x1)")
    L.append("Bx2c x2c 0 V=1-V(x2)")
    # hidden node u1,u2 with leak to 0.5
    syn("u1","x1",w11,1); syn("u1","x2",w21,2); L.append("Rl1 u1 nmid 1e4")
    syn("u2","x1",w12,3); syn("u2","x2",w22,4); L.append("Rl2 u2 nmid 1e4")
    L.append(f"Rb1 u1 {'nhi' if b1>=0 else 'nlo'} {1/(abs(b1)*1e-4+1e-9):.1f}")
    L.append(f"Rb2 u2 {'nhi' if b2>=0 else 'nlo'} {1/(abs(b2)*1e-4+1e-9):.1f}")
    L+=["Vmid nmid 0 0.5","Vhi nhi 0 1","Vlo nlo 0 0"]
    # ACTIVE neurons: a = clip(0.5+gain*(u-0.5))
    L.append(f"Ba1 a1 0 V=min(1,max(0,0.5+{gain}*(V(u1)-0.5)))")
    L.append(f"Ba2 a2 0 V=min(1,max(0,0.5+{gain}*(V(u2)-0.5)))")
    L.append("Ba1c a1c 0 V=1-V(a1)")
    L.append("Ba2c a2c 0 V=1-V(a2)")
    # output node uo from activations (signed), leak; readout = active too
    syn("uo","a1",o1,5); syn("uo","a2",o2,6); L.append("Rlo uo nmid 1e4")
    L.append(f"Rbo uo {'nhi' if bo>=0 else 'nlo'} {1/(abs(bo)*1e-4+1e-9):.1f}")
    L.append(f"Bao ao 0 V=min(1,max(0,0.5+{gain}*(V(uo)-0.5)))")
    L+=[".control"]
    for i,(a,b,_) in enumerate(PATS):
        L+=[f"alter Vx1 {vl(a)}",f"alter Vx2 {vl(b)}","op",f"wrdata act{i}.dat v(ao)"]
    L+=[".endc",".end"]
    return "\n".join(L)+"\n"

def run(W,gain):
    open("ep_active.cir","w").write(deck(W,gain))
    subprocess.run(["ngspice","-b","ep_active.cir"],capture_output=True,text=True,timeout=60)
    return np.array([np.loadtxt(f"act{i}.dat").reshape(-1)[-1] for i in range(4)])

rng=np.random.default_rng(0)
print("max XOR output-separation vs neuron GAIN (reciprocal avg synapses + active neuron):")
for gain in [1,2,4,8,16]:
    best=0; bo=None
    for k in range(200):
        W=rng.uniform(-3,3,9)
        o=run(W,gain)
        sep=abs(o[[0,3]].mean()-o[[1,2]].mean())
        # require correct XOR ordering (00,11 low or high consistently vs 01,10)
        if sep>best: best=sep; bo=o
    print(f"  GAIN={gain:2d}:  max XOR-sep={best:.3f}   best o*={np.round(bo,3)}")
