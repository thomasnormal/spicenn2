#!/usr/bin/env python3
"""
RISK 3: two-phase Equilibrium Propagation training of a toy task, in ngspice.

Network (all reciprocal triode conductances, signed via differential pairs):
  inputs x1,x2 (and complements x1b=1-x1, x2b=1-x2) -> single output node o.
  weight w_i realized as a differential conductance pair (G+_i to x_i, G-_i to x_ib).
  bias realized as a pair to fixed rails vhi(0.6)/vlo(0.4).
  output node has a leak to 0.5 to fix the operating point.

Two phases per pattern (equilibrium found with .op; RISK1 proved the fixed point is unique):
  FREE  : clamp inputs, output o floats -> read o*  and node voltages.
  NUDGE : additionally tie o to target t through a weak conductance beta -> read o^beta.

Local EP update for each conductance k spanning terminals (p,q):
  signal_k = (v_p^beta - v_q^beta)^2 - (v_p* - v_q*)^2
  dVg_k    = -eta * signal_k        (gradient DESCENT on 1/2 (o-t)^2; sign verified empirically)
No transpose, no backward pass: each conductance updates from the squared voltage drop
*across itself* in the two phases. The controller only presents voltages and nudges gates.
"""
import numpy as np, subprocess, os, sys

KP, WL, VT = 50e-6, 2.0, 0.2     # device: G = KP*WL*(Vg-VT-Vsource)
LO, HI = 0.4, 0.6                # logic 0/1 as voltages (small swing -> stays in triode)
RB_NUDGE = 33e3                  # nudge conductance ~ 1/Rb (beta)
RLEAK = 50e3                     # output leak to 0.5
NUDGE_SIGN = +1                  # set t toward target; combined with update sign

# ---- task: OR (linearly separable), targets as voltages ----
PATS = [(0,0,0),(0,1,1),(1,0,1),(1,1,1)]   # (x1,x2,OR)
def vlevel(b): return HI if b else LO

# conductances: name -> (terminalA_node, gate_var).  terminalB is always 'o'.
# weights w1: (1p to x1),(1n to x1b); w2: (2p,2n); bias: (bp to vhi),(bn to vlo)
CONDS = ['1p','1n','2p','2n','bp','bn']
TERM = {'1p':'x1','1n':'x1b','2p':'x2','2n':'x2b','bp':'vhi','bn':'vlo'}

def deck(vg):
    """build ngspice deck for current gate voltages vg (dict name->volts)."""
    L = ["EP training step"]
    L.append(".model NSYN NMOS (LEVEL=1 VTO=0.2 KP=50u LAMBDA=0 GAMMA=0 PHI=0.7)")
    # input sources (altered per pattern in .control)
    for s in ['x1','x2','x1b','x2b']:
        L.append(f"V{s} {s} 0 0.5")
    L.append("Vhi vhi 0 0.6")
    L.append("Vlo vlo 0 0.4")
    L.append("Vt vt 0 0.5")
    # device conductances o<->terminal, gate set by stored weight
    for c in CONDS:
        L.append(f"M{c} o vg{c} {TERM[c]} 0 NSYN W=200u L=100u")
        L.append(f"Vg{c} vg{c} 0 {vg[c]:.5f}")
    L.append(f"Rleak o nref {RLEAK}")
    L.append("Vnref nref 0 0.5")
    L.append(f"Rb o vt 1e12")        # nudge resistor, altered on/off
    # control: 4 patterns x (free, nudge)
    L.append(".control")
    for i,(a,b,_) in enumerate(PATS):
        x1,x2 = vlevel(a),vlevel(b)
        L += [f"alter Vx1 {x1}", f"alter Vx2 {x2}",
              f"alter Vx1b {round(1.0-x1,3)}", f"alter Vx2b {round(1.0-x2,3)}",
              "alter Rb 1e12", "op", f"wrdata f{i}.dat v(o)"]
        t = vlevel(PATS[i][2])
        L += [f"alter Vt {t}", f"alter Rb {RB_NUDGE}", "op", f"wrdata n{i}.dat v(o)"]
    L.append(".endc"); L.append(".end")
    return "\n".join(L)+"\n"

def readval(fn):
    return float(np.loadtxt(fn).reshape(-1)[-1])   # last col = value

def run_step(vg):
    open("ep_step.cir","w").write(deck(vg))
    r = subprocess.run(["ngspice","-b","ep_step.cir"],capture_output=True,text=True,timeout=120)
    free=[readval(f"f{i}.dat") for i in range(4)]
    nud =[readval(f"n{i}.dat") for i in range(4)]
    return np.array(free), np.array(nud)

def main():
    eta = float(os.environ.get("ETA","0.5"))
    iters = int(os.environ.get("ITERS","40"))
    sign = float(os.environ.get("USIGN","-1"))      # update sign (descent); flip if loss rises
    rng = np.random.default_rng(0)
    vg = {c: 1.15+0.05*rng.standard_normal() for c in CONDS}  # init near triode, small spread
    targets = np.array([vlevel(p[2]) for p in PATS])
    print(f"eta={eta} iters={iters} sign={sign} RB={RB_NUDGE} init_vg~1.15")
    hist=[]
    for it in range(iters):
        free,nud = run_step(vg)
        loss = float(np.mean((free-targets)**2))
        hist.append(loss)
        # accumulate local EP update over patterns
        dvg = {c:0.0 for c in CONDS}
        beta = 1.0/RB_NUDGE
        for i,(a,b,_) in enumerate(PATS):
            x1,x2 = vlevel(a),vlevel(b)
            tv = {'x1':x1,'x2':x2,'x1b':round(1-x1,3),'x2b':round(1-x2,3),'vhi':0.6,'vlo':0.4}
            for c in CONDS:
                vp = tv[TERM[c]]
                dfree = (vp-free[i])**2
                dnud  = (vp-nud[i])**2
                sig = (dnud - dfree)/beta              # (1/beta)[ (dv^beta)^2 - (dv*)^2 ]
                dvg[c] += sign*eta*sig
        for c in CONDS:
            vg[c] = float(np.clip(vg[c]+dvg[c]/len(PATS), 0.8, 1.6))
        if it%5==0 or it==iters-1:
            acc = float(np.mean((free>0.5)==(targets>0.5)))
            print(f"it{it:3d} loss={loss:.5f} acc={acc:.2f} o*={np.round(free,3)} "
                  f"on^={np.round(nud,3)}")
    print("final loss",hist[-1],"min loss",min(hist))
    np.savetxt("ep_loss.dat",np.array(hist))
    print("free outputs:",np.round(free,3)," targets:",targets)

if __name__=="__main__":
    main()
