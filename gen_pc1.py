#!/usr/bin/env python3
"""SINGLE-LAYER fully-analog continuous Predictive Coding, in ONE ngspice .tran.
Proves every non-negotiable of the goal on a linearly-separable task (here: OR / AND):
  - weights are CHARGE ON CAPS (wp_i,wn_i); the synapse reads them, a product cell charges them.
  - learning is PHYSICS: each weight cap integrates its LOCAL product  eps_o . x_in_i  continuously
    (no two-phase, no controller, no optimizer) -- this is the PC weight rule dW/dt ~ eps . presyn.
  - hard-clamped output: x_o pinned to the label, eps_o = x_o - mu_o is an explicit node.
The controller only writes the PWL schedule (inputs + label), cycling the 4 patterns.
Output mu_o = W.x_in (the prediction) is read per pattern to check it learns the target.
"""
import os, numpy as np, subprocess
TASK=os.environ.get("TASK","or")     # or | and
KMAP=float(os.environ.get("KMAP","0.30")); VW0=float(os.environ.get("VW0","0.5"))
IND=float(os.environ.get("IND","0.30")); TD=float(os.environ.get("TD","0.25"))
VBSYN=os.environ.get("VBSYN","0.6"); WL=os.environ.get("WL","1000u"); RCS=os.environ.get("RCS","300k")
GBL=os.environ.get("GBL","0.9")      # learning product-cell tail (learning-rate)
CWW=os.environ.get("CWW","30p")      # weight cap
RWL=os.environ.get("RWL","2g")       # weak weight-CM hold to VW0
RMO=os.environ.get("RMO","40k"); SGN=float(os.environ.get("SGN","1"))
TH=float(os.environ.get("TH","300")); NEP=int(os.environ.get("NEP","120"))  # ns/pattern, #epochs
TAG=os.environ.get("RUNTAG",str(os.getpid()))
PATS=[(0,0),(0,1),(1,0),(1,1)]
def tgt(p): return (p[0] or p[1]) if TASK=="or" else (p[0] and p[1])
def ts(b): return 1.0 if b else -1.0
SUB=f""".model NNR NMOS (LEVEL=1 VTO=0.2 KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)
.model PNR PMOS (LEVEL=1 VTO=-0.2 KP=40u LAMBDA=0.02 GAMMA=0 PHI=0.7)
.subckt gsyn inp inn wp wn outp outn vdd vbn
Mtail nt vbn 0 0 NNR W=400u L=100u
M5 na wp nt 0 NNR W=200u L=100u
M6 nb wn nt 0 NNR W=200u L=100u
M1 outp inp na 0 NNR W=200u L=100u
M2 outn inn na 0 NNR W=200u L=100u
M3 outn inp nb 0 NNR W=200u L=100u
M4 outp inn nb 0 NNR W=200u L=100u
.ends
.subckt cmld p n vdd
Rcs1 p cmx {RCS}
Rcs2 n cmx {RCS}
MLp p cmx vdd vdd PNR W={WL} L=100u
MLn n cmx vdd vdd PNR W={WL} L=100u
.ends
.subckt esub xp xn mp mn ep en vdd vbn
Mt nt vbn 0 0 NNR W=400u L=100u
M1x ep xn nt 0 NNR W=200u L=100u
M2x en xp nt 0 NNR W=200u L=100u
M1m ep mp nt 0 NNR W=200u L=100u
M2m en mn nt 0 NNR W=200u L=100u
RLp vdd ep 50k
RLn vdd en 50k
.ends
.subckt gprod xp xn yp yn wp wn vdd vbn
* chopper-free Gilbert product -> differential current onto (wp,wn): integrates  x.y
Mtail nt vbn 0 0 NNR W=400u L=100u
M5 na yp nt 0 NNR W=200u L=100u
M6 nb yn nt 0 NNR W=200u L=100u
M1 iop xp na 0 NNR W=200u L=100u
M2 ion xn na 0 NNR W=200u L=100u
M3 ion xp nb 0 NNR W=200u L=100u
M4 iop xn nb 0 NNR W=200u L=100u
Mlp iop iop vdd vdd PNR W=200u L=100u
Mln ion ion vdd vdd PNR W=200u L=100u
Mpu_wp wp iop vdd vdd PNR W=200u L=100u
MpRn nrn ion vdd vdd PNR W=200u L=100u
MnRn nrn nrn 0 0 NNR W=200u L=100u
Mpd_wp wp nrn 0 0 NNR W=200u L=100u
Mpu_wn wn ion vdd vdd PNR W=200u L=100u
MpRp nrp iop vdd vdd PNR W=200u L=100u
MnRp nrp nrp 0 0 NNR W=200u L=100u
Mpd_wn wn nrp 0 0 NNR W=200u L=100u
.ends"""
def stepped(vals):   # list of (t,val) -> PWL
    pts=[]
    for k,(t,v) in enumerate(vals):
        if k>0: pts.append((t-0.1,vals[k-1][1]))
        pts.append((t,v))
    return "PWL("+" ".join(f"{t:.1f}n {v:.4f}" for t,v in pts)+")"
def gen(seed):
    rng=np.random.default_rng(seed); w0=[0.3*rng.standard_normal() for _ in range(2)]
    NP=4*NEP; TEND=NP*TH
    x1=[];x2=[];vp=[];vn=[]
    for g in range(NP):
        p=PATS[g%4]; t=g*TH
        x1.append((t,0.5+IND*ts(p[0]))); x2.append((t,0.5+IND*ts(p[1])))
        td=TD*ts(tgt(p)); vp.append((t,0.5+td)); vn.append((t,0.5-td))
    L=["single-layer continuous PC",SUB,"Vdd vdd 0 1.0",f"Vbsyn vbsyn 0 {VBSYN}",f"Vgbl gbl 0 {GBL}","Vgm gm 0 0.6",
       f"Vx1p a_xp1 0 {stepped(x1)}",f"Vx1n a_xn1 0 {stepped([(t,1-v) for t,v in x1])}",
       f"Vx2p a_xp2 0 {stepped(x2)}",f"Vx2n a_xn2 0 {stepped([(t,1-v) for t,v in x2])}",
       f"Vxop xop 0 {stepped(vp)}",f"Vxon xon 0 {stepped(vn)}"]   # HARD-CLAMP output to label
    ic=[]
    for i in (1,2):
        gp=np.clip(VW0+KMAP*w0[i-1],0.3,0.9); gn=np.clip(VW0-KMAP*w0[i-1],0.3,0.9)
        L+=[f"Cwp{i} wp{i} 0 {CWW}",f"Cwn{i} wn{i} 0 {CWW}",
            f"Rwp{i} wp{i} wcm {RWL}",f"Rwn{i} wn{i} wcm {RWL}",     # weak hold of weight CM
            f"X_syn{i} a_xp{i} a_xn{i} wp{i} wn{i} mop mon vdd vbsyn gsyn"]    # forward: mu_o += w_i x_i
        ic.append(f".ic v(wp{i})={gp:.4f} v(wn{i})={gn:.4f}")
    L+=[f"Vwcm wcm 0 {VW0}","Xcm_mo mop mon vdd cmld",f"Rmo mop mon {RMO}",   # Rmo sets output gain (avoid rail)
        "Xeo xop xon mop mon eop eon vdd gm esub"]                  # eps_o = label - mu_o
    # LEARNING: each weight cap integrates  eps_o . x_in_i   (continuous, local, no optimizer)
    ep,en=("eop","eon") if SGN>0 else ("eon","eop")
    for i in (1,2):
        L+=[f"X_lrn{i} {ep} {en} a_xp{i} a_xn{i} wp{i} wn{i} vdd gbl gprod"]
    L+=ic
    L+=[f".tran 2n {int(TEND)}n uic",".control","run",
        "wrdata pc1{T}.dat v(mop) v(mon) v(xop) v(wp1) v(wn1) v(wp2) v(wn2)".replace("{T}",TAG),
        ".endc",".end"]
    open(f"pc1_{TAG}.cir","w").write("\n".join(L)+"\n")
if __name__=="__main__":
    seed=int(os.environ.get("SEED","0")); gen(seed)
    subprocess.run(["ngspice","-b",f"pc1_{TAG}.cir"],capture_output=True,text=True,timeout=300)
    d=np.loadtxt(f"pc1{TAG}.dat"); t=d[:,0]*1e9; mo=d[:,1]-d[:,3]; xo=d[:,5]
    print(f"single-layer continuous PC, task={TASK}, seed={seed}")
    def acc(ep):
        ok=0; outs=[]
        for m in range(4):
            g=ep*4+m; i=np.argmin(np.abs(t-(g*TH+TH-2))); outs.append(mo[i]*1e3)
            ok+=((mo[i]>0)==(ts(tgt(PATS[m]))>0))
        return ok,outs
    for ep in [0,10,30,60,NEP-1]:
        a,o=acc(ep); print(f"  epoch {ep:3d}: {a}/4  mu_o(mV)={np.round(o,1)}")
