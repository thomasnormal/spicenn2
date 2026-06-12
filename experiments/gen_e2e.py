#!/usr/bin/env python3
"""CONTINUOUS-TIME end-to-end EP that trains XOR in a SINGLE ngspice .tran.
No numpy in the training loop: weights are differential voltages on caps (gating the dsyn
synapse tails), physically updated by 4-quadrant cells that correlate the local pre/post
activities and chopper the two-phase difference onto the caps.  The controller only writes
the PWL schedule (inputs, target, phase clock).

Network (reciprocal, energy-based, current-mode differential -- same family as ep_dtcm):
  2 inputs -> H hidden diff-pair neurons -> 1 linear output.
  hidden<->output share weight caps (reciprocal), so the circuit relaxes to an energy min.
Schedule: cycle the 4 XOR patterns, each held KCYC free/nudge cycles, repeated NEPOCH times.
"""
import os, numpy as np
H=int(os.environ.get("H","4"))
TH=float(os.environ.get("TH","150"))          # half-period ns (free | nudge)
KCYC=int(os.environ.get("KCYC","1"))          # free/nudge cycles per pattern presentation
NEPOCH=int(os.environ.get("NEPOCH","150"))    # sweeps through all 4 patterns
CW=os.environ.get("CW","8p"); CN=os.environ.get("CN","0.2p")
RC=os.environ.get("RC","2g")                  # weight-cap CMFB leak (large -> weight holds)
VW0=float(os.environ.get("VW0","0.5")); KMAP=float(os.environ.get("KMAP","0.10"))
XSW=float(os.environ.get("XSW","0.30")); TSW=float(os.environ.get("TSW","0.10"))
WLOAD=os.environ.get("WLOAD","3000u"); NUDW=os.environ.get("NUDW","60u")
RLNEU=os.environ.get("RLNEU","800k"); VBN=os.environ.get("VBN","0.30")
NDIV=float(os.environ.get("NDIV","0.0"))      # per-hidden-neuron tail-bias diversity (breaks symmetry)
BWSCALE=float(os.environ.get("BWSCALE","1.0"))  # backward(out->hid) synapse strength vs forward: <1 breaks the
# reciprocal self-reinforcing degenerate loop (hidden driven more by INPUT than by output feedback)
BWW=max(20,int(200*BWSCALE))                    # backward-synapse transistor width (scaled)
GBN=os.environ.get("GBN","0.7")               # update-cell tail bias (sets learning rate)
GBNH=os.environ.get("GBNH","")  # hidden-layer update LR (default=GBN)
GBNB=os.environ.get("GBNB","")  # bias update LR (default=GBN); throttle to let features form first
GBN1=os.environ.get("GBN1","")                # if set, anneal tail bias GBN->GBN1 over the run (in-circuit LR decay)
CMW=os.environ.get("CMW","700u")              # CMFB sink width (strong -> absorbs common injection)
SO=os.environ.get("SO","-1"); SH=os.environ.get("SH","-1")   # update sign: output / hidden layer
SEED=int(os.environ.get("SEED","0")); UPD=int(os.environ.get("UPD","1"))
TAG=os.environ.get("RUNTAG","")               # suffix for output files (parallel-safe)
PATS=[(0,0,0),(0,1,1),(1,0,1),(1,1,0)]
def tsign(b): return +1.0 if b else -1.0
rng=np.random.default_rng(SEED)
WEIGHTS=[]
for j in range(1,H+1):
    for i in (1,2): WEIGHTS.append(f"w{i}{j}")
    WEIGHTS+=[f"o{j}",f"bh{j}"]
WEIGHTS.append("bo")
if os.environ.get("WINIT"):                                  # load initial weights from file (WEIGHTS order)
    sc=float(os.environ.get("WSCALE","1.0"))                 # scale loaded weights (warm-start refinement demo)
    vv=np.loadtxt(os.environ["WINIT"]).reshape(-1)*sc; w0={k:float(vv[i]) for i,k in enumerate(WEIGHTS)}
else:
    w0={k:float(0.4*rng.standard_normal()) for k in WEIGHTS} # initial weights (in numpy units)
def wg(wv): return np.clip(VW0+KMAP*wv,0.3,0.9), np.clip(VW0-KMAP*wv,0.3,0.9)

def stepped(vals_at):    # vals_at: list of (t_ns, value) step changes -> PWL string
    pts=[];
    for k,(t,v) in enumerate(vals_at):
        if k>0: pts.append((t-0.1, vals_at[k-1][1]))
        pts.append((t, v))
    return "PWL("+" ".join(f"{t:.1f}n {v:.4f}" for t,v in pts)+")"

NP=4*NEPOCH                       # total pattern presentations
PWIN=KCYC*2*TH                    # ns per pattern presentation
TEND=NP*PWIN
# input / target schedules (step at each pattern window)
ax1=[]; ax2=[]; vtp=[]; vtn=[]
for g in range(NP):
    p=PATS[g%4]; t0=g*PWIN
    ax1.append((t0,0.5+XSW*tsign(p[0]))); ax2.append((t0,0.5+XSW*tsign(p[1])))
    td=TSW*tsign(p[2]); vtp.append((t0,0.5+td)); vtn.append((t0,0.5-td))
# phase clocks: continuous square wave, nudge in 2nd half of each 2TH cycle
def clk(nudge_hi):
    pts=[(0.0, 2.0 if not nudge_hi else 0.0)]; t=0.0; ncyc=int(TEND/(2*TH))
    for c in range(ncyc):
        a,b=(2.0,0.0) if nudge_hi else (0.0,2.0)   # (nudge,free)
        pts+=[(t+TH-0.5,b),(t+TH+0.5,a),(t+2*TH-0.5,a),(t+2*TH+0.5,b)]; t+=2*TH
    return "PWL("+" ".join(f"{x:.1f}n {v}" for x,v in pts)+")"

L=["continuous-time end-to-end EP XOR (single .tran)",
".model NNR NMOS (LEVEL=1 VTO=0.2  KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)",
".model PNR PMOS (LEVEL=1 VTO=-0.2 KP=40u  LAMBDA=0.02 GAMMA=0 PHI=0.7)",
""".subckt dsyn inp inn outp outn wp wn vdd
M1 outp inp tp 0 NNR W=200u L=100u
M2 outn inn tp 0 NNR W=200u L=100u
Mtp tp wp 0 0 NNR W=200u L=100u
M3 outn inp tn 0 NNR W=200u L=100u
M4 outp inn tn 0 NNR W=200u L=100u
Mtn tn wn 0 0 NNR W=200u L=100u
.ends""",
f""".subckt dsynb inp inn outp outn wp wn vdd
M1 outp inp tp 0 NNR W={BWW}u L=100u
M2 outn inn tp 0 NNR W={BWW}u L=100u
Mtp tp wp 0 0 NNR W={BWW}u L=100u
M3 outn inp tn 0 NNR W={BWW}u L=100u
M4 outp inn tn 0 NNR W={BWW}u L=100u
Mtn tn wn 0 0 NNR W={BWW}u L=100u
.ends
.subckt dneuron up un ap an vdd vbn
M1 an up tn 0 NNR W=200u L=100u
M2 ap un tn 0 NNR W=200u L=100u
Mt tn vbn 0 0 NNR W=200u L=100u
RLa vdd ap {RLNEU}
RLb vdd an {RLNEU}
.ends""",
""".subckt gupd xp xn yp yn wp wn psig psigb vdd vbn
Mtail nt vbn 0 0 NNR W=400u L=100u
M5 na yp nt 0 NNR W=200u L=100u
M6 nb yn nt 0 NNR W=200u L=100u
M1 iop xp na 0 NNR W=200u L=100u
M2 ion xn na 0 NNR W=200u L=100u
M3 ion xp nb 0 NNR W=200u L=100u
M4 iop xn nb 0 NNR W=200u L=100u
Mlp iop iop vdd vdd PNR W=200u L=100u
Mln ion ion vdd vdd PNR W=200u L=100u
MopA ca iop vdd vdd PNR W=200u L=100u
MopB cb ion vdd vdd PNR W=200u L=100u
Msw1 ca psig  wp 0 NNR W=300u L=100u
Msw2 cb psig  wn 0 NNR W=300u L=100u
Msw3 ca psigb wn 0 NNR W=300u L=100u
Msw4 cb psigb wp 0 NNR W=300u L=100u
.ends""",
# DIFFERENTIAL-ONLY update cell: pushes I_iop onto wp & pulls I_ion from wp (and mirror on wn),
# so net common-mode injection is ZERO (no CM runaway). Input chopper swaps yA/yB by phase for
# the two-phase (nudge-free) sign. wp += (I_iop - I_ion); wn += (I_ion - I_iop).
""".subckt gupd2 xp xn yp yn wp wn psig psigb vdd vbn
Msa yA yp psig 0 NNR W=300u L=100u
Msb yA yn psigb 0 NNR W=300u L=100u
Msc yB yn psig 0 NNR W=300u L=100u
Msd yB yp psigb 0 NNR W=300u L=100u
Mtail nt vbn 0 0 NNR W=400u L=100u
M5 na yA nt 0 NNR W=200u L=100u
M6 nb yB nt 0 NNR W=200u L=100u
M1 iop xp na 0 NNR W=200u L=100u
M2 ion xn na 0 NNR W=200u L=100u
M3 ion xp nb 0 NNR W=200u L=100u
M4 iop xn nb 0 NNR W=200u L=100u
Mlp iop iop vdd vdd PNR W=200u L=100u
Mln ion ion vdd vdd PNR W=200u L=100u
Mpu_p wp iop vdd vdd PNR W=200u L=100u
Mpu_n wn ion vdd vdd PNR W=200u L=100u
MpRn nrn ion vdd vdd PNR W=200u L=100u
MnRn nrn nrn 0 0 NNR W=200u L=100u
Mpd_p wp nrn 0 0 NNR W=200u L=100u
MpRp nrp iop vdd vdd PNR W=200u L=100u
MnRp nrp nrp 0 0 NNR W=200u L=100u
Mpd_n wn nrp 0 0 NNR W=200u L=100u
.ends""",
"Vdd vdd 0 1.0", "Vbn vbn 0 "+VBN,
("Vgbn gbn 0 "+GBN) if not GBN1 else f"Vgbn gbn 0 PWL(0n {GBN} {int(TEND)}n {GBN1})",  # const or annealed LR
f"Vgbnh gbnh 0 {GBNH if GBNH else GBN}",  # hidden-layer update rail
f"Vgbnb gbnb 0 {GBNB if GBNB else GBN}",  # bias update rail (throttle)
"Vbphi bp_hi 0 0.8","Vbplo bp_lo 0 0.2",
f"Vax1 a_xp1 0 {stepped(ax1)}", f"Vax1n a_xn1 0 {stepped([(t,1.0-v) for t,v in ax1])}",
f"Vax2 a_xp2 0 {stepped(ax2)}", f"Vax2n a_xn2 0 {stepped([(t,1.0-v) for t,v in ax2])}",
f"Vtp vtp 0 {stepped(vtp)}", f"Vtn vtn 0 {stepped(vtn)}",
f"Vpsig psig 0 {clk(True)}", f"Vpsigb psigb 0 {clk(False)}", f"Vnen nen 0 {clk(True)}"]

ic=[]   # .ic lines for weight caps and state nodes
def wcap(key,sgn):
    gp,gn=wg(w0[key])
    L.append(f"Cwp_{key} wp_{key} 0 {CW}"); L.append(f"Cwn_{key} wn_{key} 0 {CW}")
    L.append(f"Rcp_{key} wp_{key} cm_{key} {RC}"); L.append(f"Rcn_{key} wn_{key} cm_{key} {RC}")
    # WEAK saturation CMFB sink (gate=sensed CM, source=gnd): removes equal current from both
    # caps (common-mode only, differential preserved). Kept weak (small W) so its CM setpoint
    # stays near VW0 even at gentle update injection -- a strong sink crashes CM and starves
    # the synapse tails. Sink current ~ (CM-Vt)^2 self-limits if CM rises.
    L.append(f"Msp_{key} wp_{key} cm_{key} 0 0 NNR W={CMW} L=100u")
    L.append(f"Msn_{key} wn_{key} cm_{key} 0 0 NNR W={CMW} L=100u")
    ic.append(f".ic v(wp_{key})={gp:.4f} v(wn_{key})={gn:.4f} v(cm_{key})={VW0}")
def syn(tag,ip,inn,op,on,key,sub="dsyn"):
    L.append(f"X_{tag} {ip} {inn} {op} {on} wp_{key} wn_{key} vdd {sub}")
CELL=os.environ.get("CELL","gupd")            # update cell: gupd (common-injecting) | gupd2 (differential-only)
def upd(tag,xp,xn,yp,yn,key,sgn,rail="gbn"):
    ps,psb=("psig","psigb") if sgn>0 else ("psigb","psig")
    if UPD: L.append(f"Xu_{tag} {xp} {xn} {yp} {yn} wp_{key} wn_{key} {ps} {psb} vdd {rail} {CELL}")

# weight caps
for k in WEIGHTS: wcap(k, 1)
# hidden layer
for j in range(1,H+1):
    up,un=f"up_h{j}",f"un_h{j}"
    bj=float(VBN)+(NDIV*(2.0*(j-1)/max(H-1,1)-1.0) if NDIV>0 else 0.0)   # heterogeneous tail bias
    L+=[f"Mlp_h{j} {up} {up} vdd vdd PNR W={WLOAD} L=100u",
        f"Mln_h{j} {un} {un} vdd vdd PNR W={WLOAD} L=100u",
        f"Chp_h{j} {up} 0 {CN}", f"Chn_h{j} {un} 0 {CN}",
        f"Vbnh{j} bnh{j} 0 {bj:.4f}",
        f"Xn_h{j} {up} {un} ap_h{j} an_h{j} vdd bnh{j} dneuron"]
    ic.append(f".ic v({up})=0.5 v({un})=0.5 v(ap_h{j})=0.5 v(an_h{j})=0.5")
    for i in (1,2):
        syn(f"i{i}{j}",f"a_xp{i}",f"a_xn{i}",up,un,f"w{i}{j}")
        upd(f"i{i}{j}",f"a_xp{i}",f"a_xn{i}",f"ap_h{j}",f"an_h{j}",f"w{i}{j}",int(SH),"gbnh")
    syn(f"bk{j}","up_o","un_o",up,un,f"o{j}","dsynb")      # reciprocal backward (shared o{j}), scalable W
    syn(f"bh{j}","bp_hi","bp_lo",up,un,f"bh{j}")           # hidden bias
    upd(f"bh{j}","bp_hi","bp_lo",f"ap_h{j}",f"an_h{j}",f"bh{j}",int(SH),"gbnb")
# output layer
L+=[f"Mlp_o up_o up_o vdd vdd PNR W={WLOAD} L=100u", f"Mln_o un_o un_o vdd vdd PNR W={WLOAD} L=100u",
    f"Cop up_o 0 {CN}", f"Con un_o 0 {CN}"]
ic.append(".ic v(up_o)=0.5 v(un_o)=0.5")
for j in range(1,H+1):
    syn(f"fo{j}",f"ap_h{j}",f"an_h{j}","up_o","un_o",f"o{j}")     # forward (shared o{j})
    upd(f"o{j}",f"ap_h{j}",f"an_h{j}","up_o","un_o",f"o{j}",int(SO))
syn("bo","bp_hi","bp_lo","up_o","un_o","bo")
upd("bo","bp_hi","bp_lo","up_o","un_o","bo",int(SO),"gbnb")
# output nudge toward target during nudge phase
L+=[f"Mnp up_o nen vtp 0 NNR W={NUDW} L=100u", f"Mnn un_o nen vtn 0 NNR W={NUDW} L=100u"]

L+=ic
rdw=" ".join(f"v(wp_{k}) v(wn_{k})" for k in WEIGHTS)
L+=[f".tran 1n {int(TEND)}n uic",".control","run",
    f"wrdata e2e_w{TAG}.dat {rdw}",
    f"wrdata e2e_o{TAG}.dat v(up_o) v(un_o) v(a_xp1) v(a_xp2) v(vtp)",
    ".endc",".end"]
open(f"e2e{TAG}.cir","w").write("\n".join(L)+"\n")
np.savetxt(f"e2e_w0{TAG}.dat",np.array([w0[k] for k in WEIGHTS]))
print(f"wrote e2e.cir  H={H} weights={len(WEIGHTS)} seed={SEED} sim={TEND/1000:.1f}us  (UPD={UPD})")
print("WEIGHTS order:"," ".join(WEIGHTS))
