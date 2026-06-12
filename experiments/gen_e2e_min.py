#!/usr/bin/env python3
"""De-risk the CONTINUOUS-TIME end-to-end EP loop on the simplest system: ONE weight,
linear unit, in a single ngspice .tran.  The weight is a differential voltage on caps
(vwp,vwn) that gate the dsyn synapse tail transistors -- exactly as in ep_dtcm, but now
the caps are charged by a PHYSICAL 4-quadrant update cell (Gilbert) choppered by a phase
clock, instead of by numpy.  Target is presented as a voltage and the output is softly
nudged toward it in the nudge phase.  If vwp-vwn converges so o -> w*x tracks t, the loop
is real and we scale to XOR.

Schedule per cycle (period 2*TH ns):  [0,TH] free (no nudge), [TH,2TH] nudge.
EP weight update integrates  (x . o)_nudge - (x . o)_free  onto the weight caps.
"""
import os
TH=float(os.environ.get("TH","200"))         # half-period (ns)
NCYC=int(os.environ.get("NCYC","80"))         # number of free/nudge cycles
CW=os.environ.get("CW","8p")                  # weight cap
CN=os.environ.get("CN","0.2p")                # node cap (fast relaxation)
RC=os.environ.get("RC","200meg")              # CMFB leak on weight cap (large -> differential holds)
VW0=float(os.environ.get("VW0","0.5"))        # weight-cap common mode / init
XSW=float(os.environ.get("XSW","0.30"))       # input differential half-swing
TSW=float(os.environ.get("TSW","0.12"))       # target differential half-swing
WLOAD=os.environ.get("WLOAD","3000u")
NUDW=os.environ.get("NUDW","60u")             # nudge pass-transistor width (sets beta)
SGN=os.environ.get("SGN","1")                 # update sign: 1 -> psig,psigb ; -1 -> swapped
GUP=os.environ.get("GUP","g")                 # update cell: g=Gilbert
PS,PSB=("psig","psigb") if SGN=="1" else ("psigb","psig")
def clk(hi_in_nudge):
    # PWL square wave aligned to the free/nudge schedule; value `hi` during nudge window
    pts=[(0.0, 0.0 if hi_in_nudge else 2.0)]; t=0.0
    for c in range(NCYC):
        a,b=(2.0,0.0) if hi_in_nudge else (0.0,2.0)   # nudge value, free value
        pts+=[(t+TH-0.5,b),(t+TH+0.5,a),(t+2*TH-0.5,a),(t+2*TH+0.5,b)]
        t+=2*TH
    return "PWL("+" ".join(f"{x:.1f}n {v}" for x,v in pts)+")"
S=f"""continuous-time end-to-end EP -- minimal 1-weight linear regression
.model NNR NMOS (LEVEL=1 VTO=0.2  KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)
.model PNR PMOS (LEVEL=1 VTO=-0.2 KP=40u  LAMBDA=0.02 GAMMA=0 PHI=0.7)
.subckt dsyn inp inn outp outn wp wn vdd
M1 outp inp tp 0 NNR W=200u L=100u
M2 outn inn tp 0 NNR W=200u L=100u
Mtp tp wp 0 0 NNR W=200u L=100u
M3 outn inp tn 0 NNR W=200u L=100u
M4 outp inn tn 0 NNR W=200u L=100u
Mtn tn wn 0 0 NNR W=200u L=100u
.ends
* 4-quadrant Gilbert update integrator: charges (wp,wn) by (x.y)_nudge-(x.y)_free
.subckt gupd xp xn yp yn wp wn psig psigb vdd vbn
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
.ends
Vdd vdd 0 1.0
Vbn vbn 0 0.30
Vgbn gbn 0 0.7
* --- input x (constant differential here: a fixed regression sample) ---
Vxp a_xp 0 {0.5+XSW}
Vxn a_xn 0 {0.5-XSW}
* --- target t (constant differential) ---
Vtp vtp 0 {0.5+TSW}
Vtn vtn 0 {0.5-TSW}
* --- phase clocks ---
Vpsig  psig  0 {clk(True)}
Vpsigb psigb 0 {clk(False)}
Vnen   nen   0 {clk(True)}
* --- weight caps (the trainable parameter) + CMFB to hold common mode ---
Cwp wp 0 {CW}
Cwn wn 0 {CW}
Rcp wp cmn {RC}
Rcn wn cmn {RC}
Msp wp cmn 0 0 NNR W=700u L=100u
Msn wn cmn 0 0 NNR W=700u L=100u
* --- forward synapse: o = w * x ---
Mlp_o up_o up_o vdd vdd PNR W={WLOAD} L=100u
Mln_o un_o un_o vdd vdd PNR W={WLOAD} L=100u
Co_p up_o 0 {CN}
Co_n un_o 0 {CN}
Xsyn a_xp a_xn up_o un_o wp wn vdd dsyn
* --- nudge: in nudge phase, pull output softly toward target (switch transistors) ---
Mnp up_o nen vtp 0 NNR W={NUDW} L=100u
Mnn un_o nen vtn 0 NNR W={NUDW} L=100u
* --- update cell: correlate input activity x with output activity o (SGN sets polarity) ---
Xupd a_xp a_xn up_o un_o wp wn {PS} {PSB} vdd gbn gupd
.ic v(wp)={VW0} v(wn)={VW0} v(up_o)=0.5 v(un_o)=0.5 v(cmn)={VW0}
.tran 1n {int(NCYC*2*TH)}n uic
.control
run
wrdata e2e_min.dat v(wp) v(wn) v(up_o) v(un_o)
.endc
.end
"""
open("e2e_min.cir","w").write(S); print("wrote e2e_min.cir  (NCYC=%d TH=%g -> %g us sim)"%(NCYC,TH,NCYC*2*TH/1000))
