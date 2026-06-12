#!/usr/bin/env python3
"""Generate a Spectre SPICE-mode deck for the EP update cell (no behavioral sources):
Gilbert multiplier -> PMOS current mirrors -> chopper(psign) -> differential weight cap.
5 free/nudge cycles; phi_i: free 0.2 / nudge 0.25 ; phi_j: free 0.3 / nudge 0.35."""
import os
NCYC=int(os.environ.get("NCYC","8")); TH=100  # cycles, half-period ns
CW=os.environ.get("CW","20p"); RC=os.environ.get("RC","500k")
# phi_i, phi_j free/nudge values (env-settable for sign / zero-signal tests)
PIF=float(os.environ.get("PIF","0.2")); PIN=float(os.environ.get("PIN","0.25"))
PJF=float(os.environ.get("PJF","0.3")); PJN=float(os.environ.get("PJN","0.35"))
def pwl(free,nud):
    pts=[(0,free)]; t=0
    for c in range(NCYC):
        pts+=[(t+TH,free),(t+TH+0.01,nud),(t+2*TH,nud)]
        if c<NCYC-1: pts+=[(t+2*TH+0.01,free)]
        t+=2*TH
    return "PWL("+" ".join(f"{a}n {b}" for a,b in pts)+")"
# differential inputs vxp=0.5+phi_i/2 etc.
pi_f,pi_n=PIF,PIN; pj_f,pj_n=PJF,PJN
S=f"""simulator lang=spice
* EP update cell (Spectre)
.model NNR NMOS (LEVEL=1 VTO=0.2  KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)
.model PNR PMOS (LEVEL=1 VTO=-0.2 KP=40u  LAMBDA=0.02 GAMMA=0 PHI=0.7)
Vdd vdd 0 2.0
Vbn vbn 0 0.75
Vxp vxp 0 {pwl(0.5+pi_f/2,0.5+pi_n/2)}
Vxn vxn 0 {pwl(0.5-pi_f/2,0.5-pi_n/2)}
Vyp vyp 0 {pwl(0.5+pj_f/2,0.5+pj_n/2)}
Vyn vyn 0 {pwl(0.5-pj_f/2,0.5-pj_n/2)}
Vpsig  psig  0 {pwl(0,2)}
Vpsigb psigb 0 {pwl(2,0)}
* Gilbert core
Mtail nt vbn 0 0 NNR W=400u L=100u
M5 na vyp nt 0 NNR W=200u L=100u
M6 nb vyn nt 0 NNR W=200u L=100u
M1 iop vxp na 0 NNR W=200u L=100u
M2 ion vxn na 0 NNR W=200u L=100u
M3 ion vxp nb 0 NNR W=200u L=100u
M4 iop vxn nb 0 NNR W=200u L=100u
* PMOS diode loads + mirror copies down to cap nodes
Mlp iop iop vdd vdd PNR W=200u L=100u
Mln ion ion vdd vdd PNR W=200u L=100u
MopA ca iop vdd vdd PNR W=200u L=100u
MopB cb ion vdd vdd PNR W=200u L=100u
* chopper = NMOS pass transistors (gate rail-to-rail), swap by psign
Msw1 ca psig  vwp 0 NNR W=300u L=100u
Msw2 cb psig  vwn 0 NNR W=300u L=100u
Msw3 ca psigb vwn 0 NNR W=300u L=100u
Msw4 cb psigb vwp 0 NNR W=300u L=100u
* differential weight caps + CMFB sinks: gate = sensed common mode (vwp+vwn)/2 -> self-matches
* the source current and holds CM, while the DIFFERENTIAL integrates freely.
Rc1 vwp cmn {RC}
Rc2 vwn cmn {RC}
Msp vwp cmn 0 0 NNR W=700u L=100u
Msn vwn cmn 0 0 NNR W=700u L=100u
Cwp vwp 0 {CW}
Cwn vwn 0 {CW}
.ic v(vwp)=0.5 v(vwn)=0.5
.tran 0.05n {NCYC*2*TH}n uic
.end
"""
open("ep_cell.scs","w").write(S); print("wrote ep_cell.scs")
