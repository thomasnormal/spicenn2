#!/usr/bin/env python3
"""Single-NMOS squarer EP update cell (Xyce). Replaces the 7-T Gilbert with ONE NMOS:
gate-source = dv = phi_i-phi_j  ->  I ~ (dv-Vt)^2 (saturation squarer).
Mirror -> chopper(phase) -> differential weight cap + CMFB. The two-phase difference
integrates I(dv_nudge)-I(dv_free) ~ (dv_b^2 - dv_*^2) onto the weight."""
import os
NCYC=int(os.environ.get("NCYC","8")); TH=100
DVF=float(os.environ.get("DVF","0.30")); DVN=float(os.environ.get("DVN","0.40"))  # dv free/nudge
CW=os.environ.get("CW","20p"); RC=os.environ.get("RC","500k")
def pwl(free,nud):
    pts=[(0,free)]; t=0
    for c in range(NCYC):
        pts+=[(t+TH,free),(t+TH+0.01,nud),(t+2*TH,nud)]
        if c<NCYC-1: pts+=[(t+2*TH+0.01,free)]
        t+=2*TH
    return "PWL("+" ".join(f"{a}n {b}" for a,b in pts)+")"
S=f"""single-NMOS squarer EP update cell (Xyce)
.model NNR NMOS (LEVEL=1 VTO=0.2  KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)
.model PNR PMOS (LEVEL=1 VTO=-0.2 KP=40u  LAMBDA=0.02 GAMMA=0 PHI=0.7)
Vdd vdd 0 2.0
Vg vg 0 {pwl(0.5+DVF/2,0.5+DVN/2)}
Vs vs 0 {pwl(0.5-DVF/2,0.5-DVN/2)}
Vpsig  psig  0 {pwl(0,2)}
Vpsigb psigb 0 {pwl(2,0)}
* --- single NMOS squarer: Vgs = vg-vs = dv ; I ~ (dv-Vt)^2 ---
Msq dsq vg vs 0 NNR W=1000u L=100u
* PMOS diode + mirror copy of the square current to cs
Mlsq dsq dsq vdd vdd PNR W=200u L=100u
MopS cs  dsq vdd vdd PNR W=200u L=100u
* chopper: route the (single) square current to vwp (nudge) or vwn (free)
Msw1 cs psig  vwp 0 NNR W=300u L=100u
Msw2 cs psigb vwn 0 NNR W=300u L=100u
* differential weight cap + CMFB sinks (gate=sensed CM)
Rc1 vwp cmn {RC}
Rc2 vwn cmn {RC}
Msp vwp cmn 0 0 NNR W=700u L=100u
Msn vwn cmn 0 0 NNR W=700u L=100u
Cwp vwp 0 {CW}
Cwn vwn 0 {CW}
.options DEVICE gmin=1e-8 voltagelimiterflag=1
.options TIMEINT reltol=1e-3 abstol=1e-6
.ic V(vwp)=0.5 V(vwn)=0.5
.tran 0.05n {NCYC*2*TH}n UIC
.print tran V(vwp) V(vwn)
.end
"""
open("nmos_cell.cir","w").write(S); print("wrote nmos_cell.cir")
