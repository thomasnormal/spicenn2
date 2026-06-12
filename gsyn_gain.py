#!/usr/bin/env python3
"""Isolate the gsyn forward gain (the ~300x/2-stage attenuation root cause).  Drive one gsyn with a
swept differential input at a FIXED weight, measure d(outp-outn)/d(vin) under different loads:
 (a) cmld (current CMFB load)   (b) cmld with bigger Rcs   (c) active diff-amp load (gm*ro)  via .dc."""
import os, subprocess, numpy as np
VBSYN=os.environ.get("VBSYN","0.45"); WdP=float(os.environ.get("WP","0.62")); WdN=float(os.environ.get("WN","0.38"))
SUB=""".model NNR NMOS (LEVEL=1 VTO=0.2 KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)
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
MLp p cmx vdd vdd PNR W=1000u L=100u
MLn n cmx vdd vdd PNR W=1000u L=100u
.ends"""
def deck(load,rcs="300k"):
    sub=SUB.replace("{RCS}",rcs)
    L=[f"gsyn gain test load={load}",sub,"Vdd vdd 0 1.0",f"Vbsyn vbsyn 0 {VBSYN}",
       "Vinp inp 0 0.5","Vinn inn 0 0.5",f"Vwp wp 0 {WdP}",f"Vwn wn 0 {WdN}",
       "Vcm vcm 0 0.5",
       "Xs inp inn wp wn outp outn vdd vbsyn gsyn"]
    if load=="cmld": L+=["Xl outp outn vdd cmld"]
    elif load=="cmld_hi": L+=["Xl outp outn vdd cmld"]  # rcs passed bigger
    elif load=="res":  L+=["Rp outp vcm 200k","Rn outn vcm 200k"]
    elif load=="activeload":
        # current-source PMOS loads (high ro) + a CMFB-free fixed bias; high differential impedance
        L+=["Mlp outp pb vdd vdd PNR W=300u L=200u","Mln outn pb vdd vdd PNR W=300u L=200u","Vpb pb 0 0.55"]
    L+=[".control",
        "dc Vinp 0.40 0.60 0.005",
        "let din = v(inp)-0.5",          # differential input (inn fixed at 0.5)
        "let dout = v(outp)-v(outn)",
        "wrdata gg_{T}.dat dout din".replace("{T}",os.environ.get("RUNTAG","x")),
        ".endc",".end"]
    return "\n".join(L)+"\n"
def run(load,rcs="300k"):
    open("gg.cir","w").write(deck(load,rcs))
    subprocess.run(["ngspice","-b","gg.cir"],capture_output=True,text=True,timeout=60)
    d=np.loadtxt(f"gg_{os.environ.get('RUNTAG','x')}.dat")
    dout=d[:,0]; din=d[:,1]
    # gain near center
    mid=len(din)//2; g=(dout[mid+2]-dout[mid-2])/(din[mid+2]-din[mid-2])
    return g, dout.max()-dout.min()
for load,rcs in [("cmld","300k"),("cmld","3meg"),("res","300k"),("activeload","300k")]:
    g,swing=run(load,rcs)
    print(f"  load={load:10s} rcs={rcs:5s}: gain d(out)/d(in)={g:+.3f}  output swing={swing*1e3:.1f}mV (input swing 200mV)")
