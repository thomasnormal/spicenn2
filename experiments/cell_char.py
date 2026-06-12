#!/usr/bin/env python3
# Characterize the Gilbert synapse (gsyn) multiply linearity: sweep differential feature (fd) x weight (wd)
# over a grid via one PWL .tran, measure I(outp)-I(outn), fit to the ideal product fd*wd -> R^2.
# RDEG env adds source degeneration. Goal: see if degeneration lifts R^2 toward ~0.99.
import os, subprocess, numpy as np
RDEG=os.environ.get("RDEG","")
FCM=float(os.environ.get("FCM","0.6"))   # feature common-mode
WCM=float(os.environ.get("WCM","0.5"))   # weight common-mode (VW0)
VOUT=float(os.environ.get("VOUT","0.7")) # output hold voltage
SW=float(os.environ.get("SW","0.25"))    # half-swing of fd and wd
VBSYN=os.environ.get("VBSYN","0.45")
gsyn = (""".subckt gsyn inp inn wp wn outp outn vdd vbn
Mtail nt vbn 0 0 NNR W=400u L=100u
M5 na wp nt 0 NNR W=200u L=100u
M6 nb wn nt 0 NNR W=200u L=100u
M1 outp inp na 0 NNR W=200u L=100u
M2 outn inn na 0 NNR W=200u L=100u
M3 outn inp nb 0 NNR W=200u L=100u
M4 outp inn nb 0 NNR W=200u L=100u
.ends""" if not RDEG else f""".subckt gsyn inp inn wp wn outp outn vdd vbn
Mtail nt vbn 0 0 NNR W=400u L=100u
M5 na wp s5 0 NNR W=200u L=100u
Rd5 s5 nt {RDEG}
M6 nb wn s6 0 NNR W=200u L=100u
Rd6 s6 nt {RDEG}
M1 outp inp s1 0 NNR W=200u L=100u
Rd1 s1 na {RDEG}
M2 outn inn s2 0 NNR W=200u L=100u
Rd2 s2 na {RDEG}
M3 outn inp s3 0 NNR W=200u L=100u
Rd3 s3 nb {RDEG}
M4 outp inn s4 0 NNR W=200u L=100u
Rd4 s4 nb {RDEG}
.ends""")
# grid of operating points
fds=np.linspace(-SW,SW,9); wds=np.linspace(-SW,SW,9)
pts=[(fd,wd) for wd in wds for fd in fds]
Ts=50e-9; dt=2e-9
def pwl(vals):
    s=[]
    for k,v in enumerate(vals):
        t0=k*Ts; s+=[(t0,v),(t0+Ts-dt,v)]
    s.append((len(vals)*Ts,vals[-1]))
    return "PWL("+" ".join(f"{t:.10g} {v:.6f}" for t,v in s)+")"
inp=[FCM+fd/2 for fd,wd in pts]; inn=[FCM-fd/2 for fd,wd in pts]
wp =[WCM+wd/2 for fd,wd in pts]; wn =[WCM-wd/2 for fd,wd in pts]
L=["* gsyn characterization",
   ".model NNR NMOS (LEVEL=1 VTO=0.2 KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)",
   gsyn, "Vdd vdd 0 1.0", f"Vbsyn vbsyn 0 {VBSYN}",
   f"Vinp inp 0 {pwl(inp)}", f"Vinn inn 0 {pwl(inn)}",
   f"Vwp wp 0 {pwl(wp)}", f"Vwn wn 0 {pwl(wn)}",
   f"Vop outp 0 {VOUT}", f"Von outn 0 {VOUT}",
   "Xs inp inn wp wn outp outn vdd vbsyn gsyn",
   ".control", f"tran {dt} {len(pts)*Ts} uic",
   "wrdata cell_char.txt i(Vop) i(Von)", "echo CHAR_OK", ".endc", ".end"]
open("cell_char.cir","w").write("\n".join(L)+"\n")
subprocess.run(["ngspice","-b","cell_char.cir"],capture_output=True,timeout=120)
d=np.loadtxt("cell_char.txt")
t=d[:,0]; iop=d[:,1]; ion=d[:,3]
# sample mid-hold of each slot
idx=[int((k+0.7)*Ts/dt) for k in range(len(pts))]
idx=[min(i,len(t)-1) for i in idx]
Iout=-(iop[idx]-ion[idx])   # I into output = -I(Vsource)
prod=np.array([fd*wd for fd,wd in pts])
# R^2 of Iout vs linear in prod
A=np.polyfit(prod,Iout,1); pred=np.polyval(A,prod)
ssr=np.sum((Iout-pred)**2); sst=np.sum((Iout-Iout.mean())**2)
r2=1-ssr/sst if sst>0 else 0
# also check it's a clean PRODUCT: corr with fd*wd vs spurious fd, wd terms
fdv=np.array([fd for fd,wd in pts]); wdv=np.array([wd for fd,wd in pts])
B=np.linalg.lstsq(np.c_[prod,fdv,wdv,np.ones_like(prod)],Iout,rcond=None)[0]
print(f"RDEG={RDEG or '0':>6}: multiply R^2={r2:.4f}  gain={A[0]*1e6:.1f}uA  "
      f"spurious(fd,wd)/prod={abs(B[1])+abs(B[2]):.2e}/{abs(B[0]):.2e}")
