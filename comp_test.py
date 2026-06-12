#!/usr/bin/env python3
# Standalone validation of the subtractive common-mode competition cell.
# Drive C=3 output keystones with FIXED hidden-activation voltages (several scenarios via
# PWL slots) and fixed output weights, targets=0. Log y_c, the per-class error nodes d2e_c,
# and the common-mode node cmv. Expectation: WITHOUT competition the errors delta_c=(d2e_c-1.2)
# carry a nonzero common mode (mean!=0); WITH competition the cell subtracts the mean so
# sum_c delta_c ~ 0 while the *spread* of delta_c (the discriminative part) is preserved.
import os, numpy as np
COMP=os.environ.get("COMP","subtractive"); C=3; H=3
VC=1.8; Gs=1.2; VREFO=0.5; RTO=7e3; RCM=RTO/C
# three scenarios of hidden activations (rows) -> different y_c per scenario
HV=np.array([[2.2,0.8,0.8],[0.8,2.2,0.8],[1.4,1.4,1.4]])   # scenario x hidden
# fixed output weights (C x H): make class c respond to hidden c (identity-ish) + bias
Wts=np.array([[0.8,-0.4,-0.4],[-0.4,0.8,-0.4],[-0.4,-0.4,0.8]]); Bo=np.array([0.0,0.0,0.0])
Ts=0.3; dt=0.05; N=len(HV)
def pwl(vals):
    pts=[]
    for k in range(N):
        v=vals[k]; t0=k*Ts; pts+=[(t0,v),(t0+Ts-dt,v)]
    pts.append((N*Ts,pts[-1][1])); return " ".join(f"{t:.3f} {v:.3f}" for t,v in pts)
L=[]; w=lambda s:L.append(s)
w("* competition cell standalone test")
w(".model NSYN NMOS (LEVEL=1 VTO=0.2 KP=50u LAMBDA=0 GAMMA=0 PHI=0.7)")
w(".model NMIR NMOS (LEVEL=1 VTO=0.5 KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)")
w(".model PMIR PMOS (LEVEL=1 VTO=-0.5 KP=40u LAMBDA=0.02 GAMMA=0 PHI=0.7)")
w("Vvdd vdd 0 3"); w(f"Vrefo vrefo 0 {VREFO}"); w(f"Vvc vcn 0 {VC}")
w("Vrefd vrefd 0 1.2"); w("Vboff boff 0 1.9"); w(f"Vbx bx 0 dc 1.5")
for h in range(1,H+1): w(f"Vhv{h} hv_{h} 0 PWL({pwl(HV[:,h-1])})")
for c in range(1,C+1): w(f"Vtg{c} tg{c} 0 dc 0")   # targets = 0
def sub(tag,cp,cn,vout,Rt,vrefn):
    w(f"Msa_{tag} {cn} {cn} 0 0 NMIR W=20u L=1u"); w(f"Msb_{tag} {vout} {cn} 0 0 NMIR W=20u L=1u")
    w(f"Msc_{tag} {cp} {cp} 0 0 NMIR W=20u L=1u"); w(f"Msd_{tag} dpf_{tag} {cp} 0 0 NMIR W=20u L=1u")
    w(f"Mse_{tag} dpf_{tag} dpf_{tag} vdd vdd PMIR W=60u L=1u"); w(f"Msf_{tag} {vout} dpf_{tag} vdd vdd PMIR W=60u L=1u")
    w(f"R_{tag} {vout} {vrefn} {Rt}")
def syn(idx,vin,cp,cn,val):
    w(f"Vg_{idx} g_{idx} 0 {VC+Gs*val:.3f}"); w(f"Mp_{idx} {vin} g_{idx} {cp} 0 NSYN W=10u L=1u"); w(f"Mn_{idx} {vin} vcn {cn} 0 NSYN W=10u L=1u")
inputs_o=[(str(j),f"hv_{j}") for j in range(1,H+1)]+[("b","bx")]
for c in range(1,C+1):
    for j,vin in inputs_o:
        val = Wts[c-1][int(j)-1] if j!="b" else Bo[c-1]
        syn(f"2_{c}_{j}",vin,f"colpo{c}",f"colno{c}",val)
    sub(f"o{c}",f"colpo{c}",f"colno{c}",f"y_{c}",RTO,"vrefo")
errsrc={c:"vrefd" for c in range(1,C+1)}; errsrc_n={c:"vrefd" for c in range(1,C+1)}
if COMP=="subtractive":
    for c in range(1,C+1):
        w(f"Mcmp_{c} cm_ps colno{c} 0 0 NMIR W=20u L=1u")
        w(f"Mcmq_{c} cm_ps tg{c}    0 0 NMIR W=10u L=1u")
        w(f"Mcmr_{c} cm_ns dpf_o{c} vdd vdd PMIR W=60u L=1u")
        w(f"Mcms_{c} cm_ns boff     vdd vdd PMIR W=10u L=1u")
    w("Mcmpd cm_ps cm_ps vdd vdd PMIR W=60u L=1u"); w("Mcmnd cm_ns cm_ns 0 0 NMIR W=20u L=1u")
    w("Mcmsrc cmv cm_ps vdd vdd PMIR W=60u L=1u"); w("Mcmsnk cmv cm_ns 0 0 NMIR W=20u L=1u"); w(f"Rcm cmv vrefd {RCM}")
    w("Mcmcp cmvn cm_ns vdd vdd PMIR W=60u L=1u"); w("Mcmcn cmvn cm_ps 0 0 NMIR W=20u L=1u"); w(f"Rcmn cmvn vrefd {RCM}")
    errsrc={c:"cmv" for c in range(1,C+1)}; errsrc_n={c:"cmvn" for c in range(1,C+1)}
else:
    w("Vcmv cmv 0 1.2")
for c in range(1,C+1):
    w(f"Msf2_{c} d2e_{c} dpf_o{c} vdd vdd PMIR W=60u L=1u"); w(f"Msb2_{c} d2e_{c} colno{c} 0 0 NMIR W=20u L=1u")
    w(f"Moff_{c} d2e_{c} boff vdd vdd PMIR W=10u L=1u");    w(f"Mtge_{c} d2e_{c} tg{c} 0 0 NMIR W=10u L=1u")
    w(f"Rde_{c} d2e_{c} {errsrc[c]} {RTO}")
w(".control"); w(f"  tran {dt} {N*Ts} uic")
yt=" ".join(f"v(y_{c})" for c in range(1,C+1)); et=" ".join(f"v(d2e_{c})" for c in range(1,C+1))
w(f"  wrdata comp_trace.txt {yt} {et} v(cmv)"); w("  echo COMP_OK")
w(".endc"); w(".end")
open("comp_test.cir","w").write("\n".join(L)+"\n")
print(f"COMP={COMP} RCM={RCM:.0f} scenarios={N}")
