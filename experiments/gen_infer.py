#!/usr/bin/env python3
# Frozen-weight inference, env-aware so it matches whatever the trainer used.
import numpy as np, os as _os
def ef(k,d): return float(_os.environ.get(k,d))
def ei(k,d): return int(_os.environ.get(k,d))
H=ei("H",6); VC=ef("VC","1.8"); Gs=ef("GS","1.2")
VREFH=ef("VREFH","0.5"); VREFO=ef("VREFO","0.5"); RTH=ef("RTH","8e3"); RTO=ef("RTO","7e3"); RH=ef("RH","2.3e3")
VG0=ef("VG0","0.7"); LO=ef("LO","0.7"); HI=ef("HI","1.3"); VTOSYN=ef("VTOSYN","0.5")
ACT=_os.environ.get("ACT","relu2"); VMID=ef("VMID","0.55"); VTB=ef("VTB","1.05"); TW=ef("TW","8"); VMIDSP=ef("VMIDSP","0.0")
_raw=np.loadtxt(_os.environ.get("WFILE","weights_full.txt")); _avg=ei("AVGW","1")
if _raw.ndim==1: _raw=_raw[None,:]
W=_raw[-_avg:,1::2].mean(axis=0)
_pert=ef("PERTURB","0.0")
if _pert>0:
    _rng=np.random.default_rng(ei("PSEED","0")); W=W+_rng.normal(0,_pert,size=W.shape)
Ts=0.2; dt=0.02
pat=[((0,0),0.0),((0,1),1.0),((1,0),1.0),((1,1),0.0)]
def pwl(sel):
    pts=[]; n=len(pat)
    for k in range(n):
        v=sel(pat[k]); t0=k*Ts; pts+=[(t0,v),(t0+Ts-dt,v)]
    pts.append((n*Ts,pts[-1][1])); return " ".join(f"{t:.4f} {v:.4f}" for t,v in pts)
A1=pwl(lambda s: HI if s[0][0] else LO); A2=pwl(lambda s: HI if s[0][1] else LO); TG=pwl(lambda s:s[1])
L=[]; w=lambda s:L.append(s)
w("* frozen inference")
w(f".model NSYN NMOS (LEVEL=1 VTO={VTOSYN} KP=50u LAMBDA=0 GAMMA=0 PHI=0.7)")
w(".model NMIR NMOS (LEVEL=1 VTO=0.5 KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)")
w(".model PMIR PMOS (LEVEL=1 VTO=-0.5 KP=40u LAMBDA=0.02 GAMMA=0 PHI=0.7)")
w(".model NREL NMOS (LEVEL=1 VTO=0.5 KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)")
w("Vvdd vdd 0 3"); w(f"Vrefh vrefh 0 {VREFH}"); w(f"Vrefo vrefo 0 {VREFO}"); w(f"Vg0 vg0 0 {VG0}"); w(f"Vvc vcn 0 {VC}")
if ACT=="tanh": w(f"Vvmid vmid 0 {VMID}"); w(f"Vvtb vtb 0 {VTB}")
w(f"Va1 a1 0 PWL({A1})"); w(f"Va2 a2 0 PWL({A2})"); w(f"Vtg tg 0 PWL({TG})"); w(f"Vbx bx 0 dc {HI}")
def syn(idx,vin,cp,cn,val):
    w(f"Vg_{idx} g_{idx} 0 {val:.4f}"); w(f"Mp_{idx} {vin} g_{idx} {cp} 0 NSYN W=10u L=1u"); w(f"Mn_{idx} {vin} vcn {cn} 0 NSYN W=10u L=1u")
def sub(tag,cp,cn,vout,Rt,vrefn):
    w(f"Msa_{tag} {cn} {cn} 0 0 NMIR W=20u L=1u"); w(f"Msb_{tag} {vout} {cn} 0 0 NMIR W=20u L=1u")
    w(f"Msc_{tag} {cp} {cp} 0 0 NMIR W=20u L=1u"); w(f"Msd_{tag} dpf_{tag} {cp} 0 0 NMIR W=20u L=1u")
    w(f"Mse_{tag} dpf_{tag} dpf_{tag} vdd vdd PMIR W=60u L=1u"); w(f"Msf_{tag} {vout} dpf_{tag} vdd vdd PMIR W=60u L=1u")
    w(f"R_{tag} {vout} {vrefn} {Rt}")
k=0
for j in range(1,H+1):
    for i,vin in (("1","a1"),("2","a2"),("b","bx")): syn(f"1_{j}_{i}",vin,f"colp{j}",f"coln{j}",W[k]); k+=1
    sub(f"h{j}",f"colp{j}",f"coln{j}",f"z1_{j}",RTH,"vrefh")
    if ACT=="tanh":
        vmj = f"vmid_{j}" if VMIDSP>0 else "vmid"
        if VMIDSP>0: w(f"Vvm_{j} {vmj} 0 {round(VMID+VMIDSP*((j-1)/max(1,H-1)-0.5),4)}")
        w(f"Mr_{j}  hdd_{j} z1_{j} tail_{j} 0 NREL W=12u L=1u")
        w(f"Mr2_{j} dmp_{j}  {vmj}  tail_{j} 0 NREL W=12u L=1u")
        w(f"Mtl_{j} tail_{j} vtb 0 0 NREL W={TW}u L=1u")
        w(f"Mdm_{j} dmp_{j} dmp_{j} vdd vdd PMIR W=40u L=1u")
    else:
        w(f"Mr_{j} hdd_{j} z1_{j} 0 0 NREL W=12u L=1u")
    w(f"Mrp1_{j} hdd_{j} hdd_{j} vdd vdd PMIR W=40u L=1u"); w(f"Mrp2_{j} hv_{j} hdd_{j} vdd vdd PMIR W=40u L=1u"); w(f"Rhv_{j} hv_{j} vg0 {RH}")
for j in range(1,H+1): syn(f"2_{j}",f"hv_{j}","colpo","colno",W[k]); k+=1
syn("2_b","bx","colpo","colno",W[k]); k+=1
sub("o","colpo","colno","y",RTO,"vrefo")
w(".control"); w(f"  tran {dt} {len(pat)*Ts} uic")
hvt=" ".join(f"v(hv_{j})" for j in range(1,min(H,6)+1))
w(f"  wrdata xor_infer_trace.txt v(y) v(a1) v(a2) v(tg) "+hvt); w("  echo INFER_OK")
w(".endc"); w(".end")
open("xor_infer.cir","w").write("\n".join(L)+"\n")
print(f"infer H={H} weights={k}")
