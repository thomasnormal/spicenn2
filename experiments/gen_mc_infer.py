#!/usr/bin/env python3
# Frozen-weight multi-class inference: forward-only deck, presents a held-out test set,
# logs y_c per pattern. Mirrors gen_mc.py's forward construction and weight order exactly.
import numpy as np, os as _os
def ef(k,d): return float(_os.environ.get(k,d))
def ei(k,d): return int(_os.environ.get(k,d))
D=ei("D",4); H=ei("H",8); C=ei("C",3)
VC=ef("VC","1.8"); Gs=ef("GS","1.2")
VREFH=ef("VREFH","0.5"); VREFO=ef("VREFO","0.5"); RTH=ef("RTH","8e3"); RTO=ef("RTO","7e3"); RH=ef("RH","8e3")
VG0=ef("VG0","0.7"); VTOSYN=ef("VTOSYN","0.2")
INLO=ef("INLO","0.5"); INHI=ef("INHI","1.5")
NTE=ei("NTE",20)

TAG=_os.environ.get("TAG","")
DATAFILE_TE=_os.environ.get("DATAFILE_TE","")
if DATAFILE_TE:
    _d=np.load(DATAFILE_TE); Xte=_d["X"].astype(float); Yte=_d["Y"].astype(int)
    mn=np.zeros(Xte.shape[1]); mx=np.ones(Xte.shape[1])
else:
    dat=np.load("mc_data.npz"); mn=dat["mn"]; mx=dat["mx"]; SPREAD=float(dat["spread"])
    def make_blobs(C,D,n,seed,spread):
        cen=np.random.default_rng(2024).normal(0,1.5,(C,D))
        rng=np.random.default_rng(seed); X=[];Y=[]
        for c in range(C): X.append(cen[c]+rng.normal(0,spread,(n,D))); Y+=[c]*n
        return np.vstack(X), np.array(Y)
    Xte,Yte=make_blobs(C,D,NTE,ei("TESTSEED",2),SPREAD)
def scale_fwd(X): return INLO+(X-mn)/(mx-mn+1e-9)*(INHI-INLO)
Xf=scale_fwd(Xte)
np.save("mc_test_labels.npy", Yte)

# frozen weights (last training row), same order as gen_mc allgp
_raw=np.loadtxt(_os.environ.get("WFILE","mc_weights.txt")); _avg=ei("AVGW","1")
if _raw.ndim==1: _raw=_raw[None,:]
W=_raw[-_avg:,1::2].mean(axis=0)

Ts=0.2; dt=0.02; Npat=len(Xf)
def pwl_from(vals):
    pts=[]
    for k in range(Npat):
        v=vals[k]; t0=k*Ts; pts+=[(t0,v),(t0+Ts-dt,v)]
    pts.append((Npat*Ts,pts[-1][1])); return " ".join(f"{t:.4f} {v:.4f}" for t,v in pts)

L=[]; w=lambda s:L.append(s)
w("* frozen multi-class inference")
w(f".model NSYN NMOS (LEVEL=1 VTO={VTOSYN} KP=50u LAMBDA=0 GAMMA=0 PHI=0.7)")
w(".model NMIR NMOS (LEVEL=1 VTO=0.5 KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)")
w(".model PMIR PMOS (LEVEL=1 VTO=-0.5 KP=40u LAMBDA=0.02 GAMMA=0 PHI=0.7)")
w(".model NREL NMOS (LEVEL=1 VTO=0.5 KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)")
w("Vvdd vdd 0 3"); w(f"Vrefh vrefh 0 {VREFH}"); w(f"Vrefo vrefo 0 {VREFO}"); w(f"Vg0 vg0 0 {VG0}"); w(f"Vvc vcn 0 {VC}")
w(f"Vbx bx 0 dc {INHI}")
ACT=_os.environ.get("ACT","relu2")
VMID=ef("VMID","0.55"); VTB=ef("VTB","1.05"); TW=ef("TW","20"); VMIDSP=ef("VMIDSP","0.0")
if ACT=="tanh": w(f"Vvmid vmid 0 {VMID}"); w(f"Vvtb vtb 0 {VTB}")
_used_feats=None
if int(_os.environ.get("DIRECT","0")) and int(_os.environ.get("ROUTSP","0")) and _os.path.exists("mc_sel.txt"):
    _used_feats=set()
    with open("mc_sel.txt") as _sf:
        for _line in _sf: _used_feats.update(int(x)+1 for x in _line.split())
for i in range(1,D+1):
    if _used_feats is not None and i not in _used_feats: continue
    w(f"Va{i} a{i} 0 PWL({pwl_from([float(x[i-1]) for x in Xf])})")
_SYNW=ef("SYNW","10"); _RSDEG=_os.environ.get("RSDEG","")
def syn(idx,vin,cp,cn,val):
    w(f"Vg_{idx} g_{idx} 0 {val:.4f}")
    if _RSDEG:   # match training source-degeneration
        w(f"Mp_{idx} {vin} g_{idx} sp_{idx} 0 NSYN W={_SYNW}u L=1u"); w(f"Rsp_{idx} sp_{idx} {cp} {_RSDEG}")
        w(f"Mn_{idx} {vin} vcn sn_{idx} 0 NSYN W={_SYNW}u L=1u"); w(f"Rsn_{idx} sn_{idx} {cn} {_RSDEG}")
    else:
        w(f"Mp_{idx} {vin} g_{idx} {cp} 0 NSYN W={_SYNW}u L=1u"); w(f"Mn_{idx} {vin} vcn {cn} 0 NSYN W={_SYNW}u L=1u")
def sub(tag,cp,cn,vout,Rt,vrefn):
    w(f"Msa_{tag} {cn} {cn} 0 0 NMIR W=20u L=1u"); w(f"Msb_{tag} {vout} {cn} 0 0 NMIR W=20u L=1u")
    w(f"Msc_{tag} {cp} {cp} 0 0 NMIR W=20u L=1u"); w(f"Msd_{tag} dpf_{tag} {cp} 0 0 NMIR W=20u L=1u")
    w(f"Mse_{tag} dpf_{tag} dpf_{tag} vdd vdd PMIR W=60u L=1u"); w(f"Msf_{tag} {vout} dpf_{tag} vdd vdd PMIR W=60u L=1u")
    w(f"R_{tag} {vout} {vrefn} {Rt}")
DIRECT=int(_os.environ.get("DIRECT","0"))
inputs_h=[(str(i),f"a{i}") for i in range(1,D+1)]+[("b","bx")]
inputs_o=([(str(i),f"a{i}") for i in range(1,D+1)] if DIRECT
          else [(str(j),f"hv_{j}") for j in range(1,H+1)])+[("b","bx")]
# ROUTSP sparse readout: load the per-class feature selection saved by gen_mc (identical connectivity).
ROUTSP=int(_os.environ.get("ROUTSP","0"))
if ROUTSP and DIRECT:
    oc_inputs={}
    with open("mc_sel.txt") as _f:
        for c,_line in enumerate(_f,1):
            _top=[int(x) for x in _line.split()]
            oc_inputs[c]=[(str(i+1),f"a{i+1}") for i in _top]+[("b","bx")]
else:
    oc_inputs={c: inputs_o for c in range(1,C+1)}
CONN=_os.environ.get("CONN","dense")
if CONN=="patch2x2":
    patches=[[1,2,5,6],[3,4,7,8],[9,10,13,14],[11,12,15,16]]; nf=max(1,H//len(patches))
    hid_inputs={j:[(str(i),f"a{i}") for i in patches[(j-1)//nf % len(patches)]]+[("b","bx")] for j in range(1,H+1)}
elif CONN=="rf3x3":
    fields=[[4*pr+pc+1, 4*pr+pc+2, 4*pr+pc+5, 4*pr+pc+6] for pr in range(3) for pc in range(3)]; nf=max(1,H//9)
    hid_inputs={j:[(str(i),f"a{i}") for i in fields[(j-1)//nf % 9]]+[("b","bx")] for j in range(1,H+1)}
else:
    hid_inputs={j:inputs_h for j in range(1,H+1)}
k=0
for j in ([] if DIRECT else range(1,H+1)):
    for i,vin in hid_inputs[j]: syn(f"1_{j}_{i}",vin,f"colp{j}",f"coln{j}",W[k]); k+=1
    sub(f"h{j}",f"colp{j}",f"coln{j}",f"z1_{j}",RTH,"vrefh")
    if ACT=="tanh":
        vmj=f"vmid_{j}" if VMIDSP>0 else "vmid"
        if VMIDSP>0: w(f"Vvm_{j} {vmj} 0 {round(VMID+VMIDSP*((j-1)/max(1,H-1)-0.5),4)}")
        w(f"Mr_{j}  hdd_{j} z1_{j} tail_{j} 0 NREL W=12u L=1u")
        w(f"Mr2_{j} dmp_{j}  {vmj}  tail_{j} 0 NREL W=12u L=1u")
        w(f"Mtl_{j} tail_{j} vtb 0 0 NREL W={TW}u L=1u")
        w(f"Mdm_{j} dmp_{j} dmp_{j} vdd vdd PMIR W=40u L=1u")
    else:
        w(f"Mr_{j} hdd_{j} z1_{j} 0 0 NREL W=12u L=1u")
    w(f"Mrp1_{j} hdd_{j} hdd_{j} vdd vdd PMIR W=40u L=1u"); w(f"Mrp2_{j} hv_{j} hdd_{j} vdd vdd PMIR W=40u L=1u"); w(f"Rhv_{j} hv_{j} vg0 {RH}")
for c in range(1,C+1):
    for j,vin in oc_inputs[c]: syn(f"2_{c}_{j}",vin,f"colpo{c}",f"colno{c}",W[k]); k+=1
    sub(f"o{c}",f"colpo{c}",f"colno{c}",f"y_{c}",RTO,"vrefo")
w(".control"); w(f"  tran {dt} {Npat*Ts} uic")
yt=" ".join(f"v(y_{c})" for c in range(1,C+1))
if ei("PROBE_RAILS",0):
    yt+=" "+" ".join(f"v(colpo{c}) v(colno{c})" for c in range(1,C+1))
w(f"  wrdata mc_infer_trace{TAG}.txt {yt}"); w("  echo INFER_OK")
w(".endc"); w(".end")
open(f"mc_infer{TAG}.cir","w").write("\n".join(L)+"\n")
print(f"infer D={D} H={H} C={C} weights_used={k} npat={Npat} avg={_avg}")
