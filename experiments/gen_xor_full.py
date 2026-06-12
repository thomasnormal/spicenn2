#!/usr/bin/env python3
# XOR fully-transistor trainer, fully parameterized via env vars for the candidate sweep.
import random, sys, os as _os
random.seed(int(_os.environ.get("SEED","7")))
def ef(k,d): return float(_os.environ.get(k,d))
def ei(k,d): return int(_os.environ.get(k,d))
H    = ei("H",6)
Ts   = ef("TS","1.0"); dt = ef("DTFRAC","0.15")*Ts
NTRAIN = int(sys.argv[1]) if len(sys.argv)>1 else 80
Nstp = NTRAIN
Ke   = float(sys.argv[2]) if len(sys.argv)>2 else 0.15
Cg   = float(sys.argv[3]) if len(sys.argv)>3 else 2e-3
VC   = ef("VC","1.8"); Gs=ef("GS","1.2"); INIT=ef("INIT","0.6")
VREFH= ef("VREFH","0.5"); VREFO=ef("VREFO","0.5")
RTH  = ef("RTH","8e3"); RTO=ef("RTO","7e3"); RH=ef("RH","2.3e3")
VG0  = ef("VG0","0.7"); LO=ef("LO","0.7"); HI=ef("HI","1.3")
LOu,HIu = 0.3,1.0; Vd0=ef("VD0","1.2"); bb=0.05
VTOSYN=ef("VTOSYN","0.5"); RDECAY=ef("RDECAY","0.0"); BWD=_os.environ.get("BWD","fa")
ACT=_os.environ.get("ACT","relu2")            # relu2 (square-law, current) | tanh (bounded diff-pair)
VMID=ef("VMID","0.55"); VTB=ef("VTB","1.05"); TW=ef("TW","20")  # diff-pair center, tail bias, tail width
VMIDSP=ef("VMIDSP","0.0")   # spread of per-unit tanh centers -> diverse hidden features
pat=[((0,0),0.0),((0,1),1.0),((1,0),1.0),((1,1),0.0)]
def pwl(sel):
    pts=[]
    for k in range(Nstp):
        v=sel(pat[k%4]); t0=k*Ts; pts+=[(t0,v),(t0+Ts-dt,v)]
    pts.append((Nstp*Ts,pts[-1][1]))
    return " ".join(f"{t:.4f} {v:.4f}" for t,v in pts)
A1=pwl(lambda s: HI if s[0][0] else LO); A2=pwl(lambda s: HI if s[0][1] else LO)
AU1=pwl(lambda s: HIu if s[0][0] else LOu); AU2=pwl(lambda s: HIu if s[0][1] else LOu); TG=pwl(lambda s:s[1])
L=[]; ic=[]; w=lambda s:L.append(s)
w("* XOR fully-transistor trainer (parameterized)")
w(f".model NSYN NMOS (LEVEL=1 VTO={VTOSYN} KP=50u  LAMBDA=0    GAMMA=0 PHI=0.7)")
w(".model NMIR NMOS (LEVEL=1 VTO=0.5 KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)")
w(".model PMIR PMOS (LEVEL=1 VTO=-0.5 KP=40u LAMBDA=0.02 GAMMA=0 PHI=0.7)")
w(".model NREL NMOS (LEVEL=1 VTO=0.5 KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)")
w(f".param Cg={Cg} Ke={Ke} Vd0={Vd0} VG0={VG0} VC={VC} bb={bb}")
w("Vvdd vdd 0 3"); w(f"Vrefh vrefh 0 {VREFH}"); w(f"Vrefo vrefo 0 {VREFO}")
w(f"Vg0 vg0 0 {VG0}"); w(f"Vvc vcn 0 {VC}")
if ACT=="tanh": w(f"Vvmid vmid 0 {VMID}"); w(f"Vvtb vtb 0 {VTB}")
w(f"Va1 a1 0 PWL({A1})"); w(f"Va2 a2 0 PWL({A2})"); w(f"Vtg tg 0 PWL({TG})")
w(f"Vau1 au1 0 PWL({AU1})"); w(f"Vau2 au2 0 PWL({AU2})"); w(f"Vaub aub 0 dc {HIu}"); w(f"Vbx bx 0 dc {HI}")
def synapse(idx,vin,cp,cn,w0):
    gp=f"gp_{idx}"; w(f"C_{gp} {gp} 0 {{Cg}}")
    w(f"Mp_{idx} {vin} {gp} {cp} 0 NSYN W=10u L=1u")
    w(f"Mn_{idx} {vin} vcn {cn} 0 NSYN W=10u L=1u")
    if RDECAY>0: w(f"Rdk_{idx} {gp} vcn {RDECAY}")
    ic.append((gp, round(VC+Gs*w0,4))); return gp
def subtractor(tag,cp,cn,vout,Rt,vrefn):
    w(f"Msa_{tag} {cn} {cn} 0 0 NMIR W=20u L=1u"); w(f"Msb_{tag} {vout} {cn} 0 0 NMIR W=20u L=1u")
    w(f"Msc_{tag} {cp} {cp} 0 0 NMIR W=20u L=1u"); w(f"Msd_{tag} dpf_{tag} {cp} 0 0 NMIR W=20u L=1u")
    w(f"Mse_{tag} dpf_{tag} dpf_{tag} vdd vdd PMIR W=60u L=1u"); w(f"Msf_{tag} {vout} dpf_{tag} vdd vdd PMIR W=60u L=1u")
    w(f"R_{tag} {vout} {vrefn} {Rt}")
def ota(tag,ep,en,actg,gp,tw=20):
    w(f"Mt_{tag} ts_{tag} {actg} 0 0 NMIR W={tw}u L=1u")
    w(f"Mo1_{tag} da_{tag} {en} ts_{tag} 0 NMIR W=20u L=1u"); w(f"Mo2_{tag} {gp} {ep} ts_{tag} 0 NMIR W=20u L=1u")
    w(f"Mo3_{tag} da_{tag} da_{tag} vdd vdd PMIR W=60u L=1u"); w(f"Mo4_{tag} {gp} da_{tag} vdd vdd PMIR W=60u L=1u")
def ota_relu(tag,ep,en,xg,rg,gp):
    # tail conducts only when input xg is on AND unit-on rg is high ~ x_i * ReLU'. Wide to offset series stacking.
    w(f"Mtt_{tag} ts_{tag} {rg} mid_{tag} 0 NMIR W=48u L=1u")
    w(f"Mtb_{tag} mid_{tag} {xg} 0 0 NMIR W=48u L=1u")
    w(f"Mo1_{tag} da_{tag} {en} ts_{tag} 0 NMIR W=20u L=1u"); w(f"Mo2_{tag} {gp} {ep} ts_{tag} 0 NMIR W=20u L=1u")
    w(f"Mo3_{tag} da_{tag} da_{tag} vdd vdd PMIR W=60u L=1u"); w(f"Mo4_{tag} {gp} da_{tag} vdd vdd PMIR W=60u L=1u")
def comparator(j):
    # near-rail "unit-on" signal uon_j: high when z1_j > 0.5 (the ReLU threshold), else low. Step ReLU'.
    w(f"Mct_{j} ctail_{j} vcbias 0 0 NMIR W=10u L=1u")
    w(f"Mca_{j} cna_{j} z1_{j} ctail_{j} 0 NSYN W=20u L=1u")
    w(f"Mcb_{j} uon_{j} vthr   ctail_{j} 0 NSYN W=20u L=1u")
    w(f"Mcp1_{j} cna_{j} cna_{j} vdd vdd PMIR W=40u L=1u")
    w(f"Mcp2_{j} uon_{j} cna_{j} vdd vdd PMIR W=40u L=1u")
hidw={}
for j in range(1,H+1):
    cps={}
    for i,vin in (("1","a1"),("2","a2"),("b","bx")):
        cps[i]=synapse(f"1_{j}_{i}",vin,f"colp{j}",f"coln{j}",round(random.uniform(-INIT,INIT),3))
    hidw[j]=cps; subtractor(f"h{j}",f"colp{j}",f"coln{j}",f"z1_{j}",RTH,"vrefh")
    if ACT=="tanh":
        # bounded neuron: differential pair. Tail current caps M1's current -> hv saturates.
        # per-unit vmid (VMIDSP spread) so units transition at different z1 -> diverse features.
        vmj = f"vmid_{j}" if VMIDSP>0 else "vmid"
        if VMIDSP>0: w(f"Vvm_{j} {vmj} 0 {round(VMID+VMIDSP*((j-1)/max(1,H-1)-0.5),4)}")  # deterministic spread
        w(f"Mr_{j}  hdd_{j} z1_{j} tail_{j} 0 NREL W=12u L=1u")
        w(f"Mr2_{j} dmp_{j}  {vmj}  tail_{j} 0 NREL W=12u L=1u")
        w(f"Mtl_{j} tail_{j} vtb 0 0 NREL W={TW}u L=1u")
        w(f"Mdm_{j} dmp_{j} dmp_{j} vdd vdd PMIR W=40u L=1u")
    else:
        w(f"Mr_{j} hdd_{j} z1_{j} 0 0 NREL W=12u L=1u")
    w(f"Mrp1_{j} hdd_{j} hdd_{j} vdd vdd PMIR W=40u L=1u"); w(f"Mrp2_{j} hv_{j} hdd_{j} vdd vdd PMIR W=40u L=1u")
    w(f"Rhv_{j} hv_{j} vg0 {RH}")
outw={}
for j in range(1,H+1): outw[j]=synapse(f"2_{j}",f"hv_{j}","colpo","colno",round(random.uniform(-INIT,INIT),3))
outw["b"]=synapse("2_b","bx","colpo","colno",round(random.uniform(-INIT,INIT),3))
subtractor("o","colpo","colno","y",RTO,"vrefo")
w("Vrefd vrefd 0 1.2"); w("Vboff boff 0 1.9")
w("Msf2 d2e dpf_o vdd vdd PMIR W=60u L=1u"); w("Msb2 d2e colno 0 0 NMIR W=20u L=1u")
w("Moff d2e boff vdd vdd PMIR W=10u L=1u"); w("Mtge d2e tg 0 0 NMIR W=10u L=1u"); w(f"Rde d2e vrefd {RTO}")
OWTW=ef("OWTW","20")  # slow the output layer so hidden XOR-features form before the output collapses
for j in range(1,H+1): ota(f"o{j}","d2e","vrefd",f"hv_{j}",outw[j],tw=OWTW)
ota("ob","d2e","vrefd","bx",outw["b"],tw=OWTW)
au={"1":"au1","2":"au2","b":"aub"}
if BWD=="exact":
    rpm=_os.environ.get("RPMODE","step")
    for j in range(1,H+1):
        if rpm=="one": w(f"Brp_{j} rp_{j} 0 V = 1")
        else: w(f"Brp_{j} rp_{j} 0 V = 0.5*(1+(v(z1_{j})-0.5)/sqrt((v(z1_{j})-0.5)*(v(z1_{j})-0.5)+bb*bb))")
        w(f"Bd1_{j} d1_{j} 0 V = (v(d2e)-1.2)*(v({outw[j]})-VC)*v(rp_{j})")
        w(f"Bdp_{j} d1p_{j} 0 V = Vd0 + Ke*v(d1_{j})"); w(f"Bdn_{j} d1n_{j} 0 V = Vd0 - Ke*v(d1_{j})")
    for j in range(1,H+1):
        for i in ("1","2","b"): ota(f"u{j}{i}",f"d1p_{j}",f"d1n_{j}",au[i],hidw[j][i])
elif BWD=="trans":
    # FULLY-TRANSISTOR exact backward: transpose-read w2_j*delta2 (4-T diff cell sharing the
    # output weight gate) -> keystone -> d1pre_j ; ReLU' supplied by the z1-gated OTA tail.
    RTbk=ef("RTBK","3e3")
    # buffer the error node (matched low-Vt source followers) so the transpose read can't load d2e
    w("Vref12 vref12 0 1.2")
    w("Mbfe vdd d2e    d2eb   0 NSYN W=200u L=1u"); w("Rbfe d2eb 0 20e3")
    w("Mbfr vdd vref12 vrefdb 0 NSYN W=200u L=1u"); w("Rbfr vrefdb 0 20e3")
    for j in range(1,H+1):
        gpj=outw[j]
        w(f"MpTa_{j} d2eb   {gpj} ctp_{j} 0 NSYN W=10u L=1u")
        w(f"MnTb_{j} vrefdb vcn  ctp_{j} 0 NSYN W=10u L=1u")
        w(f"MnTa_{j} d2eb   vcn  ctn_{j} 0 NSYN W=10u L=1u")
        w(f"MpTb_{j} vrefdb {gpj} ctn_{j} 0 NSYN W=10u L=1u")
        subtractor(f"bk{j}", f"ctp_{j}", f"ctn_{j}", f"d1pre_{j}", RTbk, "vrefd")
    for j in range(1,H+1):
        for i in ("1","2","b"): ota_relu(f"u{j}{i}", f"d1pre_{j}", "vrefd", au[i], f"z1_{j}", hidw[j][i])
elif BWD=="transrp":
    # DIAGNOSTIC: transistor transpose multiply, but clean step ReLU' (isolates which half fails)
    RTbk=ef("RTBK","32e3")
    w("Vref12 vref12 0 1.2")
    w("Mbfe vdd d2e d2eb 0 NSYN W=200u L=1u"); w("Rbfe d2eb 0 20e3")
    w("Mbfr vdd vref12 vrefdb 0 NSYN W=200u L=1u"); w("Rbfr vrefdb 0 20e3")
    for j in range(1,H+1):
        gpj=outw[j]
        w(f"MpTa_{j} d2eb {gpj} ctp_{j} 0 NSYN W=10u L=1u"); w(f"MnTb_{j} vrefdb vcn ctp_{j} 0 NSYN W=10u L=1u")
        w(f"MnTa_{j} d2eb vcn ctn_{j} 0 NSYN W=10u L=1u"); w(f"MpTb_{j} vrefdb {gpj} ctn_{j} 0 NSYN W=10u L=1u")
        subtractor(f"bk{j}", f"ctp_{j}", f"ctn_{j}", f"d1pre_{j}", RTbk, "vrefd")
    for j in range(1,H+1):
        w(f"Brp_{j} rp_{j} 0 V = 0.5*(1+(v(z1_{j})-0.5)/sqrt((v(z1_{j})-0.5)*(v(z1_{j})-0.5)+bb*bb))")
        w(f"Bd1_{j} d1_{j} 0 V = (v(d1pre_{j})-1.2)*v(rp_{j})")
        w(f"Bdp_{j} d1p_{j} 0 V = Vd0 + Ke*v(d1_{j})"); w(f"Bdn_{j} d1n_{j} 0 V = Vd0 - Ke*v(d1_{j})")
    for j in range(1,H+1):
        for i in ("1","2","b"): ota(f"u{j}{i}",f"d1p_{j}",f"d1n_{j}",au[i],hidw[j][i])
elif BWD in ("trans2","trans2rp"):
    # FULLY-TRANSISTOR exact backward with DIFFERENTIAL error so the transpose survives delta2<0.
    RTbk=ef("RTBK","16e3")
    # build d2e_n = 1.2 - delta2 (= vrefd + (tg - y)) via mirror copy of the error, roles swapped
    w("Mqa d2en_ps colno 0 0 NMIR W=20u L=1u")        # sense I-_out
    w("Mqb d2en_ps tg    0 0 NMIR W=10u L=1u")        # sense I_tg
    w("Mpd d2en_ps d2en_ps vdd vdd PMIR W=60u L=1u")  # PMOS diode (sum)
    w("Mpm d2e_n   d2en_ps vdd vdd PMIR W=60u L=1u")  # mirror -> source (I-_out+I_tg) into d2e_n
    w("Mqc d2en_ns dpf_o vdd vdd PMIR W=60u L=1u")    # sense I+_out
    w("Mqd d2en_ns boff  vdd vdd PMIR W=10u L=1u")    # sense I_off
    w("Mnd d2en_ns d2en_ns 0 0 NMIR W=20u L=1u")      # NMOS diode (sum)
    w("Mnm d2e_n   d2en_ns 0 0 NMIR W=20u L=1u")      # mirror -> sink (I+_out+I_off) from d2e_n
    w(f"Rden d2e_n vrefd {RTO}")
    # matched low-Vt followers: both buffered drains stay above the column for either sign of delta2
    w("Mbep vdd d2e   d2e_pb 0 NSYN W=200u L=1u"); w("Rbep d2e_pb 0 20e3")
    w("Mben vdd d2e_n d2e_nb 0 NSYN W=200u L=1u"); w("Rben d2e_nb 0 20e3")
    for j in range(1,H+1):
        gpj=outw[j]
        w(f"MpTa_{j} d2e_pb {gpj} ctp_{j} 0 NSYN W=10u L=1u"); w(f"MnTb_{j} d2e_nb vcn ctp_{j} 0 NSYN W=10u L=1u")
        w(f"MnTa_{j} d2e_pb vcn  ctn_{j} 0 NSYN W=10u L=1u"); w(f"MpTb_{j} d2e_nb {gpj} ctn_{j} 0 NSYN W=10u L=1u")
        subtractor(f"bk{j}", f"ctp_{j}", f"ctn_{j}", f"d1pre_{j}", RTbk, "vrefd")
    if BWD=="trans2rp":
        for j in range(1,H+1):
            w(f"Brp_{j} rp_{j} 0 V = 0.5*(1+(v(z1_{j})-0.5)/sqrt((v(z1_{j})-0.5)*(v(z1_{j})-0.5)+bb*bb))")
            w(f"Bd1_{j} d1_{j} 0 V = (v(d1pre_{j})-1.2)*v(rp_{j})")
            w(f"Bdp_{j} d1p_{j} 0 V = Vd0 + Ke*v(d1_{j})"); w(f"Bdn_{j} d1n_{j} 0 V = Vd0 - Ke*v(d1_{j})")
        for j in range(1,H+1):
            for i in ("1","2","b"): ota(f"u{j}{i}",f"d1p_{j}",f"d1n_{j}",au[i],hidw[j][i])
    elif ACT=="tanh" and _os.environ.get("TBWD","comp")=="plain":
        for j in range(1,H+1):
            for i in ("1","2","b"): ota(f"u{j}{i}", f"d1pre_{j}", "vrefd", au[i], hidw[j][i])
    else:
        # comparator ReLU'-style gate (proven backward). For tanh, threshold = VMID (center).
        w(f"Vthr vthr 0 {VMID if ACT=='tanh' else 0.5}"); w(f"Vcbias vcbias 0 {ef('VCBIAS','1.0')}")
        for j in range(1,H+1): comparator(j)
        for j in range(1,H+1):
            for i in ("1","2","b"): ota_relu(f"u{j}{i}", f"d1pre_{j}", "vrefd", au[i], f"uon_{j}", hidw[j][i])
else:
    rfa=random.Random(int(_os.environ.get("FASEED","1"))); fa=[0]+[rfa.choice([-1,1]) for _ in range(H)]
    for j in range(1,H+1):
        epj,enj=("d2e","vrefd") if fa[j]>0 else ("vrefd","d2e")
        for i in ("1","2","b"): ota(f"u{j}{i}",epj,enj,au[i],hidw[j][i])
WN=ef("WNOISE","0.0")
if WN>0:
    for g in [hidw[j][i] for j in range(1,H+1) for i in ("1","2","b")]+[outw[j] for j in range(1,H+1)]+[outw["b"]]:
        w(f"In_{g} {g} 0 TRNOISE({WN} {round(dt,4)} 0 0)")
ICW=_os.environ.get("ICW","")
if ICW:
    import numpy as _np
    _wv=_np.loadtxt(ICW); _wv=_wv[-1,1::2] if _wv.ndim>1 else _wv[1::2]
    ic=[(n,round(float(_wv[k]),4)) for k,(n,_) in enumerate(ic)]
w(".ic "+" ".join(f"v({n})={v}" for n,v in ic))
w(".control")
if WN>0: w(f"  set rndseed={ei('NSEED','1')}")
w(f"  tran {dt} {Nstp*Ts} uic")
hvt=" ".join(f"v(hv_{j})" for j in range(1,min(H,6)+1))
w(f"  wrdata xor_full_trace.txt v(y) v(a1) v(a2) v(tg) "+hvt)
allgp=[hidw[j][i] for j in range(1,H+1) for i in ("1","2","b")]+[outw[j] for j in range(1,H+1)]+[outw["b"]]
w("  wrdata weights_full.txt "+" ".join(f"v({g})" for g in allgp)); w("  echo FULL_OK")
w(".endc"); w(".end")
open("xor_full.cir","w").write("\n".join(L)+"\n")
print(f"H={H} NT={NTRAIN} VTOSYN={VTOSYN} VC={VC} GS={Gs} RTH={RTH} VREFH={VREFH} RH={RH} INIT={INIT} LO={LO} HI={HI} RDECAY={RDECAY}")
