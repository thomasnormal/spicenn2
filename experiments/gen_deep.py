#!/usr/bin/env python3
# N-LAYER sparse locally-connected in-circuit trainer (generalizes gen_mc.py to depth>1).
# Forward (sparse synapses + keystone + tanh neuron per hidden layer, linear keystone output),
# transpose-read backprop with per-hidden-layer common-mode subtraction (CMSUB) + tanh' gate,
# OTA weight-cap updates, output-error common-mode subtraction (OCMSUB). Zero behavioural sources.
# Topology = a 2x2 stride-2 pyramid (locally connected, fan-in/out<=4) defined by ARCH, matching
# build_sparse.py. Proves multi-layer sparse backprop trains in real ngspice.
import sys, os as _os, numpy as np, random
def ef(k,d): return float(_os.environ.get(k,d))
def ei(k,d): return int(_os.environ.get(k,d))
NTRAIN=int(sys.argv[1]) if len(sys.argv)>1 else 1000
Ke=float(sys.argv[2]) if len(sys.argv)>2 else 0.3
Cg=float(sys.argv[3]) if len(sys.argv)>3 else 5e-4
random.seed(ei("SEED",1)); np.random.seed(ei("SEED",1))
RES=ei("RES",4); C=ei("C",3)
VC=ef("VC","1.8"); Gs=ef("GS","1.2"); INIT=ef("INIT","0.3")
VREFH=ef("VREFH","0.5"); VREFO=ef("VREFO","0.5"); RTH=ef("RTH","8e3"); RTO=ef("RTO","7e3"); RH=ef("RH","8e3")
VG0=ef("VG0","0.7"); VTOSYN=ef("VTOSYN","0.2"); RDECAY=ef("RDECAY","0.0"); RTbk=ef("RTBK","16e3")
VCBIAS=ef("VCBIAS","1.0"); VULK=ef("VULK","0.6"); RULK=ef("RULK","60e3")
INLO=ef("INLO","0.3"); INHI=ef("INHI","1.7")
VMID=ef("VMID","0.5"); VTB=ef("VTB","1.4"); TW=ef("TW","32"); HWTW=ef("HWTW","48")
OWTW=ef("OWTW","20"); RCMB=ef("RCMB","10e3"); ROCM=ef("ROCM","10e3")
CMSUB=ei("CMSUB",1); OCMSUB=ei("OCMSUB",1)
Ts=ef("TS","0.3"); dt=ef("DTFRAC","0.34")*Ts

# ---- data ----
DF=_os.environ.get("DATAFILE","digits_train.npz")
_d=np.load(DF); X=_d["X"].astype(float); Y=_d["Y"].astype(int)
D=X.shape[1]
MLP=_os.environ.get("MLP","")   # e.g. "48,48,48" -> fully-connected deep MLP (dense or FANIN-sparse), D inputs
if not MLP: assert D==RES*RES, f"D={D} != RES^2={RES*RES}"
# present each pattern; cycle order
order=list(range(len(Y)))
seq=[order[k%len(order)] for k in range(NTRAIN)]
seqX=[X[k] for k in seq]; seqY=[Y[k] for k in seq]
Nstp=NTRAIN

# ---- pyramid connectivity (2x2 stride-2, channel growth), fan-in<=4 ----
ARCH=_os.environ.get("ARCH","4:2:2:4,16:2:2:4")   # per layer: Cout:ksize:stride:fanin
def four_regular(nL,nR,k,seed):
    r=np.random.default_rng(seed); edges=set()
    for _ in range(k):
        perm=r.permutation(nR)
        for L in range(nL): edges.add((L,int(perm[L%nR])))
    return edges
# build layer connectivity: returns list of layers; each layer = list over output neurons of input-index lists
def build_layers(RES,ARCH):
    layers=[]; H=W=RES; Cin=1
    for li,part in enumerate(ARCH.split(",")):
        Cout,ks,st,fi=[int(x) for x in part.split(":")]
        Hout=(H-ks)//st+1; Wout=(W-ks)//st+1
        nin=H*W*Cin; nout=Hout*Wout*Cout
        conn=[[] for _ in range(nout)]; r=np.random.default_rng(li+1)
        for io in range(Hout):
            for jo in range(Wout):
                rf=[]
                for di in range(ks):
                    for dj in range(ks):
                        ii=io*st+di; jj=jo*st+dj
                        for c in range(Cin): rf.append((ii*W+jj)*Cin+c)
                rf=np.array(rf); fo=np.zeros(len(rf))
                for co in range(Cout):
                    o=(io*Wout+jo)*Cout+co
                    cand=sorted(range(len(rf)), key=lambda x:(fo[x], r.random()))[:fi]
                    for ci in cand: conn[o].append(int(rf[ci])); fo[ci]+=1
        layers.append((conn,nin,nout)); H,W,Cin=Hout,Wout,Cout
    return layers,H*W*Cin
def build_mlp(D,widths,fanin):
    # fully-connected (or FANIN-sparse) deep MLP: layers of given widths, each unit reads `fanin`
    # (or all) of the previous layer. Returns same (conn,nin,nout) format as build_layers.
    layers=[]; nin=D
    for li,wd in enumerate(widths):
        conn=[[] for _ in range(wd)]; rr=np.random.default_rng(100+li)
        for o in range(wd):
            if fanin and nin>fanin: conn[o]=sorted(int(x) for x in rr.choice(nin,fanin,replace=False))
            else:                   conn[o]=list(range(nin))
        layers.append((conn,nin,wd)); nin=wd
    return layers,nin
if MLP:
    _wd=[int(x) for x in MLP.split(",")]; _FANIN=ei("FANIN",0)
    hidden_layers,nfeat=build_mlp(D,_wd,_FANIN)
else:
    hidden_layers,nfeat=build_layers(RES,ARCH)
# output readout: each class reads ROUTK features (sparse), or ALL if ROUTK>=nfeat (dense)
ROUTK=ei("ROUTK",4)
r=np.random.default_rng(99); out_conn=[list(r.choice(nfeat,size=min(ROUTK,nfeat),replace=False)) for _ in range(C)]

L=[]; w=lambda s:L.append(s); ic=[]
w("* N-layer sparse in-circuit trainer (pyramid)")
w(f".param Cg={Cg}")
w(f".model NSYN NMOS (LEVEL=1 VTO={VTOSYN} KP=50u LAMBDA=0 GAMMA=0 PHI=0.7)")
w(".model NMIR NMOS (LEVEL=1 VTO=0.5 KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)")
w(".model PMIR PMOS (LEVEL=1 VTO=-0.5 KP=40u LAMBDA=0.02 GAMMA=0 PHI=0.7)")
w(".model NREL NMOS (LEVEL=1 VTO=0.5 KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)")
w("Vvdd vdd 0 3"); w(f"Vrefh vrefh 0 {VREFH}"); w(f"Vrefo vrefo 0 {VREFO}"); w(f"Vg0 vg0 0 {VG0}"); w(f"Vvc vcn 0 {VC}")
w("Vrefd vrefd 0 1.2"); w("Vboff boff 0 1.9")
w(f"Vthr vthr 0 {VMID}"); w(f"Vcbias vcbias 0 {VCBIAS}"); w(f"Vvmid vmid 0 {VMID}"); w(f"Vvtb vtb 0 {VTB}")
if VULK>0: w(f"Vulk vulk 0 {VULK}")
w(f"Vbx bx 0 dc {INHI}")
def pwl(vals):
    pts=[]
    for k in range(Nstp):
        v=vals[k]; t0=k*Ts; pts+=[(t0,v),(t0+Ts-dt,v)]
    pts.append((Nstp*Ts,pts[-1][1])); return " ".join(f"{t:.4f} {v:.4f}" for t,v in pts)
def scale(x): return INLO+(x-X.min())/(X.max()-X.min()+1e-9)*(INHI-INLO)
for i in range(D):
    w(f"Va{i} a{i} 0 PWL({pwl([float(scale(s[i])) for s in seqX])})")
TGHI=ef("TGHI","1.0"); TGLO=ef("TGLO","0.0")
for c in range(C):
    w(f"Vtg{c} tg{c} 0 PWL({pwl([TGHI if (lab==c) else TGLO for lab in seqY])})")

# ---- cells (verbatim from gen_mc) ----
def synapse(idx,vin,cp,cn,w0):
    gp=f"gp_{idx}"; w(f"C_{gp} {gp} 0 {{Cg}}")
    w(f"Mp_{idx} {vin} {gp} {cp} 0 NSYN W=10u L=1u"); w(f"Mn_{idx} {vin} vcn {cn} 0 NSYN W=10u L=1u")
    if RDECAY>0: w(f"Rdk_{idx} {gp} vcn {RDECAY}")
    ic.append((gp, round(VC+Gs*w0,4))); return gp
def subtractor(tag,cp,cn,vout,Rt,vrefn):
    w(f"Msa_{tag} {cn} {cn} 0 0 NMIR W=20u L=1u"); w(f"Msb_{tag} {vout} {cn} 0 0 NMIR W=20u L=1u")
    w(f"Msc_{tag} {cp} {cp} 0 0 NMIR W=20u L=1u"); w(f"Msd_{tag} dpf_{tag} {cp} 0 0 NMIR W=20u L=1u")
    w(f"Mse_{tag} dpf_{tag} dpf_{tag} vdd vdd PMIR W=60u L=1u"); w(f"Msf_{tag} {vout} dpf_{tag} vdd vdd PMIR W=60u L=1u")
    w(f"R_{tag} {vout} {vrefn} {Rt}")
def tanh_neuron(tag,zin):
    w(f"Mr_{tag}  hdd_{tag} {zin} tail_{tag} 0 NREL W=12u L=1u")
    w(f"Mr2_{tag} dmp_{tag}  vmid  tail_{tag} 0 NREL W=12u L=1u")
    w(f"Mtl_{tag} tail_{tag} vtb 0 0 NREL W={TW}u L=1u")
    w(f"Mdm_{tag} dmp_{tag} dmp_{tag} vdd vdd PMIR W=40u L=1u")
    w(f"Mrp1_{tag} hdd_{tag} hdd_{tag} vdd vdd PMIR W=40u L=1u"); w(f"Mrp2_{tag} hv_{tag} hdd_{tag} vdd vdd PMIR W=40u L=1u")
    w(f"Rhv_{tag} hv_{tag} vg0 {RH}")
def ota(tag,ep,en,actg,gp,tw):
    w(f"Mt_{tag} ts_{tag} {actg} 0 0 NMIR W={tw}u L=1u")
    w(f"Mo1_{tag} da_{tag} {en} ts_{tag} 0 NMIR W=20u L=1u"); w(f"Mo2_{tag} {gp} {ep} ts_{tag} 0 NMIR W=20u L=1u")
    w(f"Mo3_{tag} da_{tag} da_{tag} vdd vdd PMIR W=60u L=1u"); w(f"Mo4_{tag} {gp} da_{tag} vdd vdd PMIR W=60u L=1u")
def ota_relu(tag,ep,en,xg,rg,gp):   # update gated by input xg AND derivative-proxy rg
    w(f"Mtt_{tag} ts_{tag} {rg} mid_{tag} 0 NMIR W={HWTW}u L=1u")
    w(f"Mtb_{tag} mid_{tag} {xg} 0 0 NMIR W={HWTW}u L=1u")
    w(f"Mo1_{tag} da_{tag} {en} ts_{tag} 0 NMIR W=20u L=1u"); w(f"Mo2_{tag} {gp} {ep} ts_{tag} 0 NMIR W=20u L=1u")
    w(f"Mo3_{tag} da_{tag} da_{tag} vdd vdd PMIR W=60u L=1u"); w(f"Mo4_{tag} {gp} da_{tag} vdd vdd PMIR W=60u L=1u")
def comparator(tag,zin):
    w(f"Mct_{tag} ctail_{tag} vcbias 0 0 NMIR W=10u L=1u")
    w(f"Mca_{tag} cna_{tag} {zin} ctail_{tag} 0 NSYN W=20u L=1u")
    w(f"Mcb_{tag} uon_{tag} vthr ctail_{tag} 0 NSYN W=20u L=1u")
    w(f"Mcp1_{tag} cna_{tag} cna_{tag} vdd vdd PMIR W=40u L=1u"); w(f"Mcp2_{tag} uon_{tag} cna_{tag} vdd vdd PMIR W=40u L=1u")
    if VULK>0: w(f"Rulk_{tag} uon_{tag} vulk {RULK}")
DBR=ef("DBR","16e3")   # diffbuf load resistor (sets backward-buffer gain)
def diffbuf(tag,vp,vn,outp,outn):
    # Re-center a hidden delta (vp=dpre, vn=d1bar) into a vrefd-centered, low-Z DIFFERENTIAL pair
    # (outp,outn) = vrefd +/- k*(vp-vn). THE depth fix: the deeper transpose multiplier needs its
    # inputs balanced around vrefd (its saturation operating point) like the output's d2e_pb/d2e_nb;
    # reading the raw single-ended dpre (which sits ~0.64, in triode) gave cosine 0.1. v->I via NMIR
    # transconductors, PMOS-mirror the vp/vn currents, pull-up by one & pull-down by the other.
    w(f"MgP_{tag} gP_{tag} {vp} 0 0 NMIR W=20u L=1u");  w(f"MgPm_{tag} gP_{tag} gP_{tag} vdd vdd PMIR W=40u L=1u")
    w(f"MgN_{tag} gN_{tag} {vn} 0 0 NMIR W=20u L=1u");  w(f"MgNm_{tag} gN_{tag} gN_{tag} vdd vdd PMIR W=40u L=1u")
    w(f"MoPa_{tag} {outp} gP_{tag} vdd vdd PMIR W=40u L=1u"); w(f"MoPb_{tag} {outp} {vn} 0 0 NMIR W=20u L=1u")
    w(f"RoP_{tag} {outp} vrefd {DBR}")
    w(f"MoNa_{tag} {outn} gN_{tag} vdd vdd PMIR W=40u L=1u"); w(f"MoNb_{tag} {outn} {vp} 0 0 NMIR W=20u L=1u")
    w(f"RoN_{tag} {outn} vrefd {DBR}")

# ---- FORWARD: hidden layers ----
# node naming: layer L neuron n -> activation node act_{L}_{n}; layer0 inputs = a{i}
gpw={}   # gpw[(L,n)] = list of (input_node, gp_node) for backward transpose
actnode=lambda Lid,n: (f"a{n}" if Lid==0 else f"hv_L{Lid}_{n}")
prev_size=D
for Lid,(conn,nin,nout) in enumerate(hidden_layers,start=1):
    for n in range(nout):
        gps=[]
        for k,ipx in enumerate(conn[n]):
            vin=actnode(Lid-1,ipx)
            gp=synapse(f"{Lid}_{n}_{k}", vin, f"colp_L{Lid}_{n}", f"coln_L{Lid}_{n}", round(random.uniform(-INIT,INIT),3))
            gps.append((ipx,gp))
        # bias synapse (reads bx)
        gpb=synapse(f"{Lid}_{n}_b","bx",f"colp_L{Lid}_{n}",f"coln_L{Lid}_{n}", round(ef("BHID","0.0")+random.uniform(-INIT,INIT),3))
        gpw[(Lid,n)]=gps  # (input neuron idx in prev layer, gp)
        subtractor(f"h_L{Lid}_{n}", f"colp_L{Lid}_{n}", f"coln_L{Lid}_{n}", f"z_L{Lid}_{n}", RTH, "vrefh")
        tanh_neuron(f"L{Lid}_{n}", f"z_L{Lid}_{n}")
nhid=len(hidden_layers)

# ---- FORWARD: output keystones (linear) ----
outgp={}
for c in range(C):
    gps=[]
    for k,f in enumerate(out_conn[c]):
        vin=actnode(nhid,f)
        gp=synapse(f"o_{c}_{k}", vin, f"colpo{c}", f"colno{c}", round(random.uniform(-INIT,INIT),3))
        gps.append((f,gp))
    gpb=synapse(f"o_{c}_b","bx",f"colpo{c}",f"colno{c}",round(random.uniform(-INIT,INIT),3))
    outgp[c]=gps
    subtractor(f"o{c}", f"colpo{c}", f"colno{c}", f"y_{c}", RTO, "vrefo")

# ---- output error blocks (trans2 differential) + OCMSUB ----
oref="vrefd"
if OCMSUB:
    oref="d2ebar"
    for c in range(C): w(f"Rocm_{c} d2e_{c} d2ebar {ROCM}")
    w("Cocm d2ebar 0 50f")
for c in range(C):
    w(f"Msf2_{c} d2e_{c} dpf_o{c} vdd vdd PMIR W=60u L=1u"); w(f"Msb2_{c} d2e_{c} colno{c} 0 0 NMIR W=20u L=1u")
    w(f"Moff_{c} d2e_{c} boff vdd vdd PMIR W=10u L=1u"); w(f"Mtge_{c} d2e_{c} tg{c} 0 0 NMIR W=10u L=1u"); w(f"Rde_{c} d2e_{c} vrefd {RTO}")
    w(f"Mqa_{c} d2en_ps_{c} colno{c} 0 0 NMIR W=20u L=1u"); w(f"Mqb_{c} d2en_ps_{c} tg{c} 0 0 NMIR W=10u L=1u")
    w(f"Mpd_{c} d2en_ps_{c} d2en_ps_{c} vdd vdd PMIR W=60u L=1u"); w(f"Mpm_{c} d2e_n_{c} d2en_ps_{c} vdd vdd PMIR W=60u L=1u")
    w(f"Mqc_{c} d2en_ns_{c} dpf_o{c} vdd vdd PMIR W=60u L=1u"); w(f"Mqd_{c} d2en_ns_{c} boff vdd vdd PMIR W=10u L=1u")
    w(f"Mnd_{c} d2en_ns_{c} d2en_ns_{c} 0 0 NMIR W=20u L=1u"); w(f"Mnm_{c} d2e_n_{c} d2en_ns_{c} 0 0 NMIR W=20u L=1u")
    w(f"Rden_{c} d2e_n_{c} vrefd {RTO}")
    w(f"Mbep_{c} vdd d2e_{c} d2e_pb_{c} 0 NSYN W=200u L=1u"); w(f"Rbep_{c} d2e_pb_{c} 0 20e3")
    w(f"Mben_{c} vdd d2e_n_{c} d2e_nb_{c} 0 NSYN W=200u L=1u"); w(f"Rben_{c} d2e_nb_{c} 0 20e3")

# ---- output weight update (OTA, gate by feature activation) ----
FREEZE=ei("FREEZE",0)
if not FREEZE:
    for c in range(C):
        for k,(f,gp) in enumerate(outgp[c]): ota(f"oU_{c}_{k}", f"d2e_{c}", oref, actnode(nhid,f), gp, OWTW)

# ---- BACKWARD through hidden layers (top-down). Each hidden neuron gets a delta node dpre_{L}_{n}.
# transpose read: for neuron (L,n), sum over consumers m in layer L+1 that read it: W[L+1][m,n]*delta_{L+1}[m].
# We need buffered +/- error of each consumer. For the top hidden layer consumers are the C outputs
# (use d2e_pb/d2e_nb). For deeper layers consumers are hidden neurons (buffer their dpre).
# build consumer map: consumers[(L,n)] = list of (gp_of_that_synapse, consumer_posbuf, consumer_negbuf)
# First make buffered delta nodes for each hidden neuron after we compute dpre (need 2-pass): we
# build top layer first (consumers=outputs), then each lower layer using the layer above's buffers.

# reverse maps
def consumers_for_layer(Lid):
    # returns dict: prev-neuron-index n -> list of (gp, m) for layer Lid synapses reading n
    cons={}
    cn=hidden_layers[Lid-1][0]
    for m in range(len(cn)):
        for ipx in cn[m]:
            cons.setdefault(ipx,[]).append((m,))
    return cons

# buffered error nodes per layer: top = outputs already (d2e_pb_c/d2e_nb_c). For hidden layer Lid we
# produce dpre_{Lid}_{n}, then buffer to dpb_{Lid}_{n}/dnb_{Lid}_{n} for the layer below.
def gp_of(Lid,m,prev_n):
    # gp node of the synapse in layer Lid, neuron m, that reads prev neuron prev_n
    for ipx,gp in gpw[(Lid,m)]:
        if ipx==prev_n: return gp
    return None

DIFFBUF=ei("DIFFBUF",1)   # buffer hidden deltas into vrefd-centered differential pairs (depth fix)
for Lid in range(nhid,0,-1):
    conn,nin,nout=hidden_layers[Lid-1]
    # consumers live in layer Lid+1 (hidden) or the output layer (if Lid==nhid)
    for n in range(nout):
        # gather transpose contributions into ctp/ctn
        contribs=[]
        if Lid==nhid:
            # consumers = outputs reading feature n
            for c in range(C):
                for k,(f,gp) in enumerate(outgp[c]):
                    if f==n: contribs.append((gp, f"d2e_pb_{c}", f"d2e_nb_{c}"))
        else:
            cn2=hidden_layers[Lid][0]  # layer Lid+1 connectivity
            # THE depth fix: read the UPPER layer's BUFFERED, vrefd-centered differential delta pair
            # (dpb,dnb) from diffbuf -- NOT the raw single-ended dpre. Raw dpre sits ~0.64 (triode for
            # the square-law multiplier) and is high-Z; reading it gave stage2 cosine 0.1. The buffered
            # pair is balanced around vrefd (multiplier saturation point) like the output's d2e_pb/nb.
            nbref = f"d1bar_L{Lid+1}" if CMSUB else "vrefd"
            for m in range(len(cn2)):
                if n in cn2[m]:
                    gp=gp_of(Lid+1,m,n)
                    if DIFFBUF:
                        _pb,_nb=(f"dpb_L{Lid+1}_{m}",f"dnb_L{Lid+1}_{m}")
                        if ei("DBSGN",0): _pb,_nb=_nb,_pb   # swap to test transpose sign
                        contribs.append((gp,_pb,_nb))
                    else:       contribs.append((gp, f"dpre_L{Lid+1}_{m}", nbref))
        if not contribs:
            # no consumer -> tie dpre to vrefd (no update)
            w(f"Rno_L{Lid}_{n} dpre_L{Lid}_{n} vrefd 1e3");
        else:
            for gp,pb,nb in contribs:
                w(f"MpTa_L{Lid}_{n}_{contribs.index((gp,pb,nb))} {pb} {gp} ctp_L{Lid}_{n} 0 NSYN W=10u L=1u")
                w(f"MnTb_L{Lid}_{n}_{contribs.index((gp,pb,nb))} {nb} vcn ctp_L{Lid}_{n} 0 NSYN W=10u L=1u")
                w(f"MnTa_L{Lid}_{n}_{contribs.index((gp,pb,nb))} {pb} vcn ctn_L{Lid}_{n} 0 NSYN W=10u L=1u")
                w(f"MpTb_L{Lid}_{n}_{contribs.index((gp,pb,nb))} {nb} {gp} ctn_L{Lid}_{n} 0 NSYN W=10u L=1u")
            subtractor(f"bk_L{Lid}_{n}", f"ctp_L{Lid}_{n}", f"ctn_L{Lid}_{n}", f"dpre_L{Lid}_{n}", RTbk, "vrefd")
    # CMSUB for this layer
    href="vrefd"
    if CMSUB:
        href=f"d1bar_L{Lid}"
        for n in range(nout): w(f"Rcm_L{Lid}_{n} dpre_L{Lid}_{n} {href} {RCMB}")
        w(f"Ccm_L{Lid} {href} 0 50f")
    # DIFFBUF: re-center each delta into a vrefd-centered low-Z differential pair (dpb,dnb) for the
    # layer BELOW to read (Lid>=2 feeds layer Lid-1). This is THE depth-transpose fix.
    if DIFFBUF and Lid>=2:
        for n in range(nout):
            diffbuf(f"L{Lid}_{n}", f"dpre_L{Lid}_{n}", href, f"dpb_L{Lid}_{n}", f"dnb_L{Lid}_{n}")
    # update gate f'(z): GATE=none (plain ota, no deriv gate), hv (gate by activation -- self-supplied
    # derivative, no comparator cell), comp (comparator step). hv was ~free & robust at depth-1.
    GATE=_os.environ.get("GATE","none")
    for n in range(nout):
        if GATE=="comp": comparator(f"L{Lid}_{n}", f"z_L{Lid}_{n}")
        if not FREEZE:
            for k,(ipx,gp) in enumerate(gpw[(Lid,n)]):
                if   GATE=="hv":   ota_relu(f"uU_L{Lid}_{n}_{k}", f"dpre_L{Lid}_{n}", href, actnode(Lid-1,ipx), f"hv_L{Lid}_{n}", gp)
                elif GATE=="comp": ota_relu(f"uU_L{Lid}_{n}_{k}", f"dpre_L{Lid}_{n}", href, actnode(Lid-1,ipx), f"uon_L{Lid}_{n}", gp)
                else:              ota(f"uU_L{Lid}_{n}_{k}", f"dpre_L{Lid}_{n}", href, actnode(Lid-1,ipx), gp, HWTW)
        # dpre_L{Lid}_{n} is used directly by the layer below's transpose read (no buffer needed)

# ---- noise (optional) ----
WNOISE=ef("WNOISE","0.0")
allgp=[gp for (Lid,n) in gpw for ipx,gp in gpw[(Lid,n)]]+[gp for c in outgp for f,gp in outgp[c]]
if WNOISE>0:
    for g in allgp: w(f"In_{g} {g} 0 TRNOISE({WNOISE} {round(ef('NOISEDT',str(round(dt,4))),4)} 0 0)")

w(".ic "+" ".join(f"v({n})={v}" for n,v in ic))
w(".control")
if WNOISE>0: w(f"  set rndseed={ei('NSEED','1')}")
w(f"  tran {dt} {Nstp*Ts} uic")
yt=" ".join(f"v(y_{c})" for c in range(C))
w(f"  wrdata deep_trace.txt {yt}")
if ei("PROBE",0):   # per-layer backward-fidelity probe: dpre/z/hv per neuron + targets
    tgs=" ".join(f"v(tg{c})" for c in range(C))
    dps=" ".join(f"v(dpre_L{Lid}_{n})" for Lid in range(1,nhid+1) for n in range(hidden_layers[Lid-1][2]))
    zs =" ".join(f"v(z_L{Lid}_{n})"    for Lid in range(1,nhid+1) for n in range(hidden_layers[Lid-1][2]))
    hvs=" ".join(f"v(hv_L{Lid}_{n})"   for Lid in range(1,nhid+1) for n in range(hidden_layers[Lid-1][2]))
    w(f"  wrdata deep_probe.txt {tgs} {dps} {zs} {hvs}")
w("  wrdata deep_weights.txt "+" ".join(f"v({g})" for g in allgp))
w("  echo DEEP_OK")
w(".endc"); w(".end")
open("deep.cir","w").write("\n".join(L)+"\n")
nmos=sum(1 for s in L if s[:1]=="M")
print(f"RES={RES} ARCH={ARCH} C={C} layers={nhid+1} nfeat={nfeat} weights={len(allgp)} MOS={nmos}")
