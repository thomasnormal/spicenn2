#!/usr/bin/env python3
# Multi-class fully-transistor trainer (zero B-sources).
# Generalizes the working trans2 XOR deck (gen_xor_full.py) to:
#   D inputs, H hidden ReLU units, C output classes,
# with an optional shared subtractive common-mode competition stage (COMP=subtractive).
# The forward path, the transpose-read backward (summed over all C classes by KCL at each
# hidden transpose column), and the OTA weight update all run continuously over a long .tran.
import random, sys, os as _os
import numpy as np
random.seed(int(_os.environ.get("SEED","7")))

def ef(k,d): return float(_os.environ.get(k,d))
def ei(k,d): return int(_os.environ.get(k,d))

D = ei("D",4); H = ei("H",8); C = ei("C",3)
Ts = ef("TS","0.3"); dt = ef("DTFRAC","0.34")*Ts
NTRAIN = int(sys.argv[1]) if len(sys.argv)>1 else 600
Nstp = NTRAIN
Ke   = float(sys.argv[2]) if len(sys.argv)>2 else 0.3
Cg   = float(sys.argv[3]) if len(sys.argv)>3 else 5e-4
VC   = ef("VC","1.8"); Gs=ef("GS","1.2"); INIT=ef("INIT","0.25")
INITH= ef("INITH","0.6")   # hidden input-weight init spread (wider -> live input signal)
BHID = ef("BHID","0.6")    # positive hidden ReLU-bias init (units start alive)
VREFH= ef("VREFH","0.5"); VREFO=ef("VREFO","0.5")
RTH  = ef("RTH","8e3"); RTO=ef("RTO","7e3"); RH=ef("RH","8e3")
VG0  = ef("VG0","0.7"); Vd0=ef("VD0","1.2"); bb=0.05
VTOSYN=ef("VTOSYN","0.2"); RDECAY=ef("RDECAY","0.0"); RSDEG=_os.environ.get("RSDEG",""); ORDECAY=ef("ORDECAY","0.0")
DIFFOUT=ei("DIFFOUT",0); RBAR=ef("RBAR","50e3")   # differential output readout: Mn reads hvbar (feature common-mode) -> cancels it
DIFFHID=ei("DIFFHID",0)   # DIFFERENTIAL hidden output (tanh only): Mn reads per-unit complement hv_n (dmp side) -> clean common-mode-free feature
RTbk = ef("RTBK","16e3"); VCBIAS=ef("VCBIAS","1.0")
VULK = ef("VULK","0.55"); RULK=ef("RULK","60e3")   # leaky-ReLU' floor: holds uon at VULK when off
ACT  = _os.environ.get("ACT","relu2")   # relu2 (square-law) | tanh (bounded diff-pair, center-biased)
VMID = ef("VMID","0.55"); VTB=ef("VTB","1.05"); TW=ef("TW","20"); VMIDSP=ef("VMIDSP","0.0")  # tanh params
HWTW = ef("HWTW","48")   # hidden-update OTA tail width = hidden learning rate (lower=slower/stabler)
OWTW = ef("OWTW","20")     # output-update OTA tail width; <20 slows output learning so hidden
                            # features form before the output collapses to the bias-only basin
COMP = _os.environ.get("COMP","none")           # none | subtractive
RCM  = ef("RCM", str(RTO/C))                       # common-mode gain ~ RTO/C subtracts the MEAN error
# forward input band and update-gate band
INLO=ef("INLO","0.5"); INHI=ef("INHI","1.5"); AULO=0.3; AUHI=1.0

# ---------------- task data (blobs, matches surrogate make_blobs) ----------------
def make_blobs(C,D,n,seed,spread):
    cen=np.random.default_rng(2024).normal(0,1.5,(C,D))
    rng=np.random.default_rng(seed); X=[];Y=[]
    for c in range(C): X.append(cen[c]+rng.normal(0,spread,(n,D))); Y+=[c]*n
    return np.vstack(X), np.array(Y)
SPREAD=ef("SPREAD","0.9"); NPER=ei("NPER",8)
TAG=_os.environ.get("TAG","")   # per-instance suffix for deck/output files so respawned duplicates don't clobber
DATAFILE=_os.environ.get("DATAFILE","")   # if set: load X (already in [0,1]),Y from npz instead of blobs
if DATAFILE:
    _d=np.load(DATAFILE); Xtr=_d["X"].astype(float); Ytr=_d["Y"].astype(int)
    mn=np.zeros(Xtr.shape[1]); mx=np.ones(Xtr.shape[1])   # data pre-scaled to [0,1] -> fixed band map
else:
    Xtr,Ytr = make_blobs(C,D,NPER,1,SPREAD)
    mn=Xtr.min(0); mx=Xtr.max(0)
def scale_fwd(X): return INLO+(X-mn)/(mx-mn+1e-9)*(INHI-INLO)
def scale_au(X):  return AULO+(X-mn)/(mx-mn+1e-9)*(AUHI-AULO)
Xf=scale_fwd(Xtr); Xa=scale_au(Xtr)
DIRECT=ei("DIRECT",0)   # defined early (also re-set later, same value) for input-pruning logic below
# ROUTSP: compute the per-class top-K L1 feature selection EARLY (before input sources are emitted) so
# unused features can be pruned from the deck. Written once to mc_sel.txt; reused on resume.
if ei("DIRECT",0) and ei("ROUTSP",0) and not _os.path.exists("mc_sel.txt"):
    from sklearn.linear_model import LogisticRegression as _LR0
    # top-K of plain L2 weights (NOT L1-saga, which selects a much worse subset -> ~10-17pt lower ideal)
    _m0=_LR0(max_iter=600).fit(Xtr,Ytr); _K0=ei("ROUTSP",0)
    with open("mc_sel.txt","w") as _f:
        for _c in range(C):
            _top=sorted(int(i) for i in np.argsort(np.abs(_m0.coef_[_c]))[::-1][:_K0])
            _f.write(" ".join(map(str,_top))+"\n")
# present patterns cycling; shuffle a fixed order so classes interleave
order=list(range(len(Xtr))); random.Random(3).shuffle(order)
np.savez("mc_data.npz", mn=mn, mx=mx, cen=np.random.default_rng(2024).normal(0,1.5,(C,D)),
         spread=SPREAD, C=C, D=D, INLO=INLO, INHI=INHI)

# ---------------- PWL builders ----------------
def pwl_from(vals):
    """vals[k] = value held during slot k (one number per slot)."""
    pts=[]
    for k in range(Nstp):
        v=vals[k%len(vals)]; t0=k*Ts; pts+=[(t0,v),(t0+Ts-dt,v)]
    pts.append((Nstp*Ts,pts[-1][1]))
    return " ".join(f"{t:.4f} {v:.4f}" for t,v in pts)

# per-slot sequences indexed by the shuffled presentation order
seqX = [Xf[order[k%len(order)]] for k in range(len(order))]      # forward inputs
seqA = [Xa[order[k%len(order)]] for k in range(len(order))]      # update-gate inputs
seqY = [Ytr[order[k%len(order)]] for k in range(len(order))]     # labels
# UENGATE: per-slot in-circuit update-GATE (hinge realized correctly). The forward+common-mode stream for
# ALL classes every slot (unchanged); only the weight-cap charging is gated. Controller writes UENFILE =
# one gate per training pattern (1=update, 0=skip-confident). Default (no file) = all-on = stock behavior.
UENFILE=_os.environ.get("UENFILE","")
_gate=None
if UENFILE and _os.path.exists(UENFILE):
    _gv=np.loadtxt(UENFILE).astype(int).reshape(-1)
    _gate=[int(_gv[order[k%len(order)]]) if order[k%len(order)]<len(_gv) else 1 for k in range(len(order))]
UENGATE = _gate is not None
TAILGND = "uvss" if UENGATE else "0"

L=[]; ic=[]; w=lambda s:L.append(s)
w("* Multi-class fully-transistor trainer (parameterized, trans2 backward)")
w(f".model NSYN NMOS (LEVEL=1 VTO={VTOSYN} KP=50u  LAMBDA=0    GAMMA=0 PHI=0.7)")
w(".model NMIR NMOS (LEVEL=1 VTO=0.5 KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)")
w(".model PMIR PMOS (LEVEL=1 VTO=-0.5 KP=40u LAMBDA=0.02 GAMMA=0 PHI=0.7)")
w(".model NREL NMOS (LEVEL=1 VTO=0.5 KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)")
w(f".param Cg={Cg} Ke={Ke} Vd0={Vd0} VG0={VG0} VC={VC} bb={bb}")
w("Vvdd vdd 0 3")
w(f"Vrefh vrefh 0 {VREFH}"); w(f"Vrefo vrefo 0 {VREFO}")
w(f"Vg0 vg0 0 {VG0}"); w(f"Vvc vcn 0 {VC}")
w("Vrefd vrefd 0 1.2"); w("Vboff boff 0 1.9")
w(f"Vthr vthr 0 {VMID if ACT=='tanh' else 0.5}"); w(f"Vcbias vcbias 0 {VCBIAS}")  # tanh: ReLU' gate at center
if ACT=="tanh": w(f"Vvmid vmid 0 {VMID}"); w(f"Vvtb vtb 0 {VTB}")
if VULK>0: w(f"Vulk vulk 0 {VULK}")
w(f"Vbx bx 0 dc {INHI}"); w(f"Vaub aub 0 dc {AUHI}")
# input sources (forward a_i and update-gate au_i). In DIRECT mode au_i is unused (no hidden layer) so
# skip it -> halves input PWL sources. With ROUTSP, only stream features actually read by some class.
_used_feats=None
if DIRECT and ei("ROUTSP",0) and _os.path.exists("mc_sel.txt"):
    _used_feats=set()
    with open("mc_sel.txt") as _f:
        for _line in _f: _used_feats.update(int(x)+1 for x in _line.split())
for i in range(1,D+1):
    if _used_feats is not None and i not in _used_feats: continue
    w(f"Va{i} a{i} 0 PWL({pwl_from([float(x[i-1]) for x in seqX])})")
    if not DIRECT:
        w(f"Vau{i} au{i} 0 PWL({pwl_from([float(x[i-1]) for x in seqA])})")
# one-hot target sources tg_c. TGHI/TGLO place the on/off targets inside the keystone's reachable
# output range so target-class weights settle at a graded fixed point instead of railing (which
# collapses per-example margins). Default 1/0 = classic one-hot.
TGHI=ef("TGHI","1.0"); TGLO=ef("TGLO","0.0")
for c in range(1,C+1):
    w(f"Vtg{c} tg{c} 0 PWL({pwl_from([TGHI if (lab==c-1) else TGLO for lab in seqY])})")
if UENGATE:   # gated tail-rail: Muen sinks uvss to gnd when uen high (update); floats when low (no update)
    w(f"Vuen uen 0 PWL({pwl_from([1.8 if g else 0.0 for g in _gate])})")
    w("Muen uvss uen 0 0 NMIR W=4000u L=1u")

# ---------------- device cells (verbatim from the proven deck) ----------------
SYNW=ef("SYNW","10")   # synapse FET width: scale DOWN for high-fan-in layers so the summed current
def synapse(idx,vin,cp,cn,w0):   # stays within the keystone mirror range (avoids the fan-in crush)
    gp=f"gp_{idx}"; w(f"C_{gp} {gp} 0 {{Cg}}"); _isout = idx.startswith("2_")
    if _isout:                    # ---- OUTPUT readout: DIFFHID / DIFFOUT differential, or plain (NO RSDEG) ----
        if DIFFHID:               # per-unit complement hv_n_{j}
            _hj=idx.split("_")[2]
            w(f"Mp_{idx} {vin} {gp} {cp} 0 NSYN W={SYNW}u L=1u")
            w(f"Mn_{idx} hv_n_{_hj} {gp} {cn} 0 NSYN W={SYNW}u L=1u")
        elif DIFFOUT:             # Mn reads hvbar (global common-mode), same weight gp -> cancels common-mode
            w(f"Mp_{idx} {vin} {gp} {cp} 0 NSYN W={SYNW}u L=1u")
            w(f"Mn_{idx} hvbar {gp} {cn} 0 NSYN W={SYNW}u L=1u")
        else:
            w(f"Mp_{idx} {vin} {gp} {cp} 0 NSYN W={SYNW}u L=1u")
            w(f"Mn_{idx} {vin} vcn {cn} 0 NSYN W={SYNW}u L=1u")
    elif RSDEG:                   # ---- HIDDEN: source degeneration LINEARIZES the projection (w*x) ----
        w(f"Mp_{idx} {vin} {gp} sp_{idx} 0 NSYN W={SYNW}u L=1u"); w(f"Rsp_{idx} sp_{idx} {cp} {RSDEG}")
        w(f"Mn_{idx} {vin} vcn sn_{idx} 0 NSYN W={SYNW}u L=1u"); w(f"Rsn_{idx} sn_{idx} {cn} {RSDEG}")
    else:
        w(f"Mp_{idx} {vin} {gp} {cp} 0 NSYN W={SYNW}u L=1u")
        w(f"Mn_{idx} {vin} vcn {cn} 0 NSYN W={SYNW}u L=1u")
    # RDECAY leaks weight -> vcn (centers it, cancels the feature common-mode bias that collapses the
    # keystone). ORDECAY>0 = OUTPUT-only decay ("2_*"), so a FROZEN random hidden ("1_*") keeps its features.
    _dk = (ORDECAY if ORDECAY>0 else RDECAY) if _isout else (0 if ORDECAY>0 else RDECAY)
    if _dk>0: w(f"Rdk_{idx} {gp} vcn {_dk}")
    ic.append((gp, round(VC+Gs*w0,4))); return gp
def subtractor(tag,cp,cn,vout,Rt,vrefn):
    w(f"Msa_{tag} {cn} {cn} 0 0 NMIR W=20u L=1u"); w(f"Msb_{tag} {vout} {cn} 0 0 NMIR W=20u L=1u")
    w(f"Msc_{tag} {cp} {cp} 0 0 NMIR W=20u L=1u"); w(f"Msd_{tag} dpf_{tag} {cp} 0 0 NMIR W=20u L=1u")
    w(f"Mse_{tag} dpf_{tag} dpf_{tag} vdd vdd PMIR W=60u L=1u"); w(f"Msf_{tag} {vout} dpf_{tag} vdd vdd PMIR W=60u L=1u")
    w(f"R_{tag} {vout} {vrefn} {Rt}")
def ota(tag,ep,en,actg,gp,tw=20):
    w(f"Mt_{tag} ts_{tag} {actg} {TAILGND} 0 NMIR W={tw}u L=1u")
    w(f"Mo1_{tag} da_{tag} {en} ts_{tag} 0 NMIR W=20u L=1u"); w(f"Mo2_{tag} {gp} {ep} ts_{tag} 0 NMIR W=20u L=1u")
    w(f"Mo3_{tag} da_{tag} da_{tag} vdd vdd PMIR W=60u L=1u"); w(f"Mo4_{tag} {gp} da_{tag} vdd vdd PMIR W=60u L=1u")
def ota_relu(tag,ep,en,xg,rg,gp):
    # HWTW scales the hidden-update tail current = hidden-layer learning rate. Lower -> slower,
    # more stable hidden learning (the default 48u takes ~80%-of-norm steps and overshoots/collapses).
    w(f"Mtt_{tag} ts_{tag} {rg} mid_{tag} 0 NMIR W={HWTW}u L=1u")
    w(f"Mtb_{tag} mid_{tag} {xg} 0 0 NMIR W={HWTW}u L=1u")
    w(f"Mo1_{tag} da_{tag} {en} ts_{tag} 0 NMIR W=20u L=1u"); w(f"Mo2_{tag} {gp} {ep} ts_{tag} 0 NMIR W=20u L=1u")
    w(f"Mo3_{tag} da_{tag} da_{tag} vdd vdd PMIR W=60u L=1u"); w(f"Mo4_{tag} {gp} da_{tag} vdd vdd PMIR W=60u L=1u")
def comparator(j):
    w(f"Mct_{j} ctail_{j} vcbias 0 0 NMIR W=10u L=1u")
    w(f"Mca_{j} cna_{j} z1_{j} ctail_{j} 0 NSYN W=20u L=1u")
    w(f"Mcb_{j} uon_{j} vthr   ctail_{j} 0 NSYN W=20u L=1u")
    w(f"Mcp1_{j} cna_{j} cna_{j} vdd vdd PMIR W=40u L=1u")
    w(f"Mcp2_{j} uon_{j} cna_{j} vdd vdd PMIR W=40u L=1u")
    # leaky ReLU': a weak resistor holds uon_j at a positive floor (VULK) when the unit is
    # below threshold, so dead units still receive a trickle of update gain and can recover
    # (breaks the dead-ReLU lockout). When the unit is on, Mcp2 overpowers the leak -> uon high.
    if VULK>0: w(f"Rulk_{j} uon_{j} vulk {RULK}")

inputs_h = [(str(i),f"a{i}") for i in range(1,D+1)] + [("b","bx")]
au = {str(i):f"au{i}" for i in range(1,D+1)}; au["b"]="aub"
# DIRECT: a hidden-free LINEAR multi-class delta-rule classifier -- output keystones read the
# inputs a_i directly. The output weight update is an EXACT in-circuit gradient (error x input),
# with no approximate transpose-read backward, so it does not suffer the 2-vs-1 hidden collapse.
# A linear 16->3 map already separates MNIST-4x4 triples at ~94%, so this is the right baseline.
DIRECT=ei("DIRECT",0)

# ---------------- first-layer connectivity (dense or sparse 2x2 receptive fields) -------------
CONN=_os.environ.get("CONN","dense")
if CONN=="patch2x2":
    # D=16 as a 4x4 grid (input i = 4*row+col+1). Non-overlapping 2x2 patches; H//4 filters/patch.
    patches=[[1,2,5,6],[3,4,7,8],[9,10,13,14],[11,12,15,16]]
    nf=max(1,H//len(patches))
    hid_inputs={j:[(str(i),f"a{i}") for i in patches[(j-1)//nf % len(patches)]]+[("b","bx")] for j in range(1,H+1)}
elif CONN=="rf3x3":
    # 4x4 grid, OVERLAPPING stride-1 2x2 receptive fields -> 3x3 = 9 positions. Each hidden unit
    # sees a 2x2 field (4 pixels)+bias. H//9 filters per position. (the requested 16->9->C topology)
    fields=[[4*pr+pc+1, 4*pr+pc+2, 4*pr+pc+5, 4*pr+pc+6] for pr in range(3) for pc in range(3)]
    nf=max(1,H//9)
    hid_inputs={j:[(str(i),f"a{i}") for i in fields[(j-1)//nf % 9]]+[("b","bx")] for j in range(1,H+1)}
else:
    hid_inputs={j:inputs_h for j in range(1,H+1)}

# ---------------- forward: hidden layer ----------------
hidw={}
for j in ([] if DIRECT else range(1,H+1)):
    cps={}
    for i,vin in hid_inputs[j]:
        # positive ReLU-bias init (BHID) lifts every hidden unit above the activation
        # threshold so it starts ALIVE and input-sensitive (avoids dead-ReLU bootstrap lock);
        # input weights use a wider spread INITH so hv carries real input signal at t=0.
        if i=="b": w0=BHID
        else:      w0=round(random.uniform(-INITH,INITH),3)
        cps[i]=synapse(f"1_{j}_{i}",vin,f"colp{j}",f"coln{j}",w0)
    hidw[j]=cps
    subtractor(f"h{j}",f"colp{j}",f"coln{j}",f"z1_{j}",RTH,"vrefh")
    if ACT=="tanh":
        # bounded neuron = differential pair: z1_j vs vmid, tail current (VTB/TW) caps the swing so
        # hv saturates (sigmoid/tanh). Units are CENTER-biased -> z1 swings both sides of vmid for
        # different inputs, so the nonlinearity is actually exercised (unlike always-on ReLU2).
        vmj=f"vmid_{j}" if VMIDSP>0 else "vmid"
        if VMIDSP>0: w(f"Vvm_{j} {vmj} 0 {round(VMID+VMIDSP*((j-1)/max(1,H-1)-0.5),4)}")
        w(f"Mr_{j}  hdd_{j} z1_{j} tail_{j} 0 NREL W=12u L=1u")
        w(f"Mr2_{j} dmp_{j}  {vmj}  tail_{j} 0 NREL W=12u L=1u")
        w(f"Mtl_{j} tail_{j} vtb 0 0 NREL W={TW}u L=1u")
        w(f"Mdm_{j} dmp_{j} dmp_{j} vdd vdd PMIR W=40u L=1u")
    else:
        w(f"Mr_{j} hdd_{j} z1_{j} 0 0 NREL W=12u L=1u")
    w(f"Mrp1_{j} hdd_{j} hdd_{j} vdd vdd PMIR W=40u L=1u"); w(f"Mrp2_{j} hv_{j} hdd_{j} vdd vdd PMIR W=40u L=1u")
    w(f"Rhv_{j} hv_{j} vg0 {RH}")
    if DIFFHID and ACT=="tanh":   # complementary output hv_n from the dmp (vmid) side of the diff pair
        w(f"Mrp2n_{j} hv_n_{j} dmp_{j} vdd vdd PMIR W=40u L=1u"); w(f"Rhvn_{j} hv_n_{j} vg0 {RH}")

# ---------------- forward: C output keystones ----------------
if DIFFOUT and not DIRECT:   # hvbar = in-spice common-mode of the hidden activations (resistor average)
    for j in range(1,H+1): w(f"Rbar_{j} hv_{j} hvbar {RBAR}")
    w("Cbar hvbar 0 50f")
inputs_o = ([(str(i),f"a{i}") for i in range(1,D+1)] if DIRECT
            else [(str(j),f"hv_{j}") for j in range(1,H+1)]) + [("b","bx")]
# ROUTSP=K: SPARSE readout -- each class reads only its top-K (L1-selected) features instead of all D.
# Breaks the deck-size ceiling: N=512 features but K~64-128/class -> small deck + high ideal. The L1
# selection is saved to mc_sel.txt so gen_mc_infer builds the IDENTICAL connectivity (weight order match).
ROUTSP=ei("ROUTSP",0)
if ROUTSP and DIRECT:
    # REUSE an existing mc_sel.txt if present (so checkpoint-resume chunks keep the SAME connectivity);
    # only compute the L1 selection on the first chunk.
    if _os.path.exists("mc_sel.txt"):
        oc_inputs={}
        with open("mc_sel.txt") as _f:
            for c,_line in enumerate(_f,1):
                _top=[int(x) for x in _line.split()]
                oc_inputs[c]=[(str(i+1),f"a{i+1}") for i in _top]+[("b","bx")]
    else:
        from sklearn.linear_model import LogisticRegression as _LR
        _m=_LR(max_iter=400,penalty='l1',solver='saga',C=0.5).fit(Xtr,Ytr)
        oc_inputs={}
        with open("mc_sel.txt","w") as _f:
            for c in range(1,C+1):
                _top=sorted(int(i) for i in np.argsort(np.abs(_m.coef_[c-1]))[::-1][:ROUTSP])
                oc_inputs[c]=[(str(i+1),f"a{i+1}") for i in _top]+[("b","bx")]
                _f.write(" ".join(map(str,_top))+"\n")
else:
    oc_inputs={c: inputs_o for c in range(1,C+1)}
outw={}
for c in range(1,C+1):
    outw[c]={}
    for j,vin in oc_inputs[c]:
        outw[c][j]=synapse(f"2_{c}_{j}",vin,f"colpo{c}",f"colno{c}",round(random.uniform(-INIT,INIT),3))
    subtractor(f"o{c}",f"colpo{c}",f"colno{c}",f"y_{c}",RTO,"vrefo")

# ---------------- competition (shared subtractive common-mode) ----------------
# CMnode integrates the SUM of the raw class errors (sum_c d2e_c - C*1.2) and feeds it back
# equally to every class error reference, implementing delta_c -> delta_c - mean_k(delta_k).
# Built from current mirrors only (zero B-sources). For COMP=none the references are vrefd.
errsrc   = {c:"vrefd" for c in range(1,C+1)}     # ref for d2e_c   (positive error node)
errsrc_n = {c:"vrefd" for c in range(1,C+1)}     # ref for d2e_n_c (negative error node)
if COMP=="subtractive":
    # Sum each class's positive-error contributors into a shared node and its negative-error
    # contributors into another; the difference encodes sum_c delta_c. Mirror it back as a
    # common-mode reference shift (cmH high when mean error +, cmL its complement).
    # cm node: cmH = vrefd + RCM * sum_c (I+_c - I-_c) / scale  (shared, sunk/sourced into refs)
    # Realized as: build a shared current-sum rail, convert to a reference voltage pair.
    for c in range(1,C+1):
        # replicate class c error currents into the shared common-mode rails
        w(f"Mcmp_{c} cm_ps colno{c} 0 0 NMIR W=20u L=1u")     # sense I-_c
        w(f"Mcmq_{c} cm_ps tg{c}    0 0 NMIR W=10u L=1u")     # sense I_tg,c
        w(f"Mcmr_{c} cm_ns dpf_o{c} vdd vdd PMIR W=60u L=1u") # sense I+_c
        w(f"Mcms_{c} cm_ns boff     vdd vdd PMIR W=10u L=1u") # sense I_off,c
    # cm_ps collects sum(I-_c + I_tg,c); cm_ns collects sum(I+_c + I_off,c)
    w("Mcmpd cm_ps cm_ps vdd vdd PMIR W=60u L=1u")            # PMOS diode on cm_ps
    w("Mcmnd cm_ns cm_ns 0 0 NMIR W=20u L=1u")               # NMOS diode on cm_ns
    # common-mode voltage cmv = vrefd + RCM*( sum I+_c+I_off - sum I-_c - I_tg )/C-ish
    #   = vrefd + (gain)*sum_c delta_c  (a shared node)
    w("Mcmsrc cmv cm_ps vdd vdd PMIR W=60u L=1u")             # source sum(I-_c+I_tg) ... see sign below
    w("Mcmsnk cmv cm_ns 0 0 NMIR W=20u L=1u")                 # sink sum(I+_c+I_off)
    w(f"Rcm cmv vrefd {RCM}")
    # cmv = vrefd + RCM*( sum(I-_c+I_tg) - sum(I+_c+I_off) ) = vrefd - RCM*sum_c delta_c
    # so (vrefd - cmv) ~ +sum delta. To subtract mean error from each class we shift the
    # per-class reference: use cmv as the d2e reference (raises d2e baseline when mean err<0)
    # and a complementary node cmvn for d2e_n.
    w("Mcmcp cmvn cm_ns vdd vdd PMIR W=60u L=1u")             # complementary
    w("Mcmcn cmvn cm_ps 0 0 NMIR W=20u L=1u")
    w(f"Rcmn cmvn vrefd {RCM}")
    errsrc   = {c:"cmv"  for c in range(1,C+1)}
    errsrc_n = {c:"cmvn" for c in range(1,C+1)}

# ---------------- per-class error blocks (trans2 differential) ----------------
for c in range(1,C+1):
    # d2e_c = ref + (y_c - t_c)
    w(f"Msf2_{c} d2e_{c} dpf_o{c} vdd vdd PMIR W=60u L=1u"); w(f"Msb2_{c} d2e_{c} colno{c} 0 0 NMIR W=20u L=1u")
    _offw=ef("OFFW","10")  # NOOFF/OFFW: width of the d2e offset PMOS. numpy ablation: this offset current
    if ei("NOOFF",0)==0: w(f"Moff_{c} d2e_{c} boff vdd vdd PMIR W={_offw}u L=1u")  # drives the slow-drift collapse
    w(f"Mtge_{c} d2e_{c} tg{c} 0 0 NMIR W=10u L=1u")
    w(f"Rde_{c} d2e_{c} {errsrc[c]} {RTO}")
    # d2e_n_c = ref - (y_c - t_c) via mirror copy with roles swapped
    w(f"Mqa_{c} d2en_ps_{c} colno{c} 0 0 NMIR W=20u L=1u")
    w(f"Mqb_{c} d2en_ps_{c} tg{c}    0 0 NMIR W=10u L=1u")
    w(f"Mpd_{c} d2en_ps_{c} d2en_ps_{c} vdd vdd PMIR W=60u L=1u")
    w(f"Mpm_{c} d2e_n_{c}   d2en_ps_{c} vdd vdd PMIR W=60u L=1u")
    w(f"Mqc_{c} d2en_ns_{c} dpf_o{c} vdd vdd PMIR W=60u L=1u")
    w(f"Mqd_{c} d2en_ns_{c} boff     vdd vdd PMIR W=10u L=1u")
    w(f"Mnd_{c} d2en_ns_{c} d2en_ns_{c} 0 0 NMIR W=20u L=1u")
    w(f"Mnm_{c} d2e_n_{c}   d2en_ns_{c} 0 0 NMIR W=20u L=1u")
    w(f"Rden_{c} d2e_n_{c} {errsrc_n[c]} {RTO}")
    # matched low-Vt source-follower buffers
    w(f"Mbep_{c} vdd d2e_{c}   d2e_pb_{c} 0 NSYN W=200u L=1u"); w(f"Rbep_{c} d2e_pb_{c} 0 20e3")
    w(f"Mben_{c} vdd d2e_n_{c} d2e_nb_{c} 0 NSYN W=200u L=1u"); w(f"Rben_{c} d2e_nb_{c} 0 20e3")

# ---------------- analog MOMENTUM (optional): gradient -> leaky cap -> transconductor -> weight cap.
# weight = integral of a low-passed (EMA) gradient = momentum SGD. Adds 1 cap + 1 leak R + a 2nd OTA
# per weight. Python showed momentum ~= Adam (+2.4 of +2.6) under noise at this 1-extra-cap cost.
MOM=ei("MOM",0); Cmom=ef("CMOM",str(Cg*3)); Rmom=ef("RMOM","80e3"); MOMTW=ef("MOMTW","20")
if MOM: w("Vvmom1 vmom1 0 1.0")   # constant tail-gate bias for the momentum->weight transconductor
def upd(tag,ep,en,actg,gp,tw,gated=False,rg=None):
    # gated=True uses the ReLU'-style ota_relu (hidden, rg=gate node); else plain ota.
    if MOM:
        m=f"mom_{tag}"; w(f"C_mom_{tag} {m} 0 {Cmom}"); w(f"R_mom_{tag} {m} vrefd {Rmom}"); ic.append((m,1.2))
        if gated: ota_relu(f"{tag}_g", ep, en, actg, rg, m)
        else:     ota(f"{tag}_g", ep, en, actg, m, tw)
        ota(f"{tag}_w", m, "vrefd", "vmom1", gp, MOMTW)   # m drives the weight cap (integrates momentum)
    else:
        if gated: ota_relu(tag, ep, en, actg, rg, gp)
        else:     ota(tag, ep, en, actg, gp, tw)

# ---------------- output weight update (OTA per class/hidden) ----------------
# NOBIAS: freeze the per-class output bias (the bx synapse) at its small init, so no class can
# win unconditionally by ramping its own bias -- kills the "output predicts one class" basin.
NOBIAS=ei("NOBIAS",0); FREEZE=ei("FREEZE",0)   # FREEZE: no weight updates (for backward-fidelity probe)
# OCMSUB: subtract the OUTPUT error common-mode (same resistor-average trick as CMSUB, on the output).
# d2ebar = mean_c(d2e_c); using it as the output-update reference makes the update see
# (y_c - t_c) - mean_k(y_k - t_k) -- a competitive/softmax-like error. Without it, large-C outputs
# can't separate in the compressed keystone range and collapse to one winning class. Zero B-source.
OCMSUB=ei("OCMSUB",0); oref="vrefd"
# DREFSH: shift the OTA update reference by a constant to cancel the FIXED common error offset baked
# into the trans2 error block (boff miscalibration vs the actual y operating range). Measured +0.409
# for the 8x8 features; this constant push collapsed every weight to the floor. With it cancelled the
# update sees the true (y-t) (discriminative part is already corr-0.98 faithful).
DREFSH=ef("DREFSH","0.0")
if DREFSH!=0:
    oref="orefsh"; w(f"Vorefsh orefsh 0 {round(1.2+DREFSH,4)}")
elif OCMSUB and not FREEZE:
    oref="d2ebar"
    for c in range(1,C+1): w(f"Rocm_{c} d2e_{c} d2ebar {ef('ROCM','10e3')}")
    w("Cocm d2ebar 0 50f")
# PERAZ: per-class dynamic auto-zero (IDEAS2 #16). Each class gets its OWN slow leaky tracker of its
# error d2e_c (oraz_c = slow average), used as that class's update reference. Removes the PER-CLASS DC
# offset (which shifts each class's equilibrium and starves the weak classes) while leaving the
# per-example AC signal. Long tau (RAZ*CAZ) so it tracks DC, not signal. O(C) extra parts.
PERAZ=ei("PERAZ",0); RAZ=ef("RAZ","10e3"); CAZ=ef("CAZ","2e-3")
# GLOBALAZ: ONE shared auto-zero node (resistor-averages all d2e_c into a single oraz_g cap) used as the
# update reference for EVERY class. Model-found fix for the C=10 collapse: the PER-CLASS oraz creates
# per-class feedback -> rich-get-richer drift; a GLOBAL one cancels common-mode offset WITHOUT the drift
# (fast-model C=10: per-class collapses 20-37%, global climbs to 67% stable). Also fewer caps (1 vs C).
GLOBALAZ=ei("GLOBALAZ",0); RAZG=ef("RAZG","60e3")
if GLOBALAZ and not FREEZE:
    for c in range(1,C+1): w(f"R_azg_{c} d2e_{c} oraz_g {RAZG}")
    w(f"C_azg oraz_g 0 {CAZ}"); ic.append(("oraz_g",1.2))
for c in (range(1,C+1) if not FREEZE else []):
    cref=oref
    if GLOBALAZ: cref="oraz_g"
    elif PERAZ:
        cref=f"oraz_{c}"; w(f"R_az_{c} d2e_{c} {cref} {RAZ}"); w(f"C_az_{c} {cref} 0 {CAZ}"); ic.append((cref,1.2))
    for j,vin in oc_inputs[c]:
        if NOBIAS and j=="b": continue
        upd(f"o{c}_{j}", f"d2e_{c}", cref, vin, outw[c][j], OWTW)

# ---------------- backward: transpose read, summed over classes by KCL ----------------
for j in ([] if DIRECT else range(1,H+1)):
    for c in range(1,C+1):
        gpj=outw[c][str(j)]
        w(f"MpTa_{j}_{c} d2e_pb_{c} {gpj} ctp_{j} 0 NSYN W=10u L=1u")
        w(f"MnTb_{j}_{c} d2e_nb_{c} vcn  ctp_{j} 0 NSYN W=10u L=1u")
        w(f"MnTa_{j}_{c} d2e_pb_{c} vcn  ctn_{j} 0 NSYN W=10u L=1u")
        w(f"MpTb_{j}_{c} d2e_nb_{c} {gpj} ctn_{j} 0 NSYN W=10u L=1u")
    subtractor(f"bk{j}", f"ctp_{j}", f"ctn_{j}", f"d1pre_{j}", RTbk, "vrefd")

# ---------------- backward common-mode subtraction ----------------
# The transpose-read backward d1pre_j is dominated by a COMMON-MODE offset: the output error
# (y_c - t_c) has a large mean over classes (compressed outputs sit above the mostly-zero one-hot
# targets), and sum_c W2_cj*(common error) drifts EVERY hidden cap the same way -> collapse.
# Fix: d1bar = resistor-average of all d1pre_j (= vrefd + mean signal); use it as the hidden-update
# reference so each unit sees (d1pre_j - d1bar) = its signal MINUS the shared offset. Zero-cost,
# zero B-source: a resistor star to a high-impedance node.
CMSUB=ei("CMSUB",0); href="vrefd"
if CMSUB and not DIRECT:
    href="d1bar"
    for j in range(1,H+1): w(f"Rcm_{j} d1pre_{j} d1bar {ef('RCMB','20e3')}")
    w("Ccm d1bar 0 50f")   # settle to the average (DC: resistor star -> mean of d1pre_j)

# ---------------- hidden weight update (comparator ReLU' + gated OTA) ----------------
FRZH=ei("FRZH",0)   # freeze hidden weights only (random features): trains output layer alone
# NOGATE: drop the comparator ReLU' gate entirely and use the plain output-style OTA for the hidden
# update (gated by the input only). For the tanh neuron uon_j is already saturated-on (measured), so
# the gate carries ~no information -- removing it gives the same training with ~6 fewer MOS per hidden
# unit (no comparator, no leaky-ReLU resistor, smaller update OTA). Simpler AND identical behavior.
NOGATE=ei("NOGATE",0)
# GATE selects the backward activation-derivative gate f'(z):
#   comp = separate comparator step 1[z>vmid] (baseline, ~5 MOS/unit)
#   none = no gate (f'=1), plain OTA gated by input only (cheapest)
#   hv   = gate by the activation hv_j itself (0 extra MOS -- hv already exists; smooth, monotone
#          proxy for the derivative that's "self-supplied" by the neuron, no comparator cell)
GATE=_os.environ.get("GATE","none" if NOGATE else "comp")
need_comp = (GATE=="comp")
for j in ([] if (DIRECT or not need_comp) else range(1,H+1)): comparator(j)
for j in ([] if (DIRECT or FREEZE or FRZH) else range(1,H+1)):
    for i,_ in hid_inputs[j]:
        if   GATE=="none": upd(f"u{j}_{i}", f"d1pre_{j}", href, au[i], hidw[j][i], HWTW)
        elif GATE=="hv":   upd(f"u{j}_{i}", f"d1pre_{j}", href, au[i], hidw[j][i], HWTW, gated=True, rg=f"hv_{j}")
        else:              upd(f"u{j}_{i}", f"d1pre_{j}", href, au[i], hidw[j][i], HWTW, gated=True, rg=f"uon_{j}")

# ---------------- weight-cap noise (trnoise current; zero B-source) ----------------
# Stochastic weight perturbation: an independent TRNOISE current source into each weight cap.
# Acts as SGLD-style exploration -- kicks units out of the dead/degenerate basin and breaks the
# symmetry that starves competition at a cold start.
WNOISE=ef("WNOISE","0.0")
allgp=([] if DIRECT else [hidw[j][i] for j in range(1,H+1) for i,_ in hid_inputs[j]])+[outw[c][j] for c in range(1,C+1) for j,_ in oc_inputs[c]]
if WNOISE>0:
    for g in allgp:
        w(f"In_{g} {g} 0 TRNOISE({WNOISE} {round(ef('NOISEDT',str(round(dt,4))),4)} 0 0)")

# ---------------- initial conditions + analysis ----------------
ICW=_os.environ.get("ICW","")
if ICW:
    _wv=np.loadtxt(ICW); _wv=_wv[-1,1::2] if _wv.ndim>1 else _wv[1::2]
    # only the first len(_wv) ic entries are the trainable WEIGHT caps (allgp order); any extra entries
    # (e.g. PERAZ oraz_c auto-zero caps) keep their default IC so resume doesn't overflow the weights file.
    ic=[(n,round(float(_wv[k]),4)) if k<len(_wv) else (n,_v) for k,(n,_v) in enumerate(ic)]
w(".ic "+" ".join(f"v({n})={v}" for n,v in ic))
w(".control")
if WNOISE>0: w(f"  set rndseed={ei('NSEED','1')}")
w(f"  tran {dt} {Nstp*Ts} uic")
yt=" ".join(f"v(y_{c})" for c in range(1,C+1))
at=" ".join(f"v(a{i})" for i in range(1,D+1) if _used_feats is None or i in _used_feats)
tt=" ".join(f"v(tg{c})" for c in range(1,C+1))
w(f"  wrdata mc_trace{TAG}.txt {yt} {at} {tt}")
if ei("PROBED2E",0):   # log the per-class error node d2e_c (+ oref) to check for per-class update bias
    w("  wrdata mc_d2e.txt "+" ".join(f"v(d2e_{c})" for c in range(1,C+1))+f" v({oref})")
if ei("PROBE",0) and not DIRECT:   # backward-fidelity probe: log circuit backward d1pre_j, z1_j, hv_j
    pr=" ".join(f"v(d1pre_{j})" for j in range(1,H+1))+" "+" ".join(f"v(z1_{j})" for j in range(1,H+1))+" "+" ".join(f"v(hv_{j})" for j in range(1,H+1))+" "+" ".join(f"v(uon_{j})" for j in range(1,H+1))
    w(f"  wrdata mc_probe.txt {pr}")
w(f"  wrdata mc_weights{TAG}.txt "+" ".join(f"v({g})" for g in allgp))
w("  echo MC_OK")
w(".endc"); w(".end")
open(f"mc{TAG}.cir","w").write("\n".join(L)+"\n")
nmos=sum(1 for s in L if s[:1]=="M")
print(f"D={D} H={H} C={C} COMP={COMP} NT={NTRAIN} Cg={Cg} spread={SPREAD} npat={len(order)} MOS={nmos} weights={len(allgp)}")
