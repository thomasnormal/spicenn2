#!/usr/bin/env python3
"""FULLY IN-SPICE learner for 2D nonlinear tasks (circles/rings/spirals).
Both INFERENCE and TRAINING are analog (transistors/R/C/V only, no behavioral sources):
  - HIDDEN: fixed RANDOM analog features (random weight VOLTAGES -> degenerated gsyn -> dneuron -> a_h).
  - READOUT: weights = CHARGE ON CAPS (wp/wn); gsyn reads them; a physical gprod cell charges them by the
    LOCAL product eps_c . a_h_j  (continuous PC weight rule dW/dt ~ eps.presyn).  esub computes eps = label - mu.
  - ONE .tran: phase1 TRAIN (output hard-clamped to one-hot label, learning tail GBL on);
               phase2 EVAL  (clamp neutral, GBL=0 -> caps frozen) -> read mu_o per test example -> argmax.
The CONTROLLER only writes the PWL schedule (which input+label to present when) = cycling the data.  Nothing
computes a gradient or a weight update outside SPICE.
"""
import os, numpy as np, subprocess
TASK=os.environ.get("TASK","circles"); C=int(os.environ.get("C","2")); H=int(os.environ.get("H","48"))
SEED=int(os.environ.get("SEED","0")); NTR=int(os.environ.get("NTR","48")); NTE=int(os.environ.get("NTE","60"))
NEP=int(os.environ.get("NEP","16")); SCALE=float(os.environ.get("SCALE","1.5")); RWS=float(os.environ.get("RWS","1.5"))
IND=float(os.environ.get("IND","0.30")); KMAP=float(os.environ.get("KMAP","0.30")); VW0=0.5; TD=float(os.environ.get("TD","0.20"))
RDEG=os.environ.get("RDEG","12k"); GDEG=os.environ.get("GDEG","12k"); WL=os.environ.get("WL","8000u"); RCS=os.environ.get("RCS","300k")
RDEGR=os.environ.get("RDEGR",RDEG); WLRO=os.environ.get("WLRO","3000u"); RPD=os.environ.get("RPD","150k")   # readout cmld: weaker pull-up + pull-down -> mu-CM near 0.5
RNODE=os.environ.get("RNODE","20k"); RMO=os.environ.get("RMO","30k"); VBSYN=os.environ.get("VBSYN","0.6")
VBNEU=os.environ.get("VBNEU","0.35"); GBL=os.environ.get("GBL","0.85"); CWW=os.environ.get("CWW","20p")
RWL=os.environ.get("RWL","2g"); TH=float(os.environ.get("TH","120")); STEP=os.environ.get("STEP","2n")
CSH=os.environ.get("CSH","1e-13"); SO=float(os.environ.get("SO","-1")); SGN=float(os.environ.get("SGN","1"))
TAG=os.environ.get("RUNTAG",str(os.getpid())); rng=np.random.default_rng(SEED)
DEPLOYW=None   # if set (C x H+1), caps are .ic'd to these weights (readout-capacity / hold test)
LEARNON=False  # if True, keep learning ON even when DEPLOYW set (hold test: does it hold the good weights?)
# ---------- data ----------
def gen():
    rg=np.random.default_rng(7); X=[];y=[]
    if TASK=="circles":
        for c in range(2):
            r=(rg.uniform(0.2,0.6,1500) if c==0 else rg.uniform(1.0,1.5,1500)); th=rg.uniform(0,2*np.pi,1500)
            X+=list(np.c_[r*np.cos(th),r*np.sin(th)]); y+=[c]*1500
    elif TASK=="blobs":   # TRIVIAL linearly-separable diagnostic
        for c in range(C):
            cx=-2.2 if c==0 else 2.2
            X+=list(np.c_[rg.normal(cx,0.35,1500), rg.normal(0,0.35,1500)]); y+=[c]*1500
    elif TASK=="rings":
        sep=float(os.environ.get("RINGSEP","1.2")); w=float(os.environ.get("RINGW","0.30")); base=float(os.environ.get("RINGBASE","0.25"))
        for c in range(C):
            r=rg.uniform(sep*c+base,sep*c+base+w,1500); th=rg.uniform(0,2*np.pi,1500); X+=list(np.c_[r*np.cos(th),r*np.sin(th)]); y+=[c]*1500
    elif TASK=="spirals":
        for c in range(C):
            t=np.linspace(0.2,1,1500); r=t*2.5; th=t*1.8*np.pi+c*2*np.pi/C+rg.normal(0,0.08,1500); X+=list(np.c_[r*np.cos(th),r*np.sin(th)]); y+=[c]*1500
    return np.array(X),np.array(y)
X,y=gen(); m=X.mean(0); s=X.std(0)+1e-6; X=(X-m)/s*SCALE; X=np.c_[X,np.ones(len(X))]; N=X.shape[1]
Xtr=[];ytr=[];Xte=[];yte=[]
for c in range(C):
    i=np.where(y==c)[0]; rng.shuffle(i)
    for k in i[:NTR]: Xtr.append(X[k]); ytr.append(c)
    for k in i[NTR:NTR+NTE]: Xte.append(X[k]); yte.append(c)
Xtr=np.array(Xtr); ytr=np.array(ytr); Xte=np.array(Xte); yte=np.array(yte)
WIH=rng.standard_normal((H,N))*RWS   # FIXED random hidden weights (become fixed voltages)
WIH[:,N-1]*=float(os.environ.get("BIASW","1.0"))   # scale the hidden bias-input weight (source of per-feature DC)
def wv(w): return float(np.clip(VW0+KMAP*w,0.3,0.9))
# ---------- cells (all pure-transistor; gsyn degenerated for linearity) ----------
_g=f""".subckt gsyn inp inn wp wn outp outn vdd vbn
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
.ends"""
_gR=f""".subckt gsynR inp inn wp wn outp outn vdd vbn
Mtail nt vbn 0 0 NNR W=400u L=100u
M5 na wp s5 0 NNR W=200u L=100u
Rd5 s5 nt {RDEGR}
M6 nb wn s6 0 NNR W=200u L=100u
Rd6 s6 nt {RDEGR}
M1 outp inp s1 0 NNR W=200u L=100u
Rd1 s1 na {RDEGR}
M2 outn inn s2 0 NNR W=200u L=100u
Rd2 s2 na {RDEGR}
M3 outn inp s3 0 NNR W=200u L=100u
Rd3 s3 nb {RDEGR}
M4 outp inn s4 0 NNR W=200u L=100u
Rd4 s4 nb {RDEGR}
.ends"""
_gP=f""".subckt gsynP inp inn wp wn outp outn vdd vbn
Mtail nt vbn 0 0 NNR W=400u L=100u
M5 na wp s5 0 NNR W=200u L=100u
Rd5 s5 nt {os.environ.get("RDEGP","1k")}
M6 nb wn s6 0 NNR W=200u L=100u
Rd6 s6 nt {os.environ.get("RDEGP","1k")}
M1 outp inp na 0 NNR W=200u L=100u
M2 outn inn na 0 NNR W=200u L=100u
M3 outn inp nb 0 NNR W=200u L=100u
M4 outp inn nb 0 NNR W=200u L=100u
.ends"""
SUB=f""".model NNR NMOS (LEVEL=1 VTO=0.2 KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)
.model PNR PMOS (LEVEL=1 VTO=-0.2 KP=40u LAMBDA=0.02 GAMMA=0 PHI=0.7)
{_g}
{_gR}
{_gP}
.subckt cmld p n vdd
Rcs1 p cmx {RCS}
Rcs2 n cmx {RCS}
MLp p cmx vdd vdd PNR W={WL} L=100u
MLn n cmx vdd vdd PNR W={WL} L=100u
.ends
.subckt cmldW p n vdd
Rcs1 p cmx {RCS}
Rcs2 n cmx {RCS}
MLp p cmx vdd vdd PNR W={WLRO} L=100u
MLn n cmx vdd vdd PNR W={WLRO} L=100u
Rpd1 p 0 {RPD}
Rpd2 n 0 {RPD}
.ends
.subckt cmldR p n vdd vref vbn
Rcs1 p cmx {RCS}
Rcs2 n cmx {RCS}
Mt nt vbn 0 0 NNR W=200u L=100u
Ma da cmx nt 0 NNR W=100u L=100u
Mb db vref nt 0 NNR W=100u L=100u
MLa da da vdd vdd PNR W=100u L=100u
MLb db da vdd vdd PNR W=100u L=100u
MLp p db vdd vdd PNR W={WL} L=100u
MLn n db vdd vdd PNR W={WL} L=100u
.ends
.subckt dneuron up un ap an vdd vbn
M1 an up tn 0 NNR W=200u L=100u
M2 ap un tn 0 NNR W=200u L=100u
Mt tn vbn 0 0 NNR W=200u L=100u
Xcm ap an vdd cmld
.ends
.subckt esub xp xn mp mn ep en vdd vbn
Mtx ntx vbn 0 0 NNR W=400u L=100u
M1x ep xn ntx 0 NNR W=200u L=100u
M2x en xp ntx 0 NNR W=200u L=100u
Mtm ntm vbn 0 0 NNR W=400u L=100u
M1m ep mp ntm 0 NNR W=200u L=100u
M2m en mn ntm 0 NNR W=200u L=100u
RLp vdd ep 50k
RLn vdd en 50k
.ends
.subckt gprod xp xn yp yn wp wn vdd vbn
Mtail nt vbn 0 0 NNR W=400u L=100u
M5 na yp sy5 0 NNR W=200u L=100u
Rg5 sy5 nt {GDEG}
M6 nb yn sy6 0 NNR W=200u L=100u
Rg6 sy6 nt {GDEG}
M1 iop xp sx1 0 NNR W=200u L=100u
Rg1 sx1 na {GDEG}
M2 ion xn sx2 0 NNR W=200u L=100u
Rg2 sx2 na {GDEG}
M3 ion xp sx3 0 NNR W=200u L=100u
Rg3 sx3 nb {GDEG}
M4 iop xn sx4 0 NNR W=200u L=100u
Rg4 sx4 nb {GDEG}
Mlp iop iop vdd vdd PNR W=200u L=100u
Mln ion ion vdd vdd PNR W=200u L=100u
Mpu_wp wp iop vdd vdd PNR W=200u L=100u
MpRn nrn ion vdd vdd PNR W=200u L=100u
MnRn nrn nrn 0 0 NNR W=200u L=100u
Mpd_wp wp nrn 0 0 NNR W=200u L=100u
Mpu_wn wn ion vdd vdd PNR W=200u L=100u
MpRp nrp iop vdd vdd PNR W=200u L=100u
MnRp nrp nrp 0 0 NNR W=200u L=100u
Mpd_wn wn nrp 0 0 NNR W=200u L=100u
.ends
.subckt comp inp inn out vdd vbn
Mt nt vbn 0 0 NNR W=200u L=100u
M1 d1 inp nt 0 NNR W=200u L=100u
M2 out inn nt 0 NNR W=200u L=100u
M3 d1 d1 vdd vdd PNR W=200u L=100u
M4 out d1 vdd vdd PNR W=200u L=100u
.ends"""
def stepped(vals):
    pts=[]
    for k,(t,v) in enumerate(vals):
        if k>0: pts.append((t-0.1,vals[k-1][1]))
        pts.append((t,v))
    return "PWL("+" ".join(f"{t:.1f}n {v:.5f}" for t,v in pts)+")"
# ---------- schedule: phase1 train (NEP*NTR presentations), phase2 eval (NTE) ----------
EVAL_BLOCKS=[]   # list of (start_slot, [labels]) for interval-eval early-stop
def build():
    global EVAL_BLOCKS
    EVALEV=int(os.environ.get("EVALEV","6"))   # run an eval block every EVALEV epochs (controller monitors -> early-stop)
    slots=[]   # (x, label or None=eval)
    EVAL_BLOCKS=[]
    if int(os.environ.get("DUMPALL","0")):    # forward-only over ALL examples (train+test), dump a_h to extract features
        global ALL_X,ALL_Y
        ALL_X=np.concatenate([Xtr,Xte]); ALL_Y=np.concatenate([ytr,yte])
        st=len(slots); EVAL_BLOCKS.append((st,list(ALL_Y)))
        for k in range(len(ALL_Y)): slots.append((ALL_X[k],None))
    elif DEPLOYW is not None and not LEARNON:   # capacity/hold test: no interval eval, just train(frozen)+eval
        for ep in range(NEP):
            for k in rng.permutation(len(ytr)): slots.append((Xtr[k],ytr[k]))
        st=len(slots); EVAL_BLOCKS.append((st,list(yte)))
        for k in range(len(yte)): slots.append((Xte[k],None))
    else:
        for ep in range(NEP):
            for k in rng.permutation(len(ytr)): slots.append((Xtr[k],ytr[k]))
            if (ep+1)%EVALEV==0 or ep==NEP-1:
                st=len(slots); EVAL_BLOCKS.append((st,list(yte)))
                for k in range(len(yte)): slots.append((Xte[k],None))
    NP=len(slots); TEND=NP*TH; NTRAIN=NP-sum(len(b[1]) for b in EVAL_BLOCKS)
    inp=[[] for _ in range(N)]; clp=[[] for _ in range(C)]; gblv=[]; ccmv=[]; _tc=[0]
    CCM_HI=float(os.environ.get("CCM_HI","0.5")); CCM_LO=float(os.environ.get("CCM_LO",os.environ.get("CLAMPCM","0.5")))  # clamp-CM schedule tracks mu-CM drift
    for g,(x,lab) in enumerate(slots):
        t=g*TH
        if lab is None:                                                          # EVAL slot: learning OFF, clamp neutral
            gblv.append((t,0.0)); ccmv.append((t,CCM_LO))
            for c in range(C): clp[c].append((t,0.5))
        else:
            if int(os.environ.get("GBLANN","0")):
                _lo=float(os.environ.get("GBLLO","0.35")); _gb=_lo+(float(GBL)-_lo)*((1-_tc[0]/max(1,NTRAIN))**float(os.environ.get("ANNEXP","1.5")))
            else: _gb=float(GBL)
            ccmv.append((t, CCM_HI+(CCM_LO-CCM_HI)*(_tc[0]/max(1,NTRAIN))))   # schedule clamp-CM HI->LO over training
            _tc[0]+=1
            _sf=float(os.environ.get("SETTLE","0"))   # SETTLE>0: hold GBL=0 for first SETTLE*TH (let forward settle), then charge
            if _sf>0: gblv.append((t,0.0)); gblv.append((t+_sf*TH,_gb))
            else: gblv.append((t,_gb))
            for c in range(C): clp[c].append((t,0.5+(TD if c==lab else -TD)))
        for i in range(N): inp[i].append((t,0.5+IND*float(x[i])))
    L=["fully in-spice PC learner",SUB,"Vdd vdd 0 1.0",f"Vbsyn vbsyn 0 {VBSYN}",f"Vbneu vbneu 0 {VBNEU}","Vgm gm 0 0.6",f"Vwcm wcm 0 {VW0}"]
    L+=[f"Vgbl gbl 0 {stepped(gblv)}"]
    for i in range(N):
        L+=[f"Vxp{i} a_xp{i} 0 {stepped(inp[i])}",f"Vxn{i} a_xn{i} 0 {stepped([(t,1-v) for t,v in inp[i]])}"]
    for c in range(C):   # label clamp CM follows the ccmv schedule (matches drifting mu-CM -> kills esub offset)
        L+=[f"Vxop{c} xop{c} 0 {stepped([(t,cc+(v-0.5)) for (t,v),(_,cc) in zip(clp[c],ccmv)])}",
            f"Vxon{c} xon{c} 0 {stepped([(t,cc-(v-0.5)) for (t,v),(_,cc) in zip(clp[c],ccmv)])}"]
    # ----- HIDDEN layer (fixed-random, or TRAINED when TRAIN_HIDDEN=1) -----
    TRH=int(os.environ.get("TRAIN_HIDDEN","0"))   # 1 = train hidden weights too (2-layer in-circuit PC)
    CWWH=os.environ.get("CWWH",CWW); RWLH=os.environ.get("RWLH",RWL)
    ic=[]   # collected .ic for all trainable caps (hidden + readout)
    L+=[f"Vahcm ahcm 0 {os.environ.get('AHCM','0.666')}"]   # a_h CM ref for the AHSCALE attenuator
    if TRH: L+=[f"Vbbk vbbk 0 {os.environ.get('VBBK','0.5')}"]   # backward-synapse tail bias (only when 2-layer)
    POLY=int(os.environ.get("POLY","0"))   # >0 = DETERMINISTIC polynomial features (gsyn products) instead of random hidden
    for j in (range(H) if not POLY else range(0)):
        for i in range(N):
            wp,wn=f"wih_p_{i}_{j}",f"wih_n_{i}_{j}"
            if TRH:   # hidden weight = charge on caps, init to random WIH, leak to wcm
                L+=[f"Cwihp_{i}_{j} {wp} 0 {CWWH}",f"Cwihn_{i}_{j} {wn} 0 {CWWH}",
                    f"Rwihp_{i}_{j} {wp} wcm {RWLH}",f"Rwihn_{i}_{j} {wn} wcm {RWLH}"]
                ic+=[f".ic v({wp})={wv(WIH[j,i]):.4f} v({wn})={wv(-WIH[j,i]):.4f}"]
            else:
                L+=[f"Vwih_p_{i}_{j} {wp} 0 {wv(WIH[j,i])}",f"Vwih_n_{i}_{j} {wn} 0 {wv(-WIH[j,i])}"]
        xp,xn=f"xhp{j}",f"xhn{j}"
        L+=[f"Xcm_xh{j} {xp} {xn} vdd cmld",f"Rnd_xh{j} {xp} {xn} {RNODE}",f"Cxh{j}p {xp} 0 {CSH}",f"Cxh{j}n {xn} 0 {CSH}"]
        for i in range(N): L+=[f"Xih_{i}_{j} a_xp{i} a_xn{i} wih_p_{i}_{j} wih_n_{i}_{j} {xp} {xn} vdd vbsyn gsyn"]
        _ahs=float(os.environ.get("AHSCALE","1.0"))   # attenuate a_h amplitude into both cells' LINEAR range -> delta-rule ~ ridge
        if _ahs<0.999:
            Ra=10000.0; Rb=Ra*_ahs/(1-_ahs)
            L+=[f"Xn{j} {xp} {xn} ahr_p{j} ahr_n{j} vdd vbneu dneuron",
                f"Rahp{j} ahr_p{j} ahp{j} {Ra:.0f}",f"Rahq{j} ahp{j} ahcm {Rb:.0f}",
                f"Rahn{j} ahr_n{j} ahn{j} {Ra:.0f}",f"Rahm{j} ahn{j} ahcm {Rb:.0f}"]
        else:
            L+=[f"Xn{j} {xp} {xn} ahp{j} ahn{j} vdd vbneu dneuron"]
    # ----- learnable READOUT (cap weights + gprod learning), with a BIAS weight (constant feature) -----
    ABIAS=float(os.environ.get("ABIAS","0.30"))   # constant "bias feature" the readout learns a weight for
    L+=[f"Vahb_p ahbias_p 0 {0.5+ABIAS}",f"Vahb_n ahbias_n 0 {0.5-ABIAS}"]
    if POLY:   # polynomial monomial features x^a y^b (1<=a+b<=POLY), each = a gsyn product (all CM-normalized via cmld)
        PUN=float(os.environ.get("PUNITY","0.3"))   # the "unity" differential for degree-1 buffering / odd-degree multiplicand
        L+=[f"Vpunp punp 0 {0.5+PUN}",f"Vpunn punn 0 {0.5-PUN}"]   # constant ~"1"
        PMIN=int(os.environ.get("POLYMIN","1"))   # min total degree (circles/rings need only degree-2: x^2,xy,y^2)
        monos=[(a,tot-a) for tot in range(PMIN,POLY+1) for a in range(tot+1)]
        raw={(1,0):("a_xp0","a_xn0"),(0,1):("a_xp1","a_xn1")}   # raw inputs (CM 0.5)
        nd={}
        def getm(a,b):
            if (a,b) in nd: return nd[(a,b)]
            if (a,b) in raw: n1=raw[(a,b)]; n2=("punp","punn")   # buffer raw input * unity -> cmld CM
            else:
                p,q=((1,0),(a-1,b)) if a>0 else ((0,1),(a,b-1))
                n1=raw[p]; n2=getm(*q)   # multiply RAW input (full-scale) by the lower monomial -> products don't vanish
            k=f"mono{a}_{b}"; fp,fn=k+"p",k+"n"; _pg="gsynP" if int(os.environ.get("POLYGAIN","0")) else "gsyn"
            L.extend([f"Xpoly_{k} {n1[0]} {n1[1]} {n2[0]} {n2[1]} {fp} {fn} vdd vbsyn {_pg}",
                      f"Xcm_{k} {fp} {fn} vdd cmld",f"Rpoly_{k} {fp} {fn} {RNODE}",
                      f"Cpp_{k} {fp} 0 {CSH}",f"Cpn_{k} {fn} 0 {CSH}"])
            PN=int(os.environ.get("POLYNORM","0"))
            if PN>0 and a+b>=int(os.environ.get("PNMIN","2")):   # CASCADE PN dneurons to amplify the tiny product back to full scale (each ~10x gain)
                cur=(fp,fn)
                for st in range(PN):
                    np_,nn_=f"{k}_s{st}p",f"{k}_s{st}n"; L.append(f"Xpn{st}_{k} {cur[0]} {cur[1]} {np_} {nn_} vdd vbneu dneuron"); cur=(np_,nn_)
                nd[(a,b)]=cur; return cur
            nd[(a,b)]=(fp,fn); return (fp,fn)
        feats=[getm(a,b) for (a,b) in monos]+[("ahbias_p","ahbias_n")]
    else:
        feats=[(f"ahp{j}",f"ahn{j}") for j in range(H)]+[("ahbias_p","ahbias_n")]   # H features + 1 bias
    if int(os.environ.get("CMSUB","1")): L+=[f"Cmocmp mocmp 0 {CSH}",f"Cmocmn mocmn 0 {CSH}"]   # output common-mode avg nodes
    CMFB=int(os.environ.get("CMFB","0")); VREF=os.environ.get("VREF","0.5")
    if CMFB: L+=[f"Vcmref cmref 0 {VREF}"]
    for c in range(C):
        if int(os.environ.get("CMW","0")): rload=f"Xcm_mo{c} mop{c} mon{c} vdd cmldW"   # passive weaker pull-up -> mu-CM near 0.5
        elif CMFB: rload=f"Xcm_mo{c} mop{c} mon{c} vdd cmref vbneu cmldR"
        else: rload=f"Xcm_mo{c} mop{c} mon{c} vdd cmld"
        L+=[rload,f"Rmo{c} mop{c} mon{c} {RMO}",f"Cmo{c}p mop{c} 0 {CSH}",f"Cmo{c}n mon{c} 0 {CSH}"]
        for j,(ap,an) in enumerate(feats):
            wp,wn=f"who_p_{j}_{c}",f"who_n_{j}_{c}"
            L+=[f"Cwp_{j}_{c} {wp} 0 {CWW}",f"Cwn_{j}_{c} {wn} 0 {CWW}",
                f"Rwp_{j}_{c} {wp} wcm {RWL}",f"Rwn_{j}_{c} {wn} wcm {RWL}",
                f"Xsyn_{j}_{c} {ap} {an} {wp} {wn} mop{c} mon{c} vdd vbsyn gsynR"]
            if DEPLOYW is not None: ic+=[f".ic v({wp})={wv(DEPLOYW[c][j]):.4f} v({wn})={wv(-DEPLOYW[c][j]):.4f}"]
            else: ic+=[f".ic v({wp})={VW0} v({wn})={VW0}"]   # start at zero weight
        if int(os.environ.get("CMSUB","1")):   # subtract OUTPUT common-mode (GLOBALAZ): error on CENTERED mu_o
            L+=[f"Rcmp{c} mop{c} mocmp {os.environ.get('RCM','40k')}",f"Rcmn{c} mon{c} mocmn {os.environ.get('RCM','40k')}",
                f"Xcms{c} mop{c} mon{c} mocmp mocmn cmop{c} cmon{c} vdd gm esub",   # cmo_c = mu_c - mean_c(mu)
                f"Xeo{c} xop{c} xon{c} cmop{c} cmon{c} eop{c} eon{c} vdd gm esub"]  # eps_c = label - centered mu_c
        else:
            L+=[f"Xeo{c} xop{c} xon{c} mop{c} mon{c} eop{c} eon{c} vdd gm esub"]   # eps_c = label - mu_c (regression)
    # learning: gprod charges each readout cap by eps_c . feature_j.
    # AC-COUPLE the feature into the gprod (cap blocks the per-feature DC mean -> learning sees only the
    # discriminative VARIATION, killing the common-mode drift).  Forward readout still uses the full a_h.
    ACC=int(os.environ.get("ACC","1")); CAC=os.environ.get("CAC","2p"); RAC=os.environ.get("RAC","20meg")
    ep_,en_=("eop","eon") if SGN>0 else ("eon","eop")
    if ACC:
        L+=["Vahref vahref 0 0.5"]
        gfeats=[]
        for j,(ap,an) in enumerate(feats):
            if j==H: gfeats.append((ap,an)); continue   # bias feature is pure DC -> don't AC-couple it
            yp,yn=f"yac_p_{j}",f"yac_n_{j}"
            L+=[f"Cac_p_{j} {ap} {yp} {CAC}",f"Rac_p_{j} {yp} vahref {RAC}",
                f"Cac_n_{j} {an} {yn} {CAC}",f"Rac_n_{j} {yn} vahref {RAC}"]
            gfeats.append((yp,yn))
    else:
        gfeats=feats
    if int(os.environ.get("ACOMP","0")):   # compress each a_h feature with a dneuron (tanh) into the gprod -> matches prototype's compressive feature
        cf=[]
        for j,(ap,an) in enumerate(gfeats):
            cp,cn=f"acmp_p_{j}",f"acmp_n_{j}"
            L+=[f"Xacmp_{j} {ap} {an} {cp} {cn} vdd vbneu dneuron"]
            cf.append((cp,cn))
        gfeats=cf
    if int(os.environ.get("ATR","0")):   # GRADIENT-AWARE feature: pass a_h through a gsyn REPLICA (same as readout a_h-stage)
        L+=[f"Vwfp wfp 0 {0.5+float(os.environ.get('WFREF','0.10'))}",f"Vwfn wfn 0 {0.5-float(os.environ.get('WFREF','0.10'))}"]
        tf=[]
        for j,(ap,an) in enumerate(gfeats):
            tp,tn=f"atr_p_{j}",f"atr_n_{j}"
            L+=[f"Xatr_{j} {ap} {an} wfp wfn {tp} {tn} vdd vbsyn gsynR",   # gsyn(a_h, w_ref) -> T(a_h) = readout's a_h transconductance
                f"Xcm_atr{j} {tp} {tn} vdd cmld",f"Ratr{j} {tp} {tn} {os.environ.get('RATR','40k')}"]
            tf.append((tp,tn))
        gfeats=tf
    # HINGE: gate learning OFF for an example once it is confidently correct (stop-when-correct).
    # score = sum_c (xop_c-xon_c)*(cmo_c)  (clamp polarity . centered output) ~ correct-class margin.
    # OTA comparator -> conf high when margin > 0 -> pull gblg low (turns off all gprod tails for that example).
    TAIL="gbl"
    if int(os.environ.get("HINGE","0")):
        VTH=float(os.environ.get("VTH","0.50"))   # working gate threshold (margin needed to gate, late training)
        VTHHI=float(os.environ.get("VTHHI","0.95")); VTHFRAC=float(os.environ.get("VTHFRAC","0.5"))   # SCHEDULE:
        vthv=[(g*TH, VTHHI if (DEPLOYW is None and g<NP*VTHFRAC) else VTH) for g in range(NP)]   # gate OFF early (bootstrap) -> ON (refine)
        for c in range(C): L+=[f"Xsc{c} xop{c} xon{c} mop{c} mon{c} sop son vdd vbsyn gsyn"]   # raw mo: CM cancels (zero-mean label)
        L+=[f"Xcm_s sop son vdd cmld",f"Rs sop son {os.environ.get('RS','40k')}",
            f"Vth sref 0 {stepped(vthv)}",
            f"Xcomp sref sop cdum conf vdd vbneu dneuron",                       # SOFT gate (low-gain dneuron): conf high when margin>thr
            f"Rgbl gbl gblg {os.environ.get('RGBL','30k')}",
            f"Vmsrc msrc 0 {os.environ.get('MSRC','0.18')}",                      # offset so Mgate is OFF until conf>MSRC+Vt
            f"Mgate gblg conf msrc 0 NNR W=600u L=100u",                          # conf high -> gblg->msrc(~0.18<Vt) -> tail OFF (learning off)
            f"Cconf conf 0 {os.environ.get('CCONF','2p')}",f"Csop sop 0 {os.environ.get('CSOP','2p')}",
            f"Cgblg gblg 0 {os.environ.get('CGBLG','2p')}"]                       # light filter (dneuron is non-stiff)
        TAIL="gblg"
    FZ=int(os.environ.get("FORCEZERO","0"))   # tie gprod eps inputs to a common ref -> eps=0 forced (localize offset: esub vs gprod)
    for c in range(C):
        for j,(ap,an) in enumerate(gfeats):
            xp_,xn_=(("gm","gm") if FZ else (f"{ep_}{c}",f"{en_}{c}"))
            L+=[f"Xlrn_{j}_{c} {xp_} {xn_} {ap} {an} who_p_{j}_{c} who_n_{j}_{c} vdd {TAIL} gprod"]
    # ===== 2-LAYER: analog backward error eps_h = W_ho^T . eps_o, then hidden gprod update =====
    if TRH and not FZ and DEPLOYW is None:
        RNH=os.environ.get("RNODEH",RNODE)
        for j in range(H):   # explicit hidden-error neuron: sum_c gsyn(eps_o_c, who_jc) -> ehp/ehn
            L+=[f"Xcm_eh{j} ehp{j} ehn{j} vdd cmld",f"Reh{j} ehp{j} ehn{j} {RNH}",
                f"Ceh{j}p ehp{j} 0 {CSH}",f"Ceh{j}n ehn{j} 0 {CSH}"]
            for c in range(C):   # backward synapse: input=eps_o, weight=who_jc (the SAME readout caps, read in reverse)
                L+=[f"Xbk_{j}_{c} {ep_}{c} {en_}{c} who_p_{j}_{c} who_n_{j}_{c} ehp{j} ehn{j} vdd vbbk gsyn"]
        ehp_,ehn_="ehp","ehn"
        if int(os.environ.get("CMSUBH","1")):   # subtract BACKWARD common-mode (resistor-star avg across hidden units) -> kills the offset that decays all hidden weights
            RCMB=os.environ.get("RCMB","10k")
            L+=[f"Cehbp ehbarp 0 {CSH}",f"Cehbn ehbarn 0 {CSH}"]
            for j in range(H): L+=[f"Rehbp_{j} ehp{j} ehbarp {RCMB}",f"Rehbn_{j} ehn{j} ehbarn {RCMB}"]
            for j in range(H): L+=[f"Xehc_{j} ehp{j} ehn{j} ehbarp ehbarn cehp{j} cehn{j} vdd gm esub"]  # ceh_j = eph_j - mean_k(eph_k)
            ehp_,ehn_="cehp","cehn"
        hxp,hxn=(ehn_,ehp_) if int(os.environ.get("HSGN","0")) else (ehp_,ehn_)   # HSGN flips hidden-update sign
        for j in range(H):   # hidden update: gprod charges wih caps by eps_h_j . x_in_i
            for i in range(N):
                L+=[f"Xlrh_{i}_{j} {hxp}{j} {hxn}{j} a_xp{i} a_xn{i} wih_p_{i}_{j} wih_n_{i}_{j} vdd {TAIL} gprod"]
    L+=ic
    cols=" ".join(f"v(mop{c}) v(mon{c})" for c in range(C))
    if int(os.environ.get("POLYDUMP","0")):
        _fn=[(f"mono{a}_{b}" if a+b<2 else f"mono{a}_{b}_s1") for tot in range(1,5) for a in range(tot+1) for b in [tot-a]]
        cols+=" "+" ".join(f"v({n}p) v({n}n)" for n in _fn)
    if int(os.environ.get("DUMPAH","0")): cols+=" "+" ".join(f"v(ahp{j}) v(ahn{j})" for j in range(H))
    if int(os.environ.get("DUMPW","0")): cols+=" "+" ".join(f"v(who_p_{j}_{c}) v(who_n_{j}_{c})" for c in range(C) for j in range(H))
    if int(os.environ.get("DUMPH","0")) and TRH: cols+=" "+" ".join([f"v(ehp{j}) v(ehn{j})" for j in range(min(4,H))]+[f"v(wih_p_0_{j}) v(wih_n_0_{j})" for j in range(min(4,H))])
    savenodes=" ".join(dict.fromkeys(cols.replace("v(","").replace(")","").split()))   # only store the wrdata nodes -> ~100x less memory
    savel=[f"save {savenodes}"] if int(os.environ.get("SAVE","1")) else []
    global COLNODES; COLNODES=[s for s in cols.replace("v(","").replace(")","").split()]
    if os.environ.get("SIM","ngspice")=="spectre":
        # Spectre SPICE-mode deck: no .control; .tran card; strobe output once per TH window to keep the PSF small.
        head=["simulator lang=spice","* "+L[0]]+[l for l in L[1:]]   # title -> comment; reuse models/subckts/elements/.ic
        strobe=f"strobeperiod={TH}n strobedelay={TH-3}n outputstart=0" if int(os.environ.get("STROBE","1")) else ""
        head+=[f".options reltol={os.environ.get('RELTOL','1e-3')} gmin=1e-10",
               f".tran {STEP} {int(TEND)}n uic {strobe}",
               f".save {savenodes}" if int(os.environ.get("SAVE","1")) else "",".end"]
        open(f"ps_{TAG}.scs","w").write("\n".join(x for x in head if x)+"\n")
        return TEND
    L+=[f".options gmin=1e-10 reltol={os.environ.get('RELTOL','2e-3')} abstol={os.environ.get('ABSTOL','1e-12')} vntol={os.environ.get('VNTOL','1e-6')} cshunt=1e-14 method=gear",".control"]+savel+[f"tran {STEP} {int(TEND)}n uic",
        f"wrdata ps_{TAG}.dat {cols}",".endc",".end"]
    open(f"ps_{TAG}.cir","w").write("\n".join(L)+"\n")
    return TEND
TEND=build()
NP1,NP2=EVAL_BLOCKS[-1][0],len(EVAL_BLOCKS[-1][1])
def parse_psf(globpat,nodes):
    import glob
    f=sorted(glob.glob(globpat))[0]; txt=open(f).read()
    vi=txt.index("\nVALUE\n"); body=txt[vi+7:]; nset=set(nodes)
    times=[]; data={n:[] for n in nodes}
    for line in body.splitlines():
        line=line.strip()
        if line=="END": break
        if not line or line[0]!='"': continue
        q=line.index('"',1); name=line[1:q]; rest=line[q+1:].strip()
        if name=="time": times.append(float(rest))
        elif name in nset: data[name].append(float(rest))
    t=np.array(times)*1e9; M=np.array([data[n] for n in nodes]).T
    return t,M
def run_and_parse():
    if os.environ.get("SIM","ngspice")=="spectre":
        import glob
        raw=f"raw_{TAG}"; spath="/opt/cadence/installs/SPECTRE231/bin"
        for f in glob.glob(f"{raw}/*.tran.tran"): os.remove(f)   # avoid stale parse on re-run (DEPLOY)
        r=subprocess.run([f"{spath}/spectre","+spice",f"+mt={os.environ.get('MT','8')}",f"+lqtimeout={os.environ.get('LQ','900')}",f"ps_{TAG}.scs","-format","psfascii","-raw",raw,"+log",f"ps_{TAG}.log"],
                         capture_output=True,text=True,timeout=2400,env={**os.environ,"PATH":spath+":"+os.environ.get("PATH","")})
        try:
            return parse_psf(f"{raw}/*.tran.tran",COLNODES)
        except Exception as e:
            print("PARSE FAIL",e); print((r.stdout or "")[-1500:]); print((r.stderr or "")[-800:]); raise SystemExit
    else:
        r=subprocess.run(["ngspice","-b",f"ps_{TAG}.cir"],capture_output=True,text=True,timeout=1800)
        try:
            d=np.loadtxt(f"ps_{TAG}.dat"); return d[:,0]*1e9, d[:,1::2]
        except Exception as e:
            print("PARSE FAIL",e); print(r.stderr[-800:]); raise SystemExit
t,cols=run_and_parse()
# read EACH eval block (interval eval), debias, argmax -> report BEST (early-stop) and FINAL
def block_acc(st,labs):
    MO=[]
    for g in range(len(labs)):
        slot=st+g; row=cols[np.argmin(np.abs(t-(slot*TH+TH-3)))]
        MO.append(np.array([SO*(row[2*c]-row[2*c+1]) for c in range(C)]))
    M=np.array(MO); M=M-M.mean(0)
    return float(np.mean([int(np.argmax(M[i]))==labs[i] for i in range(len(labs))]))
accs=[block_acc(st,labs) for st,labs in EVAL_BLOCKS]
acc=accs[-1]; best=max(accs)
print(f"[IN-SPICE] TASK={TASK} C={C} H={H} NEP={NEP} -> FINAL={acc:.3f} BEST(early-stop)={best:.3f}  curve={['%.2f'%a for a in accs]}")
if int(os.environ.get("DUMPALL","0")):   # extract real analog a_h for ALL examples -> prototype learns on them
    st,labs=EVAL_BLOCKS[0]; AH=[]
    for g in range(len(labs)):
        slot=st+g; row=cols[np.argmin(np.abs(t-(slot*TH+TH-3)))]
        AH.append(np.array([row[2*C+2*j]-row[2*C+2*j+1] for j in range(H)]))
    AH=np.array(AH); ntr=len(ytr)
    np.savez(f"/tmp/ah_{TASK}.npz",Atr=AH[:ntr],Ytr=ALL_Y[:ntr],Ate=AH[ntr:],Yte=ALL_Y[ntr:])
    print(f"[DUMPALL] saved /tmp/ah_{TASK}.npz  Atr={AH[:ntr].shape} Ate={AH[ntr:].shape} |a_h|mean={np.abs(AH).mean():.4f}")
    raise SystemExit
if int(os.environ.get("DUMPAH","0")):   # SANITY (numpy, not the learning path): are the analog a_h separable?
    st,LB=EVAL_BLOCKS[-1]; LB=np.array(LB); AH=[]
    for g in range(len(LB)):
        slot=st+g; row=cols[np.argmin(np.abs(t-(slot*TH+TH-3)))]
        AH.append(np.array([row[2*C+2*j]-row[2*C+2*j+1] for j in range(H)]))
    AH=np.array(AH); Ab=np.c_[AH,np.ones(len(AH))]
    T=np.full((len(LB),C),-1.0); T[np.arange(len(LB)),LB]=1
    W=np.linalg.solve(Ab.T@Ab+1e-1*np.eye(Ab.shape[1]),Ab.T@T)
    sep=np.mean(np.argmax(Ab@W,1)==LB)
    print(f"  [SANITY] ridge on analog a_h (eval set) sep acc={sep:.3f}  |a_h|mean={np.abs(AH).mean():.4f}")
    if int(os.environ.get("DUMPW","0")):   # compare LEARNED caps vs ridge direction
        off=2*C+2*H; slot=st+len(LB)-1; row=cols[np.argmin(np.abs(t-(slot*TH+TH-3)))]
        Wl=np.zeros((C,H))
        for c in range(C):
            for j in range(H): Wl[c,j]=row[off+2*(c*H+j)]-row[off+2*(c*H+j)+1]   # who_p - who_n (differential weight)
        Wr=W[:H,:].T   # ridge weight [c,j]
        def cs(a,b): a=a.ravel();b=b.ravel(); return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
        railed=np.mean((np.abs(Wl)>0.38))   # near the 0.3/0.9 voltage rails (diff>0.38 ~ railed)
        print(f"  [WDIAG] cos(learned_caps, ridge)={cs(Wl,Wr):+.3f}  |Wlearned|mean={np.abs(Wl).mean():.4f} max={np.abs(Wl).max():.3f} railed-frac={railed:.2f}")
    if int(os.environ.get("DEPLOY","0")):   # DEPLOY ridge weights into the IN-CIRCUIT readout -> capacity test
        if int(os.environ.get("LEARNDEPLOY","0")):   # deploy the LEARNED weights (scaled) -> magnitude vs direction test
            Wd=np.vstack([Wl.T, W[H:,:]])   # (H+1, C): learned H weights + ridge bias row
            print(f"  [LEARNDEPLOY] deploying LEARNED weights (cos-to-ridge known); scaled")
        else:
            Wd=W.copy()   # (H+1, C) ridge on analog a_h
        sc=float(os.environ.get("WSCALE","1.0"))/(np.abs(Wd).max()+1e-9)
        DEPLOYW=(Wd*sc).T   # (C, H+1) -> caps; build() reads this global, sets .ic, GBL=0
        build()
        t2,c2=run_and_parse()
        MO2=[];LB2=[]
        for g in range(NP2):
            slot=NP1+g; row=c2[np.argmin(np.abs(t2-(slot*TH+TH-3)))]
            MO2.append(np.array([SO*(row[2*c]-row[2*c+1]) for c in range(C)])); LB2.append(yte[g])
        MO2=np.array(MO2); LB2=np.array(LB2); M2=MO2-MO2.mean(0)
        accd=float(np.mean([int(np.argmax(M2[i]))==LB2[i] for i in range(len(LB2))]))
        print(f"  [DEPLOY] in-circuit readout w/ RIDGE weights (scale {sc:.2f}): test acc={accd:.3f} |mu_o|mean={np.abs(MO2).mean():.4f}  <- readout CAPACITY")
        if int(os.environ.get("HOLDTEST","0")):   # start at ridge, turn LEARNING ON -> does it HOLD or drift?
            LEARNON=True; build()
            t3,c3=run_and_parse()
            MO3=[];LB3=[]
            for g in range(NP2):
                slot=NP1+g; row=c3[np.argmin(np.abs(t3-(slot*TH+TH-3)))]
                MO3.append(np.array([SO*(row[2*c]-row[2*c+1]) for c in range(C)])); LB3.append(yte[g])
            MO3=np.array(MO3); M3=MO3-MO3.mean(0)
            acch=float(np.mean([int(np.argmax(M3[i]))==np.array(LB3)[i] for i in range(len(LB3))]))
            print(f"  [HOLDTEST] start@ridge, learning ON -> test acc={acch:.3f}  (holds=>learning-OK, drifts=>wrong attractor)")
