#!/usr/bin/env python3
"""gen_pcL adapted to the 2D tasks (circles/rings/spirals): the SAME fully-in-spice 2-layer continuous PC
(Gilbert gsyn synapses, dneuron nonlinearity, esub error cells, gprod cells that PHYSICALLY charge the weight
caps -> both layers trained in-circuit by their local products eps.presyn).  Controller only writes the PWL
data schedule: train patterns are output-clamped to the label; a trailing eval block streams TEST patterns
with a NEUTRAL clamp so the read mu_o is the free forward prediction.  0 behavioral elements."""
import os, numpy as np, subprocess, re
H=int(os.environ.get("H","16")); SEED=int(os.environ.get("SEED","0")); TASK=os.environ.get("TASK","circles")
IND=float(os.environ.get("IND","0.45")); TD=float(os.environ.get("TD","0.20"))
KMAP=float(os.environ.get("KMAP","0.30")); VW0=float(os.environ.get("VW0","0.5"))
WL=os.environ.get("WL","1000u"); RCS=os.environ.get("RCS","300k"); RNODE=os.environ.get("RNODE","20k"); RNODEO=os.environ.get("RNODEO",os.environ.get("RNODE","20k"))
VBSYN=os.environ.get("VBSYN","0.45"); VBNEU=os.environ.get("VBNEU","0.35"); GMT=os.environ.get("GMT","0.6"); GMTO=os.environ.get("GMTO",GMT)
GBLH=os.environ.get("GBLH","0.85"); GBLO=os.environ.get("GBLO","0.85")
CWW=os.environ.get("CWW","30p"); RWL=os.environ.get("RWL","2g")
SGNH=float(os.environ.get("SGNH","1")); SGNO=float(os.environ.get("SGNO","1"))
TH=float(os.environ.get("TH","250")); NEP=int(os.environ.get("NEP","10"))
NTR=int(os.environ.get("NTR","80")); NTE=int(os.environ.get("NTE","200")); INSC=float(os.environ.get("INSC","1.0"))
TAG=os.environ.get("RUNTAG",str(os.getpid()))
def ts(b): return 1.0 if b>0 else -1.0
rng=np.random.default_rng(SEED)
# ---- data (npz are min-max [0,1]; map to [-1,1]*INSC) ----
dtr=np.load(f"/tmp/insp/{TASK}_train.npz"); Xtr=(2*dtr['X']-1)*INSC; Ytr=dtr['Y']
dte=np.load(f"/tmp/insp/{TASK}_test.npz");  Xte=(2*dte['X']-1)*INSC; Yte=dte['Y']
NTR=min(NTR,len(Xtr)); NTE=min(NTE,len(Xte)); Xtr,Ytr=Xtr[:NTR],Ytr[:NTR]; Xte,Yte=Xte[:NTE],Yte[:NTE]
# ---- subckts: reuse gen_pcL's proven SUB verbatim ----
src=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'gen_pcL.py')).read()
SUBT=re.search(r'SUB=f"""(.*?)"""',src,re.S).group(1); SUB=SUBT.replace("{RCS}",RCS).replace("{WL}",WL)
OTAW=os.environ.get("OTAW","20")
SUB+="""
.subckt otau ep en actg gp vdd
Mt ts actg 0 0 NNR W="""+OTAW+"""u L=1u
Mo1 da en ts 0 NNR W=20u L=1u
Mo2 gp ep ts 0 NNR W=20u L=1u
Mo3 da da vdd vdd PNR W=60u L=1u
Mo4 gp da vdd vdd PNR W=60u L=1u
.ends"""
HIN=(1,2,3) if int(os.environ.get("HBIAS","1")) else (1,2)   # i=3 is a constant BIAS input (essential for radial features)
OBIAS=int(os.environ.get("OBIAS","1"))   # output BIAS unit (the readout intercept) -- essential for the decision threshold
WK=[]
for j in range(H):
    for i in HIN: WK.append(f"wih_{i}_{j}")
    WK.append(f"who_{j}")
if OBIAS: WK.append("who_b")
w0={k:float(0.5*rng.standard_normal()) for k in WK}
WCLO=float(os.environ.get("WCLO","0.3")); WCHI=float(os.environ.get("WCHI","0.9"))  # weight-cap voltage clip (wider=bigger projection)
def wgv(v): return float(np.clip(VW0+KMAP*v,WCLO,WCHI)),float(np.clip(VW0-KMAP*v,WCLO,WCHI))
def stepped(vals):
    pts=[]
    for k,(t,v) in enumerate(vals):
        if k>0: pts.append((t-0.1,vals[k-1][1]))
        pts.append((t,v))
    return "PWL("+" ".join(f"{t:.1f}n {v:.4f}" for t,v in pts)+")"
def syn(tag,ip,inn,op,on,key,vbn="vbsyn"): return [f"X_{tag} {ip} {inn} wp_{key} wn_{key} {op} {on} vdd {vbn} gsyn"]
VBSYNO=os.environ.get("VBSYNO",VBSYN)   # SEPARATE bias for the output readout gsyn (wider linear range -> linear Sum who*a_h)
PROBEFEAT=int(os.environ.get("PROBEFEAT","0"))
def gen():
    # schedule: [NEP epochs x NTR train patterns (clamped)] then [NTE test patterns (neutral clamp = eval)]
    sched=[]
    if PROBEFEAT:   # pure-forward feature probe: train+test patterns, NO learning/clamp (eval_start=0)
        for k in range(NTR): sched.append((Xtr[k,0],Xtr[k,1],None))
        for k in range(NTE): sched.append((Xte[k,0],Xte[k,1],None))
        eval_start=0
        return _emit(sched,eval_start)
    for ep in range(NEP):
        for k in rng.permutation(NTR): sched.append((Xtr[k,0],Xtr[k,1],ts(Ytr[k])))
    eval_start=len(sched)
    for k in range(NTE): sched.append((Xte[k,0],Xte[k,1],None))
    return _emit(sched,eval_start)
def _emit(sched,eval_start):
    NP=len(sched); TEND=NP*TH
    x1=[];x2=[];vp=[];vn=[]
    for g,(a,b,s) in enumerate(sched):
        t=g*TH
        x1.append((t,0.5+IND*float(a))); x2.append((t,0.5+IND*float(b)))
        td=TD*s if s is not None else 0.0; vp.append((t,0.5+td)); vn.append((t,0.5-td))
    # During eval: OUTPUT esub tail (gm_o) and gbl learning tails drop to 0 -> output is FREE (no clamp/backward
    # to the hidden) and weights frozen; HIDDEN esub tail (gm_h) STAYS ON so the hidden still tracks the input
    # (the feedforward pass). mo2 = who.a_h is then the clean prediction.
    et=eval_start*TH
    pwl0=lambda v: f"PWL(0 {v} {et-0.1:.1f}n {v} {et:.1f}n 0)"
    L=["pcL 2D",SUB,"Vdd vdd 0 1.0",f"Vbsyn vbsyn 0 {VBSYN}",f"Vbneu vbneu 0 {VBNEU}",
       f"Vbsyn_o vbsyn_o 0 {VBSYNO}",f"Vgm_h gm_h 0 {GMT}",f"Vgm_o gm_o 0 {pwl0(GMTO)}",
       f"Vgblh gblh 0 {pwl0(GBLH)}",f"Vgblo gblo 0 {pwl0(GBLO)}",f"Vwcm wcm 0 {VW0}",
       f"Vx1p a_xp1 0 {stepped(x1)}",f"Vx1n a_xn1 0 {stepped([(t,1-v) for t,v in x1])}",
       f"Vx2p a_xp2 0 {stepped(x2)}",f"Vx2n a_xn2 0 {stepped([(t,1-v) for t,v in x2])}",
       f"Vx3p a_xp3 0 {0.5+IND:.4f}",f"Vx3n a_xn3 0 {0.5-IND:.4f}",   # constant BIAS input (x3=+1)
       f"Vaobp aobp 0 {0.5+0.3:.4f}",f"Vaobn aobn 0 {0.5-0.3:.4f}",    # constant OUTPUT-bias unit
       f"Vxop xop 0 {stepped(vp)}",f"Vxon xon 0 {stepped(vn)}"]
    ic=[]
    ICWF=os.environ.get("ICWF","")   # resume: load saved weight-cap voltages as .ic (chunked training past the window)
    sav=None
    if ICWF and os.path.exists(ICWF):
        sav={}
        for line in open(ICWF):
            p=line.split()
            if len(p)==2: sav[p[0]]=p[1]
    for k in WK:
        gp,gn=wgv(w0[k])
        L+=[f"Cwp_{k} wp_{k} 0 {CWW}",f"Cwn_{k} wn_{k} 0 {CWW}",f"Rwp_{k} wp_{k} wcm {RWL}",f"Rwn_{k} wn_{k} wcm {RWL}"]
        if sav is not None: ic.append(f".ic v(wp_{k})={sav.get('wp_'+k,f'{gp:.4f}')} v(wn_{k})={sav.get('wn_'+k,f'{gn:.4f}')}")
        else: ic.append(f".ic v(wp_{k})={gp:.4f} v(wn_{k})={gn:.4f}")
    for j in range(H):
        xp,xn=f"xhp{j}",f"xhn{j}"; mp,mn=f"mhp{j}",f"mhn{j}"
        L+=[f"Xcm_mh{j} {mp} {mn} vdd cmld",f"Xcm_xh{j} {xp} {xn} vdd cmld",
            f"Rnd_mh{j} {mp} {mn} {RNODE}",f"Rnd_xh{j} {xp} {xn} {RNODE}",f"Cxhp{j} {xp} 0 0.4p",f"Cxhn{j} {xn} 0 0.4p"]
        for i in HIN:
            L+=syn(f"ihm_{i}_{j}",f"a_xp{i}",f"a_xn{i}",mp,mn,f"wih_{i}_{j}")
            L+=syn(f"ihx_{i}_{j}",f"a_xp{i}",f"a_xn{i}",xp,xn,f"wih_{i}_{j}")
        if not int(os.environ.get("FRZHID","0")):   # ELM: no backward error into the hidden -> a_h stays CLEAN
            L+=syn(f"bk_{j}","eop","eon",xp,xn,f"who_{j}")        # forward (matches eval), so who trains on the
        L+=[f"Xn{j} {xp} {xn} ahp{j} ahn{j} vdd vbneu dneuron"]   # same features it's evaluated on
        L+=[f"Xeh{j} {xp} {xn} {mp} {mn} ehp{j} ehn{j} vdd gm_h esub"]
    L+=["Xcm_mo mop mon vdd cmld",f"Rnd_mo mop mon {RNODEO}"]
    for j in range(H): L+=syn(f"ho_{j}",f"ahp{j}",f"ahn{j}","mop","mon",f"who_{j}","vbsyn_o")
    if OBIAS: L+=syn("ho_b","aobp","aobn","mop","mon","who_b","vbsyn_o")
    # error from the FORWARD prediction (mop2/mon2, no clamp feedback) -> clean eo = target - forward = true LMS
    # gradient (the clamped node mop/mon is pulled toward the label, contaminating the direction).
    _epred = "mop2 mon2" if int(os.environ.get("FWDERR","1")) else "mop mon"
    L+=[f"Xeo xop xon {_epred} eop eon vdd gm_o esub"]
    # FORWARD-ONLY readout (pure feedforward who.a_h, NO esub clamp feedback) for clean inference at eval
    L+=["Xcm_mo2 mop2 mon2 vdd cmld",f"Rnd_mo2 mop2 mon2 {RNODEO}"]
    for j in range(H): L+=syn(f"ho2_{j}",f"ahp{j}",f"ahn{j}","mop2","mon2",f"who_{j}","vbsyn_o")
    if OBIAS: L+=syn("ho2_b","aobp","aobn","mop2","mon2","who_b","vbsyn_o")
    eh_p,eh_n=("ehp","ehn") if SGNH>0 else ("ehn","ehp"); eo_p,eo_n=("eop","eon") if SGNO>0 else ("eon","eop")
    FRZHID=int(os.environ.get("FRZHID","0"))   # ELM: omit hidden-learning gprod cells entirely (frozen random hidden) -> smaller/faster deck
    # ACGP: AC-couple the feature into the LEARNING gprod -> removes the feature DC offset (per-unit mean ~0.4)
    # that contaminates the gradient direction. The READOUT (ho2) still uses the DC-coupled a_h. -> clean LMS.
    ACGP=int(os.environ.get("ACGP","0")); CHP=os.environ.get("CHP","2p"); RHP=os.environ.get("RHP","2meg")
    if ACGP:
        L+=["Vhpref hpref 0 0.5"]
        for j in range(H):
            L+=[f"Chp_p{j} ahp{j} ahph{j} {CHP}",f"Rhp_p{j} ahph{j} hpref {RHP}",
                f"Chp_n{j} ahn{j} ahnh{j} {CHP}",f"Rhp_n{j} ahnh{j} hpref {RHP}"]
    yh=lambda j: (f"ahph{j}",f"ahnh{j}") if ACGP else (f"ahp{j}",f"ahn{j}")
    for j in range(H):
        if not FRZHID:
            for i in HIN:
                L+=[f"Xlih_{i}_{j} {eh_p}{j} {eh_n}{j} a_xp{i} a_xn{i} wp_wih_{i}_{j} wn_wih_{i}_{j} vdd gblh gprod"]
        _yp,_yn=yh(j)
        if int(os.environ.get("OTAU","0")):   # OTA update (gen_mc-style, higher fidelity): dwp~eo*ahp, dwn~eo*ahn
            L+=[f"Xou_p_{j} {eo_p} {eo_n} {_yp} wp_who_{j} vdd otau",
                f"Xou_n_{j} {eo_p} {eo_n} {_yn} wn_who_{j} vdd otau"]
        else:
            L+=[f"Xlho_{j} {eo_p} {eo_n} {_yp} {_yn} wp_who_{j} wn_who_{j} vdd gblo gprod"]
    if OBIAS: L+=[f"Xlho_b {eo_p} {eo_n} aobp aobn wp_who_b wn_who_b vdd gblo gprod"]
    L+=ic
    L+=[".options cshunt=1e-14 gmin=1e-10 reltol=2e-3",f".tran 2n {int(TEND)}n uic",".control","run",
        f"wrdata pcL2d{TAG}.dat v(mop) v(mon) v(mop2) v(mon2)"]
    if PROBEFEAT: L+=["wrdata pcL2dh"+TAG+".dat "+" ".join(f"v(ahp{j}) v(ahn{j})" for j in range(H))]
    else: L+=["wrdata pcL2dw"+TAG+".dat "+" ".join(f"v(wp_{k}) v(wn_{k})" for k in WK)]
    L+=[".endc",".end"]
    open(f"pcL2d_{TAG}.cir","w").write("\n".join(L)+"\n")
    return TEND,eval_start,NP,sched
if __name__=="__main__":
    TEND,eval_start,NP,sched=gen()
    subprocess.run(["ngspice","-b",f"pcL2d_{TAG}.cir"],capture_output=True,text=True,timeout=590)
    if PROBEFEAT:
        import warnings;warnings.filterwarnings('ignore');from sklearn.linear_model import LogisticRegression
        d=np.loadtxt(f"pcL2dh{TAG}.dat"); t=d[:,0]*1e9; v=d[:,1::2]  # interleaved t; v has 2H cols (ahp,ahn)
        feat=v[:,0::2]-v[:,1::2]   # ahp-ahn per unit -> (Ntime,H)
        def at(g): return feat[np.argmin(np.abs(t-(g*TH+TH-3)))]
        Ftr=np.array([at(k) for k in range(NTR)]); Fte=np.array([at(NTR+k) for k in range(NTE)])
        clf=LogisticRegression(max_iter=600).fit(Ftr,Ytr[:NTR])
        print(f"PROBEFEAT {TASK} H={H}: REAL in-spice hidden -> train-fit={clf.score(Ftr,Ytr[:NTR]):.3f} TEST={clf.score(Fte,Yte[:NTE]):.3f} (std={feat.std():.3f})")
        import sys; sys.exit(0)
    d=np.loadtxt(f"pcL2d{TAG}.dat"); t=d[:,0]*1e9; mo=d[:,1]-d[:,3]; mo2=d[:,5]-d[:,7]  # wrdata interleaves t per var
    def samp(g): return np.argmin(np.abs(t-(g*TH+TH-3)))
    # TRAIN acc on the LAST epoch (clamped-node mo tracks label if it learned)
    tr0=(NEP-1)*NTR; oktr=sum(((mo[samp(g)]>0)==(sched[g][2]>0)) for g in range(tr0,tr0+NTR))
    # TEST acc: forward-only mo2 on the trailing eval block
    ok=sum(((mo2[samp(eval_start+k)]>0)==(Yte[k]>0)) for k in range(NTE))
    print(f"gen_pcL2d {TASK}: H={H} NTR={NTR} NEP={NEP} SGNH={SGNH} SGNO={SGNO} -> TRAIN(last-ep)={oktr/NTR:.3f}  TEST(fwd)={ok/NTE:.3f}")
    # save final weight caps for chunked resume
    try:
        wd=np.loadtxt(f"pcL2dw{TAG}.dat")[-1]; vals=wd[1::2]
        with open(os.environ.get("SAVEW",f"pcL2dsav{TAG}.txt"),"w") as fs:
            for idx,k in enumerate(WK): fs.write(f"wp_{k} {vals[2*idx]:.5f}\nwn_{k} {vals[2*idx+1]:.5f}\n")
    except Exception as e: print(f"  [savew] {e}")
    # DIAGNOSTIC: decode final weight caps -> Python forward pass (ideal readout) to test if LEARNING worked
    try:
        wd=np.loadtxt(f"pcL2dw{TAG}.dat")[-1]  # last row = final cap voltages, interleaved t per var
        vals=wd[1::2]  # drop the time columns
        W={}
        for idx,k in enumerate(WK): W[k]=(vals[2*idx]-vals[2*idx+1])/(2*KMAP)  # w=(gp-gn)/2KMAP
        Wih=np.array([[W[f"wih_{i}_{j}"] for i in (1,2)] for j in range(H)])  # (H,2) [decode ignores bias]
        Who=np.array([W[f"who_{j}"] for j in range(H)])                       # (H,)
        moved=np.std([W[k] for k in WK])
        Z=Xte@Wih.T; A=np.tanh(Z); pred=(A@Who)>0
        accpy=(pred==(Yte>0)).mean()
        # also the IDEAL linear readout on the learned hidden features
        from sklearn.linear_model import LogisticRegression
        import warnings;warnings.filterwarnings('ignore')
        Ztr=(2*dtr['X'][:NTR]-1)*INSC@Wih.T; Atr=np.tanh(Ztr)
        idl=LogisticRegression(max_iter=500).fit(Atr,Ytr).score(np.tanh(Xte@Wih.T),Yte)
        print(f"  [decode] weight-std={moved:.3f}  Python-fwd(learned who)={accpy:.3f}  ideal-readout-on-learned-hidden={idl:.3f}")
    except Exception as e: print(f"  [decode] failed: {e}")
