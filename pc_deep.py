#!/usr/bin/env python3
"""DEEP, SPARSE, fully-in-SPICE continuous Predictive Coding — ONE .tran trains the net.
Generalises gen_pcL to L layers with SPARSE connectivity (fan-in<=K).  Goal: robust (all seeds)
circles/rings/spirals via DEPTH + SPARSITY, not width.

Predictive-coding wiring per layer l (1..L), exactly like gen_pcL:
  m_l_j  = forward prediction of neuron j from the activations a_{l-1} of its parents
  x_l_j  = value node: gets the SAME forward synapses (into x) PLUS the top-down backward error
           from layer l+1 (its children), read through the SAME shared weight caps (NO transport)
  a_l_j  = dneuron(x_l_j)              eps_l_j = esub(x_l_j, m_l_j) = local prediction error
Output layer L: x_L clamped to the one-hot label (train) / free (eval); m_L = prediction; eps_L=label-m.
Update (local, in-circuit gprod): w_{l}_{j<-p} cap  <-  eps_l_j . a_{l-1}_p.
The SAME cap w_{l}_{j<-p} is read by: forward-into-m, forward-into-x, AND backward (eps_{l}_j -> x_{l-1}_p).
Controller only writes the PWL (inputs+label) schedule.  SIM=spectre|ngspice.
"""
import os, numpy as np, subprocess, glob
SEED=int(os.environ.get("SEED","0")); rng=np.random.default_rng(SEED)   # SEED = data split + connectivity
WSEED=int(os.environ.get("WSEED",str(SEED))); wrng=np.random.default_rng(WSEED+1000)   # WSEED = weight init only (for restart-on-stall)
TASK=os.environ.get("TASK","circles"); C=int(os.environ.get("C","2"))
LAYERS=[int(x) for x in os.environ.get("LAYERS","2,8,8").split(",")]+[C]   # input..hidden..; +C output
K=int(os.environ.get("FANIN","6"))
NTR=int(os.environ.get("NTR","14")); NTE=int(os.environ.get("NTE","80")); NEP=int(os.environ.get("NEP","24"))
SCALE=float(os.environ.get("SCALE","1.5")); IND=float(os.environ.get("IND","0.30")); TD=float(os.environ.get("TD","0.20"))
KMAP=float(os.environ.get("KMAP","0.30")); VW0=0.5
TH=float(os.environ.get("TH","250")); STEP=os.environ.get("STEP","2n")
VBSYN=os.environ.get("VBSYN","0.45"); VBNEU=os.environ.get("VBNEU","0.35"); GMT=os.environ.get("GMT","0.6")
GBLH=os.environ.get("GBLH","0.85"); GBLO=os.environ.get("GBLO","0.85")
WL=os.environ.get("WL","1000u"); RCS=os.environ.get("RCS","300k"); RNODE=os.environ.get("RNODE","20k")
RDEG=os.environ.get("RDEG","12k")   # source degeneration (THE key linearizing fix from pc_spice)
CWW=os.environ.get("CWW","30p"); RWL=os.environ.get("RWL","2g"); SGNH=float(os.environ.get("SGNH","1")); SGNO=float(os.environ.get("SGNO","1"))
SIM=os.environ.get("SIM","spectre"); MT=os.environ.get("MT","8"); LQ=os.environ.get("LQ","2000")
TAG=os.environ.get("RUNTAG",str(os.getpid()))
NL=len(LAYERS)   # total layers incl input(0) and output(NL-1)
# ---------- data ----------
def gen():
    rg=np.random.default_rng(7); X=[];y=[]
    if TASK=="circles":
        for c in range(2):
            r=(rg.uniform(0.2,0.6,1500) if c==0 else rg.uniform(1.0,1.5,1500)); th=rg.uniform(0,2*np.pi,1500)
            X+=list(np.c_[r*np.cos(th),r*np.sin(th)]); y+=[c]*1500
    elif TASK=="rings":
        for c in range(C):
            r=rg.uniform(1.2*c+0.25,1.2*c+0.55,1500); th=rg.uniform(0,2*np.pi,1500); X+=list(np.c_[r*np.cos(th),r*np.sin(th)]); y+=[c]*1500
    elif TASK=="spirals":
        for c in range(C):
            t=np.linspace(0.2,1,1500); r=t*2.5; th=t*1.8*np.pi+c*2*np.pi/C+rg.normal(0,0.08,1500); X+=list(np.c_[r*np.cos(th),r*np.sin(th)]); y+=[c]*1500
    elif TASK=="digits":   # sklearn 8x8 digits, first C classes (scale-up target)
        from sklearn.datasets import load_digits
        d=load_digits(); Xd=d.data.astype(float); yd=d.target
        keep=yd<C
        return Xd[keep], yd[keep]
    elif TASK=="xor":
        for a in (0,1):
            for b in (0,1):
                X+=[[a,b]]*750; y+=[a^b]*750
    return np.array(X,float),np.array(y)
X,y=gen()
RC=float(os.environ.get("RADCOMP","0"))
if RC>0:   # compressive radial normalization (foveal/retinal gain): expand center, compress periphery
    _r=np.linalg.norm(X,axis=1,keepdims=True); X=X/(RC+_r)
m=X.mean(0); s=X.std(0)+1e-6; X=(X-m)/s*SCALE
Xtr=[];ytr=[];Xte=[];yte=[]
for c in range(C):
    i=np.where(y==c)[0]; rng.shuffle(i)
    for k in i[:NTR]: Xtr.append(X[k]); ytr.append(c)
    for k in i[NTR:NTR+NTE]: Xte.append(X[k]); yte.append(c)
Xtr=np.array(Xtr); ytr=np.array(ytr); Xte=np.array(Xte); yte=np.array(yte)
MASKSSL=int(os.environ.get("MASKSSL","0"))
SSLCLS=int(os.environ.get("SSLCLS","0"))
if SSLCLS:   # phase-2: classification on the SSL input subset (same 56 inputs the pretrained features expect)
    _var=Xtr.var(0); _mask=np.argsort(_var)[-int(os.environ.get("NMASK","8")):]
    _keep=np.array([i for i in range(Xtr.shape[1]) if i not in set(_mask.tolist())])
    Xtr=Xtr[:,_keep]; Xte=Xte[:,_keep]
if MASKSSL:   # masked self-supervised PC: inputs = unmasked pixels; clamp targets = the MASKED pixels' VALUES (no labels!)
    _var=Xtr.var(0); _mask=np.argsort(_var)[-C:]            # predict the C highest-variance pixels
    _keep=np.array([i for i in range(Xtr.shape[1]) if i not in set(_mask.tolist())])
    Ttr=np.clip(Xtr[:,_mask],-1,1); Tte=np.clip(Xte[:,_mask],-1,1)   # continuous targets in [-1,1]
    Xtr=Xtr[:,_keep]; Xte=Xte[:,_keep]
NIN=LAYERS[0]   # input dims used (first NIN of the 2D+bias)
# ---------- sparse connectivity: per layer l>=1, each neuron picks K parents in layer l-1 ----------
# RFGRID=1: spatial receptive fields. When BOTH layer l-1 and l are perfect squares, neuron (r,c) of the
# WcxWc output grid pools an RFKxRFK window of the WpxWp input grid at stride Wp/Wc (conv-like local
# pooling, the vision inductive bias the random-sparse net throws away at the flatten step). Non-square
# layers (e.g. the C-class readout) fall back to random/full fan-in. RFK=2 default (stride-2 pooling).
RFGRID=int(os.environ.get("RFGRID","0")); RFK=int(os.environ.get("RFK","2"))
def _issq(n): r=int(round(n**0.5)); return r if r*r==n else 0
parents={}   # parents[(l,j)] = list of parent indices p in layer l-1
children={}  # children[(l-1,p)] = list of (j) in layer l connected to p
for l in range(1,NL):
    nprev=LAYERS[l-1]; nl=LAYERS[l]; kk=min(int(os.environ.get("KOUT",K)) if l==NL-1 else K, nprev)   # KOUT: readout-specific fan-in (sizing law: ~C)
    Wp=_issq(nprev); Wc=_issq(nl)
    for j in range(nl):
        if RFGRID and Wp and Wc:   # spatial local window
            rc,cc=divmod(j,Wc); st=Wp/Wc; r0=int(rc*st); c0=int(cc*st)
            ps=sorted({min(r0+dr,Wp-1)*Wp+min(c0+dc,Wp-1) for dr in range(RFK) for dc in range(RFK)})
        else:
            ps=list(rng.choice(nprev,size=kk,replace=False)) if kk<nprev else list(range(nprev))
        parents[(l,j)]=ps
        for p in ps: children.setdefault((l-1,p),[]).append(j)
maxfo=max((len(v) for v in children.values()), default=0)
DALE=int(os.environ.get("DALE","0"))
dsign={}
if DALE:   # Dale's law: per-neuron FIXED sign (hidden layers; inputs & outputs excitatory). Separate rng -> main weight/data streams untouched.
    _drng=np.random.RandomState(int(os.environ.get("DSEED","0")))
    for _l in range(1,NL-1):
        for _j in range(LAYERS[_l]): dsign[(_l,_j)]=1 if _drng.rand()<0.5 else -1
# ---------- cells (verbatim from gen_pcL) ----------
_LAM=os.environ.get("LAM","0.02")
SUB=f""".model NNR NMOS (LEVEL=1 VTO=0.2 KP=120u LAMBDA={_LAM} GAMMA=0 PHI=0.7)
.model PNR PMOS (LEVEL=1 VTO=-0.2 KP=40u LAMBDA={_LAM} GAMMA=0 PHI=0.7)
.subckt gsyn inp inn wp wn outp outn vdd vbn
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
.ends
.subckt gsynL inp inn wp wn outp outn vdd vbn
Mtail nt vbn 0 0 NNR W=400u L=100u
M5 na wp s5 0 NNR W=200u L=100u
Rd5 s5 nt {os.environ.get("RDEGRO","60k")}
M6 nb wn s6 0 NNR W=200u L=100u
Rd6 s6 nt {os.environ.get("RDEGRO","60k")}
M1 outp inp s1 0 NNR W=200u L=100u
Rd1 s1 na {os.environ.get("RDEGRO","60k")}
M2 outn inn s2 0 NNR W=200u L=100u
Rd2 s2 na {os.environ.get("RDEGRO","60k")}
M3 outn inp s3 0 NNR W=200u L=100u
Rd3 s3 nb {os.environ.get("RDEGRO","60k")}
M4 outp inn s4 0 NNR W=200u L=100u
Rd4 s4 nb {os.environ.get("RDEGRO","60k")}
.ends
.subckt cmld p n vdd
Rcs1 p cmx {RCS}
Rcs2 n cmx {RCS}
MLp p cmx vdd vdd PNR W={WL} L=100u
MLn n cmx vdd vdd PNR W={WL} L=100u
.ends
.subckt dneuron up un ap an vdd vbn
M1 an up tn 0 NNR W=200u L=100u
M2 ap un tn 0 NNR W=200u L=100u
Mt tn vbn 0 0 NNR W=200u L=100u
Xcm ap an vdd cmld
.ends
.subckt nrelu up un ap an vdd vbn
* one-sided neuron: cutoff = free ReLU. Knee at v(vbn)+VT (+I.R); un pin = fixed reference gate (global relref).
Mr d1 up s1 0 NNR W=200u L=100u
Rs s1 vbn 12k
Mlp d1 d1 vdd vdd PNR W=200u L=100u
Mlo an d1 vdd vdd PNR W={os.environ.get("NRW","200u")} L=100u
Mrr d2 un s2 0 NNR W=200u L=100u
Rs2 s2 vbn 12k
Mlp2 d2 d2 vdd vdd PNR W=200u L=100u
Mlo2 ap d2 vdd vdd PNR W={os.environ.get("NRW","200u")} L=100u
* LEAKY path: weak source-to-GROUND degenerated legs conduct across the whole input range
* (not just above knee), adding a small linear slope below the knee -> leaky ReLU. NRLEAK large = off
* (~standard ReLU); ~100k = meaningful leak. Symmetric on signal (up) and reference (un) legs so the
* differential zero is preserved. Currents sum into the SAME mirror diodes d1/d2 as the rectifying legs.
Mlk d1 up slk 0 NNR W=200u L=100u
Rlk slk 0 {os.environ.get("NRLEAK","1e9")}
Mlk2 d2 un slk2 0 NNR W=200u L=100u
Rlk2 slk2 0 {os.environ.get("NRLEAK","1e9")}
Rop ap 0 50k
Ron an 0 50k
.ends
.subckt dneuronT up un ap an tn vdd vbn
M1 an up tn 0 NNR W=200u L=100u
M2 ap un tn 0 NNR W=200u L=100u
Mt tn vbn 0 0 NNR W=200u L=100u
Xcm ap an vdd cmld
.ends
.subckt esub xp xn mp mn ep en vdd vbn
Mt nt vbn 0 0 NNR W=400u L=100u
M1x ep xn s1x 0 NNR W=200u L=100u
Rs1x s1x nt {os.environ.get("EDEG","0")}
M2x en xp s2x 0 NNR W=200u L=100u
Rs2x s2x nt {os.environ.get("EDEG","0")}
M1m ep mp s1m 0 NNR W=200u L=100u
Rs1m s1m nt {os.environ.get("EDEG","0")}
M2m en mn s2m 0 NNR W=200u L=100u
Rs2m s2m nt {os.environ.get("EDEG","0")}
RLp vdd ep 50k
RLn vdd en 50k
.ends
.subckt gprod xp xn yp yn wp wn vdd vbn
Mtail nt vbn 0 0 NNR W=400u L=100u
M5 na yp sy5 0 NNR W=200u L=100u
Rg5 sy5 nt {os.environ.get("GDEG","12k")}
M6 nb yn sy6 0 NNR W=200u L=100u
Rg6 sy6 nt {os.environ.get("GDEG","12k")}
M1 iop xp sx1 0 NNR W=200u L=100u
Rg1 sx1 na {os.environ.get("GDEG","12k")}
M2 ion xn sx2 0 NNR W=200u L=100u
Rg2 sx2 na {os.environ.get("GDEG","12k")}
M3 ion xp sx3 0 NNR W=200u L=100u
Rg3 sx3 nb {os.environ.get("GDEG","12k")}
M4 iop xn sx4 0 NNR W=200u L=100u
Rg4 sx4 nb {os.environ.get("GDEG","12k")}
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
.subckt gprodC ap an bp bn yp yn wp wn ckd ckr pk vdd vbn
* CHOPPER-EP update: input mux (a when ckd | b when ckr) + output H-bridge (direct ckd | crossed ckr)
Mxa1 ap ckd xip 0 NNR W=200u L=100u
Mxa2 an ckd xin 0 NNR W=200u L=100u
Mxb1 bp ckr xip 0 NNR W=200u L=100u
Mxb2 bn ckr xin 0 NNR W=200u L=100u
Rpki xip pk 10meg
Rpkj xin pk 10meg
Mtail nt vbn 0 0 NNR W=400u L=100u
M5 na yp sy5 0 NNR W=200u L=100u
Rg5 sy5 nt 12k
M6 nb yn sy6 0 NNR W=200u L=100u
Rg6 sy6 nt 12k
M1 iop xip sx1 0 NNR W=200u L=100u
Rg1 sx1 na 12k
M2 ion xin sx2 0 NNR W=200u L=100u
Rg2 sx2 na 12k
M3 ion xip sx3 0 NNR W=200u L=100u
Rg3 sx3 nb 12k
M4 iop xin sx4 0 NNR W=200u L=100u
Rg4 sx4 nb 12k
Mlp iop iop vdd vdd PNR W=200u L=100u
Mln ion ion vdd vdd PNR W=200u L=100u
Mpu_op op iop vdd vdd PNR W=200u L=100u
MpRn nrn ion vdd vdd PNR W=200u L=100u
MnRn nrn nrn 0 0 NNR W=200u L=100u
Mpd_op op nrn 0 0 NNR W=200u L=100u
Mpu_on on ion vdd vdd PNR W=200u L=100u
MpRp nrp iop vdd vdd PNR W=200u L=100u
MnRp nrp nrp 0 0 NNR W=200u L=100u
Mpd_on on nrp 0 0 NNR W=200u L=100u
Mbd1 op ckd wp 0 NNR W=200u L=100u
Mbd2 on ckd wn 0 NNR W=200u L=100u
Mbr1 op ckr wn 0 NNR W=200u L=100u
Mbr2 on ckr wp 0 NNR W=200u L=100u
Rpko op pk 10meg
Rpkn on pk 10meg
.ends"""
def wgv(v): return float(np.clip(VW0+KMAP*v,0.3,0.9)),float(np.clip(VW0-KMAP*v,0.3,0.9))
def stepped(vals):
    pts=[]
    for k,(t,v) in enumerate(vals):
        if k>0: pts.append((t-0.1,vals[k-1][1]))
        pts.append((t,v))
    return "PWL("+" ".join(f"{t:.1f}n {v:.5f}" for t,v in pts)+")"
# node helpers (differential: every signal is _p/_n)
def ap_(l,i): return (f"a{l}p_{i}",f"a{l}n_{i}")   # activation of layer l, neuron i
def syn(tag,ip,inn,op,on,wp,wn,vb="vbsyn"): return [f"X_{tag} {ip} {inn} {wp} {wn} {op} {on} vdd {vb} gsyn"]
def gen_deck():
    EVK=int(os.environ.get("EVK","0"))   # interval eval: insert a test block every EVK epochs (0 = only final) -> per-seed early-stop
    AFL=float(os.environ.get("AFLOOR","0.6")); VBBK=float(os.environ.get("VBBK",VBSYN)); gBLH=float(GBLH); gBLO=float(GBLO)
    SYM=int(os.environ.get("SYMNUDGE","0"))   # symmetric +/-beta nudge
    CHL=int(os.environ.get("CHL","0"))   # contrastive (EP): free phase (pol=0) then clamped phase (pol=1); readout charged by (a.x)_clamp - (a.x)_free
    OV=int(os.environ.get("OVERIN","0"))   # curriculum: oversample INNER-radius train points (controller-side data cycling)
    if OV>0:
        _rtr=np.linalg.norm(Xtr[:,:2],axis=1); _thr=np.quantile(_rtr,float(os.environ.get("OVQ","0.33")))
        _inner=[k for k in range(len(Xtr)) if _rtr[k]<_thr]
    SLOTS=[]; eval_blocks=[]
    for ep in range(NEP):
        idx=list(range(len(Xtr)))
        if OV>0: idx+=list(_inner)*OV
        rng.shuffle(idx)
        for k in idx:
            if CHL: SLOTS+=[("tr",k,0),("tr",k,1)]   # FREE then CLAMPED
            elif SYM: SLOTS+=[("tr",k,1),("tr",k,-1)]
            else: SLOTS.append(("tr",k,1))
        if EVK and (ep+1)%EVK==0 and ep<NEP-1:
            eval_blocks.append(len(SLOTS))
            for k in range(len(Xte)): SLOTS.append(("te",k,0))
    ev0=len(SLOTS); eval_blocks.append(ev0)
    total_tr=sum(1 for sl in SLOTS if sl[0]=="tr")
    for k in range(len(Xte)): SLOTS.append(("te",k,0))
    NP=len(SLOTS); TEND=NP*TH
    # per-slot PWLs: inputs, label clamp, and gbl/vbbk (annealed during train, OFF during every eval block)
    inp=[[] for _ in range(NIN)]; clp_p=[[] for _ in range(C)]; gh=[]; goC=[]; goF=[]; vb=[]; ckd=[]; ckr=[]; tr_count=0
    ghL={l:[] for l in range(1,NL-1)}
    for g,(kind,k,pol) in enumerate(SLOTS):
        t=g*TH
        xv = Xtr[k] if kind=="tr" else Xte[k]
        if kind=="tr" and float(os.environ.get("AUGJIT","0"))>0: xv = xv + float(os.environ.get("AUGJIT","0"))*rng.standard_normal(len(xv))   # train-time input jitter (controller-side augmentation)
        for d in range(NIN): inp[d].append((t,0.5+IND*float(np.clip(xv[d] if d<len(xv) else 1.0,-3,3))/3.0*2))
        lab = ytr[k] if kind=="tr" else yte[k]
        tdv = TD*(1.0-float(os.environ.get("TDAN","0"))*tr_count/max(1,total_tr))   # TD anneal: strong clamp early, weak late (don't over-drive the equilibrium)
        for c in range(C):
            cl = 1 if pol==1 else (pol if pol!=0 else 0)   # CHL: pol=0 free(0), pol=1 clamp
            if MASKSSL:
                v = cl*tdv*float(Ttr[k,c]) if kind=="tr" else 0.0   # clamp to the masked pixel's VALUE (regression)
            else:
                # ZEROSUM: balanced per-output pressure. Default targets pull correct class +tdv and EACH
                # of the C-1 others -tdv -> net -(C-2)tdv downward drift = the documented C=10 rich-get-
                # richer collapse. Zero-sum spreads the negative over C-1 so each pattern's target sums to 0.
                # negative-class target scale: 1.0 = unbalanced (sum -(C-2)tdv, best at small C),
                # 1/(C-1) = zero-sum (sum 0, best at large C). ZSNEG sweeps the crossover between them.
                _zsneg = float(os.environ.get("ZSNEG", (1.0/(C-1)) if int(os.environ.get("ZEROSUM","0")) else 1.0))
                v = (cl*tdv if c==lab else -cl*tdv*_zsneg) if kind=="tr" else 0.0
            clp_p[c].append((t,0.5+v))
        if kind=="tr" and g>0:
            ckd.append((t,1.0 if pol==1 else 0.0)); ckr.append((t,1.0 if pol==0 else 0.0))
        else:
            ckd.append((t,0.0)); ckr.append((t,0.0))
        if kind!="tr" and int(os.environ.get("LWISE","0")):
            for _hl in range(1,NL-1): ghL[_hl].append((t,0.0))
        if kind=="tr":
            _anns=int(os.environ.get("ANNS","0"))   # absolute-slot anneal ramp (NEP-independent; matches early-stop semantics)
            f=min(1.0, tr_count/_anns) if _anns else tr_count/max(1,total_tr)
            tr_count+=1
            ghv=gBLH-(gBLH-AFL)*f; gov=gBLO-(gBLO-AFL)*f
            if f>float(os.environ.get("HFREEZE","1.0")): ghv=0.0   # freeze HIDDEN after this fraction -> readout refines on the good (early) features
            if int(os.environ.get("HFSLOTS","0")) and tr_count>int(os.environ.get("HFSLOTS","0")): ghv=0.0   # ABSOLUTE-slot burst (doesn't scale with NTR/NEP)
            if CHL==3:
                clamped=(pol==1)
                gh.append((t, ghv if (clamped or int(os.environ.get("HCHOP","0"))) else 0.0)); goC.append((t, gov))   # chopper tails ALWAYS on (clocks gate)
                goF.append((t, 0.0)); vb.append((t, VBBK if clamped else 0.0))
            elif CHL:
                clamped=(pol==1)
                gh.append((t, ghv if clamped else 0.0)); goC.append((t, gov if clamped else 0.0))
                goF.append((t, gov if not clamped else 0.0)); vb.append((t, VBBK if clamped else 0.0))
            else:
                gh.append((t,ghv)); goC.append((t,gov)); goF.append((t,0.0)); vb.append((t,VBBK))
                if int(os.environ.get("LWISE","0")):
                    _nh=NL-2; _f=min(0.999, tr_count/max(1,total_tr))
                    _p=float(os.environ.get("LWPOW","1.0"))   # >1: layers turn ON earlier (turn_on(l)=((l-1)/nh)^p) -> longer joint-refinement tail
                    _w=int(_nh*(_f**(1.0/_p)))+1
                    for _hl in range(1,NL-1): ghL[_hl].append((t, ghv if _hl<=_w and (_hl==_w or int(os.environ.get("LWKEEP","0"))) else 0.0))
        else:   # eval block: freeze learning + backward -> pure forward inference
            gh.append((t,0.0)); goC.append((t,0.0)); goF.append((t,0.0)); vb.append((t,0.0))
    gblh_pwl=stepped(gh); gblo_pwl=stepped(goC); gbloF_pwl=stepped(goF); vbbk_pwl=stepped(vb)
    ghL_pwl={l:stepped(v) for l,v in ghL.items()} if int(os.environ.get("LWISE","0")) else {}
    HINGE=int(os.environ.get("HINGE","0"))   # freeze-on-convergence: stop ALL learning once output is confidently correct (anti-rail/drift)
    L=["deep sparse continuous PC",SUB,"Vdd vdd 0 1.0",
       f"Vbsyn vbsyn 0 {VBSYN}",f"Vbneu vbneu 0 {VBNEU}",f"Vgm gm 0 {GMT}",
       f"Vbbk vbbk 0 {vbbk_pwl}",f"Vwcm wcm 0 {VW0}"]
    for _l,_p in ghL_pwl.items(): L+=[f"VgblhL{_l} gblhL{_l} 0 {_p}"]
    VBBKS=float(os.environ.get("VBBKS","0"))   # per-layer backward gain compensation: deeper layers get hotter bk tails
    if VBBKS>0:
        for _l in range(1,NL-1):
            _v=min(0.9, VBBK+(NL-2-_l)*VBBKS)
            _pwl=stepped([(t, _v if val>0 else 0.0) for t,val in vb])
            L+=[f"Vbbkl{_l} vbbk{_l} 0 {_pwl}"]
    if HINGE:   # gbl tails fed through R so the gate can pull them low when 'conf' is high
        L+=[f"Vgblhs gblhs 0 {gblh_pwl}",f"Rgblh gblhs gblh {os.environ.get('RGBL','30k')}",f"Cgblh gblh 0 2p",
            f"Vgblos gblos 0 {gblo_pwl}",f"Rgblo gblos gblo {os.environ.get('RGBL','30k')}",f"Cgblo gblo 0 2p"]
    else:
        L+=[f"Vgblh gblh 0 {gblh_pwl}",f"Vgblo gblo 0 {gblo_pwl}"]
    if int(os.environ.get("CHL","0")): L+=[f"Vgblof gblof 0 {gbloF_pwl}"]   # free-phase readout tail (contrastive negative term)
    if int(os.environ.get("CHL","0"))==3: L+=[f"Vckd ckd 0 {stepped(ckd)}",f"Vckr ckr 0 {stepped(ckr)}"]   # chopper clocks (controller-provided)
    if float(os.environ.get("WCLAMP","0"))>0:   # active weight-range clamp rails (diode FETs added per cap)
        CLW=float(os.environ.get("WCLAMP","0")); L+=[f"Vwch wch 0 {0.5+CLW-0.2:.3f}",f"Vwcl wcl 0 {0.5-CLW+0.2:.3f}"]
    if SIM=="spectre": L=["deep sparse continuous PC (spectre)","simulator lang=spice","* title",SUB]+L[2:]
    # input nodes a0 (clamped to data)
    for d in range(NIN):
        ap,an=ap_(0,d); L+=[f"Vin{d}p {ap} 0 {stepped(inp[d])}",f"Vin{d}n {an} 0 {stepped([(t,1-v) for t,v in inp[d]])}"]
    # label clamp nodes (output x = label)
    for c in range(C):
        L+=[f"Vxo{c}p xo{c}p 0 {stepped(clp_p[c])}",f"Vxo{c}n xo{c}n 0 {stepped([(t,1-v) for t,v in clp_p[c]])}"]
    # weight caps (shared), init random
    ic=[]; WK=[]; SINIT=int(os.environ.get("SINIT","0")); WINIT=float(os.environ.get("WINIT","0.5"))
    for l in range(1,NL):
        for j in range(LAYERS[l]):
            for pi,p in enumerate(parents[(l,j)]):
                key=f"{l}_{j}_{p}"; WK.append(key)
                if SINIT and l==1 and len(parents[(l,j)])==2:   # structured spread-angle init for square units (guaranteed-diverse, non-degenerate projections)
                    th=np.pi*j/LAYERS[l]; v=(np.cos(th) if pi==0 else np.sin(th))*WINIT*2.0
                elif SINIT and l==NL-1:
                    if int(os.environ.get("UNIRO","0")): v=(1.0 if j==1 else -1.0)*float(os.environ.get("WINITO","0.15"))   # UNIFORM-per-class readout init = near-ridge radius^2 discriminant for radial tasks
                    else: v=float(os.environ.get("WINITO","0.15"))*float(wrng.standard_normal())
                else: v=WINIT*float(wrng.standard_normal())
                if float(os.environ.get("SPERT","0"))>0: v+=float(os.environ.get("SPERT","0"))*float(wrng.standard_normal())   # perturb structured inits (gives WSEED leverage)
                if DALE: v=abs(v)   # Dale: weights start positive; the connection's sign lives in the WIRING (parent neuron's fixed sign)
                gp,gn=wgv(v)
                if os.environ.get("WLOAD"):
                    import json as _json
                    if not hasattr(gen_deck,"_wl"): gen_deck._wl=_json.load(open(os.environ["WLOAD"]))
                    if key in gen_deck._wl: gp,gn=gen_deck._wl[key]
                if int(os.environ.get("NOHID","0")) and l<NL-1:   # FIXED random hidden: weights = plain sources, NO caps
                    L+=[f"Vwp_{key} wp_{key} 0 {gp}",f"Vwn_{key} wn_{key} 0 {gn}"]
                    continue
                if int(os.environ.get("CONSOL","0")):   # synaptic consolidation: fast cap leaks toward a SLOW long-term cap (decay toward consolidated history, not neutral)
                    _rc=os.environ.get("RCON","50meg"); _cs=os.environ.get("CWS","3n")
                    L+=[f"Cwp_{key} wp_{key} 0 {CWW}",f"Cwn_{key} wn_{key} 0 {CWW}",
                        f"Cwsp_{key} wsp_{key} 0 {_cs}",f"Cwsn_{key} wsn_{key} 0 {_cs}",
                        f"Rcp_{key} wp_{key} wsp_{key} {_rc}",f"Rcn_{key} wn_{key} wsn_{key} {_rc}",
                        f"Rwp_{key} wsp_{key} wcm {RWL}",f"Rwn_{key} wsn_{key} wcm {RWL}"]
                else:
                    L+=[f"Cwp_{key} wp_{key} 0 {CWW}",f"Cwn_{key} wn_{key} 0 {CWW}",
                        f"Rwp_{key} wp_{key} wcm {RWL}",f"Rwn_{key} wn_{key} wcm {RWL}"]
                if float(os.environ.get("WCLAMP","0"))>0:   # diode clamps: hold w inside the multiplier's valid range (graceful saturation)
                    for wn_ in (f"wp_{key}",f"wn_{key}"):
                        L+=[f"Mch_{wn_} {wn_} {wn_} wch 0 NNR W=400u L=100u",
                            f"Mcl_{wn_} wcl wcl {wn_} 0 NNR W=400u L=100u"]
                ic+=[f".ic v(wp_{key})={gp:.4f} v(wn_{key})={gn:.4f}"]
                if int(os.environ.get("CONSOL","0")): ic+=[f".ic v(wsp_{key})={gp:.4f} v(wsn_{key})={gn:.4f}"]
                if int(os.environ.get("CHL","0"))==2 and l==NL-1:   # CDS: second cap pair for the FREE-phase integral
                    L+=[f"Cwq_{key} wq_{key} 0 {CWW}",f"Cwr_{key} wr_{key} 0 {CWW}",
                        f"Rwq_{key} wq_{key} wcm {RWL}",f"Rwr_{key} wr_{key} wcm {RWL}"]
                    ic+=[f".ic v(wq_{key})={gn:.4f} v(wr_{key})={gp:.4f}"]
    # hidden + output layer nodes & forward synapses
    for l in range(1,NL):
        out = (l==NL-1)
        for j in range(LAYERS[l]):
            mp,mn=f"m{l}p_{j}",f"m{l}n_{j}"
            L+=[f"Xcm_m{l}_{j} {mp} {mn} vdd cmld",f"Rnd_m{l}_{j} {mp} {mn} {RNODE}"]
            if (not out) and int(os.environ.get("NOHID","0"))>=2:
                xp,xn=mp,mn   # NOHID=2: no backward ever -> x IS m; skip the entire x-node copy (halves hidden synapse hardware)
            elif not out:
                xp,xn=f"x{l}p_{j}",f"x{l}n_{j}"
                L+=[f"Xcm_x{l}_{j} {xp} {xn} vdd cmld",f"Rnd_x{l}_{j} {xp} {xn} {RNODE}",
                    f"Cx{l}p_{j} {xp} 0 0.4p",f"Cx{l}n_{j} {xn} 0 0.4p"]
            elif int(os.environ.get("SOFTC","0")):   # SOFT clamp (beta-nudge): x_L is a FREE node pulled to the label through RCLAMP -> true small-beta simultaneous EP
                xp,xn=f"xq{j}p",f"xq{j}n"
                L+=[f"Xcm_xq{j} {xp} {xn} vdd cmld",f"Rnd_xq{j} {xp} {xn} {RNODE}",
                    f"Cxq{j}p {xp} 0 0.4p",f"Cxq{j}n {xn} 0 0.4p",
                    f"Rscp{j} {xp} xo{j}p {os.environ.get('RCLAMP','120k')}",f"Rscn{j} {xn} xo{j}n {os.environ.get('RCLAMP','120k')}"]
            else:
                xp,xn=f"xo{j}p",f"xo{j}n"   # output value = clamped label
            # forward synapses from parents into m (and into x if hidden)
            fmcell="gsynL" if (out and int(os.environ.get("LINRO","0"))) else "gsyn"   # linearized readout synapse -> delta fixed point reaches ridge
            CHL2o = out and int(os.environ.get("CHL","0"))==2
            for p in parents[(l,j)]:
                pap,pan=ap_(l-1,p); key=f"{l}_{j}_{p}"
                wr0,wr1 = (f"wp_{key}",f"wq_{key}") if CHL2o else (f"wp_{key}",f"wn_{key}")   # CDS differential read: clamped-int minus free-int
                _dmp,_dmn=(mp,mn) if dsign.get((l-1,p),1)>0 else (mn,mp)   # DALE: inhibitory parent -> crossed outputs (free sign flip)
                _dxp,_dxn=(xp,xn) if dsign.get((l-1,p),1)>0 else (xn,xp)
                L+=[f"X_fm_{key} {pap} {pan} {wr0} {wr1} {_dmp} {_dmn} vdd vbsyn {fmcell}"]
                if ((not out) and int(os.environ.get("NOHID","0"))<2) or (out and int(os.environ.get("SOFTC","0"))): L+=syn(f"fx_{key}",pap,pan,_dxp,_dxn,f"wp_{key}",f"wn_{key}")   # x copy (skipped for NOHID>=2 hidden)
            if int(os.environ.get("BIASW","0")):   # learnable per-neuron BIAS (intrinsic plasticity): the self-calibration parameter that absorbs forward offsets
                _bk=f"b_{l}_{j}"
                L+=[f"Cwbp_{_bk} wbp_{_bk} 0 {CWW}",f"Cwbn_{_bk} wbn_{_bk} 0 {CWW}",
                    f"Rwbp_{_bk} wbp_{_bk} wcm {RWL}",f"Rwbn_{_bk} wbn_{_bk} wcm {RWL}",
                    f"X_fmb_{_bk} bup bun wbp_{_bk} wbn_{_bk} {mp} {mn} vdd vbsyn gsyn"]
                ic+=[f".ic v(wbp_{_bk})=0.6 v(wbn_{_bk})=0.6"]
                if ((not out) and int(os.environ.get("NOHID","0"))<2) or (out and int(os.environ.get("SOFTC","0"))):
                    L+=[f"X_fxb_{_bk} bup bun wbp_{_bk} wbn_{_bk} {xp} {xn} vdd vbsyn gsyn"]
            # activation + error neuron
            if not out:
                aap,aan=ap_(l,j)
                if int(os.environ.get("POW4","0")):   # QUARTIC unit: a_l=(x_l^2)^2 -> spans deg-4 (spirals) in ONE layer, no inter-layer credit assignment
                    x2p,x2n=f"x2{l}p_{j}",f"x2{l}n_{j}"
                    L+=[f"Xnq1{l}_{j} {xp} {xn} {xp} {xn} {x2p} {x2n} vdd vbsyn gsyn",
                        f"Xcmq1{l}_{j} {x2p} {x2n} vdd cmld",f"Rq1{l}_{j} {x2p} {x2n} {RNODE}"]
                    s2p,s2n=x2p,x2n
                    for st in range(int(os.environ.get("Q4NORM","0"))):   # renormalize x^2 (dneuron) before squaring again -> deg-4 doesn't vanish
                        rp,rn=f"x2r{st}{l}p_{j}",f"x2r{st}{l}n_{j}"; L.append(f"Xq4n{st}{l}_{j} {s2p} {s2n} {rp} {rn} vdd vbneu dneuron"); s2p,s2n=rp,rn
                    L+=[f"Xnq2{l}_{j} {s2p} {s2n} {s2p} {s2n} {aap} {aan} vdd vbsyn gsyn",
                        f"Xcma{l}_{j} {aap} {aan} vdd cmld",f"Rna{l}_{j} {aap} {aan} {RNODE}"]
                elif int(os.environ.get("SQ","0")):   # SQUARE unit: a_l = (x_l)^2 (gsyn product) -> learns radial features (x^2+y^2) with few TRAINED units
                    L+=[f"Xn{l}_{j} {xp} {xn} {xp} {xn} {aap} {aan} vdd vbsyn gsyn",
                        f"Xcma{l}_{j} {aap} {aan} vdd cmld",f"Rna{l}_{j} {aap} {aan} {RNODE}"]
                elif int(os.environ.get("TFG","0")) in (2,3):   # expose the tail node tn = native saturation signal (device-accurate f' proxy)
                    L+=[f"Xn{l}_{j} {xp} {xn} {aap} {aan} tnn{l}_{j} vdd vbneu dneuronT"]
                elif int(os.environ.get("NEUREL","0")) and (not os.environ.get("NRLAYERS") or str(l) in os.environ["NRLAYERS"].split(",")):   # one-sided ReLU neuron (knee = v(vrl)+VT, per-neuron placement via BIASW); NRLAYERS=comma list -> ReLU only on those layers, tanh elsewhere
                    # cell inverts (signal leg mirrors onto an): swap output pins so the ReLU is upright by
                    # construction (Q-probe evidence: inverted polarity anti-classifies, and at NRW>=400u the
                    # bias-flip snap can no longer rescue it -> 0.03 acc). NRINV=1 reproduces old wiring.
                    _oap,_oan=((aan,aap) if not int(os.environ.get("NRINV","0")) else (aap,aan))
                    L+=[f"Xn{l}_{j} {xp} relref {_oap} {_oan} vdd vrl nrelu"]
                else:
                    L+=[f"Xn{l}_{j} {xp} {xn} {aap} {aan} vdd vbneu dneuron"]
            if (not out) and int(os.environ.get("LATINH","0")):
                aap3,aan3=ap_(l,j)
                L+=[f"Xli_a{l}_{j} {aap3} {aan3} ulp uln lat{l}p lat{l}n vdd vbsyn gsyn",
                    f"Xli_x{l}_{j} lat{l}p lat{l}n usp2 usn2 {xp} {xn} vdd vbsyn gsyn"]
            ehp,ehn=f"e{l}p_{j}",f"e{l}n_{j}"
            if (not out) and int(os.environ.get("NOHID","0"))>=2: pass   # NOHID=2: hidden eps unused (no bk, no hidden updates)
            else: L+=[f"Xeh{l}_{j} {xp} {xn} {mp} {mn} {ehp} {ehn} vdd gm esub"]   # eps_l_j = x - m
    # backward error feedback: eps_{l+1}_k -> x_l_p through the SHARED cap w_{l+1}_{k}_{p}
    if int(os.environ.get("BKSIGN","0")): L+=[f"Vusp usp 0 0.2",f"Vusn usn 0 0.8"]
    if int(os.environ.get("BIASW","0")): L+=[f"Vbup bup 0 {os.environ.get('BUP','0.75')}",f"Vbun bun 0 {os.environ.get('BUN','0.49')}"]   # constant unit input for learnable per-neuron biases
    if int(os.environ.get("NEUREL","0")): L+=[f"Vrelref relref 0 {os.environ.get('RELREF','0.66')}",f"Vvrl vrl 0 {os.environ.get('VRL','0.33')}"]   # ReLU reference gate + knee rail
    if int(os.environ.get("LATINH","0")):
        L+=[f"Vulp ulp 0 0.8",f"Vuln uln 0 0.2",f"Vusp2 usp2 0 0.2",f"Vusn2 usn2 0 0.8"]
        for _l in range(1,NL-1):
            L+=[f"Xcmlat{_l} lat{_l}p lat{_l}n vdd cmld",f"Rlat{_l} lat{_l}p lat{_l}n {RNODE}",
                f"Clatp{_l} lat{_l}p 0 0.4p",f"Clatn{_l} lat{_l}n 0 0.4p"]
    FPERT=int(os.environ.get("FPERT","0"))   # f' via replica perturbation: f(x+eps*bk)-f(x) ~ f'(x).bk (no explicit multiplier)
    if FPERT: L+=[f"Vuip uip 0 0.8",f"Vuin uin 0 0.2"]   # unity weight refs for the injection stage
    FGATE=int(os.environ.get("FGATE","0"))   # tanh f'-gate: modulate the inter-layer backward by (1-a^2) so saturated units aren't over-credited
    if FGATE: AREF=float(os.environ.get("AREF","0.12")); L+=[f"Varef aref 0 {0.5+AREF}",f"Varefn arefn 0 {0.5-AREF}"]
    for l in range(1,NL-1):   # only hidden layers receive top-down
        for p in range(LAYERS[l]):
            xp,xn=f"x{l}p_{p}",f"x{l}n_{p}"
            if FGATE:
                bkp,bkn=f"bk{l}p_{p}",f"bk{l}n_{p}"
                for k in children.get((l,p),[]):
                    key=f"{l+1}_{k}_{p}"; ekp,ekn=f"e{l+1}p_{k}",f"e{l+1}n_{k}"
                    L+=syn(f"bk_{key}",ekp,ekn,bkp,bkn,f"wp_{key}",f"wn_{key}","vbbk")
                # FEEDFORWARD f': gate from a_m=dneuron(m_l) (forward prediction, NO backward coupling -> no loop)
                amp,amn=f"am{l}p_{p}",f"am{l}n_{p}"; mp,mn=f"m{l}p_{p}",f"m{l}n_{p}"
                ap,an=amp,amn; a2p,a2n=f"a2{l}p_{p}",f"a2{l}n_{p}"; gp,gn=f"fg{l}p_{p}",f"fg{l}n_{p}"
                L+=[f"Xam{l}_{p} {mp} {mn} {amp} {amn} vdd vbneu dneuron",
                    f"Xcmbk{l}_{p} {bkp} {bkn} vdd cmld",f"Rbk{l}_{p} {bkp} {bkn} {RNODE}",
                    f"Xa2{l}_{p} {ap} {an} {ap} {an} {a2p} {a2n} vdd vbsyn gsyn",
                    f"Xcma2{l}_{p} {a2p} {a2n} vdd cmld",f"Ra2{l}_{p} {a2p} {a2n} {RNODE}",
                    f"Xfgsub{l}_{p} aref arefn {a2p} {a2n} {gp} {gn} vdd gm esub",
                    f"Xfgbk{l}_{p} {bkp} {bkn} {gp} {gn} {xp} {xn} vdd vbbk gsyn",
                    f"Cfa2p{l}_{p} {a2p} 0 5p",f"Cfa2n{l}_{p} {a2n} 0 5p",   # damp the f'-gate feedback ringing
                    f"Cfgp{l}_{p} {gp} 0 5p",f"Cfgn{l}_{p} {gn} 0 5p",
                    f"Cfbkp{l}_{p} {bkp} 0 5p",f"Cfbkn{l}_{p} {bkn} 0 5p"]
            elif FPERT:   # backward -> bk collector; replica pair applies f'(x); injection back into x
                bkp,bkn=f"fb{l}p_{p}",f"fb{l}n_{p}"
                for k in children.get((l,p),[]):
                    key=f"{l+1}_{k}_{p}"; ekp,ekn=f"e{l+1}p_{k}",f"e{l+1}n_{k}"
                    L+=syn(f"bk_{key}",ekp,ekn,bkp,bkn,f"wp_{key}",f"wn_{key}","vbbk")
                RXC=os.environ.get("RXC","5k"); RBC=os.environ.get("RBC","50k")
                L+=[f"Xcmfb{l}_{p} {bkp} {bkn} vdd cmld",f"Rfb{l}_{p} {bkp} {bkn} {RNODE}",f"Cfbp{l}_{p} {bkp} 0 2p",f"Cfbn{l}_{p} {bkn} 0 2p",
                    f"Rma{l}_{p} {xp} mxa{l}p_{p} {RXC}",f"Rmb{l}_{p} {bkp} mxa{l}p_{p} {RBC}",
                    f"Rmc{l}_{p} {xn} mxa{l}n_{p} {RXC}",f"Rmd{l}_{p} {bkn} mxa{l}n_{p} {RBC}",
                    f"Rme{l}_{p} {xp} mxb{l}p_{p} {RXC}",f"Rmf{l}_{p} wcm mxb{l}p_{p} {RBC}",
                    f"Rmg{l}_{p} {xn} mxb{l}n_{p} {RXC}",f"Rmh{l}_{p} wcm mxb{l}n_{p} {RBC}",
                    f"XrA{l}_{p} mxa{l}p_{p} mxa{l}n_{p} raA{l}p_{p} raA{l}n_{p} vdd vbneu dneuron",
                    f"XrB{l}_{p} mxb{l}p_{p} mxb{l}n_{p} raB{l}p_{p} raB{l}n_{p} vdd vbneu dneuron",
                    f"Xfd{l}_{p} raA{l}p_{p} raA{l}n_{p} raB{l}p_{p} raB{l}n_{p} fd{l}p_{p} fd{l}n_{p} vdd gm esub",
                    f"Cfdp{l}_{p} fd{l}p_{p} 0 2p",f"Cfdn{l}_{p} fd{l}n_{p} 0 2p",
                    f"Xinj{l}_{p} fd{l}p_{p} fd{l}n_{p} uip uin {xp} {xn} vdd vbbk gsyn"]
            elif int(os.environ.get("NOHID","0")):
                pass   # NOHID: no backward synapses at all (fixed random hidden)
            elif int(os.environ.get("BKSIGN","0")):   # SIGN-FAITHFUL backward: bk sum -> collector -> dneuron comparator (regenerates to full swing) -> inject into x. Alignment via signs, not decaying magnitudes.
                bkp,bkn=f"sb{l}p_{p}",f"sb{l}n_{p}"
                for k in children.get((l,p),[]):
                    key=f"{l+1}_{k}_{p}"; ekp,ekn=f"e{l+1}p_{k}",f"e{l+1}n_{k}"
                    L+=syn(f"bk_{key}",ekp,ekn,bkp,bkn,f"wp_{key}",f"wn_{key}","vbbk")
                L+=[f"Xcmsb{l}_{p} {bkp} {bkn} vdd cmld",f"Rsb{l}_{p} {bkp} {bkn} {RNODE}",
                    f"Csbp{l}_{p} {bkp} 0 0.4p",f"Csbn{l}_{p} {bkn} 0 0.4p",
                    f"Xsg{l}_{p} {(bkp if dsign.get((l,p),1)>0 else bkn)} {(bkn if dsign.get((l,p),1)>0 else bkp)} sg{l}p_{p} sg{l}n_{p} vdd vbneu dneuron",
                    f"Xsj{l}_{p} sg{l}p_{p} sg{l}n_{p} usp usn {xp} {xn} vdd vbbk gsyn"]
            elif int(os.environ.get("BKDFA","0")):   # DFA: output error broadcast DIRECTLY to every hidden layer via FIXED random weights (no caps, no deep transpose chain -> depth-uniform error quality)
                oc=NL-1
                for k in range(C):
                    key=f"dfa_{l}_{p}_{k}"; ekp,ekn=f"e{oc}p_{k}",f"e{oc}n_{k}"
                    v=float(os.environ.get("DFAW","1.0"))*float(wrng.standard_normal()); gp_,gn_=wgv(v)
                    L+=[f"Vwp_{key} wp_{key} 0 {gp_}",f"Vwn_{key} wn_{key} 0 {gn_}"]
                    L+=syn(f"bk_{key}",ekp,ekn,xp,xn,f"wp_{key}",f"wn_{key}","vbbk")
            else:
                _dbxp,_dbxn=(xp,xn) if dsign.get((l,p),1)>0 else (xn,xp)   # DALE: transpose of a crossed forward synapse is crossed
                for k in children.get((l,p),[]):
                    key=f"{l+1}_{k}_{p}"; ekp,ekn=f"e{l+1}p_{k}",f"e{l+1}n_{k}"
                    if int(os.environ.get("CHL","0"))==2 and l+1==NL-1:
                        L+=[f"X_bk_{key} {ekp} {ekn} wp_{key} wq_{key} {_dbxp} {_dbxn} vdd vbbk gsyn"]
                    else:
                        _bt=f"vbbk{l}" if float(os.environ.get("VBBKS","0"))>0 else "vbbk"
                        L+=syn(f"bk_{key}",ekp,ekn,_dbxp,_dbxn,f"wp_{key}",f"wn_{key}",_bt)
    if int(os.environ.get("TFG","0")):   # f'-gate reference: tref - a^2 > 0 in the linear region, ~0 when saturated
        TREF=float(os.environ.get("TREF","0.10")); L+=[f"Vtref tref 0 {0.5+TREF}",f"Vtrefn trefn 0 {0.5-TREF}"]
    if int(os.environ.get("TFG","0"))==2:   # global REPLICA dneuron (inputs at CM) -> tn0 = balanced tail reference
        L+=[f"Vtcm tcm 0 0.5",f"Xrep tcm tcm repap repan tn0 vdd vbneu dneuronT"]
    if int(os.environ.get("HCHOP","0"))==3:   # CM-matched zero ref: replica esub w/ all-mid inputs -> zero differential at the EXACT esub CM
        L+=[f"Xzref wcm wcm wcm wcm ezp ezn vdd gm esub"]
    if int(os.environ.get("TFG","0"))==3:   # glow = source ref of the per-neuron LR-gate transistor (engage threshold = glow+VTO)
        L+=[f"Vglow glow 0 {os.environ.get('GLOW','0.10')}"]
    # LEARNING: gprod charges each shared cap by eps_l_j . a_{l-1}_p
    eh=lambda l,j:((f"e{l}p_{j}",f"e{l}n_{j}") if SGNH>0 else (f"e{l}n_{j}",f"e{l}p_{j}"))
    eo=lambda j:((f"e{NL-1}p_{j}",f"e{NL-1}n_{j}") if SGNO>0 else (f"e{NL-1}n_{j}",f"e{NL-1}p_{j}"))
    for l in range(1,NL):
        out=(l==NL-1); tail="gblo" if out else (f"gblhL{l}" if int(os.environ.get("LWISE","0")) else "gblh")
        for j in range(LAYERS[l]):
            ep,en = eo(j) if out else eh(l,j)
            if out and int(os.environ.get("PERAZ","0")):   # PER-CLASS AUTO-ZERO: subtract each class's SLOW-AVG error
                # (RC low-pass of eps_c) -> high-passed error removes that class's systematic DC offset (the
                # residual per-class drift that ZEROSUM's global balance doesn't catch). gen_mc: +9..13 pts at C=10.
                RAZ=os.environ.get("RAZ","2meg"); CAZ=os.environ.get("CAZ","8p")
                lpp,lpn,eap,ean=f"azlp{j}",f"azln{j}",f"eaz{j}p",f"eaz{j}n"
                L+=[f"Razp{j} {ep} {lpp} {RAZ}",f"Cazp{j} {lpp} wcm {CAZ}",
                    f"Razn{j} {en} {lpn} {RAZ}",f"Cazn{j} {lpn} wcm {CAZ}",
                    f"Xaz{j} {ep} {en} {lpp} {lpn} {eap} {ean} vdd gm esub",   # eps - lowpass(eps)
                    f"Xcmaz{j} {eap} {ean} vdd cmld",f"Raz{j} {eap} {ean} {RNODE}"]
                ep,en=eap,ean
            if int(os.environ.get("SQ","0")) and not out:   # f'-gate for SQUARE unit: modulate hidden error by x_l (square derivative 2x)
                xp,xn=f"x{l}p_{j}",f"x{l}n_{j}"; emp,emn=f"em{l}p_{j}",f"em{l}n_{j}"
                L+=[f"Xfg{l}_{j} {ep} {en} {xp} {xn} {emp} {emn} vdd vbsyn gsyn",
                    f"Xcmem{l}_{j} {emp} {emn} vdd cmld",f"Rem{l}_{j} {emp} {emn} {RNODE}"]
                ep,en=emp,emn
            elif int(os.environ.get("TFG","0"))==3 and not out:   # CLEANEST f': gate the gprod TAIL (per-neuron LR) by the native saturation signal tnn
                tail=f"gbl{l}_{j}"                                #   eps path untouched (SGNH=-1 as ungated); 1 transistor+R+C per neuron
                L+=[f"Rgl{l}_{j} gblh {tail} {os.environ.get('RGL','30k')}",
                    f"Mgl{l}_{j} {tail} tnn{l}_{j} glow 0 NNR W={os.environ.get('WGL','600u')} L=100u",
                    f"Cgl{l}_{j} {tail} 0 1p"]
            elif int(os.environ.get("TFG","0"))==2 and not out:   # CLEAN f'-gate: g = tref - (tn - tn0); tn = neuron tail (native saturation), tn0 = replica
                gp,gn=f"tg{l}p_{j}",f"tg{l}n_{j}"; emp,emn=f"tem{l}p_{j}",f"tem{l}n_{j}"
                L+=[f"Xtg{l}_{j} tref trefn tnn{l}_{j} tn0 {gp} {gn} vdd gm esub",
                    f"Xtem{l}_{j} {ep} {en} {gp} {gn} {emp} {emn} vdd vbsyn gsyn"]
                if int(os.environ.get("EMCM","0")):   # CM-match em to the gprod's designed input (esub-style 50k pull-ups to vdd)
                    L+=[f"Rtemp{l}_{j} vdd {emp} 50k",f"Rtemn{l}_{j} vdd {emn} 50k"]
                else:
                    L+=[f"Xcmtem{l}_{j} {emp} {emn} vdd cmld",f"Rtem{l}_{j} {emp} {emn} {os.environ.get('REM',RNODE)}"]
                ep,en=emp,emn
            elif int(os.environ.get("TFG","0")) and not out:   # tanh f'-gate on the UPDATE PATH ONLY (no dynamics loop -> no ringing):
                aap,aan=ap_(l,j)                               # em = eps.(tref - a^2) ~ eps.f'(a); saturated units stop getting over-updated
                a2p,a2n=f"ta2{l}p_{j}",f"ta2{l}n_{j}"; gp,gn=f"tg{l}p_{j}",f"tg{l}n_{j}"; emp,emn=f"tem{l}p_{j}",f"tem{l}n_{j}"
                L+=[f"Xta2{l}_{j} {aap} {aan} {aap} {aan} {a2p} {a2n} vdd vbsyn gsyn",
                    f"Xcmta2{l}_{j} {a2p} {a2n} vdd cmld",f"Rta2{l}_{j} {a2p} {a2n} {RNODE}",
                    f"Xtg{l}_{j} tref trefn {a2p} {a2n} {gp} {gn} vdd gm esub",
                    f"Xtem{l}_{j} {ep} {en} {gp} {gn} {emp} {emn} vdd vbsyn gsyn",
                    f"Xcmtem{l}_{j} {emp} {emn} vdd cmld",f"Rtem{l}_{j} {emp} {emn} {os.environ.get('REM',RNODE)}"]
                ep,en=emp,emn
            if int(os.environ.get("SGNUP","0")):   # SIGN-SGD update: dneuron comparator regenerates eps to full swing -> update = sign(eps).a
                scp,scn=f"sce{l}p_{j}",f"sce{l}n_{j}"   # one comparator per NEURON, shared by its K update cells (cheap-cell amortization)
                L+=[f"Xsce{l}_{j} {ep} {en} {scp} {scn} vdd vbneu dneuron"]
                ep,en=scp,scn
            if int(os.environ.get("BIASW","0")) and not ((not out) and int(os.environ.get("NOHID","0"))):   # bias update: dW_b ~ eps . unit
                _bk=f"b_{l}_{j}"
                L+=[f"Xlrb_{_bk} {ep} {en} bup bun wbp_{_bk} wbn_{_bk} vdd {tail} gprod"]
            CHLon = out and int(os.environ.get("CHL","0"))
            if CHLon:   # contrastive: clamped charges +label.a (gblo), free charges -prediction.a (gblof) -> cap = (a.s)_clamp - (a.s)_free = gradient
                xop,xon=(f"xo{j}p",f"xo{j}n") if SGNO>0 else (f"xo{j}n",f"xo{j}p")
                if int(os.environ.get("CHL","0")) in (2,3):   # CDS/chopper: m UNCROSSED (sign reversal via differential read / H-bridge)
                    mop,mon=(f"m{NL-1}p_{j}",f"m{NL-1}n_{j}") if SGNO>0 else (f"m{NL-1}n_{j}",f"m{NL-1}p_{j}")
                else:
                    mop,mon=(f"m{NL-1}n_{j}",f"m{NL-1}p_{j}") if SGNO>0 else (f"m{NL-1}p_{j}",f"m{NL-1}n_{j}")
            if (not out) and int(os.environ.get("NOHID","0")): continue   # NOHID: no hidden update cells (fixed random hidden)
            for p in parents[(l,j)]:
                pap,pan=ap_(l-1,p); key=f"{l}_{j}_{p}"
                if CHLon and int(os.environ.get("CHL","0"))==3:   # chopper-EP: one cell, +label.a (ckd) / -pred.a (ckr), offsets cancel
                    L+=[f"Xlr_{key} {xop} {xon} {mop} {mon} {pap} {pan} wp_{key} wn_{key} ckd ckr wcm vdd gblo gprodC"]
                elif (not out) and int(os.environ.get("CHL","0"))==3 and int(os.environ.get("HCHOP","0"))==4:   # TRUE EP: Hebbian difference (a_post.a_pre)_nudged - (a_post.a_pre)_free; same nodes both phases -> offsets cancel exactly
                    aap2,aan2 = ap_(l,j)
                    p0,p1 = (aap2,aan2) if SGNH>0 else (aan2,aap2)
                    L+=[f"Xlr_{key} {p0} {p1} {p0} {p1} {pap} {pan} wp_{key} wn_{key} ckd ckr wcm vdd gblh gprodC"]
                elif (not out) and int(os.environ.get("CHL","0"))==3 and int(os.environ.get("HCHOP","0"))==3:   # CM-MATCHED zero ref (replica esub) -> offsets truly cancel
                    L+=[f"Xlr_{key} {ep} {en} ezp ezn {pap} {pan} wp_{key} wn_{key} ckd ckr wcm vdd gblh gprodC"]
                elif (not out) and int(os.environ.get("CHL","0"))==3 and int(os.environ.get("HCHOP","0"))==2:   # hidden auto-zero vs TRUE ZERO ref: offset cancels, signal NOT subtracted (no erasure)
                    L+=[f"Xlr_{key} {ep} {en} wcm wcm {pap} {pan} wp_{key} wn_{key} ckd ckr wcm vdd gblh gprodC"]
                elif (not out) and int(os.environ.get("CHL","0"))==3 and int(os.environ.get("HCHOP","0")):   # hidden auto-zero: free phase eps~0 -> offset cancels (ERASES weights - don't use)
                    L+=[f"Xlr_{key} {ep} {en} {ep} {en} {pap} {pan} wp_{key} wn_{key} ckd ckr wcm vdd gblh gprodC"]
                elif CHLon and int(os.environ.get("CHL","0"))==2:
                    L+=[f"Xlrc_{key} {xop} {xon} {pap} {pan} wp_{key} wn_{key} vdd gblo gprod",
                        f"Xlrf_{key} {mop} {mon} {pap} {pan} wq_{key} wr_{key} vdd gblof gprod"]
                elif CHLon:
                    L+=[f"Xlrc_{key} {xop} {xon} {pap} {pan} wp_{key} wn_{key} vdd gblo gprod",
                        f"Xlrf_{key} {mop} {mon} {pap} {pan} wp_{key} wn_{key} vdd gblof gprod"]
                else:
                    _dap,_dan=(pap,pan) if dsign.get((l-1,p),1)>0 else (pan,pap)   # DALE: chain-rule sign of inhibitory parent folded into the a-input
                    L+=[f"Xlr_{key} {ep} {en} {_dap} {_dan} wp_{key} wn_{key} vdd {tail} gprod"]
    if HINGE:   # score = sum_c (label_c).(m_out_c) -> high when correct; comparator -> conf high -> pull gbl tails low
        oc=NL-1
        for c in range(C): L+=[f"Xsc_{c} xo{c}p xo{c}n m{oc}p_{c} m{oc}n_{c} sop son vdd vbsyn gsyn"]
        L+=[f"Xcm_sc sop son vdd cmld",f"Rsc sop son {os.environ.get('RS','40k')}",
            f"Vsref sref 0 {os.environ.get('VTH','0.515')}",
            f"Xcomp sref sop cdum conf vdd vbneu dneuron",
            f"Vmsrc msrc 0 {os.environ.get('MSRC','0.18')}",
            f"Mgateh gblh conf msrc 0 NNR W=600u L=100u",f"Mgateo gblo conf msrc 0 NNR W=600u L=100u",
            f"Cconf conf 0 2p",f"Csop sop 0 2p"]
    MMVT=float(os.environ.get("MMVT","0"))
    if MMVT>0:   # DEVICE MISMATCH: per-instance series gate V-sources == input-pair VT offsets (real elements, not behavioral).
        # gsyn: signal pair + weight-read pair | dneuron: comparator offset (hits BKSIGN!) | esub: both pairs (error offset) | gprod: both pairs (-> static drift term dx*dy)
        rngm=np.random.RandomState(int(os.environ.get("MMSEED","0")))
        PMM={"gsyn":[0,2],"gsynL":[0,2],"dneuron":[0],"dneuronT":[0],"esub":[0,2],"gprod":[0,2],"nrelu":[0,1]}
        if os.environ.get("MMCELLS"):   # ablation: mismatch only these cell types (comma list)
            _keep=set(os.environ["MMCELLS"].split(","))
            PMM={k:v for k,v in PMM.items() if k in _keep}
        NLs=[]; nmm=0
        for ln in L:
            t=ln.split()
            if t and t[0][0]=="X" and t[-1] in PMM and "\n" not in ln:
                _sig=float(os.environ.get(f"MMVT_{t[-1].upper()}",MMVT))   # per-cell-type sigma override (area-scaling experiments)
                for pi in PMM[t[-1]]:
                    old=t[1+pi]; nn=f"mmv{nmm}"
                    NLs.append(f"Vmm{nmm} {nn} {old} {rngm.randn()*_sig:.5g}")
                    t[1+pi]=nn; nmm+=1
                NLs.append(" ".join(t))
            else: NLs.append(ln)
        L=NLs
        print(f"[pc_deep] MISMATCH MMVT={MMVT*1000:.1f}mV MMSEED={os.environ.get('MMSEED','0')} sources={nmm}")
    L+=ic
    # readout cols = output predictions m_out
    cols=" ".join(f"v(m{NL-1}p_{c}) v(m{NL-1}n_{c})" for c in range(C))
    if int(os.environ.get("GDUMP","0")) and HINGE: cols+=" v(sop) v(son) v(conf) v(gblh) v(gblo)"
    if int(os.environ.get("WSAVE","0")):
        cols+=" "+" ".join(f"v(wp_{k}) v(wn_{k})" for k in WK)
    if int(os.environ.get("DUMPD","0")):
        hk=f"1_0_{parents[(1,0)][0]}"   # a hidden weight key (layer1, neuron0, its first parent)
        cols+=" "+" ".join([f"v(a1p_{j}) v(a1n_{j})" for j in range(LAYERS[1])]
                           +[f"v(e1p_0) v(e1n_0)"]   # hidden ERROR eps_h
                           +[f"v(wp_{NL-1}_0_0) v(wn_{NL-1}_0_0)"]   # output weight
                           +[f"v(wp_{hk}) v(wn_{hk})"])   # HIDDEN weight
    if SIM=="spectre":
        sn=" ".join(dict.fromkeys(cols.replace("v(","").replace(")","").split()))
        strobe=f"strobeperiod={TH}n strobedelay={TH-3}n outputstart=0"
        L+=[f".options reltol=1e-3 gmin=1e-10",f".tran {STEP} {int(TEND)}n uic {strobe}",f".save {sn}",".end"]
        open(f"pd_{TAG}.scs","w").write("\n".join(L)+"\n")
    else:
        L+=[f".options cshunt=1e-14 gmin=1e-10 reltol=2e-3",".control",f"tran {STEP} {int(TEND)}n uic",
            f"wrdata pd{TAG}.dat {cols}",".endc",".end"]
        open(f"pd_{TAG}.cir","w").write("\n".join(L)+"\n")
    return TEND, ev0, cols, eval_blocks
def parse_psf(globpat,nodes):
    f=sorted(glob.glob(globpat))[0]; nset=set(nodes); times=[]; data={n:[] for n in nodes}
    fh=open(f)   # streaming: psfascii can be GBs; never slurp
    for line in fh:
        if line.strip()=="VALUE": break
    for line in fh:
        line=line.strip()
        if line=="END": break
        if not line or line[0]!='"': continue
        q=line.index('"',1); name=line[1:q]; rest=line[q+1:].strip()
        if name=="time": times.append(float(rest))
        elif name in nset: data[name].append(float(rest))
    fh.close()
    return np.array(times)*1e9, np.array([data[n] for n in nodes]).T
if __name__=="__main__":
    TEND,ev0,cols,eval_blocks=gen_deck()
    nodes=[s for s in cols.replace("v(","").replace(")","").split()]
    _deck=open(f"pd_{TAG}.scs" if SIM=="spectre" else f"pd_{TAG}.cir").read(); _nm=sum(1 for ln in _deck.splitlines() if ln.strip().startswith("M"))
    _nsub={}
    import re as _re
    for ln in _deck.splitlines():
        mm=_re.match(r"X\S+ .* (\w+)\s*$", ln.strip())
        if mm: _nsub[mm.group(1)]=_nsub.get(mm.group(1),0)+1
    _fets_per={"gsyn":7,"gprod":15,"gprodC":21,"dneuron":5,"dneuronT":5,"esub":5,"cmld":2}
    _nfet=_nm+sum(_fets_per.get(k,0)*v for k,v in _nsub.items())
    _ncap=sum(1 for ln in _deck.splitlines() if ln.strip().startswith("C"))
    print(f"[pc_deep] COMPONENTS: ~{_nfet} MOSFETs ({_nm} direct + subckts {_nsub}), {_ncap} caps")
    print(f"[pc_deep] TASK={TASK} LAYERS={LAYERS} K={K} maxfanout={maxfo} #w={len(parents)} seed={SEED} sim={SIM} slots={int(TEND/TH)}")
    if SIM=="spectre":
        sp="/opt/cadence/installs/SPECTRE231/bin"
        if not (int(os.environ.get("REUSE","0")) and glob.glob(f"raw_{TAG}/*.tran.tran")):   # REUSE=1: re-parse existing raw (post-proc crash salvage)
            for f in glob.glob(f"raw_{TAG}/*.tran.tran"): os.remove(f)
            r=subprocess.run([f"{sp}/spectre","+spice",f"+mt={MT}",f"+lqtimeout={LQ}",f"pd_{TAG}.scs","-format","psfascii","-raw",f"raw_{TAG}","+log",f"pd_{TAG}.log"],
                             capture_output=True,text=True,timeout=int(os.environ.get("STO","2400")),env={**os.environ,"PATH":sp+":"+os.environ.get("PATH","")})
        else:
            class _R: stdout=""
            r=_R()
        try: t,col=parse_psf(f"raw_{TAG}/*.tran.tran",nodes)
        except Exception as e: print("PARSE FAIL",e); print((r.stdout or "")[-1200:]); raise SystemExit
    else:
        subprocess.run(["ngspice","-b",f"pd_{TAG}.cir"],capture_output=True,text=True,timeout=1200)
        d=np.loadtxt(f"pd{TAG}.dat"); t=d[:,0]*1e9; col=d[:,1::2]
    # eval EACH block (interval eval) -> accuracy curve; report FINAL and BEST (early-stop)
    def block_acc(s):
        MO=[]
        for g in range(s,s+len(yte)):
            row=col[np.argmin(np.abs(t-(g*TH+TH-3)))]
            MO.append([row[2*c]-row[2*c+1] for c in range(C)])
        M=np.array(MO); M=M-M.mean(0)
        if int(os.environ.get("MASKSSL","0")):
            return float(np.mean((M>0)==(Tte[:len(M)]>0)))   # masked-pixel SIGN accuracy (chance 0.5)
        return float(np.mean(np.argmax(M,1)==yte[:len(M)]))
    if int(os.environ.get("WSAVE","0")):
        import json as _json
        _wl={}
        for _k in [n[3:] for n in nodes if n.startswith("wp_")]:
            try:
                _ip=nodes.index(f"wp_{_k}"); _in=nodes.index(f"wn_{_k}")
                _wl[_k]=[float(col[-1][_ip]),float(col[-1][_in])]
            except ValueError: pass
        _json.dump(_wl,open(f"weights_{TAG}.json","w")); print(f"[pc_deep] saved {len(_wl)} weights -> weights_{TAG}.json")
    curve=[block_acc(s) for s in eval_blocks]
    acc=curve[-1]; best=max(curve)
    if int(os.environ.get("SAVEM","0")):   # save the per-example eval margins of the BEST block (for offline committee tests)
        bs=eval_blocks[int(np.argmax(curve))]; MO=[]
        for g in range(bs,bs+len(yte)):
            row=col[np.argmin(np.abs(t-(g*TH+TH-3)))]; MO.append([row[2*c]-row[2*c+1] for c in range(C)])
        M=np.array(MO); np.save(f"mout_{TAG}.npy", M-M.mean(0))
    print(f"[pc_deep] TEST ACC = {acc:.3f}  BEST(early-stop) = {best:.3f}  curve = {[round(x,3) for x in curve]}")
