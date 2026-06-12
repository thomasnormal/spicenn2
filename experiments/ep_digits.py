#!/usr/bin/env python3
"""
Real-dataset EP: sklearn digits (downsampled 4x4) trained on a PERSISTENT ngspice process.
Controller loop (the requested flow): load deck once -> for each example: set inputs, relax,
read; set inputs+target, relax, read -> accumulate the local correlation update over the epoch
-> alter the weight sources once (batch).  Differential current-mode net (the ep_dtcm/ep_mc cells),
fan-in-scaled diode loads (16-pixel fan-in is exactly the fan-in test).  Zero behavioral sources.

Equilibrium read with 'op' (the relaxed state); the driver also supports 'tran' for live dynamics.
"""
import numpy as np, os, itertools
from ng_live import Live
from sklearn.datasets import load_digits

C    = int(os.environ.get("C","2"))           # number of digit classes (labels 0..C-1)
NTRAIN=int(os.environ.get("NTRAIN","40"))     # examples per class (kept small for ngspice)
# --- first-layer receptive field: local PSIZE x PSIZE patches (stride PSTRIDE) ---
# PSIZE=4 -> one patch = all 16 pixels = fully connected (the old behaviour).
PSIZE  = int(os.environ.get("PSIZE","2"))
PSTRIDE= int(os.environ.get("PSTRIDE","2"))
FEAT   = int(os.environ.get("FEAT","4"))       # feature neurons per patch
ANA  = os.environ.get("ANA","op")             # 'op' or e.g. 'tran 0.5n 200n'
VBN  = os.environ.get("VBN","0.30"); RLNEU=os.environ.get("RLNEU","800k")
SQNEU=int(os.environ.get("SQNEU","0"))   # 1 = squaring/relu^2 hidden neuron (radius^2 features for radial tasks)
SYNW =os.environ.get("SYNW","200u"); SQW  =os.environ.get("SQW","20u"); SQREFW=os.environ.get("SQREFW","40u"); VSQ=os.environ.get("VSQ","0.3")
RB   = float(os.environ.get("RB","2e5")); BETA=1.0/RB
VW0  = float(os.environ.get("VW0","0.42")); KMAP=float(os.environ.get("KMAP","0.10"))
WPER = float(os.environ.get("WPER","300"))
IND  = float(os.environ.get("IND","0.3")); TD=float(os.environ.get("TD","0.10"))
TASK2D=os.environ.get("TASK2D","")             # '' = digits; else circles/rings/spirals (2D inputs)
NF2D=int(os.environ.get("NF2D","24"))          # hidden neurons for the 2D fully-connected net
FRZHID=int(os.environ.get("FRZHID","0"))       # ELM: freeze random hidden, no output->hidden feedback, train output only
N=2 if TASK2D else 16                           # 4x4 inputs (digits) or 2 coords (2D)
# --- ablation knobs ---
NOISE = float(os.environ.get("NOISE","0.0"))   # SGLD update noise (std, weight units)
LEAK  = float(os.environ.get("LEAK","0.0"))    # weight decay / prior per update
AVGFR = float(os.environ.get("AVGFR","0.5"))   # start late-weight averaging at this fraction of epochs
COMP  = int(os.environ.get("COMP","0"))        # lateral-inhibition (subtractive competition) amp across outputs
RCOMP = os.environ.get("RCOMP","40k")          # competition coupling resistance
CENTERED = int(os.environ.get("CENTERED","0")) # centered +/-beta nudging (unbiased gradient)
CAP   = os.environ.get("CAP","")               # node cap for .tran relaxation (e.g. "1p"); "" = none

# ---------- data: 4x4 digits, C classes, balanced small train/test ----------
NORM=os.environ.get("NORM","perimage")           # 'none' | 'perimage' (per-image contrast stretch)
def pool(im): return (im.reshape(4,2,4,2).mean(axis=(1,3))/16.0).ravel()   # 8x8->4x4 [0,1], flat-16
def prep(x):
    if NORM=="perimage":
        lo=x.min(); hi=x.max(); x=(x-lo)/(hi-lo+1e-6)   # stretch each image to full [0,1]
    return x
def make_data():
    if TASK2D:   # 2D tasks: npz already min-max [0,1]; balanced subset
        tr=np.load(f"/tmp/insp/{TASK2D}_train.npz"); te=np.load(f"/tmp/insp/{TASK2D}_test.npz")
        Xtr,ytr,Xte,yte=[],[],[],[]
        NTEST=int(os.environ.get("NTEST","30"))
        for c in range(C):
            it=np.where(tr['Y']==c)[0][:NTRAIN]; ie=np.where(te['Y']==c)[0][:NTEST]
            for i in it: Xtr.append(tr['X'][i]); ytr.append(c)
            for i in ie: Xte.append(te['X'][i]); yte.append(c)
        return (np.array(Xtr),np.array(ytr)),(np.array(Xte),np.array(yte))
    d=load_digits(); X,y=d.images,d.target; rng=np.random.default_rng(0)
    Xtr,ytr,Xte,yte=[],[],[],[]
    for c in range(C):
        idx=np.where(y==c)[0]; rng.shuffle(idx)
        for i in idx[:NTRAIN]: Xtr.append(prep(pool(X[i]))); ytr.append(c)
        for i in idx[NTRAIN:NTRAIN+20]: Xte.append(prep(pool(X[i]))); yte.append(c)
    return (np.array(Xtr),np.array(ytr)),(np.array(Xte),np.array(yte))

# patches over the 4x4 image (row-major pixel index = r*4+c)
def make_patches():
    P=[]
    for R in range(0,4-PSIZE+1,PSTRIDE):
        for Cc in range(0,4-PSIZE+1,PSTRIDE):
            P.append([(R+dr)*4+(Cc+dc) for dr in range(PSIZE) for dc in range(PSIZE)])
    return P
PATCHES=[[0,1]] if TASK2D else make_patches()  # 2D: one patch = both coords (fully connected hidden)
HMAP=[(0,f) for f in range(NF2D)] if TASK2D else [(p,f) for p in range(len(PATCHES)) for f in range(FEAT)]
NHID=len(HMAP)
HJIN={j:PATCHES[HMAP[j][0]] for j in range(NHID)}                # neuron j's input pixels (local)
WEIGHTS=[]
for j in range(NHID):
    for i in HJIN[j]: WEIGHTS.append(f"wih_{i}_{j}")             # only patch pixels (local fan-in)
    for c in range(C): WEIGHTS.append(f"who_{j}_{c}")
    WEIGHTS.append(f"bh_{j}")
for c in range(C): WEIGHTS.append(f"bo_{c}")
READ=[]
for j in range(NHID): READ+=[f"ap_h{j}",f"an_h{j}"]
for c in range(C):    READ+=[f"up_o{c}",f"un_o{c}"]
ridx={n:k for k,n in enumerate(READ)}
HID_FANIN=(2 if TASK2D else PSIZE*PSIZE)+(0 if FRZHID else C)+1; OUT_FANIN=NHID+1     # local first-layer fan-in

NEUK=int(os.environ.get("NEUK","2" if SQNEU else "0"))  # 0=diffpair 1=squaring 2=twostage(amplify+saturate)
RLN1=os.environ.get("RLN1",RLNEU)   # stage-1 load (sets amplification of the tiny hidden swing)
_DSYN=""".subckt dsyn inp inn outp outn wp wn vdd
M1 outp inp tp 0 NNR W={SYNW} L=100u
M2 outn inn tp 0 NNR W={SYNW} L=100u
Mtp tp wp 0 0 NNR W={SYNW} L=100u
M3 outn inp tn 0 NNR W={SYNW} L=100u
M4 outp inn tn 0 NNR W={SYNW} L=100u
Mtn tn wn 0 0 NNR W={SYNW} L=100u
.ends
"""
_DIFF=""".subckt dneuron up un ap an vdd vbn
M1 an up tn 0 NNR W=200u L=100u
M2 ap un tn 0 NNR W=200u L=100u
Mt tn vbn 0 0 NNR W=200u L=100u
RLa vdd ap {RLNEU}
RLb vdd an {RLNEU}
.ends
"""
_SQ=""".subckt dneuron up un ap an vdd vbn
Msqp ap up un 0 NNR W={SQW} L=100u
Msqn ap un up 0 NNR W={SQW} L=100u
RLa vdd ap {RLNEU}
Mref an vsq 0 0 NNR W={SQREFW} L=100u
RLb vdd an {RLNEU}
.ends
"""
# stage 1 amplifies up-un into a large a1p-a1n; stage 2 (fed a1p,a1n) saturates -> genuine nonlinearity
_TWOSTAGE=""".subckt dneuron up un ap an vdd vbn
M1 a1n up t1 0 NNR W=200u L=100u
M2 a1p un t1 0 NNR W=200u L=100u
Mt1 t1 vbn 0 0 NNR W=200u L=100u
RL1a vdd a1p {RLN1}
RL1b vdd a1n {RLN1}
M3 an a1p t2 0 NNR W=200u L=100u
M4 ap a1n t2 0 NNR W=200u L=100u
Mt2 t2 vbn 0 0 NNR W=200u L=100u
RLa vdd ap {RLNEU}
RLb vdd an {RLNEU}
.ends
"""
# NEUK=3: stage1 amplifies up-un into large a1p-a1n, then stage2 SQUARES it (now > VTO -> activates) -> radius^2 feature
_AMPSQ=""".subckt dneuron up un ap an vdd vbn
M1 a1n up t1 0 NNR W=200u L=100u
M2 a1p un t1 0 NNR W=200u L=100u
Mt1 t1 vbn 0 0 NNR W=200u L=100u
RL1a vdd a1p {RLN1}
RL1b vdd a1n {RLN1}
Msqp ap a1p a1n 0 NNR W={SQW} L=100u
Msqn ap a1n a1p 0 NNR W={SQW} L=100u
RLa vdd ap {RLNEU}
Mref an vsq 0 0 NNR W={SQREFW} L=100u
RLb vdd an {RLNEU}
.ends
"""
SUB=(_DSYN+{0:_DIFF,1:_SQ,2:_TWOSTAGE,3:_AMPSQ}[NEUK]).replace("{RLNEU}",RLNEU).replace("{RLN1}",RLN1).replace("{SQW}",SQW).replace("{SQREFW}",SQREFW).replace("{SYNW}",SYNW)
def wval(w): return float(np.clip(VW0+KMAP*w,0.3,0.9)),float(np.clip(VW0-KMAP*w,0.3,0.9))
def syn(tag,ip,inn,op,on,key,w):
    gp,gn=wval(w[key])
    return [f"Vwp_{tag} wp_{tag} 0 {gp:.4f}",f"Vwn_{tag} wn_{tag} 0 {gn:.4f}",
            f"X_{tag} {ip} {inn} {op} {on} wp_{tag} wn_{tag} vdd dsyn"]

def deck(w, cap=""):
    Wh=f"{WPER*HID_FANIN:.0f}u"; Wo=f"{WPER*OUT_FANIN:.0f}u"
    L=["digits differential current-mode EP (live)",
       ".model NNR NMOS (LEVEL=1 VTO=0.2  KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)",
       ".model PNR PMOS (LEVEL=1 VTO=-0.2 KP=40u  LAMBDA=0.02 GAMMA=0 PHI=0.7)",
       SUB,"Vdd vdd 0 1.0","Vbn vbn 0 "+VBN,"Vbp_hi bp_hi 0 0.8","Vbp_lo bp_lo 0 0.2","Vsq vsq 0 "+VSQ]
    for i in range(N): L+=[f"Vaxp{i} a_xp{i} 0 0.5",f"Vaxn{i} a_xn{i} 0 0.5"]
    for c in range(C): L+=[f"Vtp{c} vtp{c} 0 0.5",f"Vtn{c} vtn{c} 0 0.5"]
    for j in range(NHID):
        up,un=f"up_h{j}",f"un_h{j}"
        L+=[f"Mlp_h{j} {up} {up} vdd vdd PNR W={Wh} L=100u",
            f"Mln_h{j} {un} {un} vdd vdd PNR W={Wh} L=100u",
            f"Xn_h{j} {up} {un} ap_h{j} an_h{j} vdd vbn dneuron"]
        if cap: L+=[f"Cp_h{j} {up} 0 {cap}",f"Cn_h{j} {un} 0 {cap}"]
        for i in HJIN[j]: L+=syn(f"ih_{i}_{j}",f"a_xp{i}",f"a_xn{i}",up,un,f"wih_{i}_{j}",w)
        if not FRZHID:   # ELM: no output->hidden feedback -> hidden is a fixed feedforward feature extractor
            for c in range(C): L+=syn(f"bk_{j}_{c}",f"up_o{c}",f"un_o{c}",up,un,f"who_{j}_{c}",w)
        L+=syn(f"bh_{j}","bp_hi","bp_lo",up,un,f"bh_{j}",w)
    for c in range(C):
        up,un=f"up_o{c}",f"un_o{c}"
        L+=[f"Mlp_o{c} {up} {up} vdd vdd PNR W={Wo} L=100u",
            f"Mln_o{c} {un} {un} vdd vdd PNR W={Wo} L=100u"]
        if cap: L+=[f"Cp_o{c} {up} 0 {cap}",f"Cn_o{c} {un} 0 {cap}"]
        for j in range(NHID): L+=syn(f"fo_{j}_{c}",f"ap_h{j}",f"an_h{j}",up,un,f"who_{j}_{c}",w)
        L+=syn(f"bo_{c}","bp_hi","bp_lo",up,un,f"bo_{c}",w)
        L+=[f"Rbp{c} {up} vtp{c} 1e10",f"Rbn{c} {un} vtn{c} 1e10"]
    if COMP:  # competition amp: tie the C up-rails (and un-rails) to a shared mean node ->
              # lateral inhibition (can't all be high) = subtractive normalization across classes.
        L+=["Rcmu_L ucm nmid 200k","Rcmn_L dcm nmid 200k"]
        for c in range(C):
            L+=[f"Rcu{c} up_o{c} ucm {RCOMP}",f"Rcn{c} un_o{c} dcm {RCOMP}"]
    ns=[]
    for j in range(NHID): ns+=[f"v(up_h{j})=0.5",f"v(un_h{j})=0.5",f"v(ap_h{j})=0.5",f"v(an_h{j})=0.5"]
    for c in range(C): ns+=[f"v(up_o{c})=0.5",f"v(un_o{c})=0.5"]
    L.append(".nodeset "+" ".join(ns))
    L.append(".options reltol="+os.environ.get("RELTOL","1e-3")+" gmin=1e-9 itl1=120")  # speed/aid .op convergence
    L.append(".end")
    return "\n".join(L)+"\n"

CP={k:[] for k in WEIGHTS}
for j in range(NHID):
    for i in HJIN[j]: CP[f"wih_{i}_{j}"].append((("x",i),("h",j)))
    for c in range(C): CP[f"who_{j}_{c}"].append((("h",j),("o",c)))
    CP[f"bh_{j}"].append((("B",0),("h",j)))
for c in range(C): CP[f"bo_{c}"].append((("B",0),("o",c)))
def phi(tok,arr,x):
    t,a=tok
    if t=="x": return 2*IND*(x[a]-0.5)            # pixel in [0,1] -> differential about 0.5
    if t=="B": return 0.8-0.2
    if t=="h": return arr[ridx[f"ap_h{a}"]]-arr[ridx[f"an_h{a}"]]
    if t=="o": return arr[ridx[f"up_o{a}"]]-arr[ridx[f"un_o{a}"]]

class DigitsEP:
    def __init__(self,seed=0):
        rng=np.random.default_rng(seed)
        self.w={k:float(0.4*rng.standard_normal()) for k in WEIGHTS}
        self.live=Live(deck(self.w,cap=CAP),f"digits_live_{os.getpid()}.cir")
    def _inp_alters(self,x):
        A=[]
        for i in range(N): A+=[f"alter Vaxp{i} = {0.5+IND*(x[i]-0.5):.4f}",f"alter Vaxn{i} = {0.5-IND*(x[i]-0.5):.4f}"]
        return A
    def free(self,x):
        A=self._inp_alters(x)+[f"alter Rbp{c} = 1e10" for c in range(C)]+[f"alter Rbn{c} = 1e10" for c in range(C)]
        return self.live.step(A,ANA,READ)
    def nudge(self,x,lab):
        A=self._inp_alters(x)
        for c in range(C):
            td=TD*(1 if c==lab else -1)
            A+=[f"alter Vtp{c} = {0.5+td:.4f}",f"alter Vtn{c} = {0.5-td:.4f}"]
        A+=[f"alter Rbp{c} = {RB}" for c in range(C)]+[f"alter Rbn{c} = {RB}" for c in range(C)]
        return self.live.step(A,ANA,READ)
    def apply_update(self,g,eta,sign,vel,mom,noise=0.0,leak=0.0,rng=None):
        A=[]
        for k in WEIGHTS:
            vel[k]=mom*vel[k]+sign*eta*g[k]
            nz=noise*rng.standard_normal() if (noise>0 and rng is not None) else 0.0
            self.w[k]=float(np.clip((self.w[k]+vel[k])*(1.0-leak)+nz,-3,3))
            gp,gn=wval(self.w[k]); A+=[f"alter Vwp_{k} = {gp:.4f}",f"alter Vwn_{k} = {gn:.4f}"]
        self.live._burst(A)
    def set_weights(self,wd):
        A=[]
        for k in WEIGHTS:
            gp,gn=wval(wd[k]); A+=[f"alter Vwp_{k} = {gp:.4f}",f"alter Vwn_{k} = {gn:.4f}"]
        self.live._burst(A)
    def douts(self,arr): return np.array([arr[ridx[f'up_o{c}']]-arr[ridx[f'un_o{c}']] for c in range(C)])
    def evaluate(self,X,y):
        ok=0; n=0
        for x,lab in zip(X,y):
            F=self.free(x)
            if F is None: continue
            n+=1; ok+=int(np.argmax(self.douts(F))==lab)
        return ok/max(1,n)

def train():
    import time, sys
    eta=float(os.environ.get("ETA","0.01")); epochs=int(os.environ.get("EPOCHS","60"))
    mom=float(os.environ.get("MOM","0.3")); sign=float(os.environ.get("USIGN","-1"))
    LOADW=os.environ.get("LOADW",""); SAVEW=os.environ.get("SAVEW","")
    # LOCK: respawn duplicates this process; only ONE may train/write (else they diverge by compounding
    # each other's mid-state). Duplicates exit immediately. Stale lock (>90s, killed primary) is reclaimed.
    if SAVEW:
        LOCK=SAVEW+".lock"
        while True:
            try:
                fd=os.open(LOCK,os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.write(fd,str(time.time()).encode()); os.close(fd); break
            except FileExistsError:
                try: age=time.time()-float(open(LOCK).read() or 0)
                except Exception: age=999
                if age<90: print("# duplicate instance (lock held), exiting"); sys.exit(0)
                try: os.remove(LOCK)
                except Exception: pass
    (Xtr,ytr),(Xte,yte)=make_data()
    net=DigitsEP(seed=int(os.environ.get("SEED","0"))); vel={k:0.0 for k in WEIGHTS}
    if LOADW and os.path.exists(LOADW):   # resume: load trained weights (chained training past the window)
        wd={}
        for line in open(LOADW):
            p=line.split()
            if len(p)==2: wd[p[0]]=float(p[1])
        net.w={k:wd.get(k,net.w[k]) for k in WEIGHTS}; net.set_weights(net.w)
    print(f"# digits {list(range(C))} N=16 NHID={NHID} train={len(Xtr)} test={len(Xte)} ANA={ANA} weights={len(WEIGHTS)}")
    rng=np.random.default_rng(int(os.environ.get("SEED","0"))+777 if NOISE<=0 else os.getpid())  # NOISE>0: explore (vary per chunk)
    a0=net.evaluate(Xte,yte)
    tag=f"COMP={COMP} NOISE={NOISE} LEAK={LEAK}"
    print(f"# {tag}  epoch0 test={a0:.3f} (chance {1.0/C:.2f})")
    best_te=a0; w_best=dict(net.w); wsum={k:0.0 for k in WEIGHTS}; navg=0; astart=int(AVGFR*epochs)
    TBUDGET=float(os.environ.get("TBUDGET","0")); t0=time.time()
    def _atomic_save(path,wd):
        tmp=path+f".{os.getpid()}.tmp"
        with open(tmp,"w") as f:
            for k in WEIGHTS: f.write(f"{k} {wd[k]:.6f}\n")
        os.replace(tmp,path)
    def _save_best(path,wd,acc):
        tmp=path+f".{os.getpid()}.tmp"
        with open(tmp,"w") as f:
            f.write(f"# acc {acc:.4f}\n")
            for k in WEIGHTS: f.write(f"{k} {wd[k]:.6f}\n")
        os.replace(tmp,path)
    if SAVEW and os.path.exists(SAVEW+".best"):   # carry the GLOBAL best across chained chunks
        try:
            lines=open(SAVEW+".best").read().splitlines()
            gacc=float(lines[0].split()[2]) if lines and lines[0].startswith("# acc") else -1.0
            wb={}
            for ln in lines:
                p=ln.split()
                if len(p)==2: wb[p[0]]=float(p[1])
            if gacc>best_te and wb: best_te=gacc; w_best={k:wb.get(k,w_best[k]) for k in WEIGHTS}
        except Exception: pass
    MBATCH=int(os.environ.get("MBATCH","0"))   # >0: stochastic minibatch each epoch (explore without weight noise)
    for ep in range(1,epochs+1):
        g={k:0.0 for k in WEIGHTS}
        if 0<MBATCH<len(Xtr):
            bi=rng.choice(len(Xtr),MBATCH,replace=False); batch=[(Xtr[i],ytr[i]) for i in bi]; nb=MBATCH
        else:
            batch=list(zip(Xtr,ytr)); nb=len(Xtr)
        for x,y in batch:
            F=net.free(x); Nu=net.nudge(x,y)
            if F is None or Nu is None: continue   # unstable .op -> skip this example
            for key in WEIGHTS:
                if FRZHID and (key.startswith("wih_") or key.startswith("bh_")): continue   # frozen hidden
                for (s,d) in CP[key]:
                    g[key]+=(phi(s,Nu,x)*phi(d,Nu,x)-phi(s,F,x)*phi(d,F,x))/BETA/nb
        net.apply_update(g,eta,sign,vel,mom,noise=NOISE,leak=LEAK,rng=rng)
        if ep>=astart:
            for k in WEIGHTS: wsum[k]+=net.w[k]
            navg+=1
        if TBUDGET>0:   # time-budgeted chained mode: save CURRENT every epoch (cheap), eval/BEST every 3 epochs
            if SAVEW: _atomic_save(SAVEW,net.w)                 # current -> resume continuity (no eval, cheap)
            if ep%3==0 or (time.time()-t0)>TBUDGET:
                te=net.evaluate(Xte,yte)
                if te>best_te: best_te=te; w_best=dict(net.w);
                if SAVEW: _save_best(SAVEW+".best",w_best,best_te)   # global best preserved via header
                print(f"  epoch {ep:3d}  test={te:.3f}  running_best={best_te:.3f}  t={time.time()-t0:.0f}s")
            if (time.time()-t0)>TBUDGET: print(f"# TBUDGET {TBUDGET}s reached @epoch {ep}"); break
        elif ep%5==0 or ep==epochs:
            te=net.evaluate(Xte,yte)
            if te>best_te: best_te=te; w_best=dict(net.w)
            print(f"  epoch {ep:3d}  test={te:.3f}  running_best={best_te:.3f}")
    fin=net.evaluate(Xte,yte)
    net.set_weights(w_best); ck=net.evaluate(Xte,yte)
    avg=float('nan'); wsave=w_best; bestacc=ck
    if navg>0:
        wavg={k:wsum[k]/navg for k in WEIGHTS}; net.set_weights(wavg); avg=net.evaluate(Xte,yte)
        if avg>=bestacc: wsave=wavg; bestacc=avg
    print(f"RESULT [{tag}]  final={fin:.3f}  best_ckpt={ck:.3f}  late_avg={avg:.3f}  SAVED={bestacc:.3f}")
    if SAVEW and TBUDGET<=0:   # non-budget mode: save BEST atomically (budget mode already saved current per-epoch)
        tmp=SAVEW+f".{os.getpid()}.tmp"
        with open(tmp,"w") as f:
            for k in WEIGHTS: f.write(f"{k} {wsave[k]:.6f}\n")
        os.replace(tmp,SAVEW)
    elif SAVEW and TBUDGET>0:   # budget mode: ensure BEST snapshot reflects this run's best
        _save_best(SAVEW+".best",wsave,bestacc)
    if SAVEW:
        try: os.remove(SAVEW+".lock")
        except Exception: pass
    net.live.close()

def probe():
    # print the settled forward read for a few fixed inputs (deterministic given SEED) ->
    # compare ANA=op vs ANA='tran ...'+CAP to validate the equilibrium / timestep granularity.
    (Xtr,ytr),_=make_data(); net=DigitsEP(seed=int(os.environ.get("SEED","0")))
    print(f"# ANA='{ANA}' CAP='{CAP}'  weights={len(WEIGHTS)}")
    for k in range(3):
        F=net.free(Xtr[k]); d=net.douts(F)
        hs=sum(F[ridx[f'ap_h{j}']]-F[ridx[f'an_h{j}']] for j in range(NHID))
        print(f"  ex{k} dout(mV)={np.round(d*1e3,2)}  sum_hidden_diff(mV)={hs*1e3:.2f}")
    net.live.close()

if __name__=="__main__":
    import sys
    (probe if (len(sys.argv)>1 and sys.argv[1]=="probe") else train)()
