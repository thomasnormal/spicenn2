#!/usr/bin/env python3
"""Orchestrated multi-class Predictive Coding on sklearn digits.
The ANALOG circuit (ngspice .op) solves each example's PC equilibrium -- hard-clamped output,
explicit error neurons, reciprocal backward feedback -> a clean, decay-free inference.  The
controller reads the explicit local errors (eps_h, eps_o) and analog activities (a_h) and applies
the LOCAL PC update  W_ih += lr*eps_h.x_in ;  W_ho += lr*eps_o.a_h  (no backprop; each weight from
its own two terminals).  Forward inference is real analog; the weight step is controller-applied
(can be moved to the physical gprod cell later).  Persistent process for speed.  Few NMOS: only the
forward/error network."""
import os, numpy as np
from ng_live import Live
from sklearn.datasets import load_digits
NIN=int(os.environ.get("NIN","16")); N=NIN; C=int(os.environ.get("C","3")); H=int(os.environ.get("H","12")); SEED=int(os.environ.get("SEED","0"))
NTR=int(os.environ.get("NTR","40")); NTE=int(os.environ.get("NTE","40")); EP=int(os.environ.get("EP","30"))
LR=float(os.environ.get("LR","0.03")); IND=float(os.environ.get("IND","0.30")); TD=float(os.environ.get("TD","0.20"))
KMAP=float(os.environ.get("KMAP","0.30")); VW0=float(os.environ.get("VW0","0.5"))
WL=os.environ.get("WL","1000u"); RCS=os.environ.get("RCS","300k"); RNODE=os.environ.get("RNODE","20k")
VBSYN=os.environ.get("VBSYN","0.45"); VBNEU=os.environ.get("VBNEU","0.35"); GMT=os.environ.get("GMT","0.6")
VBBK=os.environ.get("VBBK","0.30")   # backward-synapse tail bias: weaker than VBSYN so top-down is a gentle correction
INJ=int(os.environ.get("INJ","0"))   # 1 = 2-solve genuine-PC: inject scaled real eps_o, analog W^T backward -> eps_h
GMO=float(os.environ.get("GMO","400")); EINJ=float(os.environ.get("EINJ","1.0"))   # output-error readout gain & injection scale
LRH=float(os.environ.get("LRH","1.0"))   # hidden-layer LR multiplier (eps_h is mV-scale vs eps_o)
RLOAD=os.environ.get("RLOAD","")   # if set (e.g. 10k): replace cmld CMFB load with resistor-to-vref (HIGH gain) on every node
NGAIN=int(os.environ.get("NGAIN","0"))   # # of dneuron gain stages cascaded on the OUTPUT mu_o (each ~7x, stable; saturates at target scale)
VBG=os.environ.get("VBG","0.35")         # gain-stage tail bias
ANNEAL=int(os.environ.get("ANNEAL","1")); WCLIP=float(os.environ.get("WCLIP","3"))
TAG=os.environ.get("RUNTAG",str(os.getpid()))
TASK=os.environ.get("TASK","digits")   # NONLINEAR 2D tasks (linear≈chance, tanh-MLP H=8≈1.0) -> test DEPTH
if TASK in ("rings","circles","xor"):
    N=2; _rg=np.random.default_rng(123); _n=int(os.environ.get("NPTS","1500")); X=[];y=[]
    if TASK=="rings":     # C concentric annuli
        for c in range(C):
            r=_rg.uniform(c+0.35,c+0.9,_n//C); th=_rg.uniform(0,2*np.pi,_n//C)
            X+=list(np.c_[r*np.cos(th),r*np.sin(th)]); y+=[c]*(_n//C)
    elif TASK=="circles": # 2 concentric circles (C=2): inner vs outer ring
        for c in range(2):
            r=_rg.uniform(0.2,0.6,_n//2) if c==0 else _rg.uniform(1.0,1.5,_n//2)
            th=_rg.uniform(0,2*np.pi,_n//2); X+=list(np.c_[r*np.cos(th),r*np.sin(th)]); y+=[c]*(_n//2)
    elif TASK=="xor":     # 4 corner blobs, label = quadrant parity (C=2, classic XOR)
        ctr=[[1,1],[1,-1],[-1,1],[-1,-1]]; lab=[0,1,1,0]
        for i,ct in enumerate(ctr): X+=list(np.array(ct)+_rg.normal(0,0.35,(_n//4,2))); y+=[lab[i]]*(_n//4)
    X=np.array(X); y=np.array(y)
else:
    d=load_digits(); X=(d.images.reshape(-1,64) if NIN==64 else d.images.reshape(-1,4,2,4,2).mean(axis=(2,4)).reshape(-1,16)); y=d.target
m=X.mean(0); s=X.std(0)+1e-6; X=np.clip((X-m)/s,-3,3)/3.0
BIAS=int(os.environ.get("BIAS","0"))
if BIAS: X=np.c_[X,np.full(len(X),float(os.environ.get("BVAL","1.0")))]; N=N+1   # constant bias input -> lets hidden units threshold (needed for radial tasks)
rng=np.random.default_rng(SEED); CLS=list(range(C)); Xtr=[];ytr=[];Xte=[];yte=[]
for ci,c in enumerate(CLS):
    idx=np.where(y==c)[0]; rng.shuffle(idx)
    for i in idx[:NTR]: Xtr.append(X[i]); ytr.append(ci)
    for i in idx[NTR:NTR+NTE]: Xte.append(X[i]); yte.append(ci)
Xtr=np.array(Xtr); ytr=np.array(ytr); Xte=np.array(Xte); yte=np.array(yte)
WIH=rng.standard_normal((H,N))*0.4; WHO=rng.standard_normal((C,H))*0.4
def wgv(v): return float(np.clip(VW0+KMAP*v,0.3,0.9)),float(np.clip(VW0-KMAP*v,0.3,0.9))
SUB=f""".model NNR NMOS (LEVEL=1 VTO=0.2 KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)
.model PNR PMOS (LEVEL=1 VTO=-0.2 KP=40u LAMBDA=0.02 GAMMA=0 PHI=0.7)
.subckt gsyn inp inn wp wn outp outn vdd vbn
Mtail nt vbn 0 0 NNR W=400u L=100u
M5 na wp nt 0 NNR W=200u L=100u
M6 nb wn nt 0 NNR W=200u L=100u
M1 outp inp na 0 NNR W=200u L=100u
M2 outn inn na 0 NNR W=200u L=100u
M3 outn inp nb 0 NNR W=200u L=100u
M4 outp inn nb 0 NNR W=200u L=100u
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
.subckt esub xp xn mp mn ep en vdd vbn
Mt nt vbn 0 0 NNR W=400u L=100u
M1x ep xn nt 0 NNR W=200u L=100u
M2x en xp nt 0 NNR W=200u L=100u
M1m ep mp nt 0 NNR W=200u L=100u
M2m en mn nt 0 NNR W=200u L=100u
RLp vdd ep 50k
RLn vdd en 50k
.ends"""
def synline(tag,ip,inn,wp,wn,op,on,vb="vbsyn"): return f"X_{tag} {ip} {inn} {wp} {wn} {op} {on} vdd {vb} gsyn"
def ld(tag,p,n):   # node load: cmld (CMFB, low diff-gain) OR resistor-to-vref (HIGH diff-gain + gentle CM pull)
    if RLOAD: return [f"Rl{tag}p {p} vref {RLOAD}",f"Rl{tag}n {n} vref {RLOAD}"]
    return [f"Xcm_{tag} {p} {n} vdd cmld",f"Rnd_{tag} {p} {n} {RNODE}"]
def monode(c): return (f"mg{c}_{NGAIN-1}p",f"mg{c}_{NGAIN-1}n") if NGAIN>0 else (f"mop{c}",f"mon{c}")
RGAIN=os.environ.get("RGAIN","")   # optional load on each gain-stage output (limits swing/railing -> convergence)
RSHUNT=os.environ.get("RSHUNT","rshunt=1e9")   # gentle high-Z convergence aid (1Gohm: negligible loading)
ANL=os.environ.get("ANL","op")             # solve: "op" or "tran 1n 80n uic" (transient settling = robust for stiff high-gain)
CSHUNT=os.environ.get("CSHUNT","")         # e.g. "cshunt=1e-14": tiny node caps -> fast, robust .tran settling
CXH=os.environ.get("CXH","")               # explicit cap on hidden activity node -> dominant pole -> stable relaxation
def gainchain(c,p,n):   # cascade NGAIN dneuron stages: lifts mu_o off the noise floor (each ~7x, saturates at target)
    lines=[]; cp,cn=p,n
    for g in range(NGAIN):
        op,on=f"mg{c}_{g}p",f"mg{c}_{g}n"
        lines+=[f"Xg{c}_{g} {cp} {cn} {op} {on} vdd vbg dneuron"]
        if RGAIN: lines+=[f"Rg{c}_{g}p {op} vref {RGAIN}",f"Rg{c}_{g}n {on} vref {RGAIN}"]
        cp,cn=op,on
    return lines
def build(train=True):
    L=["pc orchestrated",SUB,"Vdd vdd 0 1.0",f"Vbsyn vbsyn 0 {VBSYN}",f"Vbbk vbbk 0 {VBBK}",f"Vbneu vbneu 0 {VBNEU}",f"Vgm gm 0 {GMT}",f"Vwcm wcm 0 {VW0}","Vref vref 0 0.5",f"Vbg vbg 0 {VBG}"]
    for i in range(N): L+=[f"Vxp{i} a_xp{i} 0 0.5",f"Vxn{i} a_xn{i} 0 0.5"]
    # weights as ALTERABLE sources
    for j in range(H):
        for i in range(N): gp,gn=wgv(WIH[j,i]); L+=[f"Vwp_ih_{i}_{j} wp_ih_{i}_{j} 0 {gp}",f"Vwn_ih_{i}_{j} wn_ih_{i}_{j} 0 {gn}"]
        for c in range(C): gp,gn=wgv(WHO[c,j]); L+=[f"Vwp_ho_{j}_{c} wp_ho_{j}_{c} 0 {gp}",f"Vwn_ho_{j}_{c} wn_ho_{j}_{c} 0 {gn}"]
    if train:
        for c in range(C): L+=[f"Vxop{c} xop{c} 0 0.5",f"Vxon{c} xon{c} 0 0.5"]   # output clamp (alter to target)
    for j in range(H):
        xp,xn=f"xhp{j}",f"xhn{j}"; mp,mn=f"mhp{j}",f"mhn{j}"
        L+=ld(f"mh{j}",mp,mn)+ld(f"xh{j}",xp,xn)
        if CXH: L+=[f"Cxh{j}p {xp} 0 {CXH}",f"Cxh{j}n {xn} 0 {CXH}"]   # dominant-pole compensation: xh = slow integrating relaxation node
        for i in range(N):
            L+=[synline(f"ihm_{i}_{j}",f"a_xp{i}",f"a_xn{i}",f"wp_ih_{i}_{j}",f"wn_ih_{i}_{j}",mp,mn)]
            L+=[synline(f"ihx_{i}_{j}",f"a_xp{i}",f"a_xn{i}",f"wp_ih_{i}_{j}",f"wn_ih_{i}_{j}",xp,xn)]
        if train:
            for c in range(C): L+=[synline(f"bk_{j}_{c}",f"eop{c}",f"eon{c}",f"wp_ho_{j}_{c}",f"wn_ho_{j}_{c}",xp,xn,"vbbk")]
        L+=[f"Xn{j} {xp} {xn} ahp{j} ahn{j} vdd vbneu dneuron"]
        L+=[f"Xeh{j} {xp} {xn} {mp} {mn} ehp{j} ehn{j} vdd gm esub"]
    INJ=int(os.environ.get("INJ","0"))
    for c in range(C):
        L+=ld(f"mo{c}",f"mop{c}",f"mon{c}")
        for j in range(H): L+=[synline(f"ho_{j}_{c}",f"ahp{j}",f"ahn{j}",f"wp_ho_{j}_{c}",f"wn_ho_{j}_{c}",f"mop{c}",f"mon{c}")]
        L+=gainchain(c,f"mop{c}",f"mon{c}"); gp,gn=monode(c)   # amplify mu_o off the noise floor, in-circuit
        if train and INJ: L+=[f"Veop{c} eop{c} 0 0.5",f"Veon{c} eon{c} 0 0.5"]   # controller-injected output error (scaled)
        elif train: L+=[f"Xeo{c} xop{c} xon{c} {gp} {gn} eop{c} eon{c} vdd gm esub"]
    L+=[f".options gmin=1e-12 reltol=1e-3 itl1=500 {RSHUNT} {CSHUNT}",".end"]   # rshunt: high-Z node convergence aid for the high-gain cascade
    return "\n".join(L)+"\n"
def inp_alters(x):
    A=[]
    for i in range(N): A+=[f"alter Vxp{i} = {0.5+IND*float(x[i]):.4f}",f"alter Vxn{i} = {0.5-IND*float(x[i]):.4f}"]
    return A
def wt_alters():
    A=[]
    for j in range(H):
        for i in range(N): gp,gn=wgv(WIH[j,i]); A+=[f"alter Vwp_ih_{i}_{j} = {gp:.4f}",f"alter Vwn_ih_{i}_{j} = {gn:.4f}"]
        for c in range(C): gp,gn=wgv(WHO[c,j]); A+=[f"alter Vwp_ho_{j}_{c} = {gp:.4f}",f"alter Vwn_ho_{j}_{c} = {gn:.4f}"]
    return A
# read CLEAN activities (not the analog esub) -> compute eps by subtraction in the controller
RD_TR=sum([[f"xhp{j}",f"xhn{j}",f"mhp{j}",f"mhn{j}",f"ahp{j}",f"ahn{j}"] for j in range(H)],[])+sum([list(monode(c)) for c in range(C)],[])
RD_EV=sum([list(monode(c)) for c in range(C)],[])
if __name__=="__main__":
    tr=Live(build(True),f"pco_tr_{TAG}.cir")
    def recreate_tr():   # ngspice occasionally dies on a bad weight state -> restart it & re-apply CURRENT weights
        global tr
        try: tr.close()
        except Exception: pass
        tr=Live(build(True),f"pco_tr_{TAG}.cir"); tr._burst(wt_alters())
    def burst_tr(cmds):
        try: tr._burst(cmds)
        except Exception: recreate_tr()
    def dead_tr(): return tr.p.poll() is not None
    ev=Live(build(False),f"pco_ev_{TAG}.cir")   # persistent eval process (reuse; just re-burst weights)
    def recreate_ev():
        global ev
        try: ev.close()
        except Exception: pass
        ev=Live(build(False),f"pco_ev_{TAG}.cir")
    def evaluate():
        try: ev._burst(wt_alters())
        except Exception: recreate_ev(); ev._burst(wt_alters())
        mos=[];labs=[]
        for e in range(len(Xte)):
            v=ev.step(inp_alters(Xte[e]),ANL,RD_EV,fout=f"pcoev_{TAG}.dat")
            if v is None:
                if ev.p.poll() is not None: recreate_ev(); ev._burst(wt_alters())
                continue
            mos.append(np.array([v[2*c]-v[2*c+1] for c in range(C)])); labs.append(yte[e])
        if not mos: return 0.0
        M=np.array(mos); M=M-M.mean(0)   # DEBIAS: subtract the per-class common offset (task-agnostic, vs zero-input)
        return float(np.mean([int(np.argmax(M[i]))==labs[i] for i in range(len(labs))]))
    print(f"orchestrated PC: C={C} H={H} N={N} train={len(Xtr)} test={len(Xte)} weights={H*N+C*H}")
    if os.environ.get("DIAG") and int(os.environ.get("INJ","0")):
        # 2-solve genuine-PC test: forward -> read mo -> inject REAL scaled eps_o -> analog W^T backward -> eps_h
        f=np.tanh; B=min(int(os.environ.get("DIAGB","60")),len(Xtr))
        GMO=float(os.environ.get("GMO","400")); EINJ=float(os.environ.get("EINJ","1.0")); NITER=int(os.environ.get("NITER","1"))
        def cs(a,b):
            a=np.ravel(np.array(a,float));b=np.ravel(np.array(b,float)); return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
        EHc=[];EHi=[];DIHc=[];DIHi=[];EOc=[];EOi=[]
        for k in range(B):
            x=Xtr[k]; lab=ytr[k]; tgt=np.array([TD*(1 if c==lab else -1) for c in range(C)])
            neutral=sum([[f"alter Veop{c} = 0.5",f"alter Veon{c} = 0.5"] for c in range(C)],[])
            v=tr.step(inp_alters(x)+neutral,ANL,RD_TR,fout=f"pcotr_{TAG}.dat")
            mo=np.array([v[6*H+2*c]-v[6*H+2*c+1] for c in range(C)])
            for it in range(NITER):   # RELAXATION: re-inject the error from the settled mu_o each iteration
                eoc=tgt-GMO*mo
                inj=sum([[f"alter Veop{c} = {0.5+EINJ*float(eoc[c]):.4f}",f"alter Veon{c} = {0.5-EINJ*float(eoc[c]):.4f}"] for c in range(C)],[])
                v=tr.step(inp_alters(x)+inj,ANL,RD_TR,fout=f"pcotr_{TAG}.dat")
                mo=np.array([v[6*H+2*c]-v[6*H+2*c+1] for c in range(C)])
            xh=np.array([v[6*j]-v[6*j+1] for j in range(H)]); mh2=np.array([v[6*j+2]-v[6*j+3] for j in range(H)])
            ehc=xh-mh2                                                  # hidden error from ANALOG backward of REAL error
            muh=WIH@x; ahi=f(muh); moi=WHO@ahi; eoi=tgt-moi; ehi=(1-ahi**2)*(WHO.T@eoi)
            EOc.append(eoc);EOi.append(eoi);EHc.append(ehc);EHi.append(ehi);DIHc.append(np.outer(ehc,x));DIHi.append(np.outer(ehi,x))
        print(f"  --- INJECTION test (analog W^T backward) GMO={GMO} EINJ={EINJ} NITER={NITER} ---")
        print(f"  eps_o (injected)      cos={cs(EOc,EOi):+.3f}")
        print(f"  eps_h (HIDDEN ERROR)  cos={cs(EHc,EHi):+.3f}   <-- was ~0.3 without injection")
        print(f"  dW_ih update (hidden) cos={cs(DIHc,DIHi):+.3f}   <-- trains the hidden layer")
        raise SystemExit
    if os.environ.get("DIAG"):
        f=np.tanh; B=min(int(os.environ.get("DIAGB","60")),len(Xtr)); DEBIAS=int(os.environ.get("DEBIAS","1"))
        def rd(x,lab):
            A=inp_alters(x)+sum([[f"alter Vxop{c} = {0.5+TD*(1 if c==lab else -1):.4f}",f"alter Vxon{c} = {0.5-TD*(1 if c==lab else -1):.4f}"] for c in range(C)],[]) if lab is not None else \
              inp_alters(x)+sum([[f"alter Vxop{c} = 0.5",f"alter Vxon{c} = 0.5"] for c in range(C)],[])
            v=tr.step(A,ANL,RD_TR,fout=f"pcotr_{TAG}.dat")
            xh=np.array([v[6*j]-v[6*j+1] for j in range(H)]); mh=np.array([v[6*j+2]-v[6*j+3] for j in range(H)])
            ah=np.array([v[6*j+4]-v[6*j+5] for j in range(H)]); mo=np.array([v[6*H+2*c]-v[6*H+2*c+1] for c in range(C)])
            return xh,mh,ah,mo
        xh0,mh0,ah0,mo0=rd(np.zeros(N),None) if DEBIAS else (0,0,0,0)   # zero-input baselines (offsets)
        EHc=[];EHi=[];AHc=[];AHi=[];MOc=[];MOi=[];EOc=[];EOi=[];DIHc=[];DIHi=[];DHOc=[];DHOi=[]
        for k in range(B):
            x=Xtr[k]; lab=ytr[k]
            xh,mh,ah,mo=rd(x,lab); xh=xh-xh0; mh=mh-mh0; ah=ah-ah0; mo=mo-mo0
            tgt=np.array([TD*(1 if c==lab else -1) for c in range(C)])
            muh=WIH@x; ahi=f(muh); moi=WHO@ahi; eoi=tgt-moi; ehi=(1-ahi**2)*(WHO.T@eoi)
            AHc.append(ah);AHi.append(ahi);MOc.append(mo);MOi.append(moi);EHc.append(xh-mh);EHi.append(ehi)
            EOc.append(tgt);EOi.append(eoi)   # store tgt; mo kept in MOc for GMO sweep
            DIHc.append(np.outer(xh-mh,x));DIHi.append(np.outer(ehi,x))
        def cs(a,b):
            a=np.ravel(np.array(a,float));b=np.ravel(np.array(b,float))
            return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
        print(f"  --- FIDELITY (circuit vs ideal PC, cos) DEBIAS={DEBIAS} ---")
        print(f"  a_h (neuron)          cos={cs(AHc,AHi):+.3f}    mu_o cos={cs(MOc,MOi):+.3f}")
        print(f"  eps_h (HIDDEN ERROR)  cos={cs(EHc,EHi):+.3f}    dW_ih cos={cs(DIHc,DIHi):+.3f}")
        print(f"  |mo_circ| rms={np.sqrt(np.mean(np.array(MOc)**2)):.3f}  |tgt| rms={np.sqrt(np.mean(np.array(EOc)**2)):.3f}  (want comparable)")
        print("  GMO sweep (readout gain on mo) -> eps_o cos / dW_ho cos:")
        MOa=np.array(MOc);AHa=np.array(AHc);TGa=np.array(EOc);EOia=np.array(EOi);AHia=np.array(AHi)
        for GMO in [1.0,10.0,50.0,100.0,200.0,400.0,800.0]:
            eoc=TGa-GMO*MOa
            dho=np.array([np.outer(eoc[k],AHa[k]) for k in range(len(MOc))])
            dhoi=np.array([np.outer(EOia[k],AHia[k]) for k in range(len(MOc))])
            print(f"    GMO={GMO:.2f}: eps_o cos={cs(eoc,EOia):+.3f}  dW_ho cos={cs(dho,dhoi):+.3f}")
        raise SystemExit
    print(f"  epoch 0 test={evaluate():.2f} (chance {1/C:.2f})")
    MB=int(os.environ.get("MB","20"))   # minibatch size (alter weights once per MB -> fast)
    OPT=os.environ.get("OPT","sgd"); _t=[0]   # Adam (= the cap-based momentum/AGC optimizer) handles noisy gradients + per-layer scale
    mHO=np.zeros_like(WHO); vHO=np.zeros_like(WHO); mIH=np.zeros_like(WIH); vIH=np.zeros_like(WIH)
    def apply_upd(gho,gih,lr):
        global WHO,WIH
        if OPT=="adam":
            _t[0]+=1; t=_t[0]; b1=0.9;b2=0.999
            mHO[:]=b1*mHO+(1-b1)*gho; vHO[:]=b2*vHO+(1-b2)*gho**2
            WHO[:]=np.clip(WHO-lr*(mHO/(1-b1**t))/(np.sqrt(vHO/(1-b2**t))+1e-8),-WCLIP,WCLIP)
            mIH[:]=b1*mIH+(1-b1)*gih; vIH[:]=b2*vIH+(1-b2)*gih**2
            WIH[:]=np.clip(WIH-lr*LRH*(mIH/(1-b1**t))/(np.sqrt(vIH/(1-b2**t))+1e-8),-WCLIP,WCLIP)
        else:
            WHO[:]=np.clip(WHO-lr*gho,-WCLIP,WCLIP); WIH[:]=np.clip(WIH-lr*LRH*gih,-WCLIP,WCLIP)
    for ep in range(1,EP+1):
        lr=LR*(0.15+0.85*(1-ep/EP)) if ANNEAL else LR
        dWIH=np.zeros_like(WIH); dWHO=np.zeros_like(WHO); nb=0
        for k in rng.permutation(len(Xtr)):
            x=Xtr[k]; lab=ytr[k]; tgt=np.array([TD*(1 if c==lab else -1) for c in range(C)])
            if INJ:
                # 2-solve genuine PC: (1) forward, read clean a_h & mo;  (2) inject scaled real eps_o, analog backward -> eps_h
                neutral=sum([[f"alter Veop{c} = 0.5",f"alter Veon{c} = 0.5"] for c in range(C)],[])
                v=tr.step(inp_alters(x)+neutral,ANL,RD_TR,fout=f"pcotr_{TAG}.dat")
                if v is None:
                    if dead_tr(): recreate_tr()
                    continue
                ah=np.array([v[6*j+4]-v[6*j+5] for j in range(H)]); mo=np.array([v[6*H+2*c]-v[6*H+2*c+1] for c in range(C)])
                eo=tgt-GMO*mo
                inj=sum([[f"alter Veop{c} = {0.5+EINJ*float(eo[c]):.4f}",f"alter Veon{c} = {0.5-EINJ*float(eo[c]):.4f}"] for c in range(C)],[])
                v=tr.step(inp_alters(x)+inj,ANL,RD_TR,fout=f"pcotr_{TAG}.dat")
                if v is None:
                    if dead_tr(): recreate_tr()
                    continue
                xh=np.array([v[6*j]-v[6*j+1] for j in range(H)]); mh=np.array([v[6*j+2]-v[6*j+3] for j in range(H)])
                eh=xh-mh
            else:
                A=inp_alters(x)+sum([[f"alter Vxop{c} = {0.5+TD*(1 if c==lab else -1):.4f}",f"alter Vxon{c} = {0.5-TD*(1 if c==lab else -1):.4f}"] for c in range(C)],[])
                v=tr.step(A,ANL,RD_TR,fout=f"pcotr_{TAG}.dat")
                if v is None: continue   # unstable solve at this weight state -> skip example
                xh=np.array([v[6*j]-v[6*j+1] for j in range(H)]); mh=np.array([v[6*j+2]-v[6*j+3] for j in range(H)])
                ah=np.array([v[6*j+4]-v[6*j+5] for j in range(H)]); mo=np.array([v[6*H+2*c]-v[6*H+2*c+1] for c in range(C)])
                eh=xh-mh; eo=tgt-mo
            dWHO+=np.outer(eo,ah); dWIH+=np.outer(eh,x); nb+=1
            if nb>=MB:
                apply_upd(dWHO/nb,dWIH/nb,lr)  # SUBTRACT (circuit inverts a_h) handled inside; Adam or SGD
                burst_tr(wt_alters()); dWIH[:]=0; dWHO[:]=0; nb=0
        if nb>0:
            apply_upd(dWHO/nb,dWIH/nb,lr); burst_tr(wt_alters())
        if ep%5==0 or ep==EP:
            print(f"  epoch {ep:2d}: test={evaluate():.2f}  lr={lr:.3f}")
    tr.close(); ev.close()
