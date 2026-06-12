#!/usr/bin/env python3
"""BATCHED analog 2-layer net on 2D nonlinear tasks (circles/rings/spirals).  The whole minibatch is
driven through ONE ngspice .tran as a PWL schedule (one example per time-slot), and every example's
settled a_h and mu_o are read at once -> ~100x fewer ngspice calls than per-example .op.
Forward is REAL analog (gsyn synapses + dneuron); the gradient is Adam-backprop in the controller using
the analog a_h/mu_o (bias term included -> needed for radial tasks).  Goal: >0.95 on each task.
This is the fast harness; the analog-backward (genuine PC) version reuses the same .tran machinery."""
import os, subprocess, numpy as np
TASK=os.environ.get("TASK","circles"); C=int(os.environ.get("C","2")); H=int(os.environ.get("H","24"))
SEED=int(os.environ.get("SEED","0")); EP=int(os.environ.get("EP","60")); MB=int(os.environ.get("MB","32"))
LR=float(os.environ.get("LR","0.02")); SCALE=float(os.environ.get("SCALE","1.5")); NTRpc=int(os.environ.get("NTR","150"))
IND=float(os.environ.get("IND","0.18")); KMAP=float(os.environ.get("KMAP","0.30")); VW0=0.5
WL=os.environ.get("WL","1000u"); RCS=os.environ.get("RCS","300k"); RNODE=os.environ.get("RNODE","20k")
VBSYN=os.environ.get("VBSYN","0.45"); VBNEU=os.environ.get("VBNEU","0.35")
TH=float(os.environ.get("TH","20")); STEP=os.environ.get("STEP","1n"); CSH=os.environ.get("CSH","1e-13")
SA=float(os.environ.get("SA","-1")); SO=float(os.environ.get("SO","-1")); WCLIP=float(os.environ.get("WCLIP","2.5"))
AHSC=float(os.environ.get("AHSC","0.30"))   # a_h compression scale: f'=1-(a_h/AHSC)^2 (real gating despite compression)
TAG=os.environ.get("RUNTAG",str(os.getpid()))
rng=np.random.default_rng(SEED)
# ---- data ----
def gen():
    rg=np.random.default_rng(7)
    X=[];y=[]
    if TASK=="circles":
        for c in range(2):
            r=(rg.uniform(0.2,0.6,1500) if c==0 else rg.uniform(1.0,1.5,1500)); th=rg.uniform(0,2*np.pi,1500)
            X+=list(np.c_[r*np.cos(th),r*np.sin(th)]); y+=[c]*1500
    elif TASK=="rings":   # C narrow, well-separated concentric rings (radially nonlinear; linear=chance)
        for c in range(C):
            _sep=float(os.environ.get("RINGSEP","0.8")); _w=float(os.environ.get("RINGW","0.30")); _base=float(os.environ.get("RINGBASE","0.25"))
            r=rg.uniform(_sep*c+_base,_sep*c+_base+_w,1500); th=rg.uniform(0,2*np.pi,1500); X+=list(np.c_[r*np.cos(th),r*np.sin(th)]); y+=[c]*1500
    elif TASK=="spirals":   # C-arm spiral (classic two-spiral when C=2); moderate winding
        for c in range(C):
            t=np.linspace(0.2,1,1500); r=t*2.5; th=t*1.8*np.pi+c*2*np.pi/C+rg.normal(0,0.08,1500); X+=list(np.c_[r*np.cos(th),r*np.sin(th)]); y+=[c]*1500
    return np.array(X),np.array(y)
X,y=gen(); m=X.mean(0); s=X.std(0)+1e-6; X=(X-m)/s*SCALE
X=np.c_[X,np.ones(len(X))]   # bias input (constant)
N=X.shape[1]   # 3 (2 coords + bias)
Xtr=[];ytr=[];Xte=[];yte=[]
for c in range(C):
    i=np.where(y==c)[0]; rng.shuffle(i)
    for k in i[:NTRpc]: Xtr.append(X[k]); ytr.append(c)
    for k in i[NTRpc:NTRpc+int(os.environ.get("NTE","200"))]: Xte.append(X[k]); yte.append(c)
Xtr=np.array(Xtr); ytr=np.array(ytr); Xte=np.array(Xte); yte=np.array(yte)
WIH=rng.standard_normal((H,N))*np.sqrt(1/N); WHO=rng.standard_normal((C,H))*np.sqrt(1/H)
if float(os.environ.get("RANDWIH","0"))>0:   # FIXED random analog hidden (frozen) -> train output only (ELM / single-layer PC on analog features)
    _rw0=float(os.environ.get("RANDWIH")); WIH=rng.standard_normal((H,N))*_rw0
    if int(os.environ.get("BIASRW","1")): WIH[:,N-1]=rng.uniform(-_rw0,_rw0,H)
FREEZEH=int(os.environ.get("FREEZEH","0")) or float(os.environ.get("RANDWIH","0"))>0
def wgv(v): return float(np.clip(VW0+KMAP*v,0.3,0.9)),float(np.clip(VW0-KMAP*v,0.3,0.9))
RDEG=os.environ.get("RDEG","")   # source-degeneration on the Gilbert pairs (linearizes: R^2 0.92->~0.99)
_gsyn=f""".subckt gsyn inp inn wp wn outp outn vdd vbn
Mtail nt vbn 0 0 NNR W=400u L=100u
M5 na wp nt 0 NNR W=200u L=100u
M6 nb wn nt 0 NNR W=200u L=100u
M1 outp inp na 0 NNR W=200u L=100u
M2 outn inn na 0 NNR W=200u L=100u
M3 outn inp nb 0 NNR W=200u L=100u
M4 outp inn nb 0 NNR W=200u L=100u
.ends""" if not RDEG else f""".subckt gsyn inp inn wp wn outp outn vdd vbn
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
RDEGN=os.environ.get("RDEGN","")   # neuron source-degeneration (linearize the diff-pair)
_dneuron=""".subckt dneuron up un ap an vdd vbn
M1 an up tn 0 NNR W=200u L=100u
M2 ap un tn 0 NNR W=200u L=100u
Mt tn vbn 0 0 NNR W=200u L=100u
Xcm ap an vdd cmld
.ends""" if not RDEGN else f""".subckt dneuron up un ap an vdd vbn
M1 an up s1 0 NNR W=200u L=100u
Rn1 s1 tn {RDEGN}
M2 ap un s2 0 NNR W=200u L=100u
Rn2 s2 tn {RDEGN}
Mt tn vbn 0 0 NNR W=200u L=100u
Xcm ap an vdd cmld
.ends"""
SUB=f""".model NNR NMOS (LEVEL=1 VTO=0.2 KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)
.model PNR PMOS (LEVEL=1 VTO=-0.2 KP=40u LAMBDA=0.02 GAMMA=0 PHI=0.7)
{_gsyn}
.subckt cmld p n vdd
Rcs1 p cmx {RCS}
Rcs2 n cmx {RCS}
MLp p cmx vdd vdd PNR W={WL} L=100u
MLn n cmx vdd vdd PNR W={WL} L=100u
.ends
{_dneuron}"""
def pwl(vals,th):  # vals: list per slot; step waveform, slot g over [g*th,(g+1)*th]
    pts=[]
    for g,v in enumerate(vals):
        t=g*th
        if g>0: pts.append((t,vals[g-1]))   # hold previous until boundary, then jump
        pts.append((t+0.01,v))
    return "PWL("+" ".join(f"{t:.2f}n {v:.4f}" for t,v in pts)+")"
MODE=os.environ.get("MODE","bp")   # bp=controller-backprop ; pc=genuine analog backward
VBBK=os.environ.get("VBBK","0.30"); ESUB=""".subckt esub xp xn mp mn ep en vdd vbn
Mt nt vbn 0 0 NNR W=400u L=100u
M1x ep xn nt 0 NNR W=200u L=100u
M2x en xp nt 0 NNR W=200u L=100u
M1m ep mp nt 0 NNR W=200u L=100u
M2m en mn nt 0 NNR W=200u L=100u
RLp vdd ep 50k
RLn vdd en 50k
.ends"""
def build(batch,inj=None):
    # inj: None=forward(bp) or pass1(pc); else dict c->per-slot eps to inject at error nodes (pc pass2)
    K=len(batch); L=["batched",SUB,(ESUB if MODE=="pc" else ""),"Vdd vdd 0 1.0",f"Vbsyn vbsyn 0 {VBSYN}",f"Vbbk vbbk 0 {VBBK}",f"Vbneu vbneu 0 {VBNEU}","Vgm gm 0 0.6"]
    for i in range(N):
        xp=[0.5+IND*float(Xtr[k][i]) for k in batch]; xn=[0.5-IND*float(Xtr[k][i]) for k in batch]
        L+=[f"Vxp{i} a_xp{i} 0 {pwl(xp,TH)}",f"Vxn{i} a_xn{i} 0 {pwl(xn,TH)}"]
    for j in range(H):
        for i in range(N): gp,gn=wgv(WIH[j,i]); L+=[f"Vwp_ih_{i}_{j} wp_ih_{i}_{j} 0 {gp}",f"Vwn_ih_{i}_{j} wn_ih_{i}_{j} 0 {gn}"]
    for c in range(C):
        for j in range(H): gp,gn=wgv(WHO[c,j]); L+=[f"Vwp_ho_{j}_{c} wp_ho_{j}_{c} 0 {gp}",f"Vwn_ho_{j}_{c} wn_ho_{j}_{c} 0 {gn}"]
    if MODE=="pc":   # error-injection nodes (PWL): neutral on pass1, scaled eps_o on pass2
        for c in range(C):
            ep=[0.5+(inj[c][g] if inj else 0.0) for g in range(K)]; en=[0.5-(inj[c][g] if inj else 0.0) for g in range(K)]
            L+=[f"Veop{c} eop{c} 0 {pwl(ep,TH)}",f"Veon{c} eon{c} 0 {pwl(en,TH)}"]
    for j in range(H):
        xp,xn=f"xhp{j}",f"xhn{j}"
        L+=[f"Xcm_xh{j} {xp} {xn} vdd cmld",f"Rnd_xh{j} {xp} {xn} {RNODE}",f"Cxh{j}p {xp} 0 {CSH}",f"Cxh{j}n {xn} 0 {CSH}"]
        for i in range(N): L+=[f"Xihx_{i}_{j} a_xp{i} a_xn{i} wp_ih_{i}_{j} wn_ih_{i}_{j} {xp} {xn} vdd vbsyn gsyn"]
        if MODE=="pc":
            mp,mn=f"mhp{j}",f"mhn{j}"; L+=[f"Xcm_mh{j} {mp} {mn} vdd cmld",f"Rnd_mh{j} {mp} {mn} {RNODE}",f"Cmh{j}p {mp} 0 {CSH}",f"Cmh{j}n {mn} 0 {CSH}"]
            for i in range(N): L+=[f"Xihm_{i}_{j} a_xp{i} a_xn{i} wp_ih_{i}_{j} wn_ih_{i}_{j} {mp} {mn} vdd vbsyn gsyn"]
            for c in range(C): L+=[f"Xbk_{j}_{c} eop{c} eon{c} wp_ho_{j}_{c} wn_ho_{j}_{c} {xp} {xn} vdd vbbk gsyn"]
        L+=[f"Xn{j} {xp} {xn} ahp{j} ahn{j} vdd vbneu dneuron"]
    for c in range(C):
        L+=[f"Xcm_mo{c} mop{c} mon{c} vdd cmld",f"Rnd_mo{c} mop{c} mon{c} {RNODE}",f"Cmo{c}p mop{c} 0 {CSH}",f"Cmo{c}n mon{c} 0 {CSH}"]
        for j in range(H): L+=[f"Xho_{j}_{c} ahp{j} ahn{j} wp_ho_{j}_{c} wn_ho_{j}_{c} mop{c} mon{c} vdd vbsyn gsyn"]
    rds=sum([[f"v(ahp{j}) v(ahn{j})"] for j in range(H)],[])+[f"v(mop{c}) v(mon{c})" for c in range(C)]
    if MODE=="pc": rds=sum([[f"v(xhp{j}) v(xhn{j}) v(mhp{j}) v(mhn{j})"] for j in range(H)],[])+rds
    L+=[f".options gmin=1e-10 reltol=2e-3 cshunt=1e-15",".control",f"tran {STEP} {K*TH:.1f}n uic",
        f"wrdata pb_{TAG}.dat {' '.join(rds)}",".endc",".end"]
    open(f"pb_{TAG}.cir","w").write("\n".join(L)+"\n"); return K
def run_forward(batch):
    build(batch); subprocess.run(["ngspice","-b",f"pb_{TAG}.cir"],capture_output=True,text=True,timeout=300)
    d=np.loadtxt(f"pb_{TAG}.dat"); t=d[:,0]*1e9; cols=d[:,1::2]; K=len(batch)
    AH=np.zeros((K,H)); MO=np.zeros((K,C))
    off=4*H if MODE=="pc" else 0   # MODE=pc wrdata: [xh,mh]x4H first, then ah, then mo
    for g in range(K):
        row=cols[np.argmin(np.abs(t-(g*TH+TH-1.0)))]
        for j in range(H): AH[g,j]=row[off+2*j]-row[off+2*j+1]
        for c in range(C): MO[g,c]=row[off+2*H+2*c]-row[off+2*H+2*c+1]
    return SA*AH, SO*MO
def run_pc(batch,inj):  # pass2: inject eps_o, read xh,mh,ah,mo -> eps_h=xh-mh
    build(batch,inj); subprocess.run(["ngspice","-b",f"pb_{TAG}.cir"],capture_output=True,text=True,timeout=300)
    d=np.loadtxt(f"pb_{TAG}.dat"); t=d[:,0]*1e9; cols=d[:,1::2]; K=len(batch)
    XH=np.zeros((K,H)); MH=np.zeros((K,H)); AH=np.zeros((K,H)); MO=np.zeros((K,C))
    for g in range(K):
        row=cols[np.argmin(np.abs(t-(g*TH+TH-1.0)))]
        for j in range(H): XH[g,j]=row[4*j]-row[4*j+1]; MH[g,j]=row[4*j+2]-row[4*j+3]
        for j in range(H): AH[g,j]=row[4*H+2*j]-row[4*H+2*j+1]
        for c in range(C): MO[g,c]=row[4*H+2*H+2*c]-row[4*H+2*H+2*c+1]
    return XH-MH, SA*AH, SO*MO
# ---- training: analog forward (batched .tran) + Adam backprop in controller ----
f=np.tanh
def evaluate():
    global Xtr
    MOS=[];LAB=[]
    for s in range(0,len(yte),MB):
        b=list(range(s,min(s+MB,len(yte))))
        sav=Xtr; Xtr=Xte; ah,mo=run_forward(b); Xtr=sav
        for ii,g in enumerate(b): MOS.append(mo[ii]); LAB.append(yte[g])
    M=np.array(MOS)
    if not int(os.environ.get("NODEBIAS","0")): M=M-M.mean(0)   # debias (hurts some radial tasks -> NODEBIAS=1)
    return float(np.mean([int(np.argmax(M[i]))==LAB[i] for i in range(len(LAB))]))
mIH=np.zeros_like(WIH); vIH=np.zeros_like(WIH); mHO=np.zeros_like(WHO); vHO=np.zeros_like(WHO); tA=[0]
def adam(g,m,v,W,lr):
    tA[0]+=0  # per-call below
GIH=float(os.environ.get("GIH","1")); GS=float(os.environ.get("GS","1"))   # gradient sign (empirical, synapse may invert)
if int(os.environ.get("NPTRAIN","0")):
    if float(os.environ.get("RANDWIH","0"))>0:
        # RANDOM-FEATURE expansion: fixed random hidden projection (no hidden training) -> train output only
        _rw=float(os.environ.get("RANDWIH")); WIH[:]=rng.standard_normal((H,N))*_rw
        if int(os.environ.get("BIASRW","1")): WIH[:,N-1]=rng.uniform(-_rw,_rw,H)   # spread bias -> diverse thresholds
        print(f"  [RANDWIH={_rw}] fixed random hidden (extreme-learning-machine style)")
    else:   # forward-ceiling test: numpy-train clean, then analog-eval
        _m1=_v1=_m2=_v2=0;_t=0
        for _ep in range(80):
            _p=rng.permutation(len(ytr))
            for _s in range(0,len(_p),MB):
                _b=_p[_s:_s+MB]; _xb=Xtr[_b]; _T=np.full((len(_b),C),-1.);_T[np.arange(len(_b)),ytr[_b]]=1
                _A=f(_xb@WIH.T); _O=_A@WHO.T; _dO=(_O-_T)/len(_b)
                _g2=_dO.T@_A; _dA=(_dO@WHO)*(1-_A**2); _g1=_dA.T@_xb; _t+=1
                _m1=0.9*_m1+0.1*_g1;_v1=0.999*_v1+0.001*_g1**2;WIH[:]=np.clip(WIH-0.02*(_m1/(1-0.9**_t))/(np.sqrt(_v1/(1-0.999**_t))+1e-8),-1.33,1.33)
                _m2=0.9*_m2+0.1*_g2;_v2=0.999*_v2+0.001*_g2**2;WHO[:]=np.clip(WHO-0.02*(_m2/(1-0.9**_t))/(np.sqrt(_v2/(1-0.999**_t))+1e-8),-1.33,1.33)
        _na=np.mean([np.argmax(WHO@f(WIH@Xte[i]))==yte[i] for i in range(len(yte))])
        print(f"  [NPTRAIN] numpy(clipped) test={_na:.3f}  -> now analog-eval those weights:")
        if int(os.environ.get("PROBE","0")):   # localize forward bottleneck: analog-hidden -> numpy-output
            AHl=[];MOl=[];LB=[]
            for s in range(0,len(yte),MB):
                b=list(range(s,min(s+MB,len(yte)))); sav=Xtr; Xtr=Xte; ah_,mo_=run_forward(b); Xtr=sav
                for ii,g in enumerate(b): AHl.append(ah_[ii]); MOl.append(mo_[ii]); LB.append(yte[g])
            AHa=np.array(AHl); MOa=np.array(MOl); LB=np.array(LB)
            accH=np.mean([np.argmax(WHO@AHa[i])==LB[i] for i in range(len(LB))])
            accHd=np.mean(np.argmax((WHO@AHa.T).T-(WHO@AHa.T).T.mean(0),1)==LB)
            M=MOa-MOa.mean(0); accO=np.mean(np.argmax(M,1)==LB)
            ahcos=float((AHa.ravel()@np.array([f(WIH@Xte[i]) for i in range(len(LB))]).ravel())/(np.linalg.norm(AHa)*np.linalg.norm([f(WIH@Xte[i]) for i in range(len(LB))])+1e-9))
            print(f"  [PROBE] analog-hidden+numpy-out={accH:.3f} (+debias {accHd:.3f})  full-analog={accO:.3f}  a_h cos={ahcos:.3f}")
    if int(os.environ.get("SEPTEST","0")):   # TRUE separability: fit a fresh linear output ON the analog a_h features
        import sys
        def collect(Xset,yset):
            A=[];L=[]
            for s in range(0,len(yset),MB):
                b=list(range(s,min(s+MB,len(yset)))); sav=Xtr; globals()['Xtr']=Xset; ah,_=run_forward(b); globals()['Xtr']=sav
                for ii,g in enumerate(b): A.append(ah[ii]); L.append(yset[g])
            return np.array(A),np.array(L)
        Atr,Ltr=collect(Xtr,ytr); Ate,Lte=collect(Xte,yte)
        for lam in [1e-3,1e-1,1.0]:
            Ab=np.c_[Atr,np.ones(len(Atr))]; T=np.full((len(Ltr),C),-1.0); T[np.arange(len(Ltr)),Ltr]=1
            W=np.linalg.solve(Ab.T@Ab+lam*np.eye(Ab.shape[1]),Ab.T@T)
            acc=np.mean(np.argmax(np.c_[Ate,np.ones(len(Ate))]@W,1)==Lte)
            acctr=np.mean(np.argmax(Ab@W,1)==Ltr)
            print(f"  [SEPTEST lam={lam}] linear fit ON analog a_h: train={acctr:.3f} test={acc:.3f}")
        # LOCAL output rule on the cached analog features (hidden frozen) = what the circuit trains in-place.
        # hinge/margin-gated softmax = the same fix that cracked spirals; this is the single-layer local rule.
        Abtr=np.c_[Atr,np.ones(len(Atr))]; Abte=np.c_[Ate,np.ones(len(Ate))]
        gmo=float(os.environ.get("GMO","12")); wd=float(os.environ.get("WD","3e-4")); mar=float(os.environ.get("MARGIN","0.08")); NEP=int(os.environ.get("NEP","400"))
        RULE=os.environ.get("RULE","hinge")   # 'mse' = pure delta rule (bare analog gprod); 'hinge' = margin-gated (needs comparator)
        Wd=np.zeros((C,Abtr.shape[1])); md=np.zeros_like(Wd); vd=np.zeros_like(Wd); t=0; best=0
        for ep in range(1,NEP+1):
            lr=0.04*(0.03+0.97*(1-ep/NEP))   # anneal LR
            p=rng.permutation(len(Ltr))
            for s in range(0,len(p),32):
                b=p[s:s+32]; o=Abtr[b]@Wd.T; G=np.zeros_like(o)
                for ii in range(len(b)):
                    if RULE=="mse":   # pure delta: eps = o - target(+-1), analog gprod-realizable, NO comparator
                        tgt=np.full(C,-1.0); tgt[Ltr[b][ii]]=1.0; G[ii]=o[ii]-tgt
                    else:             # margin-gated softmax (needs a per-output comparator)
                        z=gmo*o[ii]; z=z-z.max(); pr=np.exp(z)/np.exp(z).sum(); d=pr.copy(); d[Ltr[b][ii]]-=1
                        if int(np.argmax(o[ii]))==Ltr[b][ii] and (o[ii][Ltr[b][ii]]-np.max(np.delete(o[ii],Ltr[b][ii])))>mar: d*=0
                        G[ii]=d
                g=(G.T@Abtr[b])/len(b)+wd*Wd; t+=1
                md=0.9*md+0.1*g; vd=0.999*vd+0.001*g**2; Wd-=lr*(md/(1-0.9**t))/(np.sqrt(vd/(1-0.999**t))+1e-8)
            if ep%80==0:
                a=np.mean(np.argmax(Abte@Wd.T,1)==Lte); best=max(best,a)
        print(f"    [LOCAL-RULE={RULE} FINAL] test={np.mean(np.argmax(Abte@Wd.T,1)==Lte):.3f} best={best:.3f}")
        sys.exit(0)
    if int(os.environ.get("SETTLEPROBE","0")):   # accuracy vs RELAXATION TIME from a NEUTRAL reset before each input
        import sys
        taus=sorted(set(float(x) for x in os.environ.get("TAUS","1,2,3,4,6,9,13,18,25,34,46").split(",") if float(x)<TH))
        RESET=int(os.environ.get("RESET","1"))   # interleave a neutral (zero-input) reset slot before each example
        perT={t:[] for t in taus}; LB2=[]
        for s in range(0,len(yte),MB):
            idx=list(range(s,min(s+MB,len(yte))))
            if RESET:
                seq=[]
                for g in idx: seq.append(np.zeros(N)); seq.append(Xte[g])   # reset, example, reset, example...
                seqX=np.array(seq)
            else:
                seqX=Xte[idx]
            sav=Xtr; globals()['Xtr']=seqX; build(list(range(len(seqX)))); globals()['Xtr']=sav
            subprocess.run(["ngspice","-b",f"pb_{TAG}.cir"],capture_output=True,text=True,timeout=300)
            d=np.loadtxt(f"pb_{TAG}.dat"); tc=d[:,0]*1e9; cl=d[:,1::2]
            for jj,g in enumerate(idx):
                slot=(2*jj+1) if RESET else jj   # the example slot (preceded by its reset)
                LB2.append(yte[g])
                for tau in taus:
                    row=cl[np.argmin(np.abs(tc-(slot*TH+tau)))]
                    perT[tau].append(np.array([SO*(row[2*H+2*c]-row[2*H+2*c+1]) for c in range(C)]))
        LB2=np.array(LB2)
        print(f"  [SETTLE] accuracy vs relaxation time tau (ns after a neutral reset), RESET={RESET} TH={TH}ns RNODE={RNODE} CSH={CSH}:")
        prev=0.0
        for tau in taus:
            M=np.array(perT[tau]); M=M-M.mean(0)
            acc=float(np.mean([np.argmax(M[i])==LB2[i] for i in range(len(LB2))]))
            print(f"    tau={tau:5.1f}ns  acc={acc:.3f}  {'UP' if acc>=prev-1e-9 else 'DOWN<<'}"); prev=acc
        sys.exit(0)
print(f"batched analog-fwd + Adam-bp: TASK={TASK} C={C} H={H} N={N} train={len(ytr)} test={len(yte)} GS={GS}")
# FORWARD FIDELITY check: analog a_h/mo vs numpy at the current (random) weights
_b=list(range(min(MB,len(ytr)))); _ah,_mo=run_forward(_b)
_ahi=np.array([f(WIH@Xtr[k]) for k in _b]); _moi=np.array([WHO@f(WIH@Xtr[k]) for k in _b])
def _cs(a,b): a=a.ravel();b=b.ravel(); return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
print(f"  fidelity: a_h cos={_cs(_ah,_ahi):+.3f}  mu_o cos={_cs(_mo,_moi):+.3f}")
print(f"  epoch 0 test={evaluate():.3f} (chance {1/C:.2f})")
GMO=float(os.environ.get("GMO","10")); EINJ=float(os.environ.get("EINJ","1.0")); TD=float(os.environ.get("TD","0.2"))
order=np.arange(len(ytr)); t=0
for ep in range(1,EP+1):
    LR=(float(os.environ.get("LR","0.02")))*((0.08+0.92*(1-ep/EP)) if int(os.environ.get("ANNEAL","0")) else 1.0)
    rng.shuffle(order)
    for s in range(0,len(order),MB):
        batch=list(order[s:s+MB]); ah,mo=run_forward(batch)
        gIH=np.zeros_like(WIH); gHO=np.zeros_like(WHO)
        if MODE=="pc":
            # genuine analog backward: inject scaled eps_o, read eps_h from the real circuit
            EO=np.array([np.array([TD*(1.0 if c==ytr[k] else -1.0) for c in range(C)])-GMO*mo[ii] for ii,k in enumerate(batch)])
            inj={c:[EINJ*float(EO[ii][c]) for ii in range(len(batch))] for c in range(C)}
            eps_h,_,_=run_pc(batch,inj)
            for ii,k in enumerate(batch):
                gHO+=GS*np.outer(EO[ii],ah[ii]); gIH+=GS*GIH*np.outer(eps_h[ii],Xtr[k])
            gHO/=len(batch); gIH/=len(batch); t+=1; b1=0.9;b2=0.999
            mHO[:]=b1*mHO+(1-b1)*gHO; vHO[:]=b2*vHO+(1-b2)*gHO**2; WHO[:]=np.clip(WHO-LR*(mHO/(1-b1**t))/(np.sqrt(vHO/(1-b2**t))+1e-8),-WCLIP,WCLIP)
            mIH[:]=b1*mIH+(1-b1)*gIH; vIH[:]=b2*vIH+(1-b2)*gIH**2; WIH[:]=(WIH if FREEZEH else np.clip(WIH-LR*(mIH/(1-b1**t))/(np.sqrt(vIH/(1-b2**t))+1e-8),-WCLIP,WCLIP))
            continue
        LOSS=os.environ.get("LOSS","mse")
        for ii,k in enumerate(batch):
            if LOSS in ("ce","hinge"):   # softmax cross-entropy: zero gradient once confidently correct
                z=GMO*mo[ii]; z=z-z.max(); p=np.exp(z)/np.exp(z).sum(); do=GS*p.copy(); do[ytr[k]]-=GS
                if LOSS=="hinge":   # margin-gate: kill gradient on confidently-correct pts (removes read-error bias drift)
                    mlab=mo[ii][ytr[k]]; mrest=np.max(np.delete(mo[ii],ytr[k]))
                    if int(np.argmax(mo[ii]))==ytr[k] and (mlab-mrest)>float(os.environ.get("MARGIN","0.05")): do=do*0.0
            else:
                tgt=np.full(C,-1.0); tgt[ytr[k]]=1.0; do=GS*(mo[ii]-tgt)
            gHO+=np.outer(do,ah[ii])
            fp=np.clip(1-(ah[ii]/AHSC)**2,0.0,1.0)   # f' of the COMPRESSED neuron: a_h~G*tanh -> f'=1-(a_h/G)^2 (real gating)
            dh=GIH*fp*(WHO.T@do)
            gIH+=np.outer(dh,Xtr[k])
        gHO/=len(batch); gIH/=len(batch); t+=1; b1=0.9;b2=0.999
        mHO[:]=b1*mHO+(1-b1)*gHO; vHO[:]=b2*vHO+(1-b2)*gHO**2; WHO[:]=np.clip(WHO-LR*(mHO/(1-b1**t))/(np.sqrt(vHO/(1-b2**t))+1e-8),-WCLIP,WCLIP)
        mIH[:]=b1*mIH+(1-b1)*gIH; vIH[:]=b2*vIH+(1-b2)*gIH**2; WIH[:]=(WIH if FREEZEH else np.clip(WIH-LR*(mIH/(1-b1**t))/(np.sqrt(vIH/(1-b2**t))+1e-8),-WCLIP,WCLIP))
    if ep%5==0 or ep==EP: print(f"  epoch {ep:2d}: test={evaluate():.3f}")
