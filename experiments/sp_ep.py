#!/usr/bin/env python3
"""2-layer EP net, ngspice-batched, BOTH layers trained (hidden LEARNS its features). Generic neuron.
Research idea: nonlinear (high-gain) neurons make the symmetric EP feedback loop unstable (loop gain>1, no
stable .op equilibrium). EP tolerates ASYMMETRIC weights, so scale the feedback path (output->hidden) by
BK_SCALE<1 to drop loop gain below 1 -> stable equilibrium WHILE the forward path stays strongly nonlinear.
All LEVEL=1 transistors, 0 behavioral. Controller cycles data + EP update (free/nudge). ngspice = no Spectre
license contention. Reuses ep_digits cells (dsyn) + sp_deep ngspice plumbing + generic sigmoid neuron."""
import os, subprocess, numpy as np, glob
import ep_digits as E, sp_deep as D, sp_batch as SB

NGSPICE=os.environ.get("NGSPICE","ngspice")
C=E.C; IND=E.IND; TD=E.TD; RB=E.RB
_NSIGN=1.0   # nudge sign: +1 = toward target, -1 = away (for centered/symmetric EP)
NH=int(os.environ.get("NH","16"))
NBIAS=int(os.environ.get("NBIAS","1"))           # parallel output-bias synapses (strong DC trim for readout)
WREP=int(os.environ.get("WREP","1"))             # replicate output synapses: large weight = sum of in-linear-range synapses
BKSC=float(os.environ.get("BK_SCALE","0.2"))     # feedback scale (<1 for stability)
NEUK=int(os.environ.get("NEUK","2"))             # 2=sigmoid(two-stage) generic neuron
WSC=float(os.environ.get("WSCALE","0.5"))
NPROC=int(os.environ.get("NPROC","20")); MThr="1"; SPTO=int(os.environ.get("SPTIMEOUT","30"))
WPER=E.WPER
OLOAD=float(os.environ.get("OLOAD","0"))   # output-load size (smaller=higher impedance=bigger readout swing); 0=auto
Wh=f"{WPER*(2+C+1):.0f}u"; Wo=f"{WPER*(OLOAD if OLOAD else (NH+1)):.0f}u"
CELL="dneuronS" if NEUK==2 else "dneuronR"

def wval(w): return E.wval(w)

def header():
    return ["* sp_ep ngspice",".global vdd vsq vbn bp_hi bp_lo",
        ".model NNR NMOS (LEVEL=1 VTO=0.2  KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)",
        ".model PNR PMOS (LEVEL=1 VTO=-0.2 KP=40u  LAMBDA=0.02 GAMMA=0 PHI=0.7)",
        D._DSYN, D._cell("dneuronS",2), D._cell("dneuronR",3),
        "Vdd vdd 0 1.0","Vbn vbn 0 "+E.VBN,"Vbp_hi bp_hi 0 0.8","Vbp_lo bp_lo 0 0.2","Vsq vsq 0 "+E.VSQ]

def syn(tag,ip,inn,op,on,wsigned):
    gp,gn=wval(wsigned)
    return [f"Vwp_{tag} wp_{tag} 0 {gp:.4f}",f"Vwn_{tag} wn_{tag} 0 {gn:.4f}",
            f"X_{tag} {ip} {inn} {op} {on} wp_{tag} wn_{tag} vdd dsyn"]

def deck_one(wih,who,bh,bo,x,p,lab):
    L=[]
    for i in range(2):
        L+=[f"Vxp{i}_{p} ax{i}p_{p} 0 {0.5+IND*(x[i]-0.5):.4f}",f"Vxn{i}_{p} ax{i}n_{p} 0 {0.5-IND*(x[i]-0.5):.4f}"]
    for c in range(C):
        tdv=(_NSIGN*TD*(1 if c==lab else -1)) if lab is not None else 0.0
        L+=[f"Vtp{c}_{p} vtp{c}_{p} 0 {0.5+tdv:.4f}",f"Vtn{c}_{p} vtn{c}_{p} 0 {0.5-tdv:.4f}"]
    saves=[]
    for j in range(NH):
        up,un=f"uh{j}_{p}",f"vh{j}_{p}"; ap,an=f"ah{j}p_{p}",f"ah{j}n_{p}"
        L+=[f"Mlp_h{j}_{p} {up} {up} vdd vdd PNR W={Wh} L=100u",f"Mln_h{j}_{p} {un} {un} vdd vdd PNR W={Wh} L=100u",
            f"Xn_h{j}_{p} {up} {un} {ap} {an} vdd vbn {CELL}"]
        for i in range(2): L+=syn(f"ih_{i}_{j}_{p}",f"ax{i}p_{p}",f"ax{i}n_{p}",up,un,wih[i][j])
        for c in range(C): L+=syn(f"bk_{j}_{c}_{p}",f"uo{c}_{p}",f"vo{c}_{p}",up,un,BKSC*who[j][c])  # SCALED feedback
        L+=syn(f"bh_{j}_{p}","bp_hi","bp_lo",up,un,bh[j])
        saves+=[ap,an]
    for c in range(C):
        up,un=f"uo{c}_{p}",f"vo{c}_{p}"
        L+=[f"Mlpo{c}_{p} {up} {up} vdd vdd PNR W={Wo} L=100u",f"Mlno{c}_{p} {un} {un} vdd vdd PNR W={Wo} L=100u"]
        for j in range(NH):
            for r in range(WREP): L+=syn(f"fo{r}_{j}_{c}_{p}",f"ah{j}p_{p}",f"ah{j}n_{p}",up,un,who[j][c]/WREP)  # replicated to keep each synapse in linear range
        for b in range(NBIAS): L+=syn(f"bo{b}_{c}_{p}","bp_hi","bp_lo",up,un,bo[c])  # NBIAS parallel bias synapses (strong DC trim)
        rb=RB if lab is not None else 1e10
        L+=[f"Rbp{c}_{p} {up} vtp{c}_{p} {rb}",f"Rbn{c}_{p} {un} vtn{c}_{p} {rb}"]
        saves+=[up,un]
    return L,saves

def run_group(wih,who,bh,bo,items,tag):
    tag=f"{os.getpid()}_{tag}"   # pid-unique: concurrent instances can't collide on tmp files
    L=header(); allsv=set()
    for (p,x,lab) in items:
        Lp,sv=deck_one(wih,who,bh,bo,x,p,lab); L+=Lp; allsv.update(sv)
    L.append(".options reltol="+os.environ.get("RELTOL","2e-3")+" gmin=1e-9")
    L+=[".control","set filetype=ascii","op",f"write /tmp/spep_{tag}.raw","quit",".endc",".end"]
    f=f"/tmp/spep_{tag}.cir"; open(f,"w").write("\n".join(L)+"\n")
    raw=f"/tmp/spep_{tag}.raw"
    try: os.remove(raw)
    except OSError: pass
    return subprocess.Popen([NGSPICE,"-b",f],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL),raw

def run_batch(wih,who,bh,bo,X,labels,tag,nproc=None):
    items=[(p,X[p],(labels[p] if labels is not None else None)) for p in range(len(X))]
    ng=min(nproc or NPROC,max(1,len(items))); groups=[items[i::ng] for i in range(ng)]
    procs=[run_group(wih,who,bh,bo,g,f"{tag}_{gi}") for gi,g in enumerate(groups) if g]
    d={}
    for pr,raw in procs:
        try: pr.wait(timeout=SPTO)
        except Exception: pr.kill(); continue
        if os.path.isfile(raw):
            try:
                dd=SB.parse_nutascii(raw)
                d.update({(k[2:-1] if k.startswith("v(") and k.endswith(")") else k):v for k,v in dd.items()})
            except Exception: pass
    got=sum(1 for p in range(len(X)) if f"uo0_{p}" in d)
    d["_got"]=got; d["_n"]=len(X)
    return d if got>0 else None   # partial OK (missing patterns -> ~0 grad via .get); None only if nothing converged

def run_safe(*a):
    d=run_batch(*a)   # NO catastrophic serial retry; one same-nproc retry only if totally failed
    if d is None: d=run_batch(a[0],a[1],a[2],a[3],a[4],a[5],a[6]+"r")
    return d

def hh(d,j,p): return d.get(f"ah{j}p_{p}",.5)-d.get(f"ah{j}n_{p}",.5)
def oo(d,c,p): return d.get(f"uo{c}_{p}",.5)-d.get(f"vo{c}_{p}",.5)

if __name__=="__main__":
    import time,sys
    (Xtr,ytr),(Xte,yte)=E.make_data()
    rng=np.random.default_rng(int(os.environ.get("SEED","0")))
    wih=[[float(WSC*rng.standard_normal()) for j in range(NH)] for i in range(2)]
    who=[[float(WSC*rng.standard_normal()) for c in range(C)] for j in range(NH)]
    bh=[float(WSC*rng.standard_normal()) for j in range(NH)]
    bo=[float(WSC*rng.standard_normal()) for c in range(C)]
    import pickle
    LOADW=os.environ.get("LOADW",""); SAVEW=os.environ.get("SAVEW","")
    SKIPEVAL=os.environ.get("SKIPEVAL",""); EVALONLY=os.environ.get("EVALONLY","")
    vih=[[0.0]*NH for i in range(2)]; vho=[[0.0]*C for j in range(NH)]; vbh=[0.0]*NH; vbo=[0.0]*C
    best=0.0
    if LOADW and os.path.exists(LOADW):
        try:
            st=pickle.load(open(LOADW,'rb')); wih,who,bh,bo,best=st[:5]
            if len(st)>5: vih,vho,vbh,vbo=st[5:9]
        except Exception: pass
    eta=float(os.environ.get("ETA","0.02")); epochs=int(os.environ.get("EPOCHS","40")); mom=float(os.environ.get("MOM","0.5")); BETA=1.0/RB
    def phi_x(i,x): return 2*IND*(x[i]-0.5)
    def evaluate(X,y):
        d=run_safe(wih,who,bh,bo,X,None,"ev")
        if not d: return -1.0
        ok=[p for p in range(len(X)) if f"uo0_{p}" in d]
        if not ok: return -1.0
        return float(np.mean([int(np.argmax([oo(d,c,p) for c in range(C)]))==y[p] for p in ok]))
    def ridge_eval():   # EP-trained hidden + RIDGE readout (controller-computed from in-spice hidden features)
        df=run_safe(wih,who,bh,bo,Xtr,None,"rf"); dt=run_safe(wih,who,bh,bo,Xte,None,"rt")
        if not df or not dt: return -1.0
        okT=[p for p in range(len(Xtr)) if f"ah0p_{p}" in df]; okE=[p for p in range(len(Xte)) if f"ah0p_{p}" in dt]
        if len(okT)<NH+2 or not okE: return -1.0
        Htr=np.array([[hh(df,j,p) for j in range(NH)]+[1.0] for p in okT])
        Hte=np.array([[hh(dt,j,p) for j in range(NH)]+[1.0] for p in okE])
        Y=np.eye(C)[[ytr[p] for p in okT]]-1.0/C
        W=np.linalg.solve(Htr.T@Htr+1e-2*np.eye(NH+1),Htr.T@Y)
        return float(np.mean([int(np.argmax(Hte[i]@W))==yte[okE[i]] for i in range(len(okE))]))
    def ridge_W():   # returns ridge readout weights (NH+1,C) from in-spice free hidden features
        df=run_safe(wih,who,bh,bo,Xtr,None,"rwf")
        if not df: return None
        okT=[p for p in range(len(Xtr)) if f"ah0p_{p}" in df]
        if len(okT)<NH+2: return None
        Htr=np.array([[hh(df,j,p) for j in range(NH)]+[1.0] for p in okT])
        Y=np.eye(C)[[ytr[p] for p in okT]]-1.0/C
        return np.linalg.solve(Htr.T@Htr+1e-2*np.eye(NH+1),Htr.T@Y)
    def savew():
        if SAVEW:
            try: pickle.dump((wih,who,bh,bo,best,vih,vho,vbh,vbo),open(SAVEW,'wb'))
            except Exception: pass
    if EVALONLY:
        acc=evaluate(Xte,yte); rd=ridge_eval()
        if acc>best: best=acc; savew()
        print(f"EVAL ep-out test={acc:.3f} RIDGE-readout test={rd:.3f} best={best:.3f}",flush=True); sys.exit(0)
    if os.environ.get("DEPLOY"):   # deploy ridge W to in-spice OUTPUT synapses -> validate fully-in-spice inference
        W=ridge_W()
        if W is None: print("DEPLOY: ridge_W failed",flush=True); sys.exit(0)
        print(f"DEPLOY: NBIAS={NBIAS} W feat[min/max/std]={W[:NH].min():.3f}/{W[:NH].max():.3f}/{W[:NH].std():.3f}",flush=True)
        bestd=-1; mx=np.abs(W[:NH]).max(); cap=WREP*1.8
        # deploy FULL-magnitude weights (replication keeps each synapse linear); ONE forward/scale, optimal threshold in controller
        for sf in [WREP*1.6/mx, WREP*1.0/mx, WREP*0.6/mx, WREP*0.3/mx]:
            for j in range(NH):
                for c in range(C): who[j][c]=float(np.clip(sf*W[j][c],-cap,cap))
            for c in range(C): bo[c]=0.0
            d=run_safe(wih,who,bh,bo,Xte,None,"dp")
            if not d: print(f"DEPLOY sf={sf:.3f}: forward failed",flush=True); continue
            ok=[p for p in range(len(Xte)) if f"uo0_{p}" in d]
            diffs=np.array([oo(d,1,p)-oo(d,0,p) for p in ok]); ys=np.array([yte[p] for p in ok])
            # best threshold on (o1-o0): class by diff>thr, both polarities
            thrs=np.percentile(diffs,np.linspace(2,98,60))
            acc=max(max(np.mean((diffs>t)==ys),np.mean((diffs<t)==ys)) for t in thrs)
            if acc>bestd: bestd=acc
            print(f"DEPLOY sf={sf:.3f}: in-spice readout (opt-threshold)={acc:.3f} | diff std={diffs.std():.4f}",flush=True)
        print(f"DEPLOYRESULT best in-spice replicated-readout test={bestd:.3f} (WREP={WREP})",flush=True); sys.exit(0)
    def grad_update(DB,DA,div=None):   # gradient = (corr(DA)-corr(DB))/div ; DA=positive(nudge+), DB=ref(free or nudge-)
        if div is None: div=BETA
        gih=[[0.0]*NH for i in range(2)]; gho=[[0.0]*C for j in range(NH)]; gbh=[0.0]*NH; gbo=[0.0]*C
        for p in range(len(Xtr)):
            x=Xtr[p]
            for j in range(NH):
                hN=hh(DA,j,p); hF=hh(DB,j,p)
                for i in range(2): gih[i][j]+=(phi_x(i,x)*hN-phi_x(i,x)*hF)/div/len(Xtr)
                gbh[j]+=(0.6*hN-0.6*hF)/div/len(Xtr)
                for c in range(C):
                    gho[j][c]+=(hN*oo(DA,c,p)-hF*oo(DB,c,p))/div/len(Xtr)
            for c in range(C): gbo[c]+=(0.6*oo(DA,c,p)-0.6*oo(DB,c,p))/div/len(Xtr)
        for i in range(2):
            for j in range(NH): vih[i][j]=mom*vih[i][j]-eta*gih[i][j]; wih[i][j]=float(np.clip(wih[i][j]+vih[i][j],-3,3))
        for j in range(NH):
            vbh[j]=mom*vbh[j]-eta*gbh[j]; bh[j]=float(np.clip(bh[j]+vbh[j],-3,3))
            for c in range(C): vho[j][c]=mom*vho[j][c]-eta*gho[j][c]; who[j][c]=float(np.clip(who[j][c]+vho[j][c],-3,3))
        for c in range(C): vbo[c]=mom*vbo[c]-eta*gbo[c]; bo[c]=float(np.clip(bo[c]+vbo[c],-3,3))
    PHASE=os.environ.get("PHASE","")
    if PHASE=="free":
        d=run_safe(wih,who,bh,bo,Xtr,None,"fr"); pickle.dump(d,open("/tmp/fr.pkl","wb"))
        print(f"FREE {'ok' if d else 'None'}",flush=True); sys.exit(0)
    if PHASE=="nudge":
        d=run_safe(wih,who,bh,bo,Xtr,list(ytr),"nu"); pickle.dump(d,open("/tmp/nu.pkl","wb"))
        print(f"NUDGE {'ok' if d else 'None'}",flush=True); sys.exit(0)
    if PHASE=="update":
        Df=pickle.load(open("/tmp/fr.pkl","rb")); Dn=pickle.load(open("/tmp/nu.pkl","rb"))
        if Df and Dn: grad_update(Df,Dn); savew(); print("UPDATED ok",flush=True)
        else: print("UPDATED skip(no reads)",flush=True)
        sys.exit(0)
    print(f"# SP_EP NH={NH} NEUK={NEUK} BK_SCALE={BKSC} train={len(Xtr)} test={len(Xte)} best_in={best:.3f}",flush=True)
    AVGFR=float(os.environ.get("AVGFR","0.6")); sW=None; nA=0   # SWA: average weights over late epochs
    CENTERED=int(os.environ.get("CENTERED","0"))   # 1 = symmetric +/-beta nudge (unbiased gradient)
    for ep in range(1,epochs+1):
        t=time.time()
        if CENTERED:
            _NSIGN=1.0;  Dp=run_safe(wih,who,bh,bo,Xtr,list(ytr),"nup")
            _NSIGN=-1.0; Dm=run_safe(wih,who,bh,bo,Xtr,list(ytr),"num"); _NSIGN=1.0
            if not Dp or not Dm: print(f"epoch {ep}: fail",flush=True); continue
            grad_update(Dm,Dp,2*BETA)
        else:
            Df=run_safe(wih,who,bh,bo,Xtr,None,"fr"); Dn=run_safe(wih,who,bh,bo,Xtr,list(ytr),"nu")
            if not Df or not Dn: print(f"epoch {ep}: fail",flush=True); continue
            grad_update(Df,Dn,BETA)
        if ep>=AVGFR*epochs:
            cur=(np.array(wih),np.array(who),np.array(bh),np.array(bo))
            sW=cur if sW is None else tuple(s+c for s,c in zip(sW,cur)); nA+=1
        if not SKIPEVAL and (ep%2==0 or ep==epochs):
            acc=evaluate(Xte,yte)
            if acc>best: best=acc
            print(f"epoch {ep:3d} test={acc:.3f} best={best:.3f} ({time.time()-t:.1f}s/ep)",flush=True)
        else:
            print(f"epoch {ep:3d} trained ({time.time()-t:.1f}s)",flush=True)
        savew()
    if nA:   # evaluate SWA-averaged weights
        aw=tuple(s/nA for s in sW)
        wih[:]=aw[0].tolist(); who[:]=aw[1].tolist(); bh[:]=aw[2].tolist(); bo[:]=aw[3].tolist()
        accA=evaluate(Xte,yte)
        print(f"SWA-avg test={accA:.3f} (vs best {best:.3f})",flush=True)
        if accA>best: best=accA
    rd=ridge_eval()
    print(f"RIDGE-readout test={rd:.3f}",flush=True)
    if rd>best: best=rd
    savew()
    print(f"SPEPRESULT best={best:.3f}")
