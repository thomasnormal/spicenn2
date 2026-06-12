#!/usr/bin/env python3
"""Device-level network surrogate of pc_deep (spirals recipe), assembled from VALIDATED cell solvers
(devmodel.py) via offline-generated lookup tables at the network's operating points, plus the MEASURED
gprod grid (realistic CMs). Sequential-online training exactly like the circuit (one example per slot,
Euler cap updates). GATE: reproduce the 8-number Spectre corpus before trusting.
Modes: SOFTC soft clamp (RCLAMP), hard clamp; SGNH/SGNO; gbl anneal via (v-0.2)^2 tail scaling."""
import os, numpy as np
import devmodel as dm

# ---------------- offline tables from validated solvers ----------------
TBL=__import__("os").path.join(__import__("os").path.dirname(__file__),"devnet_tables.npz")
def build_tables():
    print("building 5D gsyn tables (in_d x in_cm x w_d x out_cm x OUT_D) — includes output conductance...")
    ind=np.linspace(-0.6,0.6,17); incm=np.array([0.45,0.50,0.55,0.65,0.78,0.85]); wd=np.linspace(-0.62,0.62,15)
    ocm=np.array([0.42,0.48,0.54,0.60,0.66,0.72]); odf=np.linspace(-0.45,0.45,9)
    I5,C5,W5,O5,D5=np.meshgrid(ind,incm,wd,ocm,odf,indexing="ij")
    Iop,Ion=dm.gsyn_currents(C5+I5/2,C5-I5/2,0.5+W5/2,0.5-W5/2,O5+D5/2,O5-D5/2,vbn=0.45)
    IopB,IonB=dm.gsyn_currents(C5+I5/2,C5-I5/2,0.5+W5/2,0.5-W5/2,O5+D5/2,O5-D5/2,vbn=0.60)
    xd=np.linspace(-0.8,0.8,65); xcm=np.array([0.45,0.55,0.65,0.75,0.85])
    XD,XC=np.meshgrid(xd,xcm,indexing="ij")
    ap,an=dm.dneuron(XC+XD/2,XC-XD/2)
    exd=np.linspace(-0.5,0.5,21); ecm=np.array([0.42,0.50,0.58,0.66,0.74])
    EX,EM,EC1,EC2=np.meshgrid(exd,exd,ecm,ecm,indexing="ij")
    epv,env=dm.esub(EC1+EX/2,EC1-EX/2,EC2+EM/2,EC2-EM/2)
    np.savez(TBL, ind=ind,incm=incm,wd=wd,ocm=ocm,odf=odf, gIop=Iop,gIon=Ion,gIopB=IopB,gIonB=IonB,
             xd=xd,xcm=xcm,neuP=ap,neuN=an, exd=exd,ecm=ecm,esP=epv,esN=env)
    print("tables saved")
if not os.path.exists(TBL) or os.environ.get("REBUILD"): build_tables()
T=np.load(TBL)
from scipy.interpolate import RegularGridInterpolator as RGI
_GIop=RGI((T["ind"],T["incm"],T["wd"],T["ocm"],T["odf"]),T["gIop"],bounds_error=False,fill_value=None)
_GIon=RGI((T["ind"],T["incm"],T["wd"],T["ocm"],T["odf"]),T["gIon"],bounds_error=False,fill_value=None)
_GIopB=RGI((T["ind"],T["incm"],T["wd"],T["ocm"],T["odf"]),T["gIopB"],bounds_error=False,fill_value=None)
_GIonB=RGI((T["ind"],T["incm"],T["wd"],T["ocm"],T["odf"]),T["gIonB"],bounds_error=False,fill_value=None)
def gsyn_tbl(ind,incm,wd,ocm=0.6,odf=0.0,bk=False):
    sh=np.broadcast_shapes(np.shape(ind),np.shape(incm),np.shape(wd),np.shape(ocm),np.shape(odf))
    pts=np.stack([np.broadcast_to(np.clip(ind,-0.6,0.6),sh),np.broadcast_to(np.clip(incm,0.45,0.85),sh),
                  np.broadcast_to(np.clip(wd,-0.62,0.62),sh),np.broadcast_to(np.clip(ocm,0.42,0.72),sh),
                  np.broadcast_to(np.clip(odf,-0.45,0.45),sh)],-1)
    return (_GIopB(pts),_GIonB(pts)) if bk else (_GIop(pts),_GIon(pts))
_NP=RGI((T["xd"],T["xcm"]),T["neuP"],bounds_error=False,fill_value=None)
_NN=RGI((T["xd"],T["xcm"]),T["neuN"],bounds_error=False,fill_value=None)
def neu_tbl(xd,xcm):
    pts=np.stack([np.clip(xd,-0.8,0.8),np.clip(xcm,0.45,0.85)],-1)
    return _NP(pts),_NN(pts)
_EP4=RGI((T["exd"],T["exd"],T["ecm"],T["ecm"]),T["esP"],bounds_error=False,fill_value=None)
_EN4=RGI((T["exd"],T["exd"],T["ecm"],T["ecm"]),T["esN"],bounds_error=False,fill_value=None)
def esub4_tbl(xdv,mdv,xcm,mcm):
    sh=np.broadcast_shapes(np.shape(xdv),np.shape(mdv),np.shape(xcm),np.shape(mcm))
    pts=np.stack([np.broadcast_to(np.clip(xdv,-0.5,0.5),sh),np.broadcast_to(np.clip(mdv,-0.5,0.5),sh),
                  np.broadcast_to(np.clip(xcm,0.42,0.74),sh),np.broadcast_to(np.clip(mcm,0.42,0.74),sh)],-1)
    return _EP4(pts),_EN4(pts)
g=np.load("fastsim/gprod_gridCM.npz")
GP=RGI((g["xv"],g["yv"]),g["Idiff"],bounds_error=False,fill_value=None)        # y at ACTIVITY CM 0.705 (layers >= 2)
g0=np.load("fastsim/gprod_grid.npz")
GP0=RGI((g0["vals"],g0["vals"]),g0["Idiff"],bounds_error=False,fill_value=None) # y at INPUT CM 0.5 (layer 1: pre-synaptic = a0!)
def cmld_ext(Ip,In,clampP=None,clampN=None,RC=None,WL=10.0,RNODE=20e3,ito=30,itd=30):
    """cmld+RNODE (+optional clamp) solved in (cm,d) coordinates — MONOTONE nested bisection (robust in all
    regimes; the alternating p/n iteration finds spurious asymmetric solutions at heavy balanced sink)."""
    shp=np.broadcast_shapes(np.shape(Ip),np.shape(In))
    Ipb=np.broadcast_to(Ip,shp); Inb=np.broadcast_to(In,shp)
    beta=40e-6*WL
    def solve_d(cm):
        lo=np.full(shp,-0.6); hi=np.full(shp,0.6)
        for _ in range(itd):
            d=0.5*(lo+hi); p=cm+d/2; n=cm-d/2
            f=dm.pI(cm,p,1.0,beta=beta)-dm.pI(cm,n,1.0,beta=beta)-(Ipb-Inb)-2*d/RNODE
            if clampP is not None: f=f-(p-clampP)/RC+(n-clampN)/RC
            lo=np.where(f>0,d,lo); hi=np.where(f<=0,d,hi)
        return 0.5*(lo+hi)
    lo=np.zeros(shp); hi=np.full(shp,1.0)
    for _ in range(ito):
        cm=0.5*(lo+hi); d=solve_d(cm); p=cm+d/2; n=cm-d/2
        f=dm.pI(cm,p,1.0,beta=beta)+dm.pI(cm,n,1.0,beta=beta)-(Ipb+Inb)
        if clampP is not None: f=f-(p-clampP)/RC-(n-clampN)/RC
        lo=np.where(f>0,cm,lo); hi=np.where(f<=0,cm,hi)
    cm=0.5*(lo+hi); d=solve_d(cm)
    return cm+d/2, cm-d/2

# ---- cmld lookup tables (built from cmld_ext solver) ----
CTBL=__import__("os").path.join(__import__("os").path.dirname(__file__),"cmld_tables.npz")
def build_cmld_tables():
    print("building cmld tables...")
    Ig=np.linspace(0,40e-6,33)
    P1,P2=np.meshgrid(Ig,Ig,indexing="ij")
    p,n=cmld_ext(P1,P2)
    cg=np.linspace(0.38,0.62,7)
    I1,I2,C1,C2=np.meshgrid(np.linspace(0,40e-6,17),np.linspace(0,40e-6,17),cg,cg,indexing="ij")
    pc,nc=cmld_ext(I1,I2,clampP=C1,clampN=C2,RC=float(os.environ.get('RCLAMP','120e3')))
    np.savez(CTBL,Ig=Ig,p=p,n=n,Ig2=np.linspace(0,40e-6,17),cg=cg,pc=pc,nc=nc)
if not os.path.exists(CTBL) or os.environ.get("REBUILD"): build_cmld_tables()
CT=np.load(CTBL)
_CP=RGI((CT["Ig"],CT["Ig"]),CT["p"],bounds_error=False,fill_value=None)
_CN=RGI((CT["Ig"],CT["Ig"]),CT["n"],bounds_error=False,fill_value=None)
_CPc=RGI((CT["Ig2"],CT["Ig2"],CT["cg"],CT["cg"]),CT["pc"],bounds_error=False,fill_value=None)
_CNc=RGI((CT["Ig2"],CT["Ig2"],CT["cg"],CT["cg"]),CT["nc"],bounds_error=False,fill_value=None)
def cmld_tbl(Ip,In):
    pts=np.stack([np.clip(Ip,0,40e-6),np.clip(In,0,40e-6)],-1); return _CP(pts),_CN(pts)
def cmld_clamp_tbl(Ip,In,cp,cn):
    pts=np.stack([np.clip(Ip,0,40e-6),np.clip(In,0,40e-6),np.broadcast_to(np.clip(cp,0.38,0.62),np.shape(Ip)),np.broadcast_to(np.clip(cn,0.38,0.62),np.shape(Ip))],-1)
    return _CPc(pts),_CNc(pts)

# ---------------- topology / data (exact pc_deep port) ----------------
SEED=int(os.environ.get("SEED","0")); rng=np.random.default_rng(SEED)
WSEED=int(os.environ.get("WSEED",str(SEED))); wrng=np.random.default_rng(WSEED+1000)
TASK=os.environ.get("TASK","spirals"); C=2
LAYERS=[int(x) for x in os.environ.get("LAYERS","2,4,4,4,4,4").split(",")]+[C]
K=int(os.environ.get("FANIN","4"))
NTR=int(os.environ.get("NTR","60")); NTE=int(os.environ.get("NTE","80")); NEP=int(os.environ.get("NEP","24"))
SCALE=float(os.environ.get("SCALE","1.0")); IND=float(os.environ.get("IND","0.30")); TD=float(os.environ.get("TD","0.1"))
KMAP=0.30; VW0=0.5
SGNH=float(os.environ.get("SGNH","-1")); SGNO=float(os.environ.get("SGNO","-1"))
WINIT=float(os.environ.get("WINIT","1.5")); WINITO=float(os.environ.get("WINITO","0.3"))
GBLH=float(os.environ.get("GBLH","0.6")); GBLO=float(os.environ.get("GBLO","0.6")); AFL=float(os.environ.get("AFLOOR","0.4"))
SOFTC=int(os.environ.get("SOFTC","1")); RCLAMP=float(os.environ.get("RCLAMP","120e3"))
EVK=int(os.environ.get("EVK","4")); TH=400e-9; CWW=float(os.environ.get("CWW","300e-12"))
GBK=float(os.environ.get("GBK","1.0"))   # backward transresistance share (same node R)
NL=len(LAYERS)
def gen():
    rg=np.random.default_rng(7); X=[];y=[]
    if TASK=="spirals":
        for c in range(C):
            t=np.linspace(0.2,1,1500); r=t*2.5; th=t*1.8*np.pi+c*2*np.pi/C+rg.normal(0,0.08,1500)
            X+=list(np.c_[r*np.cos(th),r*np.sin(th)]); y+=[c]*1500
    elif TASK=="circles":
        for c in range(2):
            r=(rg.uniform(0.2,0.6,1500) if c==0 else rg.uniform(1.0,1.5,1500)); th=rg.uniform(0,2*np.pi,1500)
            X+=list(np.c_[r*np.cos(th),r*np.sin(th)]); y+=[c]*1500
    return np.array(X,float),np.array(y)
X,y=gen(); X=(X-X.mean(0))/(X.std(0)+1e-6)*SCALE
Xtr=[];ytr=[];Xte=[];yte=[]
for c in range(C):
    i=np.where(y==c)[0]; rng.shuffle(i)
    for k in i[:NTR]: Xtr.append(X[k]); ytr.append(c)
    for k in i[NTR:NTR+NTE]: Xte.append(X[k]); yte.append(c)
Xtr=np.array(Xtr); ytr=np.array(ytr); Xte=np.array(Xte); yte=np.array(yte)
NIN=LAYERS[0]
par={}
for l in range(1,NL):
    nprev=LAYERS[l-1]; kk=min(K,nprev)
    par[l]=np.array([ (rng.choice(nprev,size=kk,replace=False) if kk<nprev else np.arange(nprev)) for j in range(LAYERS[l]) ])
W={}
for l in range(1,NL):
    Wl=np.zeros(par[l].shape)
    for j in range(LAYERS[l]):
        for c_ in range(par[l].shape[1]):
            SIN=int(os.environ.get('SINIT','0'))
            v=(WINITO if (l==NL-1 and SIN) else WINIT)*float(wrng.standard_normal())   # pc_deep: WINITO only under SINIT/UNIRO
            gp=np.clip(VW0+KMAP*v,0.3,0.9); gn=np.clip(VW0-KMAP*v,0.3,0.9)
            Wl[j,c_]=gp-gn
    W[l]=Wl
# children map for backward
child={l:[[] for _ in range(LAYERS[l])] for l in range(1,NL-1)}
for l in range(2,NL):
    for j in range(LAYERS[l]):
        for c_,p in enumerate(par[l][j]):
            if l-1>=1: child[l-1][p].append((j,c_))
# ---------------- settle + train ----------------
def encode(xv):
    return np.array([2*IND*np.clip(xv[d] if d<len(xv) else 1.0,-3,3)/3.0*2 for d in range(NIN)])
_dbg={}
def settle(a0p,a0n, yclP=None,yclN=None, train=False, relax=int(os.environ.get("RELAX","10")), damp=float(os.environ.get("DAMP","0.5")), state=None):
    """a0p/a0n: (...,NIN) input PAIR voltages. Pairs (CM-aware); UNDER-RELAXED iteration (the circuit
    co-settles continuously; undamped discrete iteration has loop gain>1 and diverges - verified vs trace)."""
    lead=a0p.shape[:-1]
    ap={0:a0p}; an={0:a0n}
    if state is not None and state.get("init"):   # WARM START: caps carry node state across slots (physical!)
        mp=state["mp"]; mn=state["mn"]; xp=state["xp"]; xn=state["xn"]
        epp=state["epp"]; epn=state["epn"]; xprev=state["xprev"]
        for l in range(1,NL-1):
            if l in xp: ap[l],an[l]=neu_tbl(xp[l]-xn[l],0.5*(xp[l]+xn[l]))
    else:
        mp={}; mn={}; xp={}; xn={}
        epp={l:np.full(lead+(LAYERS[l],),0.78) for l in range(1,NL)}
        epn={l:np.full(lead+(LAYERS[l],),0.78) for l in range(1,NL)}
        xprev={}
    for it in range(relax):
        for l in range(1,NL):
            aind=(ap[l-1]-an[l-1])[...,par[l]]; aincm=0.5*(ap[l-1]+an[l-1])[...,par[l]]
            Wb=np.broadcast_to(W[l],aind.shape)
            if l in mp: mo_cm=(0.5*(mp[l]+mn[l]))[...,None]; mo_d=(mp[l]-mn[l])[...,None]
            else: mo_cm,mo_d=0.6,0.0
            Iop,Ion=gsyn_tbl(aind,aincm,Wb,ocm=mo_cm,odf=mo_d)   # self-consistent: currents at the node's actual state
            mp[l],mn[l]=cmld_tbl(Iop.sum(-1),Ion.sum(-1))
            if l<NL-1:
                if l in xprev: xo_cm=(0.5*(xprev[l][0]+xprev[l][1]))[...,None]; xo_d=(xprev[l][0]-xprev[l][1])[...,None]
                else: xo_cm,xo_d=0.6,0.0
                Iopx,Ionx=gsyn_tbl(aind,aincm,Wb,ocm=xo_cm,odf=xo_d)   # forward-into-x at x's actual state (conductance!)
                Ixp,Ixn=Iopx.sum(-1).copy(),Ionx.sum(-1).copy()
                if train:
                    ed=(epp[l+1]-epn[l+1]); ecm=0.5*(epp[l+1]+epn[l+1])   # RAW eps (circuit line 409): SGNH applies ONLY to the gprod learning input, NOT the backward dynamics!
                    EXB=int(os.environ.get("EXACTBK","0"))
                    for p in range(LAYERS[l]):
                        for (j2,c_) in child[l][p]:
                            if EXB:   # exact solver at the ACTUAL x output voltages (triode + output conductance)
                                eip=ecm[...,j2]+ed[...,j2]/2; ein=ecm[...,j2]-ed[...,j2]/2
                                wv=W[l+1][j2,c_]
                                if l in xprev: oxp=xprev[l][0][...,p]; oxn=xprev[l][1][...,p]
                                else: oxp=np.full_like(eip,0.5); oxn=np.full_like(eip,0.5)
                                Bop,Bon=dm.gsyn_currents(eip,ein,0.5+wv/2,0.5-wv/2,oxp,oxn,vbn=0.60)
                            else:
                                if l in xprev: xcmb=0.5*(xprev[l][0][...,p]+xprev[l][1][...,p]); xdb=xprev[l][0][...,p]-xprev[l][1][...,p]
                                else: xcmb,xdb=0.6,0.0
                                Bop,Bon=gsyn_tbl(ed[...,j2],ecm[...,j2],np.broadcast_to(W[l+1][j2,c_],ed[...,j2].shape),ocm=xcmb,odf=xdb,bk=True)
                            Ixp[...,p]+=Bop; Ixn[...,p]+=Bon
                nxp,nxn=cmld_tbl(Ixp,Ixn)
                if l in xprev: nxp=damp*nxp+(1-damp)*xprev[l][0]; nxn=damp*nxn+(1-damp)*xprev[l][1]
                xprev[l]=(nxp,nxn); xp[l],xn[l]=nxp,nxn
                ap[l],an[l]=neu_tbl(xp[l]-xn[l],0.5*(xp[l]+xn[l]))
            else:
                if train and yclP is not None:
                    if SOFTC:
                        xp[l],xn[l]=cmld_clamp_tbl(Iop.sum(-1),Ion.sum(-1),yclP,yclN)
                    else:
                        xp[l],xn[l]=np.broadcast_to(yclP,mp[l].shape),np.broadcast_to(yclN,mn[l].shape)
                else: xp[l],xn[l]=mp[l],mn[l]
            if l==1: _dbg["m1d"]=mp[l]-mn[l]; _dbg["x1d"]=xp[l]-xn[l]; _dbg["x1cm"]=0.5*(xp[l]+xn[l]); _dbg["m1cm"]=0.5*(mp[l]+mn[l])
            if l==NL-1: _dbg["m6d"]=mp[l]-mn[l]; _dbg["m6cm"]=0.5*(mp[l]+mn[l]); _dbg["xqd"]=xp[l]-xn[l]; _dbg["xqcm"]=0.5*(xp[l]+xn[l])
            nep,nen=esub4_tbl(xp[l]-xn[l],mp[l]-mn[l],0.5*(xp[l]+xn[l]),0.5*(mp[l]+mn[l]))   # 4D table: full CM-imbalance law
            epp[l]=damp*nep+(1-damp)*epp[l]; epn[l]=damp*nen+(1-damp)*epn[l]
    if state is not None:
        state.update(init=True,mp=mp,mn=mn,xp=xp,xn=xn,epp=epp,epn=epn,xprev=xprev)
    return mp[NL-1]-mn[NL-1],(ap,an),(epp,epn)
def evaluate():
    A0=np.stack([encode(Xte[i]) for i in range(len(Xte))])
    mo,_,_=settle(0.5+A0/2,0.5-A0/2,train=False,relax=3)
    mo=mo-mo.mean(0)
    return float((np.argmax(mo,1)==yte).mean())
total=NEP*len(Xtr); cnt=0; curve=[]; SST={}
for ep in range(NEP):
    idx=list(range(len(Xtr))); rng.shuffle(idx)
    for k in idx:
        f=cnt/total; cnt+=1
        sc=lambda g0:((g0-(g0-AFL)*f-0.2)/(0.6-0.2))**2
        yc=np.full(C,-TD); yc[ytr[k]]=TD; yc*=2
        a0=encode(Xtr[k])[None]
        _,(apv,anv),(eppv,epnv)=settle(0.5+a0/2,0.5-a0/2,yclP=(0.5+yc/2)[None],yclN=(0.5-yc/2)[None],train=True,state=SST)
        if os.environ.get("DEBUG"):
            global dbg_m1,dbg_x1,dbg_x1cm
            dbg_m1=_dbg["m1d"][0]; dbg_x1=_dbg["x1d"][0]; dbg_x1cm=_dbg["x1cm"][0]; dbg_m1cm=_dbg["m1cm"][0]
        if os.environ.get("DEBUG"):
            a1d=apv[1][0]-anv[1][0]; e1=eppv[1][0]-epnv[1][0]; e2=eppv[2][0]-epnv[2][0]
            e6=eppv[NL-1][0]-epnv[NL-1][0]
            print(f" slot{cnt-1:3d} | m6d={_dbg['m6d'][0][0]:+.4f} (cm={_dbg['m6cm'][0][0]:.3f}) xqd={_dbg['xqd'][0][0]:+.4f} (cm={_dbg['xqcm'][0][0]:.3f}) | e6={e6[0]:+.4f} {e6[1]:+.4f} | e1_0={e1[0]:+.4f} w1_00={W[1][0,0]:+.4f}",flush=True)
        for l in range(1,NL):
            sg=SGNO if l==NL-1 else SGNH
            scale=sc(GBLO if l==NL-1 else GBLH)*TH/CWW
            ed=np.clip(sg*(eppv[l][0]-epnv[l][0]),-0.5,0.5)[...,None]
            yd=np.clip((apv[l-1][0]-anv[l-1][0])[par[l]],-0.54,0.54)
            tb=GP0 if l==1 else GP   # layer-1 pre-synaptic y = a0 at CM 0.5 -> CM-naive grid (3.7x stronger; trace-verified)
            I=tb(np.stack([np.broadcast_to(ed,yd.shape),yd],-1))
            W[l]=np.clip(W[l]+scale*I,-0.62,0.62)
    if (ep+1)%EVK==0 or ep==NEP-1:
        curve.append(round(evaluate(),3)); print(f"  ep{ep+1}: {curve[-1]}",flush=True)
if os.environ.get("DEBUG"):
    pass
print(f"[devnet {TASK} s{SEED} SOFTC={SOFTC}] BEST={max(curve):.3f} curve={curve}")
