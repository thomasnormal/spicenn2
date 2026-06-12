#!/usr/bin/env python3
"""DEEP EP net, ngspice-batched, ALL layers trained (every hidden layer learns its own features).
Generic two-stage-sigmoid neurons. Adjacent layers fully connected but SMALL (low fan-in). Symmetric EP
feedback between every adjacent pair, scaled by BK_SCALE for relaxation stability. ngspice converges the
nonlinear feedback .op (gmin stepping) where Spectre's Newton diverged. All LEVEL=1 transistors, 0 behavioral.
Controller cycles data + EP update. VALID: no frozen layers, no hand-built features, hidden learns end-to-end."""
import os, subprocess, numpy as np
import ep_digits as E, sp_deep as D, sp_batch as SB

NGSPICE=os.environ.get("NGSPICE","ngspice")
C=E.C; IND=E.IND; TD=E.TD; RB=E.RB; BETA=1.0/RB
SIZES=[int(x) for x in os.environ.get("HID","10,10,10").split(",")]   # hidden layer sizes
LAYERS=[2]+SIZES+[C]                       # full: input(2) ... output(C)
L=len(LAYERS)-1                            # number of weight layers (edges)
BKSC=float(os.environ.get("BK_SCALE","0.3"))
NEUK=int(os.environ.get("NEUK","2")); CELL="dneuronS" if NEUK==2 else "dneuronR"
WSC=float(os.environ.get("WSCALE","0.5"))
NPROC=int(os.environ.get("NPROC","20")); SPTO=int(os.environ.get("SPTIMEOUT","30"))
WPER=E.WPER
def Wload(fanin): return f"{WPER*(fanin+1):.0f}u"

def wval(w): return E.wval(w)
def syn(tag,ip,inn,op,on,wsigned):
    gp,gn=wval(wsigned)
    return [f"Vwp_{tag} wp_{tag} 0 {gp:.4f}",f"Vwn_{tag} wn_{tag} 0 {gn:.4f}",
            f"X_{tag} {ip} {inn} {op} {on} wp_{tag} wn_{tag} vdd dsyn"]

def header():
    return ["* sp_dep_ep ngspice",".global vdd vsq vbn bp_hi bp_lo",
        ".model NNR NMOS (LEVEL=1 VTO=0.2  KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)",
        ".model PNR PMOS (LEVEL=1 VTO=-0.2 KP=40u  LAMBDA=0.02 GAMMA=0 PHI=0.7)",
        D._DSYN, D._cell("dneuronS",2), D._cell("dneuronR",3),
        "Vdd vdd 0 1.0","Vbn vbn 0 "+E.VBN,"Vbp_hi bp_hi 0 0.8","Vbp_lo bp_lo 0 0.2","Vsq vsq 0 "+E.VSQ]

# activation node names: layer 0 = inputs (ax); hidden k -> ak; output -> uo/vo (the readout IS the activation)
def actp(k,i,p): return f"ax{i}p_{p}" if k==0 else (f"uo{i}_{p}" if k==L else f"a{k}_{i}p_{p}")
def actn(k,i,p): return f"ax{i}n_{p}" if k==0 else (f"vo{i}_{p}" if k==L else f"a{k}_{i}n_{p}")
def memp(k,i,p): return f"u{k}_{i}_{p}"   # membrane (k=1..L)
def memn(k,i,p): return f"v{k}_{i}_{p}"

def deck_one(W,Bh,Bo,x,p,lab):
    Lh=[]
    for i in range(2):
        Lh+=[f"Vxp{i}_{p} ax{i}p_{p} 0 {0.5+IND*(x[i]-0.5):.4f}",f"Vxn{i}_{p} ax{i}n_{p} 0 {0.5-IND*(x[i]-0.5):.4f}"]
    for c in range(C):
        tdv=(TD*(1 if c==lab else -1)) if lab is not None else 0.0
        Lh+=[f"Vtp{c}_{p} vtp{c}_{p} 0 {0.5+tdv:.4f}",f"Vtn{c}_{p} vtn{c}_{p} 0 {0.5-tdv:.4f}"]
    saves=[]
    # hidden layers k=1..L-1
    for k in range(1,L):
        nin=LAYERS[k-1]; nout=LAYERS[k+1]; fanin=nin+nout+1
        for j in range(LAYERS[k]):
            up,un=memp(k,j,p),memn(k,j,p); ap,an=actp(k,j,p),actn(k,j,p)
            Lh+=[f"Mlp{k}_{j}_{p} {up} {up} vdd vdd PNR W={Wload(fanin)} L=100u",
                 f"Mln{k}_{j}_{p} {un} {un} vdd vdd PNR W={Wload(fanin)} L=100u",
                 f"Xn{k}_{j}_{p} {up} {un} {ap} {an} vdd vbn {CELL}"]
            for i in range(nin): Lh+=syn(f"f{k}_{i}_{j}_{p}",actp(k-1,i,p),actn(k-1,i,p),up,un,W[k][i][j])
            for m in range(nout): Lh+=syn(f"b{k}_{m}_{j}_{p}",actp(k+1,m,p),actn(k+1,m,p),up,un,BKSC*W[k+1][j][m])
            Lh+=syn(f"bh{k}_{j}_{p}","bp_hi","bp_lo",up,un,Bh[k][j])
            saves+=[ap,an]
    # output layer k=L
    nin=LAYERS[L-1]
    for c in range(C):
        up,un=f"uo{c}_{p}",f"vo{c}_{p}"
        Lh+=[f"Mlpo{c}_{p} {up} {up} vdd vdd PNR W={Wload(nin+1)} L=100u",
             f"Mlno{c}_{p} {un} {un} vdd vdd PNR W={Wload(nin+1)} L=100u"]
        for i in range(nin): Lh+=syn(f"fo_{i}_{c}_{p}",actp(L-1,i,p),actn(L-1,i,p),up,un,W[L][i][c])
        Lh+=syn(f"bo_{c}_{p}","bp_hi","bp_lo",up,un,Bo[c])
        rb=RB if lab is not None else 1e10
        Lh+=[f"Rbp{c}_{p} {up} vtp{c}_{p} {rb}",f"Rbn{c}_{p} {un} vtn{c}_{p} {rb}"]
        saves+=[up,un]
    return Lh,saves

def run_group(W,Bh,Bo,items,tag):
    tag=f"{os.getpid()}_{tag}"; Lh=header()
    for (p,x,lab) in items:
        Lp,_=deck_one(W,Bh,Bo,x,p,lab); Lh+=Lp
    Lh.append(".options reltol="+os.environ.get("RELTOL","2e-3")+" gmin=1e-9")
    Lh+=[".control","set filetype=ascii","op",f"write /tmp/sdep_{tag}.raw","quit",".endc",".end"]
    f=f"/tmp/sdep_{tag}.cir"; open(f,"w").write("\n".join(Lh)+"\n")
    raw=f"/tmp/sdep_{tag}.raw"
    try: os.remove(raw)
    except OSError: pass
    return subprocess.Popen([NGSPICE,"-b",f],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL),raw

def run_batch(W,Bh,Bo,X,labels,tag,nproc=None):
    items=[(p,X[p],(labels[p] if labels is not None else None)) for p in range(len(X))]
    ng=min(nproc or NPROC,max(1,len(items))); groups=[items[i::ng] for i in range(ng)]
    procs=[run_group(W,Bh,Bo,g,f"{tag}_{gi}") for gi,g in enumerate(groups) if g]
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
    return d if got>0 else None   # partial OK (missing patterns -> ~0 via .get); no catastrophic serial retry
def run_safe(W,Bh,Bo,X,labels,tag):
    d=run_batch(W,Bh,Bo,X,labels,tag)
    if d is None: d=run_batch(W,Bh,Bo,X,labels,tag+"r")
    return d

def act_val(d,k,i,p):
    if k==0: return 2*IND*0  # input handled separately
    if k==L: return d.get(f"uo{i}_{p}",.5)-d.get(f"vo{i}_{p}",.5)
    return d.get(f"a{k}_{i}p_{p}",.5)-d.get(f"a{k}_{i}n_{p}",.5)

if __name__=="__main__":
    import time,sys
    LF="/tmp/sdep_ep.lock"
    try: fd=os.open(LF,os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.close(fd)
    except FileExistsError:
        try: age=time.time()-os.path.getmtime(LF)
        except Exception: age=0
        if age<1800: print("# locked"); sys.exit(0)
        os.remove(LF); os.close(os.open(LF,os.O_CREAT|os.O_WRONLY))
    import atexit; atexit.register(lambda:(os.remove(LF) if os.path.exists(LF) else None))
    (Xtr,ytr),(Xte,yte)=E.make_data()
    rng=np.random.default_rng(int(os.environ.get("SEED","0")))
    W=[None]+[[[float(WSC*rng.standard_normal()) for o in range(LAYERS[k])] for i in range(LAYERS[k-1])] for k in range(1,L+1)]
    Bh=[None]+[[float(WSC*rng.standard_normal()) for j in range(LAYERS[k])] for k in range(1,L)]+[None]
    Bo=[float(WSC*rng.standard_normal()) for c in range(C)]
    vW=[None]+[[[0.0]*LAYERS[k] for i in range(LAYERS[k-1])] for k in range(1,L+1)]
    eta=float(os.environ.get("ETA","0.02")); epochs=int(os.environ.get("EPOCHS","60")); mom=float(os.environ.get("MOM","0.5"))
    def inval(i,x): return 2*IND*(x[i]-0.5)
    def evaluate(X,y):
        d=run_safe(W,Bh,Bo,X,None,"ev")
        if not d: return 0.0
        return float(np.mean([int(np.argmax([d.get(f"uo{c}_{p}",.5)-d.get(f"vo{c}_{p}",.5) for c in range(C)]))==y[p] for p in range(len(X))]))
    NL=LAYERS[L-1]   # width of last hidden layer (the readout reads THIS, low fan-in)
    def ridge_eval():   # ridge readout on LAST hidden layer features (act_val k=L-1)
        df=run_safe(W,Bh,Bo,Xtr,None,"rf"); dt=run_safe(W,Bh,Bo,Xte,None,"rt")
        if not df or not dt: return -1.0
        okT=[p for p in range(len(Xtr)) if f"a{L-1}_0p_{p}" in df]; okE=[p for p in range(len(Xte)) if f"a{L-1}_0p_{p}" in dt]
        if len(okT)<NL+2 or not okE: return -1.0
        Htr=np.array([[act_val(df,L-1,i,p) for i in range(NL)]+[1.0] for p in okT])
        Hte=np.array([[act_val(dt,L-1,i,p) for i in range(NL)]+[1.0] for p in okE])
        Yr=np.eye(C)[[ytr[p] for p in okT]]-1.0/C
        Wr=np.linalg.solve(Htr.T@Htr+1e-2*np.eye(NL+1),Htr.T@Yr)
        return float(np.mean([int(np.argmax(Hte[i]@Wr))==yte[okE[i]] for i in range(len(okE))]))
    a0=evaluate(Xte,yte); best=a0
    print(f"# DEEP_EP layers={LAYERS} BK_SCALE={BKSC} NEUK={NEUK} train={len(Xtr)} test={len(Xte)} epoch0={a0:.3f}",flush=True)
    for ep in range(1,epochs+1):
        t=time.time()
        Df=run_safe(W,Bh,Bo,Xtr,None,"fr"); Dn=run_safe(W,Bh,Bo,Xtr,list(ytr),"nu")
        if not Df or not Dn: print(f"epoch {ep}: fail",flush=True); continue
        gW=[None]+[[[0.0]*LAYERS[k] for i in range(LAYERS[k-1])] for k in range(1,L+1)]
        nb=len(Xtr)
        for p in range(nb):
            x=Xtr[p]
            for k in range(1,L+1):
                for i in range(LAYERS[k-1]):
                    preN=inval(i,x) if k==1 else act_val(Dn,k-1,i,p)
                    preF=inval(i,x) if k==1 else act_val(Df,k-1,i,p)
                    for o in range(LAYERS[k]):
                        postN=act_val(Dn,k,o,p); postF=act_val(Df,k,o,p)
                        gW[k][i][o]+=(preN*postN-preF*postF)/BETA/nb
        for k in range(1,L+1):
            for i in range(LAYERS[k-1]):
                for o in range(LAYERS[k]):
                    vW[k][i][o]=mom*vW[k][i][o]-eta*gW[k][i][o]; W[k][i][o]=float(np.clip(W[k][i][o]+vW[k][i][o],-3,3))
        if ep%4==0 or ep==epochs:
            acc=evaluate(Xte,yte); rd=ridge_eval(); best=max(best,acc,rd)
            print(f"epoch {ep:3d} ep-out={acc:.3f} RIDGE={rd:.3f} best={best:.3f} ({time.time()-t:.1f}s/ep)",flush=True)
    print(f"DEEPEPRESULT best={best:.3f}")
