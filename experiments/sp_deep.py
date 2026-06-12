#!/usr/bin/env python3
"""Deep SPARSE in-SPICE net (Spectre, batched). Per user guidance: keep every neuron's fan-in AND fan-out
small (<=~8) for analog-sum precision; prefer MORE LAYERS + sparse connectivity over one wide layer.
The old 2-layer net had OUTPUT fan-in = NF (~96) -> imprecise readout (the ~0.82 rings ceiling).

Topology (configurable via LAYERS env, default for 2D): inputs(2) -> several sparse hidden layers (frozen
random, NEUK neuron) -> small last layer -> output (fan-in = last-layer size, small). Each hidden neuron reads
FANIN random units of the previous layer. Frozen hidden + TRAINED output (controller delta, no feedback loop
-> stable .op). All LEVEL=1 transistors, 0 behavioral. Reuses ep_digits cells (dsyn, dneuron via NEUK)."""
import os, subprocess, numpy as np, glob
import ep_digits as E

SPECTRE="/opt/cadence/installs/SPECTRE231/bin/spectre"
SIM=os.environ.get("SIM","spectre")          # 'spectre' or 'ngspice' (ngspice bypasses the shared Spectre license)
NGSPICE=os.environ.get("NGSPICE","ngspice")
IND=E.IND; TD=E.TD; RB=E.RB; C=E.C
LAYERS=[int(x) for x in os.environ.get("LAYERS","2,48,48,12").split(",")]  # sizes incl input(2); output C appended
FANIN=int(os.environ.get("FANIN","6"))          # max fan-in per hidden/last neuron (precision guard)
OFANIN=int(os.environ.get("OFANIN","8"))         # output fan-in (reads last layer)
WSCALE=float(os.environ.get("WSCALE","0.6"))     # random weight std
NPROC=int(os.environ.get("NPROC","6")); MThr=os.environ.get("MThr","4")
SPTIMEOUT=int(os.environ.get("SPTIMEOUT","90"))
Wh=f"{E.WPER*(FANIN+1):.0f}u"; Wo=f"{E.WPER*(OFANIN+1):.0f}u"

def build_topology(seed):
    """Return layers: list of layers; each layer = list of neurons; neuron = list of (src_layer,src_idx) inputs,
    plus random weights. Layer 0 = inputs. Output layer = C neurons reading the last hidden layer."""
    rng=np.random.default_rng(seed)
    sizes=LAYERS[:]                       # [2, h1, h2, ...]
    nlayers=len(sizes)
    conns=[None]                          # conns[k] = list per neuron of [(srcidx, w_signed)...] reading layer k-1
    # fan-out tracking to keep it bounded too
    FANIN1=int(os.environ.get("FANIN1",str(FANIN)))   # first hidden layer fan-in (=1 -> clean axis x²/y²)
    for k in range(1,nlayers):
        prev=sizes[k-1]; fo=np.zeros(prev,dtype=int)
        layer=[]
        fin=min(FANIN1 if k==1 else FANIN, prev)
        for j in range(sizes[k]):
            # pick fin sources preferring low current fan-out
            order=np.argsort(fo+rng.random(prev)*0.01)
            srcs=sorted(order[:fin].tolist())
            for s in srcs: fo[s]+=1
            ws=[float(WSCALE*rng.standard_normal()) for _ in srcs]
            layer.append(list(zip(srcs,ws)))
        conns.append(layer)
    # output layer: C neurons reading last hidden layer
    last=sizes[-1]; fin=min(OFANIN,last)
    outl=[]
    for c in range(C):
        srcs=list(range(last)) if last<=OFANIN else sorted(rng.choice(last,fin,replace=False).tolist())
        ws=[float(WSCALE*rng.standard_normal()) for _ in srcs]
        outl.append(list(zip(srcs,ws)))
    return sizes, conns, outl

def wval(w): return E.wval(w)

NEUKS=[int(x) for x in os.environ.get("NEUKS","3,2,2").split(",")]   # per-hidden-layer neuron type
def _cell(name, neuk):
    rln=E.RLNEU; rln1=os.environ.get("RLN1",E.RLNEU); sqw=E.SQW; srw=E.SQREFW
    if neuk==3:   # radial: amplify then square
        body=f"""M1 a1n up t1 0 NNR W=200u L=100u
M2 a1p un t1 0 NNR W=200u L=100u
Mt1 t1 vbn 0 0 NNR W=200u L=100u
RL1a vdd a1p {rln1}
RL1b vdd a1n {rln1}
Msqp ap a1p a1n 0 NNR W={sqw} L=100u
Msqn ap a1n a1p 0 NNR W={sqw} L=100u
RLa vdd ap {rln}
Mref an vsq 0 0 NNR W={srw} L=100u
RLb vdd an {rln}"""
    else:         # sigmoid: amplify then saturate
        body=f"""M1 a1n up t1 0 NNR W=200u L=100u
M2 a1p un t1 0 NNR W=200u L=100u
Mt1 t1 vbn 0 0 NNR W=200u L=100u
RL1a vdd a1p {rln1}
RL1b vdd a1n {rln1}
M3 an a1p t2 0 NNR W=200u L=100u
M4 ap a1n t2 0 NNR W=200u L=100u
Mt2 t2 vbn 0 0 NNR W=200u L=100u
RLa vdd ap {rln}
RLb vdd an {rln}"""
    return f".subckt {name} up un ap an vdd vbn\n{body}\n.ends\n"
def cell_for_layer(k): return "dneuronR" if NEUKS[min(k-1,len(NEUKS)-1)]==3 else "dneuronS"
def neuron_lines(prefix, up, un, ap, an, cell="dneuronR"):
    return [f"Mlp_{prefix} {up} {up} vdd vdd PNR W={Wh} L=100u",
            f"Mln_{prefix} {un} {un} vdd vdd PNR W={Wh} L=100u",
            f"Xn_{prefix} {up} {un} {ap} {an} vdd vbn {cell}"]

def deck(sizes, conns, outl, owts, X, p, nudge_lab=None, saves=None):
    """One pattern instance p. owts = current output weights (trained). Hidden weights are in conns (frozen)."""
    L=[]
    # input nodes for this pattern
    for i in range(2):
        L+=[f"Vxp{i}_{p} a0_{i}p_{p} 0 {0.5+IND*(X[i]-0.5):.4f}",
            f"Vxn{i}_{p} a0_{i}n_{p} 0 {0.5-IND*(X[i]-0.5):.4f}"]
    # activation node names: layer 0 = inputs (ap=a0_ip, an=a0_in)
    def act(k,idx): return (f"a{k}_{idx}p_{p}", f"a{k}_{idx}n_{p}") if k>0 else (f"a0_{idx}p_{p}", f"a0_{idx}n_{p}")
    # hidden layers (frozen weights from conns)
    for k in range(1,len(sizes)):
        for j in range(sizes[k]):
            up,un=f"u{k}_{j}_{p}", f"v{k}_{j}_{p}"
            ap,an=f"a{k}_{j}p_{p}", f"a{k}_{j}n_{p}"
            L+=neuron_lines(f"{k}_{j}_{p}",up,un,ap,an,cell_for_layer(k))
            for (s,wv) in conns[k][j]:
                sp,sn=act(k-1,s); gp,gn=wval(wv)
                L+=[f"Vwp{k}_{j}_{s}_{p} wp{k}_{j}_{s}_{p} 0 {gp:.4f}",f"Vwn{k}_{j}_{s}_{p} wn{k}_{j}_{s}_{p} 0 {gn:.4f}",
                    f"X{k}_{j}_{s}_{p} {sp} {sn} {up} {un} wp{k}_{j}_{s}_{p} wn{k}_{j}_{s}_{p} vdd dsyn"]
            if saves is not None: saves.add(ap); saves.add(an)
    # output layer (TRAINED weights owts[c] aligned to outl[c] sources, reading last hidden layer)
    lastk=len(sizes)-1
    for c in range(C):
        up,un=f"uo_{c}_{p}", f"vo_{c}_{p}"
        L+=[f"Mlpo_{c}_{p} {up} {up} vdd vdd PNR W={Wo} L=100u",f"Mlno_{c}_{p} {un} {un} vdd vdd PNR W={Wo} L=100u"]
        for (s,_),wv in zip(outl[c], owts[c]):
            sp,sn=act(lastk,s); gp,gn=wval(wv)
            L+=[f"Vwop_{c}_{s}_{p} wop_{c}_{s}_{p} 0 {gp:.4f}",f"Vwon_{c}_{s}_{p} won_{c}_{s}_{p} 0 {gn:.4f}",
                f"Xo_{c}_{s}_{p} {sp} {sn} {up} {un} wop_{c}_{s}_{p} won_{c}_{s}_{p} vdd dsyn"]
        # bias + nudge clamp
        L+=[f"Vtp{c}_{p} vtp{c}_{p} 0 {0.5+((TD*(1 if c==nudge_lab else -1)) if nudge_lab is not None else 0):.4f}",
            f"Vtn{c}_{p} vtn{c}_{p} 0 {0.5-((TD*(1 if c==nudge_lab else -1)) if nudge_lab is not None else 0):.4f}"]
        rb=RB if nudge_lab is not None else 1e10
        L+=[f"Rbp{c}_{p} {up} vtp{c}_{p} {rb}",f"Rbn{c}_{p} {un} vtn{c}_{p} {rb}"]
        if saves is not None: saves.add(up); saves.add(un)
    return L

def header():
    return ["// deep sparse spectre","simulator lang=spice",
            ".model NNR NMOS (LEVEL=1 VTO=0.2  KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)",
            ".model PNR PMOS (LEVEL=1 VTO=-0.2 KP=40u  LAMBDA=0.02 GAMMA=0 PHI=0.7)",
            E.SUB.split(".ends\n",1)[0]+".ends",  # dsyn only... actually need both; use full SUB
            ]

_DSYN=E.SUB.split(".ends\n",1)[0]+".ends\n"   # just the dsyn subckt
def full_header():
    h0=(["// deep sparse spectre","simulator lang=spice"] if SIM=="spectre" else ["* deep sparse ngspice",".global vdd vsq vbn bp_hi bp_lo"])
    return h0+[
            ".model NNR NMOS (LEVEL=1 VTO=0.2  KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)",
            ".model PNR PMOS (LEVEL=1 VTO=-0.2 KP=40u  LAMBDA=0.02 GAMMA=0 PHI=0.7)",
            _DSYN, _cell("dneuronR",3), _cell("dneuronS",2),
            "Vdd vdd 0 1.0","Vbn vbn 0 "+E.VBN,"Vbp_hi bp_hi 0 0.8","Vbp_lo bp_lo 0 0.2","Vsq vsq 0 "+E.VSQ]

def parse(path):
    return E_parse(path)
def E_parse(path):
    import sp_batch as SB
    d=SB.parse_nutascii(path)
    if SIM=="ngspice":   # ngspice raw names are 'v(node)' -> strip to bare 'node'
        d={(k[2:-1] if k.startswith("v(") and k.endswith(")") else k):v for k,v in d.items()}
    return d

def run_group(sizes,conns,outl,owts,items,nudge,tag):
    saves=set(); L=full_header()
    for (p,x,lab) in items:
        L+=deck(sizes,conns,outl,owts,x,p,(lab if nudge else None),saves)
    L.append(".options reltol="+os.environ.get("RELTOL","2e-3")+" gmin=1e-9")
    raw=f"/tmp/spd_{tag}.raw"
    try: os.remove(raw)
    except OSError: pass
    if SIM=="ngspice":
        L+=[".control","set filetype=ascii","op",f"write {raw}","quit",".endc",".end"]
        f=f"/tmp/spd_{tag}.cir"; open(f,"w").write("\n".join(L)+"\n")
        return subprocess.Popen([NGSPICE,"-b",f],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL), raw
    L.append(".save "+" ".join("v("+s+")" for s in sorted(saves)))
    L+=[".op",".end"]
    f=f"/tmp/spd_{tag}.scs"; open(f,"w").write("\n".join(L)+"\n")
    return subprocess.Popen([SPECTRE,f,f"+mt={MThr}","-format","nutascii","-raw",raw],
                            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL), raw

def run_batch(sizes,conns,outl,owts,X,labels,nudge,tag,nproc=None):
    items=[(p,X[p],(labels[p] if labels is not None else None)) for p in range(len(X))]
    ng=min(nproc or NPROC,max(1,len(items))); groups=[items[i::ng] for i in range(ng)]
    procs=[run_group(sizes,conns,outl,owts,g,nudge,f"{tag}_{gi}") for gi,g in enumerate(groups) if g]
    d={}
    for pr,raw in procs:
        try: pr.wait(timeout=SPTIMEOUT)
        except Exception: pr.kill(); continue
        op=raw if os.path.isfile(raw) else (glob.glob(raw+"/*")+[None])[0]
        if op:
            try: d.update(E_parse(op))
            except Exception: pass
    # completeness: every pattern must have its output node, else the gradient/eval is corrupt -> signal caller
    if any(f"uo_0_{p}" not in d for p in range(len(X))): return None
    return d

def run_safe(sizes,conns,outl,owts,X,labels,nudge,tag):
    d=run_batch(sizes,conns,outl,owts,X,labels,nudge,tag)
    if d is None: d=run_batch(sizes,conns,outl,owts,X,labels,nudge,tag+"s",nproc=1)  # serial retry fits license
    return d

def douts(d,p): return np.array([d.get(f"uo_{c}_{p}",0.5)-d.get(f"vo_{c}_{p}",0.5) for c in range(C)])

def lastact(d,sizes,p,s):
    k=len(sizes)-1
    return d.get(f"a{k}_{s}p_{p}",0.5)-d.get(f"a{k}_{s}n_{p}",0.5)

if __name__=="__main__":
    import sys,time
    LF="/tmp/sp_deep.lock"   # single-instance lock: respawn/dups exit (else they contend on the license)
    try:
        fd=os.open(LF,os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.write(fd,str(os.getpid()).encode()); os.close(fd)
    except FileExistsError:
        try: age=time.time()-os.path.getmtime(LF)
        except Exception: age=0
        if age<1800: print("# sp_deep locked, exiting"); sys.exit(0)
        os.remove(LF); os.close(os.open(LF,os.O_CREAT|os.O_WRONLY))
    import atexit
    atexit.register(lambda: (os.remove(LF) if os.path.exists(LF) else None))
    mode=sys.argv[1] if len(sys.argv)>1 else "train"
    (Xtr,ytr),(Xte,yte)=E.make_data()
    seed=int(os.environ.get("SEED","0"))
    sizes,conns,outl=build_topology(seed)
    owts=[[float(0.4*np.random.default_rng(seed+100+c).standard_normal()) for _ in outl[c]] for c in range(C)]
    print(f"# DEEP sizes={sizes}+[{C}] FANIN={FANIN} OFANIN={OFANIN} hidden_neurons={sum(sizes[1:])}",flush=True)
    def evaluate(X,y):
        d=run_safe(sizes,conns,outl,owts,X,None,False,"ev")
        if not d: return 0.0
        return float(np.mean([int(np.argmax(douts(d,p)))==y[p] for p in range(len(X))]))
    eta=float(os.environ.get("ETA","0.02")); epochs=int(os.environ.get("EPOCHS","40")); mom=float(os.environ.get("MOM","0.5"))
    BETA=1.0/RB; vel=[[0.0]*len(outl[c]) for c in range(C)]
    a0=evaluate(Xte,yte); best=a0
    print(f"# epoch0 test={a0:.3f}",flush=True)
    for ep in range(1,epochs+1):
        t=time.time()
        Df=run_safe(sizes,conns,outl,owts,Xtr,None,False,"fr")
        Dn=run_safe(sizes,conns,outl,owts,Xtr,list(ytr),True,"nu")
        if not Df or not Dn: print(f"epoch {ep}: fail",flush=True); continue
        # output-weight gradient via EP on the (frozen) last hidden activations
        for c in range(C):
            for si,(s,_) in enumerate(outl[c]):
                g=0.0
                for p in range(len(Xtr)):
                    hN=lastact(Dn,sizes,p,s); hF=lastact(Df,sizes,p,s)
                    oN=Dn.get(f"uo_{c}_{p}",.5)-Dn.get(f"vo_{c}_{p}",.5); oF=Df.get(f"uo_{c}_{p}",.5)-Df.get(f"vo_{c}_{p}",.5)
                    g+=(hN*oN-hF*oF)/BETA/len(Xtr)
                vel[c][si]=mom*vel[c][si]-eta*g
                owts[c][si]=float(np.clip(owts[c][si]+vel[c][si],-3,3))
        if ep%2==0 or ep==epochs:
            acc=evaluate(Xte,yte); best=max(best,acc)
            print(f"epoch {ep:3d} test={acc:.3f} best={best:.3f} ({time.time()-t:.1f}s/ep)",flush=True)
    print(f"SPDRESULT best={best:.3f}")
