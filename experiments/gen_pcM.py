#!/usr/bin/env python3
"""Multi-class fully-analog continuous Predictive Coding on real digit images (sklearn digits,
4x4 z-scored), ONE ngspice .tran.  Generalizes gen_pcL: N inputs, C classes, H hidden.
Weights = charge on caps, updated only by local products (W_ih<-eps_h.x_in ; W_ho<-eps_o.a_h).
Convergence/momentum: an explicit small cap on each learning-cell tail node (CTAIL) -- portable
(no cshunt flag) and acts as per-weight gradient low-pass = momentum.
Output hard-clamped to one-vs-rest target; prediction = argmax_c mu_o_c (mu_o is the free synapse
sum, separate from the clamp node, so it's readable during training)."""
import os, numpy as np
N=16
C=int(os.environ.get("C","2")); H=int(os.environ.get("H","8")); SEED=int(os.environ.get("SEED","0"))
NTR=int(os.environ.get("NTR","10"))          # train examples per class
IND=float(os.environ.get("IND","0.30")); TD=float(os.environ.get("TD","0.20"))
KMAP=float(os.environ.get("KMAP","0.30")); VW0=float(os.environ.get("VW0","0.5"))
WL=os.environ.get("WL","1000u"); RCS=os.environ.get("RCS","300k"); RNODE=os.environ.get("RNODE","20k")
VBSYN=os.environ.get("VBSYN","0.45"); VBNEU=os.environ.get("VBNEU","0.35"); GMT=os.environ.get("GMT","0.6")
GBLH=os.environ.get("GBLH","0.85"); GBLO=os.environ.get("GBLO","0.85")
CWW=os.environ.get("CWW","30p"); RWL=os.environ.get("RWL","2g"); CTAIL=os.environ.get("CTAIL","15f")
SGNH=float(os.environ.get("SGNH","1")); SGNO=float(os.environ.get("SGNO","-1"))
TH=float(os.environ.get("TH","250")); NEP=int(os.environ.get("NEP","60"))
TAG=os.environ.get("RUNTAG",str(os.getpid()))
# ---- data ----
Xz=np.load("dig_Xz.npy"); yall=np.load("dig_y.npy")
rng=np.random.default_rng(SEED); CLS=list(range(C))
NTE=int(os.environ.get("NTE","20"))          # held-out test examples per class
Xtr=[];ytr=[];Xte=[];yte=[]
for ci,c in enumerate(CLS):
    idx=np.where(yall==c)[0]; rng.shuffle(idx)
    for i in idx[:NTR]: Xtr.append(np.clip(Xz[i],-3,3)/3.0); ytr.append(ci)   # -> [-1,1]
    for i in idx[NTR:NTR+NTE]: Xte.append(np.clip(Xz[i],-3,3)/3.0); yte.append(ci)
Xtr=np.array(Xtr); ytr=np.array(ytr); Xte=np.array(Xte); yte=np.array(yte)
order=rng.permutation(len(Xtr)); Xtr=Xtr[order]; ytr=ytr[order]
WK=[]
for j in range(H):
    for i in range(N): WK.append(f"wih_{i}_{j}")
    for c in range(C): WK.append(f"who_{j}_{c}")
w0={k:float(0.4*rng.standard_normal()) for k in WK}
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
.ends
.subckt gprod xp xn yp yn wp wn vdd vbn
Mtail nt vbn 0 0 NNR W=400u L=100u
Ctail nt 0 {CTAIL}
M5 na yp nt 0 NNR W=200u L=100u
M6 nb yn nt 0 NNR W=200u L=100u
M1 iop xp na 0 NNR W=200u L=100u
M2 ion xn na 0 NNR W=200u L=100u
M3 ion xp nb 0 NNR W=200u L=100u
M4 iop xn nb 0 NNR W=200u L=100u
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
.ends"""
def stepped(vals):
    pts=[]
    for k,(t,v) in enumerate(vals):
        if k>0: pts.append((t-0.1,vals[k-1][1]))
        pts.append((t,v))
    return "PWL("+" ".join(f"{t:.1f}n {v:.4f}" for t,v in pts)+")"
def syn(tag,ip,inn,op,on,key): return [f"X_{tag} {ip} {inn} wp_{key} wn_{key} {op} {on} vdd vbsyn gsyn"]
def gen():
    NEX=len(Xtr); NP=NEX*NEP; TEND=NP*TH
    # input + target schedules
    xin=[[] for _ in range(N)]; vtp=[[] for _ in range(C)]; vtn=[[] for _ in range(C)]
    seq=[]
    for ep in range(NEP):
        for e in range(NEX):
            g=ep*NEX+e; t=g*TH; seq.append((g,ytr[e]))
            for i in range(N): xin[i].append((t,0.5+IND*float(Xtr[e][i])))
            for c in range(C):
                td=TD*(1 if c==ytr[e] else -1); vtp[c].append((t,0.5+td)); vtn[c].append((t,0.5-td))
    L=["multi-class continuous PC digits",SUB,"Vdd vdd 0 1.0",
       f"Vbsyn vbsyn 0 {VBSYN}",f"Vbneu vbneu 0 {VBNEU}",f"Vgm gm 0 {GMT}",
       f"Vgblh gblh 0 {GBLH}",f"Vgblo gblo 0 {GBLO}",f"Vwcm wcm 0 {VW0}"]
    for i in range(N):
        L+=[f"Vxp{i} a_xp{i} 0 {stepped(xin[i])}",f"Vxn{i} a_xn{i} 0 {stepped([(t,1-v) for t,v in xin[i]])}"]
    for c in range(C):
        L+=[f"Vtp{c} xop{c} 0 {stepped(vtp[c])}",f"Vtn{c} xon{c} 0 {stepped(vtn[c])}"]
    ic=[]
    for k in WK:
        gp,gn=wgv(w0[k])
        L+=[f"Cwp_{k} wp_{k} 0 {CWW}",f"Cwn_{k} wn_{k} 0 {CWW}",
            f"Rwp_{k} wp_{k} wcm {RWL}",f"Rwn_{k} wn_{k} wcm {RWL}"]
        ic.append(f".ic v(wp_{k})={gp:.4f} v(wn_{k})={gn:.4f}")
    # hidden
    for j in range(H):
        xp,xn=f"xhp{j}",f"xhn{j}"; mp,mn=f"mhp{j}",f"mhn{j}"
        L+=[f"Xcm_mh{j} {mp} {mn} vdd cmld",f"Xcm_xh{j} {xp} {xn} vdd cmld",
            f"Rnd_mh{j} {mp} {mn} {RNODE}",f"Rnd_xh{j} {xp} {xn} {RNODE}",
            f"Cxhp{j} {xp} 0 0.4p",f"Cxhn{j} {xn} 0 0.4p"]
        for i in range(N):
            L+=syn(f"ihm_{i}_{j}",f"a_xp{i}",f"a_xn{i}",mp,mn,f"wih_{i}_{j}")
            L+=syn(f"ihx_{i}_{j}",f"a_xp{i}",f"a_xn{i}",xp,xn,f"wih_{i}_{j}")
        for c in range(C):
            L+=syn(f"bk_{j}_{c}",f"eop{c}",f"eon{c}",xp,xn,f"who_{j}_{c}")     # backward sum over classes
        L+=[f"Xn{j} {xp} {xn} ahp{j} ahn{j} vdd vbneu dneuron"]
        L+=[f"Xeh{j} {xp} {xn} {mp} {mn} ehp{j} ehn{j} vdd gm esub"]
    # output per class
    for c in range(C):
        L+=[f"Xcm_mo{c} mop{c} mon{c} vdd cmld",f"Rnd_mo{c} mop{c} mon{c} {RNODE}"]
        for j in range(H): L+=syn(f"ho_{j}_{c}",f"ahp{j}",f"ahn{j}",f"mop{c}",f"mon{c}",f"who_{j}_{c}")
        L+=[f"Xeo{c} xop{c} xon{c} mop{c} mon{c} eop{c} eon{c} vdd gm esub"]
    # learning
    eh_p,eh_n=("ehp","ehn") if SGNH>0 else ("ehn","ehp")
    eo_p,eo_n=("eop","eon") if SGNO>0 else ("eon","eop")
    for j in range(H):
        for i in range(N):
            L+=[f"Xlih_{i}_{j} {eh_p}{j} {eh_n}{j} a_xp{i} a_xn{i} wp_wih_{i}_{j} wn_wih_{i}_{j} vdd gblh gprod"]
        for c in range(C):
            L+=[f"Xlho_{j}_{c} {eo_p}{c} {eo_n}{c} ahp{j} ahn{j} wp_who_{j}_{c} wn_who_{j}_{c} vdd gblo gprod"]
    L+=ic
    cols=" ".join(sum([[f"v(mop{c})",f"v(mon{c})"] for c in range(C)],[])
                  +sum([[f"v(wp_{k})",f"v(wn_{k})"] for k in WK],[]))   # also log weights (read final row)
    L+=[".options gmin=1e-10 reltol=2e-3",f".tran 2n {int(TEND)}n uic",".control","run",
        f"wrdata pcM{TAG}.dat {cols}",".endc",".end"]
    open(f"pcM_{TAG}.cir","w").write("\n".join(L)+"\n")
    return TEND,NEX,seq
def fwd_eval(wvals):
    """forward-only deck with FIXED trained weights -> test accuracy (no clamp/error/learning)."""
    L=["pc forward eval",SUB,"Vdd vdd 0 1.0",f"Vbsyn vbsyn 0 {VBSYN}",f"Vbneu vbneu 0 {VBNEU}"]
    for i in range(N): L+=[f"Vxp{i} a_xp{i} 0 0.5",f"Vxn{i} a_xn{i} 0 0.5"]
    for k in WK: L+=[f"Vwp_{k} wp_{k} 0 {wvals[k][0]:.4f}",f"Vwn_{k} wn_{k} 0 {wvals[k][1]:.4f}"]
    for j in range(H):
        xp,xn=f"xhp{j}",f"xhn{j}"
        L+=[f"Xcm_xh{j} {xp} {xn} vdd cmld",f"Rnd_xh{j} {xp} {xn} {RNODE}"]
        for i in range(N): L+=syn(f"ihx_{i}_{j}",f"a_xp{i}",f"a_xn{i}",xp,xn,f"wih_{i}_{j}")
        L+=[f"Xn{j} {xp} {xn} ahp{j} ahn{j} vdd vbneu dneuron"]
    for c in range(C):
        L+=[f"Xcm_mo{c} mop{c} mon{c} vdd cmld",f"Rnd_mo{c} mop{c} mon{c} {RNODE}"]
        for j in range(H): L+=syn(f"ho_{j}_{c}",f"ahp{j}",f"ahn{j}",f"mop{c}",f"mon{c}",f"who_{j}_{c}")
    cols=" ".join(sum([[f"v(mop{c})",f"v(mon{c})"] for c in range(C)],[]))
    ctl=[".control"]
    for e in range(len(Xte)):
        for i in range(N):
            ctl+=[f"alter Vxp{i} = {0.5+IND*float(Xte[e][i]):.4f}",f"alter Vxn{i} = {0.5-IND*float(Xte[e][i]):.4f}"]
        ctl+=["op",f"wrdata pcMev{TAG}_{e}.dat {cols}"]
    ctl+=[".endc"]
    L+=[".options gmin=1e-10"]+ctl+[".end"]
    open(f"pcMev_{TAG}.cir","w").write("\n".join(L)+"\n")
    import subprocess; subprocess.run(["ngspice","-b",f"pcMev_{TAG}.cir"],capture_output=True,text=True,timeout=300)
    ok=0
    for e in range(len(Xte)):
        v=np.loadtxt(f"pcMev{TAG}_{e}.dat").reshape(-1)[1::2]
        pred=int(np.argmin([v[2*c]-v[2*c+1] for c in range(C)]))   # synapse inverts -> predicted class has LOWEST mu_o
        ok+=(pred==yte[e])
    return ok/len(Xte)
if __name__=="__main__":
    import subprocess
    TEND,NEX,seq=gen()
    print(f"multi-class PC digits: C={C} H={H} N={N} train={len(Xtr)} weights={len(WK)} sim={TEND/1000:.0f}us")
    subprocess.run(["ngspice","-b",f"pcM_{TAG}.cir"],capture_output=True,text=True,timeout=900)
    d=np.loadtxt(f"pcM{TAG}.dat"); t=d[:,0]*1e9
    mo=np.array([d[:,1+2*c]-d[:,2+2*c] for c in range(C)])   # mu_o per class over time
    def acc(ep):
        ok=0
        for e in range(NEX):
            g=ep*NEX+e; i=np.argmin(np.abs(t-(g*TH+TH-2)))
            pred=int(np.argmax([mo[c][i] for c in range(C)]))
            ok+=(pred==ytr[e])
        return ok/NEX
    for ep in [0,5,15,30,NEP-1]:
        print(f"  epoch {ep:3d}: train acc {acc(ep):.2f}  (chance {1.0/C:.2f})")
    # read FINAL trained weights (last row) and evaluate on HELD-OUT test set
    last=d[-1]; wvals={}
    for ki,k in enumerate(WK):
        wp=last[2*(2*C+2*ki)+1]; wn=last[2*(2*C+2*ki+1)+1]; wvals[k]=(wp,wn)
    te=fwd_eval(wvals)
    print(f"  HELD-OUT test acc: {te:.2f}  ({len(Xte)} examples, chance {1.0/C:.2f})")
