#!/usr/bin/env python3
"""FULL 2-layer fully-analog continuous Predictive Coding, ONE ngspice .tran -> trains XOR.
Weights are charge on caps, SHARED across the synapses that use them (forward prediction +
backward error feedback = the transpose, physically).  Each weight cap integrates its LOCAL
product continuously:  W_ih cap <- eps_h . x_in ;  W_ho cap <- eps_o . a_h .  Hard-clamped output.
No two-phase, no controller optimizer; controller only writes the PWL (inputs + label) schedule.
Builds on the proven node recipe (gen_pc forward) + the gprod learning cell (gen_pc1).
"""
import os, numpy as np
H=int(os.environ.get("H","4")); SEED=int(os.environ.get("SEED","0"))
IND=float(os.environ.get("IND","0.30")); TD=float(os.environ.get("TD","0.20"))
KMAP=float(os.environ.get("KMAP","0.30")); VW0=float(os.environ.get("VW0","0.5"))
WL=os.environ.get("WL","1000u"); RCS=os.environ.get("RCS","300k"); RNODE=os.environ.get("RNODE","20k")
VBSYN=os.environ.get("VBSYN","0.45"); VBNEU=os.environ.get("VBNEU","0.35"); GMT=os.environ.get("GMT","0.6")
GBLH=os.environ.get("GBLH","0.85"); GBLO=os.environ.get("GBLO","0.85")   # learning rates (gprod tails)
CWW=os.environ.get("CWW","30p"); RWL=os.environ.get("RWL","2g")
SGNH=float(os.environ.get("SGNH","1")); SGNO=float(os.environ.get("SGNO","1"))
TH=float(os.environ.get("TH","250")); NEP=int(os.environ.get("NEP","160"))
TAG=os.environ.get("RUNTAG",str(os.getpid()))
PATS=[(0,0,0),(0,1,1),(1,0,1),(1,1,0)]
def ts(b): return 1.0 if b else -1.0
rng=np.random.default_rng(SEED)
WK=[]                          # weight keys
for j in range(H):
    for i in (1,2): WK.append(f"wih_{i}_{j}")
    WK.append(f"who_{j}")
w0={k:float(0.5*rng.standard_normal()) for k in WK}
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
def syn(tag,ip,inn,op,on,key):   # synapse reads the SHARED weight caps wp_<key>,wn_<key>
    return [f"X_{tag} {ip} {inn} wp_{key} wn_{key} {op} {on} vdd vbsyn gsyn"]
def gen():
    NP=4*NEP; TEND=NP*TH
    x1=[];x2=[];vp=[];vn=[]
    for g in range(NP):
        p=PATS[g%4]; t=g*TH
        x1.append((t,0.5+IND*ts(p[0]))); x2.append((t,0.5+IND*ts(p[1])))
        td=TD*ts(p[2]); vp.append((t,0.5+td)); vn.append((t,0.5-td))
    L=["full continuous 2-layer PC",SUB,"Vdd vdd 0 1.0",
       f"Vbsyn vbsyn 0 {VBSYN}",f"Vbneu vbneu 0 {VBNEU}",f"Vgm gm 0 {GMT}",
       f"Vgblh gblh 0 {GBLH}",f"Vgblo gblo 0 {GBLO}",f"Vwcm wcm 0 {VW0}",
       f"Vx1p a_xp1 0 {stepped(x1)}",f"Vx1n a_xn1 0 {stepped([(t,1-v) for t,v in x1])}",
       f"Vx2p a_xp2 0 {stepped(x2)}",f"Vx2n a_xn2 0 {stepped([(t,1-v) for t,v in x2])}",
       f"Vxop xop 0 {stepped(vp)}",f"Vxon xon 0 {stepped(vn)}"]   # HARD-CLAMP output to label
    ic=[]
    # weight caps (shared), CM held weakly to VW0
    for k in WK:
        gp,gn=wgv(w0[k])
        L+=[f"Cwp_{k} wp_{k} 0 {CWW}",f"Cwn_{k} wn_{k} 0 {CWW}",
            f"Rwp_{k} wp_{k} wcm {RWL}",f"Rwn_{k} wn_{k} wcm {RWL}"]
        ic.append(f".ic v(wp_{k})={gp:.4f} v(wn_{k})={gn:.4f}")
    # hidden layer
    for j in range(H):
        xp,xn=f"xhp{j}",f"xhn{j}"; mp,mn=f"mhp{j}",f"mhn{j}"
        L+=[f"Xcm_mh{j} {mp} {mn} vdd cmld",f"Xcm_xh{j} {xp} {xn} vdd cmld",
            f"Rnd_mh{j} {mp} {mn} {RNODE}",f"Rnd_xh{j} {xp} {xn} {RNODE}",
            f"Cxhp{j} {xp} 0 0.4p",f"Cxhn{j} {xn} 0 0.4p"]
        for i in (1,2):
            L+=syn(f"ihm_{i}_{j}",f"a_xp{i}",f"a_xn{i}",mp,mn,f"wih_{i}_{j}")
            L+=syn(f"ihx_{i}_{j}",f"a_xp{i}",f"a_xn{i}",xp,xn,f"wih_{i}_{j}")
        L+=syn(f"bk_{j}","eop","eon",xp,xn,f"who_{j}")
        L+=[f"Xn{j} {xp} {xn} ahp{j} ahn{j} vdd vbneu dneuron"]
        L+=[f"Xeh{j} {xp} {xn} {mp} {mn} ehp{j} ehn{j} vdd gm esub"]
    # output
    L+=["Xcm_mo mop mon vdd cmld",f"Rnd_mo mop mon {RNODE}"]
    for j in range(H): L+=syn(f"ho_{j}",f"ahp{j}",f"ahn{j}","mop","mon",f"who_{j}")
    L+=["Xeo xop xon mop mon eop eon vdd gm esub"]
    # LEARNING (continuous, local): W_ih <- eps_h . x_in ;  W_ho <- eps_o . a_h
    eh_p,eh_n=("ehp","ehn") if SGNH>0 else ("ehn","ehp")
    eo_p,eo_n=("eop","eon") if SGNO>0 else ("eon","eop")
    for j in range(H):
        for i in (1,2):
            L+=[f"Xlih_{i}_{j} {eh_p}{j} {eh_n}{j} a_xp{i} a_xn{i} wp_wih_{i}_{j} wn_wih_{i}_{j} vdd gblh gprod"]
        L+=[f"Xlho_{j} {eo_p} {eo_n} ahp{j} ahn{j} wp_who_{j} wn_who_{j} vdd gblo gprod"]
    L+=ic
    L+=[".options cshunt=1e-14 gmin=1e-10 reltol=2e-3",   # aid .tran convergence (timestep-too-small)
        f".tran 2n {int(TEND)}n uic",".control","run",
        f"wrdata pcL{TAG}.dat v(mop) v(mon) v(xop)",".endc",".end"]
    open(f"pcL_{TAG}.cir","w").write("\n".join(L)+"\n")
    return TEND
if __name__=="__main__":
    import subprocess
    TEND=gen()
    subprocess.run(["ngspice","-b",f"pcL_{TAG}.cir"],capture_output=True,text=True,timeout=400)
    d=np.loadtxt(f"pcL{TAG}.dat"); t=d[:,0]*1e9; mo=d[:,1]-d[:,3]
    print(f"2-layer continuous PC XOR, seed={SEED} SGNH={SGNH} SGNO={SGNO}")
    def acc(ep):
        ok=0;outs=[]
        for m in range(4):
            g=ep*4+m; i=np.argmin(np.abs(t-(g*TH+TH-2))); outs.append(mo[i]*1e3)
            ok+=((mo[i]>0)==(ts(PATS[m][2])>0))
        return ok,outs
    for ep in [0,20,50,100,NEP-1]:
        a,o=acc(ep); print(f"  epoch {ep:3d}: {a}/4  mu_o(mV)={np.round(o,1)}")
