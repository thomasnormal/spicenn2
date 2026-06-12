#!/usr/bin/env python3
"""SINGLE-LAYER MULTI-CLASS fully-analog continuous Predictive Coding, in ONE ngspice .tran.
The PURE-THESIS version of pc1_orch: NO controller arithmetic at all.  Weights = charge on caps
(wp_ci,wn_ci); each cap integrates its LOCAL product eps_c . x_i continuously via a gprod cell
(dW/dt ~ eps.presyn); output hard-clamped to one-vs-rest +-TD label; eps_c = label_c - mu_c is an
explicit esub error node.  Controller ONLY writes the PWL schedule (inputs + labels) cycling K train
examples for NEP epochs, then a FROZEN-WEIGHT (gbl->0) test pass.  8x8 digits => linearly separable
=> a zero-error fixed point EXISTS per class (unlike 4x4 gen_pcM which had none and decayed).
Cells identical to the proven gen_pc1 (OR/AND 5/5)."""
import os, numpy as np, subprocess
from sklearn.datasets import load_digits
C=int(os.environ.get("C","5")); SEED=int(os.environ.get("SEED","0")); NIN=int(os.environ.get("NIN","64"))
KTR=int(os.environ.get("KTR","30")); KTE=int(os.environ.get("KTE","30")); NEP=int(os.environ.get("NEP","18"))
KMAP=float(os.environ.get("KMAP","0.30")); VW0=float(os.environ.get("VW0","0.5"))
IND=float(os.environ.get("IND","0.30")); TD=float(os.environ.get("TD","0.25"))
VBSYN=os.environ.get("VBSYN","0.6"); WL=os.environ.get("WL","1000u"); RCS=os.environ.get("RCS","300k")
GBL=float(os.environ.get("GBL","0.9")); CWW=os.environ.get("CWW","20p"); RWL=os.environ.get("RWL","2g")
RMO=os.environ.get("RMO","40k"); SGN=float(os.environ.get("SGN","-1")); ESGN=float(os.environ.get("ESGN","1")); TH=float(os.environ.get("TH","300"))
STEP=os.environ.get("STEP","3n"); TAG=os.environ.get("RUNTAG",str(os.getpid()))
d=load_digits(); X=(d.images.reshape(-1,64) if NIN==64 else d.images.reshape(-1,4,2,4,2).mean(axis=(2,4)).reshape(-1,16))
y=d.target; N=X.shape[1]; m=X.mean(0); s=X.std(0)+1e-6; X=np.clip((X-m)/s,-3,3)/3.0
rng=np.random.default_rng(SEED); CLS=list(range(C)); tri=[];trl=[];tei=[];tel=[]
for ci,c in enumerate(CLS):
    idx=np.where(y==c)[0]; rng.shuffle(idx); per=KTR//C+1
    for i in idx[:per]: tri.append(X[i]); trl.append(ci)
    for i in idx[per:per+KTE//C+1]: tei.append(X[i]); tel.append(ci)
tri=np.array(tri); trl=np.array(trl); tei=np.array(tei); tel=np.array(tel)
op=rng.permutation(len(trl)); tri,trl=tri[op],trl[op]
KTR=len(trl); KTE=len(tel)
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
# schedule: NEP epochs over KTR train examples (learning on), then KTE test examples (gbl frozen low)
order=[]   # (xrow, label, learn)
for ep in range(NEP):
    for k in range(KTR): order.append((tri[k],trl[k],1))
for k in range(KTE): order.append((tei[k],tel[k],0))
NP=len(order); TEND=NP*TH
# build per-input and per-label PWL waveforms
xin=[[] for _ in range(N)]; lab=[[[],[]] for _ in range(C)]; gblw=[]
for g,(xr,lb,lrn) in enumerate(order):
    t=g*TH
    for i in range(N): xin[i].append((t,0.5+IND*float(xr[i])))
    for c in range(C):
        td=TD*(1.0 if c==lb else -1.0); lab[c][0].append((t,0.5+td)); lab[c][1].append((t,0.5-td))
    gblw.append((t, GBL if lrn else 0.30))   # freeze learning during test (tail below Vt)
def gen():
    L=["single-layer multi-class continuous PC",SUB,"Vdd vdd 0 1.0",f"Vbsyn vbsyn 0 {VBSYN}","Vgm gm 0 0.6",
       f"Vgbl gbl 0 {stepped(gblw)}",f"Vwcm wcm 0 {VW0}"]
    for i in range(N):
        L+=[f"Vxp{i} a_xp{i} 0 {stepped(xin[i])}",f"Vxn{i} a_xn{i} 0 {stepped([(t,1-v) for t,v in xin[i]])}"]
    for c in range(C):
        L+=[f"Vxop{c} xop{c} 0 {stepped(lab[c][0])}",f"Vxon{c} xon{c} 0 {stepped(lab[c][1])}"]
    ic=[]
    ep_,en_=("eop","eon") if SGN>0 else ("eon","eop")
    for c in range(C):
        for i in range(N):
            w0=0.30*rng.standard_normal()
            gp=np.clip(VW0+KMAP*w0,0.3,0.9); gn=np.clip(VW0-KMAP*w0,0.3,0.9)
            L+=[f"Cwp_{i}_{c} wp_{i}_{c} 0 {CWW}",f"Cwn_{i}_{c} wn_{i}_{c} 0 {CWW}",
                f"Rwp_{i}_{c} wp_{i}_{c} wcm {RWL}",f"Rwn_{i}_{c} wn_{i}_{c} wcm {RWL}",
                f"X_syn_{i}_{c} a_xp{i} a_xn{i} wp_{i}_{c} wn_{i}_{c} mop{c} mon{c} vdd vbsyn gsyn"]
            ic+=[f".ic v(wp_{i}_{c})={gp:.4f} v(wn_{i}_{c})={gn:.4f}"]
        L+=[f"Xcm_mo{c} mop{c} mon{c} vdd cmld",f"Rmo{c} mop{c} mon{c} {RMO}",
            f"Xeo{c} xop{c} xon{c} mop{c} mon{c} eop{c} eon{c} vdd gm esub"]
        for i in range(N):
            ee_p=f"eop{c}" if SGN>0 else f"eon{c}"; ee_n=f"eon{c}" if SGN>0 else f"eop{c}"
            L+=[f"X_lrn_{i}_{c} {ee_p} {ee_n} a_xp{i} a_xn{i} wp_{i}_{c} wn_{i}_{c} vdd gbl gprod"]
    L+=ic
    rd=" ".join([f"v(mop{c}) v(mon{c})" for c in range(C)])
    L+=[f".tran {STEP} {int(TEND)}n uic",".options gmin=1e-10 reltol=2e-3 cshunt=1e-14",".control","run",
        f"wrdata pc1mc{TAG}.dat {rd}",".endc",".end"]
    open(f"pc1mc_{TAG}.cir","w").write("\n".join(L)+"\n")
    return NP,TEND
if __name__=="__main__":
    NP,TEND=gen()
    print(f"continuous single-layer MC PC: C={C} N={N} weights={C*N} caps={2*C*N} presentations={NP} TEND={TEND/1000:.1f}us")
    r=subprocess.run(["ngspice","-b",f"pc1mc_{TAG}.cir"],capture_output=True,text=True,timeout=2000)
    try:
        dat=np.loadtxt(f"pc1mc{TAG}.dat")
    except Exception as e:
        print("LOAD FAIL",e); print(r.stderr[-1500:]); raise SystemExit
    t=dat[:,0]*1e9; mu=np.array([dat[:,1+2*c]-dat[:,2+2*c] for c in range(C)])  # C x time
    def acc_window(g0,g1,labels):
        ok=0; n=0
        for j,g in enumerate(range(g0,g1)):
            ts=g*TH+TH-3; idx=np.argmin(np.abs(t-ts))
            pred=int(np.argmax(ESGN*mu[:,idx])); ok+=(pred==labels[j]); n+=1
        return ok/max(n,1)
    # training accuracy at the END of each epoch's last few examples, and final frozen test
    print("  epoch train-acc (last-epoch window):")
    for ep in [0,2,5,9,NEP-1]:
        if ep>=NEP: continue
        g0=ep*KTR; g1=(ep+1)*KTR; a=acc_window(g0,g1,list(trl))
        print(f"    epoch {ep:2d}: train={a:.3f}")
    teststart=NEP*KTR
    at=acc_window(teststart,teststart+KTE,list(tel))
    print(f"  FROZEN-WEIGHT TEST (held-out): {at:.3f}  (chance {1/C:.2f})")
