#!/usr/bin/env python3
"""Physical CDS weight-update for the sklearn-digits EP net (ep_digits), to test whether the
fully-analog update generalizes beyond XOR.  Per epoch: forward every example (free,nudge) on the
persistent ngspice process, then run a chopper-free Gilbert product-cell bank on the FREE activities
and on the NUDGE activities and subtract the cap reads (correlated double sampling -> offset cancels).
The multiply is physical transistors; CDS is the offset-cancel read.  mode=check -> gradient fidelity
(cos/sign vs numpy);  mode=train -> short physical-CDS training, report test accuracy."""
import os, sys, numpy as np, subprocess
import ep_digits as D
WEIGHTS,CP,phi,wval,KMAP=D.WEIGHTS,D.CP,D.phi,D.wval,D.KMAP
PHISC=float(os.environ.get("PHISC","0.35")); CWU=os.environ.get("CWU","400p")
THU=float(os.environ.get("THU","600")); GBNU=os.environ.get("GBNU","1.1")
TAG=os.environ.get("RUNTAG",str(os.getpid()))
GPROD=""".subckt gprod xp xn yp yn wp wn vdd vbn
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
def _pwl1(seq):
    pts=[]; t=0.0
    for v in seq:
        vv=0.5+max(-0.4,min(0.4,v*PHISC)); pts+=[(t,vv),(t+THU-0.5,vv)]; t+=THU
    return "PWL("+" ".join(f"{a:.1f}n {b:.4f}" for a,b in pts)+")"
def cds_run(acts_list, xlist):
    """acts_list[e]=READ-array for example e; run gprod bank summing product over examples."""
    ne=len(acts_list)
    L=["digits cds product run",".model NNR NMOS (LEVEL=1 VTO=0.2 KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)",
       ".model PNR PMOS (LEVEL=1 VTO=-0.2 KP=40u LAMBDA=0.02 GAMMA=0 PHI=0.7)",
       GPROD,"Vdd vdd 0 1.0",f"Vgbn gbn 0 {GBNU}"]; ic=[]
    for key in WEIGHTS:
        s,d=CP[key][0]
        sseq=[phi(s,acts_list[e],xlist[e]) for e in range(ne)]
        dseq=[phi(d,acts_list[e],xlist[e]) for e in range(ne)]
        L+=[f"Vxp_{key} xp_{key} 0 {_pwl1(sseq)}",f"Vxn_{key} xn_{key} 0 {_pwl1([-x for x in sseq])}",
            f"Vyp_{key} yp_{key} 0 {_pwl1(dseq)}",f"Vyn_{key} yn_{key} 0 {_pwl1([-x for x in dseq])}",
            f"Cwp_{key} wp_{key} 0 {CWU}",f"Cwn_{key} wn_{key} 0 {CWU}",
            f"Xu_{key} xp_{key} xn_{key} yp_{key} yn_{key} wp_{key} wn_{key} vdd gbn gprod"]
        ic.append(f".ic v(wp_{key})=0.5 v(wn_{key})=0.5")
    L+=ic; cols=" ".join(f"v(wp_{key}) v(wn_{key})" for key in WEIGHTS)
    L+=[f".tran 1n {int(ne*THU)}n uic",".control","run",f"wrdata pd{TAG}.dat {cols}",".endc",".end"]
    open(f"physd_{TAG}.cir","w").write("\n".join(L)+"\n")
    subprocess.run(["ngspice","-b",f"physd_{TAG}.cir"],capture_output=True,text=True,timeout=300)
    row=np.loadtxt(f"pd{TAG}.dat")[-1]
    return {key:(row[4*i+1]-row[4*i+3]) for i,key in enumerate(WEIGHTS)}
def phys_grad(Fs, Ns, xs):
    wf=cds_run(Fs,xs); wn=cds_run(Ns,xs)
    return {key:(wn[key]-wf[key])/(2*KMAP) for key in WEIGHTS}
def numpy_grad(Fs, Ns, xs):
    g={k:0.0 for k in WEIGHTS}
    for e in range(len(xs)):
        for key in WEIGHTS:
            s,d=CP[key][0]
            g[key]+=(phi(s,Ns[e],xs[e])*phi(d,Ns[e],xs[e])-phi(s,Fs[e],xs[e])*phi(d,Fs[e],xs[e]))
    return g

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "check"
    (Xtr,ytr),(Xte,yte)=D.make_data()
    net=D.DigitsEP(seed=int(os.environ.get("SEED","0")))
    if mode=="check":
        ne=int(os.environ.get("NE","10"))
        Fs=[net.free(Xtr[e]) for e in range(ne)]; Ns=[net.nudge(Xtr[e],ytr[e]) for e in range(ne)]
        gp=phys_grad(Fs,Ns,Xtr[:ne]); gn=numpy_grad(Fs,Ns,Xtr[:ne])
        a=np.array([gn[k] for k in WEIGHTS]); b=np.array([gp[k] for k in WEIGHTS])
        cos=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-18)); sign=float(np.mean(np.sign(a)==np.sign(b)))
        print(f"DIGITS net (NHID={D.NHID} weights={len(WEIGHTS)} C={D.C}) physical-CDS fidelity: cos={cos:.3f} sign={100*sign:.0f}%")
    elif mode=="train":
        eta=float(os.environ.get("ETA","6")); epochs=int(os.environ.get("EPOCHS","20"))
        mom=float(os.environ.get("MOM","0.5")); sign=float(os.environ.get("USIGN","-1"))
        vel={k:0.0 for k in WEIGHTS}; rng=np.random.default_rng(7)
        a0=net.evaluate(Xte,yte)
        print(f"# DIGITS physical-CDS training: classes {list(range(D.C))} NHID={D.NHID} weights={len(WEIGHTS)} train={len(Xtr)} test={len(Xte)}")
        print(f"# epoch0 test={a0:.3f} (chance {1.0/D.C:.2f})"); best=a0
        for ep in range(1,epochs+1):
            Fs=[net.free(Xtr[e]) for e in range(len(Xtr))]; Ns=[net.nudge(Xtr[e],ytr[e]) for e in range(len(Xtr))]
            g=phys_grad(Fs,Ns,Xtr)                       # <-- PHYSICAL transistor-computed gradient
            net.apply_update(g,eta,sign,vel,mom,rng=rng)
            if ep%2==0 or ep==epochs:
                te=net.evaluate(Xte,yte); best=max(best,te)
                print(f"  epoch {ep:3d}  test={te:.3f}  best={best:.3f}  (PHYSICAL update)")
        print(f"RESULT physical-CDS digits: final={net.evaluate(Xte,yte):.3f} best={best:.3f} chance={1.0/D.C:.2f}")
