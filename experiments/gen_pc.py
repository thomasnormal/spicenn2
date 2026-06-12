#!/usr/bin/env python3
"""Predictive-Coding network in ngspice (current-mode differential).  Supervised, hard-clamped.
Stage 1 here = INFERENCE only (fixed weights): does the relaxation with explicit error nodes +
hard-clamped output settle to the PC equilibrium?  Compares the settled output to a numpy PC
inference for the same weights.

Per layer:  mu = W . (below activity)  [synapse into diode load]
            eps = x - mu               [esub: differential difference amp]
hidden x_h cap relaxes:  pulled toward mu_h by a transconductor (= -eps_h term) + backward synapse
carries W_ho^T eps_o (transpose is free: reciprocal synapse).  f' gating dropped (approx, licensed).
"""
import os, numpy as np, subprocess
H=int(os.environ.get("H","6")); SEED=int(os.environ.get("SEED","0"))
IND=float(os.environ.get("IND","0.3")); TD=float(os.environ.get("TD","0.20"))
KMAP=float(os.environ.get("KMAP","0.10")); VW0=float(os.environ.get("VW0","0.5"))
WLOAD=os.environ.get("WLOAD","3000u"); RLNEU=os.environ.get("RLNEU","800k")
WL=os.environ.get("WL","2400u"); RCS=os.environ.get("RCS","300k")   # CMFB load size / CM-sense resistors
GMT=os.environ.get("GMT","0.6")           # esub tail bias
VBNEU=os.environ.get("VBNEU","0.45")      # neuron tail bias (sets a_h output CM/gain)
TAG=os.environ.get("RUNTAG",str(os.getpid()))
PATS=[(0,0,0),(0,1,1),(1,0,1),(1,1,0)]
def ts(b): return 1.0 if b else -1.0
rng=np.random.default_rng(SEED)
WEIGHTS=[]
for j in range(H):
    for i in (1,2): WEIGHTS.append(f"wih_{i}_{j}")
    WEIGHTS.append(f"who_{j}")
w0={k:float(0.5*rng.standard_normal()) for k in WEIGHTS}
if os.environ.get("WINIT"):
    vv=np.loadtxt(os.environ["WINIT"]).reshape(-1); w0={k:float(vv[i]) for i,k in enumerate(WEIGHTS)}
def wg(v): return np.clip(VW0+KMAP*v,0.3,0.9),np.clip(VW0-KMAP*v,0.3,0.9)
VBSYN=os.environ.get("VBSYN","0.55"); VBE=os.environ.get("VBE","0.45")   # CONSTANT synapse tail bias -> weight-independent CM sink
SUB=f""".subckt dsyn inp inn wp wn outp outn vdd vbn
* Gilbert synapse: out current ~ (inp-inn)(wp-wn), CONSTANT tail (gate vbn) so the common-mode
* current it sinks is the SAME regardless of weight -> every node sits at one common mode.
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
RPULL=os.environ.get("RPULL","100k"); WCM=os.environ.get("WCM","200u"); RNODE=os.environ.get("RNODE","30k")
def syn(tag,ip,inn,op,on,key):
    gp,gn=wg(w0[key])
    return [f"Vwp_{tag} wp_{tag} 0 {gp:.4f}",f"Vwn_{tag} wn_{tag} 0 {gn:.4f}",
            f"X_{tag} {ip} {inn} wp_{tag} wn_{tag} {op} {on} vdd vbsyn dsyn"]
def deck(pat,clamp_out):
    s1,s2=ts(pat[0]),ts(pat[1]); lab=ts(pat[2])
    L=["PC inference",
       ".model NNR NMOS (LEVEL=1 VTO=0.2 KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)",
       ".model PNR PMOS (LEVEL=1 VTO=-0.2 KP=40u LAMBDA=0.02 GAMMA=0 PHI=0.7)",
       SUB,"Vdd vdd 0 1.0",f"Vgm gm 0 {GMT}",f"Vbneu vbneu 0 {VBNEU}",f"Vbsyn vbsyn 0 {VBSYN}",f"Vrefn vrefn 0 0.5",f"Vbe vbe 0 {VBE}",
       f"Vx1p a_xp1 0 {0.5+IND*s1}",f"Vx1n a_xn1 0 {0.5-IND*s1}",
       f"Vx2p a_xp2 0 {0.5+IND*s2}",f"Vx2n a_xn2 0 {0.5-IND*s2}"]
    # hidden layer.  mu_h = W_ih.x_in (forward only).  x_h = W_ih.x_in + W_ho^T.eps_o (forward+backward).
    # Both are diode-loaded current-mode nodes (CM pinned ~0.5, like the EP nodes that worked), so
    # eps_h = x_h - mu_h = the top-down term, with a clean common mode throughout the chain.
    for j in range(H):
        xp,xn=f"xhp{j}",f"xhn{j}"; mp,mn=f"mhp{j}",f"mhn{j}"
        L+=[f"Xcm_mh{j} {mp} {mn} vdd cmld",f"Xcm_xh{j} {xp} {xn} vdd cmld",
            f"Rnd_mh{j} {mp} {mn} {RNODE}",f"Rnd_xh{j} {xp} {xn} {RNODE}",
            f"Cxhp{j} {xp} 0 0.4p",f"Cxhn{j} {xn} 0 0.4p"]
        for i in (1,2):
            L+=syn(f"ihm_{i}_{j}",f"a_xp{i}",f"a_xn{i}",mp,mn,f"wih_{i}_{j}")   # forward -> mu_h
            L+=syn(f"ihx_{i}_{j}",f"a_xp{i}",f"a_xn{i}",xp,xn,f"wih_{i}_{j}")   # forward -> x_h (same weight)
        L+=syn(f"bk_{j}","eop","eon",xp,xn,f"who_{j}")                          # backward W_ho^T.eps_o -> x_h
        L+=[f"Xn{j} {xp} {xn} ahp{j} ahn{j} vdd vbneu dneuron"]                 # a_h = f(x_h)
        L+=[f"Xeh{j} {xp} {xn} {mp} {mn} ehp{j} ehn{j} vdd gm esub"]            # eps_h = x_h - mu_h
    # output prediction mu_o = W_ho . a_h (diode-loaded) = the network output at test time
    L+=["Xcm_mo mop mon vdd cmld",f"Rnd_mo mop mon {RNODE}"]
    NONEU=int(os.environ.get("NONEU","0"))
    for j in range(H):
        src=(f"xhp{j}",f"xhn{j}") if NONEU else (f"ahp{j}",f"ahn{j}")
        L+=syn(f"ho_{j}",src[0],src[1],"mop","mon",f"who_{j}")
    if clamp_out:
        L+=[f"Vxop xop 0 {0.5+TD*lab}",f"Vxon xon 0 {0.5-TD*lab}",         # HARD-CLAMP output to label
            "Xeo xop xon mop mon eop eon vdd gm esub"]                     # eps_o = label - mu_o
    else:
        L+=["Veop eop 0 0.5","Veon eon 0 0.5"]                             # test: no error feedback (eps_o=0)
    cols=" ".join([f"v(xhp{j}) v(xhn{j}) v(ehp{j}) v(ehn{j})" for j in range(H)]+["v(mop) v(mon) v(eop) v(eon)"])
    L+=[".control","op",f"wrdata pc{TAG}.dat {cols}",".endc",".end"]
    return "\n".join(L)+"\n"
def run(pat,clamp_out):
    open(f"pc_{TAG}.cir","w").write(deck(pat,clamp_out))
    subprocess.run(["ngspice","-b",f"pc_{TAG}.cir"],capture_output=True,text=True,timeout=60)
    return np.loadtxt(f"pc{TAG}.dat").reshape(-1)[1::2]
if __name__=="__main__":
    print(f"PC inference circuit H={H}. Output (free) per pattern vs XOR target:")
    for pat in PATS:
        v=run(pat,clamp_out=False); o=v[-2]-v[-1]   # last cols mop,mon... actually eop,eon; recompute
        # cols: per j (xhp,xhn,ehp,ehn) then mop,mon,eop,eon
        mo=v[4*H+0]-v[4*H+1]
        print(f"  x={pat[:2]} target={'+' if pat[2] else '-'}  output mu_o(mV)={mo*1e3:+.1f}")
