#!/usr/bin/env python3
"""
FULLY-DIFFERENTIAL transistor current-mode EP (simpler, more robust, more scalable).
vs ep_tcm.py: the differential signal IS the signed representation -> complement is free,
common-mode/offset is REJECTED (no zero-weight offset), and the synapse drops the mirror
(6T crossed diff pairs instead of two 5T OTAs). ~40% fewer transistors and offset-robust.

state node = differential pair (up,un) loaded by RL to vdd; value = up-un.
neuron     = diff-pair amp (up,un)->(ap,an), increasing, complement free.
synapse    = dsyn: w+ diff pair + w- crossed diff pair, signed by (wp,wn), CM-rejecting.
Symmetric hidden<->output coupling shares the weight. Only the local correlation update
runs on the controller. Everything in the circuit is MOSFETs/resistors.
"""
import numpy as np, subprocess, os
NHID=int(os.environ.get("NHID","6"))
RLNODE=os.environ.get("RLNODE","25k"); RLNEU=os.environ.get("RLNEU","800k")
VBN=os.environ.get("VBN","0.30"); RB=float(os.environ.get("RB","2e5")); BETA=1.0/RB
NEU=os.environ.get("NEU","base")              # neuron type: base | hi | two
NDIV=float(os.environ.get("NDIV","0.0"))      # per-neuron tail-bias diversity (heterogeneous neurons)
TAG=os.environ.get("RUNTAG",str(os.getpid()))
VBN_F=float(VBN)
WLOAD=os.environ.get("WLOAD","2000u")
VW0=float(os.environ.get("VW0","0.55")); KMAP=float(os.environ.get("KMAP","0.10"))
IND=float(os.environ.get("IND","0.3"))        # input differential half-swing
TD=float(os.environ.get("TD","0.06"))         # target differential half-swing
PATS=[(0,0,0),(0,1,1),(1,0,1),(1,1,0)]
def tsign(b): return +1 if b else -1
WEIGHTS=[]
for j in range(1,NHID+1):
    for i in (1,2): WEIGHTS.append(f"w{i}{j}")
    WEIGHTS+=[f"o{j}",f"bh{j}"]
WEIGHTS.append("bo")
READ=sum(([f"ap_h{j}",f"an_h{j}"] for j in range(1,NHID+1)),[])+["up_o","un_o"]

SUB=""".subckt dsyn inp inn outp outn wp wn vdd
M1 outp inp tp 0 NNR W=200u L=100u
M2 outn inn tp 0 NNR W=200u L=100u
Mtp tp wp 0 0 NNR W=200u L=100u
M3 outn inp tn 0 NNR W=200u L=100u
M4 outp inn tn 0 NNR W=200u L=100u
Mtn tn wn 0 0 NNR W=200u L=100u
.ends
.subckt dneuron up un ap an vdd vbn
M1 an up tn 0 NNR W=200u L=100u
M2 ap un tn 0 NNR W=200u L=100u
Mt tn vbn 0 0 NNR W=200u L=100u
RLa vdd ap {RLNEU}
RLb vdd an {RLNEU}
.ends
.subckt dneuron2 up un ap an vdd vbn
M1 m2 up t1 0 NNR W=200u L=100u
M2 m1 un t1 0 NNR W=200u L=100u
Mt1 t1 vbn 0 0 NNR W=200u L=100u
RL1 vdd m1 {RLNEU}
RL2 vdd m2 {RLNEU}
M3 an m1 t2 0 NNR W=200u L=100u
M4 ap m2 t2 0 NNR W=200u L=100u
Mt2 t2 vbn 0 0 NNR W=200u L=100u
RL3 vdd ap {RLNEU}
RL4 vdd an {RLNEU}
.ends
""".replace("{RLNEU}",RLNEU)
NEUSUB="dneuron2" if NEU=="two" else "dneuron"

def wg(w): return f"{np.clip(VW0+KMAP*w,0.3,0.9):.4f}",f"{np.clip(VW0-KMAP*w,0.3,0.9):.4f}"
def syn(tag,ip,inn,op,on,key,w):
    gp,gn=wg(w[key])
    return [f"Vwp_{tag} wp_{tag} 0 {gp}",f"Vwn_{tag} wn_{tag} 0 {gn}",
            f"X_{tag} {ip} {inn} {op} {on} wp_{tag} wn_{tag} vdd dsyn"]

def deck(w):
    L=["fully-differential transistor current-mode EP",
       ".model NNR NMOS (LEVEL=1 VTO=0.2  KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)",
       ".model PNR PMOS (LEVEL=1 VTO=-0.2 KP=40u  LAMBDA=0.02 GAMMA=0 PHI=0.7)",
       SUB,"Vdd vdd 0 1.0","Vbn vbn 0 "+VBN,
       "Vbp_hi bp_hi 0 0.8","Vbp_lo bp_lo 0 0.2",      # fixed differential for bias
       "Vtp vtp 0 0.5","Vtn vtn 0 0.5"]
    # input differential clamps
    for i in (1,2):
        L+=[f"Vaxp{i} a_xp{i} 0 0.5",f"Vaxn{i} a_xn{i} 0 0.5"]
    for j in range(1,NHID+1):
        up,un=f"up_h{j}",f"un_h{j}"
        L+=[f"Mlp_h{j} {up} {up} vdd vdd PNR W="+WLOAD+" L=100u",
            f"Mln_h{j} {un} {un} vdd vdd PNR W="+WLOAD+" L=100u"]   # diode PMOS loads (hold CM)
        bj=VBN_F + (NDIV*(2.0*(j-1)/max(NHID-1,1)-1.0) if NDIV>0 else 0.0)   # per-neuron tail bias
        L.append(f"Vbnh{j} bnh{j} 0 {bj:.4f}")
        L.append(f"Xn_h{j} {up} {un} ap_h{j} an_h{j} vdd bnh{j} {NEUSUB}")
        for i in (1,2):
            L+=syn(f"i{i}{j}",f"a_xp{i}",f"a_xn{i}",up,un,f"w{i}{j}",w)
        L+=syn(f"bk{j}","up_o","un_o",up,un,f"o{j}",w)         # backward (shared o_j)
        L+=syn(f"bh{j}","bp_hi","bp_lo",up,un,f"bh{j}",w)      # bias
    # output (linear differential): diode PMOS loads + forward synapses + bias
    L+=["Mlp_o up_o up_o vdd vdd PNR W="+WLOAD+" L=100u","Mln_o un_o un_o vdd vdd PNR W="+WLOAD+" L=100u"]
    for j in range(1,NHID+1):
        L+=syn(f"fo{j}",f"ap_h{j}",f"an_h{j}","up_o","un_o",f"o{j}",w)
    L+=syn("bo_","bp_hi","bp_lo","up_o","un_o","bo",w)
    # differential nudge on output
    L+=["Rbp up_o vtp 1e10","Rbn un_o vtn 1e10"]
    L.append(".control"); cols=" ".join(f"v({n})" for n in READ)
    for k,(p,q,_) in enumerate(PATS):
        s1,s2=tsign(p),tsign(q)
        L+=[f"alter Vaxp1 {0.5+IND*s1}",f"alter Vaxn1 {0.5-IND*s1}",
            f"alter Vaxp2 {0.5+IND*s2}",f"alter Vaxn2 {0.5-IND*s2}",
            "alter Rbp 1e10","alter Rbn 1e10","op",f"wrdata gf{TAG}_{k}.dat {cols}"]
        td=TD*tsign(PATS[k][2])
        L+=[f"alter Vtp {0.5+td}",f"alter Vtn {0.5-td}",
            f"alter Rbp {RB}",f"alter Rbn {RB}","op",f"wrdata gn{TAG}_{k}.dat {cols}"]
    L+=[".endc",".end"]; return "\n".join(L)+"\n"

def rd(fn): return np.loadtxt(fn).reshape(-1)[1::2]
def run(w):
    open(f"ep_dtcm_{TAG}.cir","w").write(deck(w))
    subprocess.run(["ngspice","-b",f"ep_dtcm_{TAG}.cir"],capture_output=True,text=True,timeout=180)
    return (np.array([rd(f"gf{TAG}_{k}.dat") for k in range(4)]),
            np.array([rd(f"gn{TAG}_{k}.dat") for k in range(4)]))
idx={n:k for k,n in enumerate(READ)}
def dout(arr): return arr[idx['up_o']]-arr[idx['un_o']]          # output differential

def phi(tok,arr,pat):
    if tok=="HI": return 0.8-0.2
    if tok=="x1": return 2*IND*tsign(pat[0])
    if tok=="x2": return 2*IND*tsign(pat[1])
    if tok=="O":  return arr[idx['up_o']]-arr[idx['un_o']]
    return arr[idx[f"ap_{tok}"]]-arr[idx[f"an_{tok}"]]
def corr_pairs():
    P={k:[] for k in WEIGHTS}
    for j in range(1,NHID+1):
        for i in (1,2): P[f"w{i}{j}"].append((f"x{i}",f"h{j}"))
        P[f"o{j}"].append((f"h{j}","O")); P[f"bh{j}"].append(("HI",f"h{j}"))
    P["bo"].append(("HI","O")); return P
CP=corr_pairs()
import os as _os
UPDATE=_os.environ.get("UPDATE","corr")   # update primitive (what computes the synapse increment)
VTU=float(_os.environ.get("VTU","0.0"))   # NMOS threshold for rectified-square variants
def relu(x): return x if x>0 else 0.0
def prim(a,b):
    # the per-synapse update primitive ~ phi_i*phi_j, computed different physical ways:
    if UPDATE=="corr":    return a*b                                  # exact product (Gilbert, ref)
    if UPDATE=="quarter": return 0.25*((a+b)**2-(a-b)**2)             # 2 NMOS squarers -> exact product
    if UPDATE=="sqdiff":  return -0.5*(a-b)**2                        # 1 NMOS sat squarer on (phi_i-phi_j)
    if UPDATE=="sqsum":   return  0.5*(a+b)**2                        # 1 NMOS sat squarer on (phi_i+phi_j)
    if UPDATE=="rectdiff":return -0.5*relu(a-b-VTU)**2 +0.5*relu(b-a-VTU)**2  # real NMOS: rectified, both polarities
    if UPDATE=="triode":  return a*relu(b)                            # 1 NMOS triode: (overdrive~a)*(Vds~b>=0)
    return a*b
def ep_grad(w):
    F,N=run(w); g={k:0.0 for k in WEIGHTS}
    for k,pat in enumerate(PATS):
        for key in WEIGHTS:
            for (s,d) in CP[key]:
                f=prim(phi(s,F[k],pat),phi(d,F[k],pat)); n=prim(phi(s,N[k],pat),phi(d,N[k],pat))
                g[key]+=((n-f)/BETA)/4.0
    return g,F

# ------- PHYSICAL update: transistor cells charge the weight caps (no numpy prim) -------
# quarter-square sqdiff cell: two NMOS squarers fed +dv/2,-dv/2 (gates dvp,dvn; src vsref) summed
# -> I ~ const + dv^2/2 (linear & const cancel exactly).  Output chopper push-pulls the two-phase
# difference -(dv_nudge^2 - dv_free^2) differentially onto (wp,wn).  dv = phi_s - phi_d.
SQD_SUB=""".subckt sqd dvp dvn wp wn psig psigb vdd vsref
Msq1 isum dvp vsref 0 NNR W=400u L=100u
Msq2 isum dvn vsref 0 NNR W=400u L=100u
Mld  isum isum vdd vdd PNR W=300u L=100u
Mpsrc psrc isum vdd vdd PNR W=300u L=100u
Mpref nref isum vdd vdd PNR W=300u L=100u
Mnref nref nref 0 0 NNR W=300u L=100u
Mpu_wp psrc psigb wp 0 NNR W=400u L=100u
Mpu_wn psrc psig  wn 0 NNR W=400u L=100u
Msk_wn wn nref ssn 0 NNR W=300u L=100u
Mss_wn ssn psigb 0 0 NNR W=400u L=100u
Msk_wp wp nref ssp 0 NNR W=300u L=100u
Mss_wp ssp psig 0 0 NNR W=400u L=100u
.ends"""
# Chopper-FREE product cell (differential push-pull, no chopper) for CDS: wp += (iop-ion)=product,
# wn += -(product).  Run on free then nudge activities and subtract the reads -> offset cancels
# (correlated double sampling), with NO chopper charge-injection.
GPROD_SUB=""".subckt gprod xp xn yp yn wp wn vdd vbn
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
# CHOPPER-STABILIZED cell: Gilbert product (NO input chopper) -> differential push-pull current
# (A=+prod, B=-prod, differential so zero common-mode injection) -> OUTPUT chopper swaps A,B between
# free/nudge.  Offset delta appears equally in free & nudge -> CANCELS in wp-wn; the product survives
# as the two-phase difference.  Combines gupd's offset-cancel with gupd2's no-CM-injection.
GUPD3_SUB=""".subckt gupd3 xp xn yp yn wp wn psig psigb vdd vbn
Mtail nt vbn 0 0 NNR W=400u L=100u
M5 na yp nt 0 NNR W=200u L=100u
M6 nb yn nt 0 NNR W=200u L=100u
M1 iop xp na 0 NNR W=200u L=100u
M2 ion xn na 0 NNR W=200u L=100u
M3 ion xp nb 0 NNR W=200u L=100u
M4 iop xn nb 0 NNR W=200u L=100u
Mlp iop iop vdd vdd PNR W=200u L=100u
Mln ion ion vdd vdd PNR W=200u L=100u
Mpu_A nA iop vdd vdd PNR W=200u L=100u
MpRn nrn ion vdd vdd PNR W=200u L=100u
MnRn nrn nrn 0 0 NNR W=200u L=100u
Mpd_A nA nrn 0 0 NNR W=200u L=100u
Mpu_B nB ion vdd vdd PNR W=200u L=100u
MpRp nrp iop vdd vdd PNR W=200u L=100u
MnRp nrp nrp 0 0 NNR W=200u L=100u
Mpd_B nB nrp 0 0 NNR W=200u L=100u
Msw1 nA psigb wp 0 NNR W=400u L=100u
Msw2 nB psigb wn 0 NNR W=400u L=100u
Msw3 nA psig  wn 0 NNR W=400u L=100u
Msw4 nB psig  wp 0 NNR W=400u L=100u
.ends"""
GUPD2_SUB=""".subckt gupd2 xp xn yp yn wp wn psig psigb vdd vbn
Msa yA yp psig 0 NNR W=300u L=100u
Msb yA yn psigb 0 NNR W=300u L=100u
Msc yB yn psig 0 NNR W=300u L=100u
Msd yB yp psigb 0 NNR W=300u L=100u
Mtail nt vbn 0 0 NNR W=400u L=100u
M5 na yA nt 0 NNR W=200u L=100u
M6 nb yB nt 0 NNR W=200u L=100u
M1 iop xp na 0 NNR W=200u L=100u
M2 ion xn na 0 NNR W=200u L=100u
M3 ion xp nb 0 NNR W=200u L=100u
M4 iop xn nb 0 NNR W=200u L=100u
Mlp iop iop vdd vdd PNR W=200u L=100u
Mln ion ion vdd vdd PNR W=200u L=100u
Mpu_p wp iop vdd vdd PNR W=200u L=100u
Mpu_n wn ion vdd vdd PNR W=200u L=100u
MpRn nrn ion vdd vdd PNR W=200u L=100u
MnRn nrn nrn 0 0 NNR W=200u L=100u
Mpd_p wp nrn 0 0 NNR W=200u L=100u
MpRp nrp iop vdd vdd PNR W=200u L=100u
MnRp nrp nrp 0 0 NNR W=200u L=100u
Mpd_n wn nrp 0 0 NNR W=200u L=100u
.ends"""
THU=float(_os.environ.get("THU","200")); CWU=_os.environ.get("CWU","40p")
GBNU=_os.environ.get("GBNU","0.85"); CMWU=_os.environ.get("CMWU","120u")
PHISC=float(_os.environ.get("PHISC","0.5"))
def _pwlphi(seq):   # seq: [(phi_free,phi_nudge)] per pattern -> PWL rail (free 1st half, nudge 2nd half)
    pts=[]; t=0.0
    for ff,nn in seq:
        vf=0.5+max(-0.4,min(0.4,ff*PHISC)); vn=0.5+max(-0.4,min(0.4,nn*PHISC))
        pts+=[(t,vf),(t+THU-0.5,vf),(t+THU+0.5,vn),(t+2*THU,vn)]; t+=2*THU
    return "PWL("+" ".join(f"{a:.1f}n {b:.4f}" for a,b in pts)+")"
def _clk(nud):
    pts=[(0.0,0.0 if nud else 2.0)]; t=0.0
    for c in range(4):
        a,b=(2.0,0.0) if nud else (0.0,2.0)
        pts+=[(t+THU-0.5,b),(t+THU+0.5,a),(t+2*THU-0.5,a),(t+2*THU+0.5,b)]; t+=2*THU
    return "PWL("+" ".join(f"{x:.1f}n {v}" for x,v in pts)+")"
CELLU=_os.environ.get("CELLU","gupd2")   # physical update cell: gupd2 (product) | sqd (quarter-square sqdiff)
SQSC=float(_os.environ.get("SQSC","0.5")); VBSQ=float(_os.environ.get("VBSQ","0.45"))
def _pwldv(seq):    # seq:[(dv_free,dv_nudge)] -> rail 0.5 + dv*SQSC/2 (free 1st half, nudge 2nd)
    pts=[]; t=0.0
    for ff,nn in seq:
        vf=0.5+max(-0.42,min(0.42,ff*SQSC/2)); vn=0.5+max(-0.42,min(0.42,nn*SQSC/2))
        pts+=[(t,vf),(t+THU-0.5,vf),(t+THU+0.5,vn),(t+2*THU,vn)]; t+=2*THU
    return "PWL("+" ".join(f"{a:.1f}n {b:.4f}" for a,b in pts)+")"
def _pwl1(seq):     # seq:[v per pattern] -> rail 0.5+v*PHISC over 4 windows of THU each (no sub-phase)
    pts=[]; t=0.0
    for v in seq:
        vv=0.5+max(-0.4,min(0.4,v*PHISC)); pts+=[(t,vv),(t+THU-0.5,vv)]; t+=THU
    return "PWL("+" ".join(f"{a:.1f}n {b:.4f}" for a,b in pts)+")"
def _cds_run(acts):   # acts[p][node]-> activity; run gprod bank, return {key:(wp-wn)} differential caps
    L=["cds product run",".model NNR NMOS (LEVEL=1 VTO=0.2 KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)",
       ".model PNR PMOS (LEVEL=1 VTO=-0.2 KP=40u LAMBDA=0.02 GAMMA=0 PHI=0.7)",
       GPROD_SUB,"Vdd vdd 0 1.0",f"Vgbn gbn 0 {GBNU}"]; ic=[]
    for key in WEIGHTS:
        s,d=CP[key][0]
        sseq=[phi(s,acts[p],PATS[p]) for p in range(4)]; dseq=[phi(d,acts[p],PATS[p]) for p in range(4)]
        L+=[f"Vxp_{key} xp_{key} 0 {_pwl1(sseq)}",f"Vxn_{key} xn_{key} 0 {_pwl1([-x for x in sseq])}",
            f"Vyp_{key} yp_{key} 0 {_pwl1(dseq)}",f"Vyn_{key} yn_{key} 0 {_pwl1([-x for x in dseq])}",
            f"Cwp_{key} wp_{key} 0 {CWU}",f"Cwn_{key} wn_{key} 0 {CWU}",
            f"Xu_{key} xp_{key} xn_{key} yp_{key} yn_{key} wp_{key} wn_{key} vdd gbn gprod"]
        ic.append(f".ic v(wp_{key})=0.5 v(wn_{key})=0.5")
    L+=ic; cols=" ".join(f"v(wp_{key}) v(wn_{key})" for key in WEIGHTS)
    L+=[f".tran 1n {int(4*THU)}n uic",".control","run",f"wrdata pu{TAG}.dat {cols}",".endc",".end"]
    open(f"physupd_{TAG}.cir","w").write("\n".join(L)+"\n")
    subprocess.run(["ngspice","-b",f"physupd_{TAG}.cir"],capture_output=True,text=True,timeout=180)
    row=np.loadtxt(f"pu{TAG}.dat")[-1]
    return {key:(row[4*i+1]-row[4*i+3]) for i,key in enumerate(WEIGHTS)}
def physupd_cds(w):
    F,N=run(w)
    wf=_cds_run(F); wn=_cds_run(N)        # offset is identical in both -> cancels in the difference
    g={key:(wn[key]-wf[key])/(2*KMAP) for key in WEIGHTS}
    return g,F
def physupd_grad(w):
    if CELLU=="cds": return physupd_cds(w)
    F,N=run(w)
    L=["physical EP update cell bank",
       ".model NNR NMOS (LEVEL=1 VTO=0.2  KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)",
       ".model PNR PMOS (LEVEL=1 VTO=-0.2 KP=40u  LAMBDA=0.02 GAMMA=0 PHI=0.7)",
       (SQD_SUB if CELLU=="sqd" else GUPD3_SUB if CELLU=="gupd3" else GUPD2_SUB),"Vdd vdd 0 1.0",f"Vgbn gbn 0 {GBNU}",
       f"Vvsref vsref 0 {0.5-VBSQ:.4f}",
       f"Vpsig psig 0 {_clk(True)}",f"Vpsigb psigb 0 {_clk(False)}"]
    ic=[]
    for key in WEIGHTS:
        s,d=CP[key][0]
        L+=[f"Cwp_{key} wp_{key} 0 {CWU}",f"Cwn_{key} wn_{key} 0 {CWU}",
            f"Rcp_{key} wp_{key} cm_{key} 2g",f"Rcn_{key} wn_{key} cm_{key} 2g",
            f"Msp_{key} wp_{key} cm_{key} 0 0 NNR W={CMWU} L=100u",
            f"Msn_{key} wn_{key} cm_{key} 0 0 NNR W={CMWU} L=100u"]
        if CELLU=="sqd":
            dvseq=[(phi(s,F[p],PATS[p])-phi(d,F[p],PATS[p]), phi(s,N[p],PATS[p])-phi(d,N[p],PATS[p])) for p in range(4)]
            L+=[f"Vdvp_{key} dvp_{key} 0 {_pwldv(dvseq)}",
                f"Vdvn_{key} dvn_{key} 0 {_pwldv([(-a,-b) for a,b in dvseq])}",
                f"Xu_{key} dvp_{key} dvn_{key} wp_{key} wn_{key} psig psigb vdd vsref sqd"]
        else:
            sseq=[(phi(s,F[p],PATS[p]),phi(s,N[p],PATS[p])) for p in range(4)]
            dseq=[(phi(d,F[p],PATS[p]),phi(d,N[p],PATS[p])) for p in range(4)]
            L+=[f"Vxp_{key} xp_{key} 0 {_pwlphi(sseq)}",
                f"Vxn_{key} xn_{key} 0 {_pwlphi([(-a,-b) for a,b in sseq])}",
                f"Vyp_{key} yp_{key} 0 {_pwlphi(dseq)}",
                f"Vyn_{key} yn_{key} 0 {_pwlphi([(-a,-b) for a,b in dseq])}",
                f"Xu_{key} xp_{key} xn_{key} yp_{key} yn_{key} wp_{key} wn_{key} psig psigb vdd gbn {CELLU}"]
        ic.append(f".ic v(wp_{key})=0.5 v(wn_{key})=0.5 v(cm_{key})=0.5")
    L+=ic
    cols=" ".join(f"v(wp_{key}) v(wn_{key})" for key in WEIGHTS)
    L+=[f".tran 1n {int(4*2*THU)}n uic",".control","run",f"wrdata pu{TAG}.dat {cols}",".endc",".end"]
    open(f"physupd_{TAG}.cir","w").write("\n".join(L)+"\n")
    subprocess.run(["ngspice","-b",f"physupd_{TAG}.cir"],capture_output=True,text=True,timeout=180)
    row=np.loadtxt(f"pu{TAG}.dat")[-1]
    g={key:(row[4*i+1]-row[4*i+3])/(2*KMAP) for i,key in enumerate(WEIGHTS)}   # physical grad (w-units)
    return g,F
PHYSUPD=int(_os.environ.get("PHYSUPD","0"))
def grad(w): return physupd_grad(w) if PHYSUPD else ep_grad(w)

def main():
    import sys; mode=sys.argv[1] if len(sys.argv)>1 else "train"
    rng=np.random.default_rng(int(os.environ.get("SEED","0")))
    tgt=np.array([TD*tsign(p[2]) for p in PATS])
    if mode=="fwd":
        w={k:float(0.5*rng.standard_normal()) for k in WEIGHTS}
        F,_=run(w); print("dout per pattern (mV):",np.round([dout(F[k]) for k in range(4)],4)*1e3); return
    if mode=="gradcheck":
        d=0.05; coss=[]
        for t in range(int(os.environ.get("T","4"))):
            w={k:float(0.5*rng.standard_normal()) for k in WEIGHTS}; gfd={}
            for k in WEIGHTS:
                wp=dict(w);wp[k]+=d; wm=dict(w);wm[k]-=d
                Fp,_=run(wp); Fm,_=run(wm)
                lp=np.mean([(dout(Fp[i])-tgt[i])**2 for i in range(4)])
                lm=np.mean([(dout(Fm[i])-tgt[i])**2 for i in range(4)])
                gfd[k]=(lp-lm)/(2*d)
            gep,_=ep_grad(w); a=np.array([gep[k] for k in WEIGHTS]); b=np.array([gfd[k] for k in WEIGHTS])
            coss.append(float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-18)))
        print(f"NHID={NHID} MEANCOS {np.mean(coss):+.3f} (min {min(coss):+.3f})"); return
    eta0=float(os.environ.get("ETA","0.02")); iters=int(os.environ.get("ITERS","300"))
    mom=float(os.environ.get("MOM","0.5")); sign=float(os.environ.get("USIGN","1"))
    decay=float(os.environ.get("DECAY","0.0"))           # L2 leak: makes the solution a stable attractor
    anneal=os.environ.get("ANNEAL","none")               # none|cos|exp : LR schedule -> stable final iterate
    floor=float(os.environ.get("AFLOOR","0.0"))          # fraction of eta0 the schedule decays to
    freeze=int(os.environ.get("FREEZE","0"))             # controller stops nudging once task is solved w/ margin
    margin=float(os.environ.get("MARGIN","0.4"))         # require min|dout| > margin*TD to call it "solved"
    hold=int(os.environ.get("HOLD","3"))                 # consecutive solved iters before freezing
    nrestart=int(os.environ.get("NRESTART","1"))         # controller retries from a fresh init if an attempt stalls
    stallw=int(os.environ.get("STALLW","80"))            # iters of no best-loss improvement (still unsolved) => stall
    def attempt(w):                                      # run one training attempt; returns (frozen,facc,bacc,o,hist)
        vel={k:0.0 for k in WEIGHTS}; best=9; bacc=0; frozen=False; good=0; lastimp=0; hist=[]; o=None
        for it in range(iters):
            frac=it/max(iters-1,1)
            if   anneal=="cos": eta=eta0*(floor+(1-floor)*0.5*(1+np.cos(np.pi*frac)))
            elif anneal=="exp": eta=eta0*(floor+(1-floor)*float(np.exp(-3*frac)))
            else:               eta=eta0
            g,F=grad(w); o=np.array([dout(F[k]) for k in range(4)])
            loss=float(np.mean((o-tgt)**2)); hist.append(loss)
            acc=float(np.mean((o>0)==(tgt>0)))
            if loss<best: best=loss; bacc=acc; lastimp=it
            # controller's stopping rule: all 4 patterns correct w/ margin for `hold` consecutive reads -> freeze caps.
            good = good+1 if (acc==1.0 and float(np.min(np.abs(o)))>margin*TD) else 0
            if freeze and good>=hold: frozen=True
            if not frozen:
                for k in WEIGHTS:
                    vel[k]=mom*vel[k]+sign*eta*g[k]
                    w[k]=float(np.clip(w[k]+vel[k]-eta*decay*w[k],-3,3))
            if it%30==0 or it==iters-1 or frozen:
                print(f"it{it:3d} loss={loss:.6f} best={best:.6f} acc={acc:.2f} eta={eta:.4f} {'FROZEN ' if frozen else ''}dout(mV)={np.round(o*1e3,1)}")
            if frozen: break
            if acc<1.0 and (it-lastimp)>stallw: print(f"  (stall at it{it}, restarting)"); break
        facc=float(np.mean((o>0)==(tgt>0)))
        return frozen,facc,bacc,o,hist
    print(f"DTCM-TRAIN NHID={NHID} eta={eta0} anneal={anneal} decay={decay} freeze={freeze} restarts<={nrestart} seed={os.environ.get('SEED','0')}")
    facc=0.0; bacc=0.0; o=None; hist=[]; used=0
    for a in range(nrestart):
        used=a+1
        w={k:float(0.4*rng.standard_normal()) for k in WEIGHTS}   # fresh init each attempt (rng advances)
        frozen,facc,bacc,o,hist=attempt(w)
        if frozen or facc==1.0: break                            # solved (and held) -> done
    print(f"FINAL_acc={facc:.2f} (stable) best_acc={bacc:.2f} attempts={used} dout(mV)={np.round(o*1e3,1)} tgt(mV)={np.round(tgt*1e3,1)}")
    np.savetxt("ep_dtcm_loss.dat",np.array(hist))
    if os.environ.get("WSAVE") and facc==1.0:
        np.savetxt(os.environ["WSAVE"],np.array([w[k] for k in WEIGHTS]))
        print("saved weights ("+str(len(WEIGHTS))+") to "+os.environ["WSAVE"]+" order: "+" ".join(WEIGHTS))

if __name__=="__main__": main()
