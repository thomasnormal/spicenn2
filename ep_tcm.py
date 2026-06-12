#!/usr/bin/env python3
"""
FULLY-TRANSISTOR current-mode EP (zero behavioral sources).
Cells:  OTA transconductor synapse (5T, validated tc_cell.cir) + diff-pair neuron (validated
        neuron2.cir).  State node = leak resistor (RC).  Signed weight = differential OTA pair
        (w+ driven by a_pre, w- by ac_pre) sharing one stored signed weight via gate voltages.
Symmetric hidden<->output coupling (forward+backward OTAs share weight) -> valid energy.
Only the local correlation weight-update runs on the controller (reads node voltages, writes
gate voltages). Everything in the circuit is MOSFETs/resistors.
"""
import numpy as np, subprocess, os
NHID=int(os.environ.get("NHID","4"))
RLEAK=os.environ.get("RLEAK","150k")
RB=float(os.environ.get("RB","1e5")); BETA=1.0/RB
VW0=float(os.environ.get("VW0","0.5")); KMAP=float(os.environ.get("KMAP","0.10"))
VBN=os.environ.get("VBN","0.30"); RLN=os.environ.get("RLN","800k")
ALO,AHI=0.2,0.8; TLO,THI=float(os.environ.get("TLO","0.4")),float(os.environ.get("THI","0.6"))
PATS=[(0,0,0),(0,1,1),(1,0,1),(1,1,0)]
def xv(b): return AHI if b else ALO
def tv(b): return THI if b else TLO
HID=[f"h{j}" for j in range(1,NHID+1)]
# signed weights (scalars): input i->hidden j (w_i_j); hidden j<->output (o_j); biases bh_j, bo
WEIGHTS=[]
for j in range(1,NHID+1):
    for i in (1,2): WEIGHTS.append(f"w{i}{j}")
    WEIGHTS+= [f"o{j}", f"bh{j}"]
WEIGHTS.append("bo")
READ=[f"a_h{j}" for j in range(1,NHID+1)]+["u_o","a_o"]

SUB=""".subckt ota inp outp wg vdd vref
M1 d1 inp tl 0 NNR W=200u L=100u
M2 outp vref tl 0 NNR W=200u L=100u
Mt tl wg 0 0 NNR W=200u L=100u
M3 d1 d1 vdd vdd PNR W=500u L=100u
M4 outp d1 vdd vdd PNR W=500u L=100u
.ends
.subckt neuron u a ac vdd vref vbn
M1 ac u  tn 0 NNR W=200u L=100u
M2 a  vref tn 0 NNR W=200u L=100u
Mt tn vbn 0 0 NNR W=200u L=100u
RL1 vdd ac {RLN}
RL2 vdd a  {RLN}
.ends
""".replace("{RLN}",RLN)

def wg(w):  # signed weight -> (gate+, gate-) tail voltages
    return f"{np.clip(VW0+KMAP*w,0.25,0.85):.4f}", f"{np.clip(VW0-KMAP*w,0.25,0.85):.4f}"

def signed_syn(tag, a_pre, ac_pre, outp, wkey, w):
    gp,gn=wg(w[wkey])
    return [f"Vgp_{tag} gp_{tag} 0 {gp}", f"Vgn_{tag} gn_{tag} 0 {gn}",
            f"Xp_{tag} {a_pre} {outp} gp_{tag} vdd vref ota",
            f"Xn_{tag} {ac_pre} {outp} gn_{tag} vdd vref ota"]

def deck(w):
    L=["fully-transistor current-mode EP",
       ".model NNR NMOS (LEVEL=1 VTO=0.2  KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)",
       ".model PNR PMOS (LEVEL=1 VTO=-0.2 KP=40u  LAMBDA=0.02 GAMMA=0 PHI=0.7)",
       SUB, "Vdd vdd 0 1.0","Vref vref 0 0.5","Vbn vbn 0 "+VBN,"Vmid nmid 0 0.5",
       "Vt vt 0 0.5","Vhi vhi 0 0.8","Vlo vlo 0 0.2",
       "Va_x1 a_x1 0 0.5","Vac_x1 ac_x1 0 0.5","Va_x2 a_x2 0 0.5","Vac_x2 ac_x2 0 0.5"]
    # hidden neurons + their incoming synapses
    for j in range(1,NHID+1):
        u=f"u_h{j}"
        L.append(f"Rlk_h{j} {u} nmid {RLEAK}")
        L.append(f"Xn_h{j} {u} a_h{j} ac_h{j} vdd vref vbn neuron")
        for i in (1,2):
            L+=signed_syn(f"i{i}{j}",f"a_x{i}",f"ac_x{i}",u,f"w{i}{j}",w)
        # backward from output (shared weight o_j)
        L+=signed_syn(f"bo{j}","a_o","ac_o",u,f"o{j}",w)
        # bias
        L+=signed_syn(f"bh{j}","vhi","vlo",u,f"bh{j}",w)
    # output neuron + forward synapses from hidden + bias
    L.append(f"Rlk_o u_o nmid {RLEAK}")
    L.append("Xn_o u_o a_o ac_o vdd vref vbn neuron")
    for j in range(1,NHID+1):
        L+=signed_syn(f"fo{j}",f"a_h{j}",f"ac_h{j}","u_o",f"o{j}",w)
    L+=signed_syn("bo_","vhi","vlo","u_o","bo",w)
    L.append("Rb u_o vt 1e10")
    L.append(".control"); cols=" ".join(f"v({n})" for n in READ)
    for k,(p,q,_) in enumerate(PATS):
        x1,x2=xv(p),xv(q)
        L+=[f"alter Va_x1 {x1}",f"alter Vac_x1 {round(1.0-x1,3)}",
            f"alter Va_x2 {x2}",f"alter Vac_x2 {round(1.0-x2,3)}",
            "alter Rb 1e10","op",f"wrdata tf{k}.dat {cols}"]
        L+=[f"alter Vt {tv(PATS[k][2])}",f"alter Rb {RB}","op",f"wrdata tn{k}.dat {cols}"]
    L+=[".endc",".end"]; return "\n".join(L)+"\n"

def rd(fn): return np.loadtxt(fn).reshape(-1)[1::2]
def run(w):
    open("ep_tcm.cir","w").write(deck(w))
    subprocess.run(["ngspice","-b","ep_tcm.cir"],capture_output=True,text=True,timeout=180)
    return (np.array([rd(f"tf{k}.dat") for k in range(4)]),
            np.array([rd(f"tn{k}.dat") for k in range(4)]))
idx={n:k for k,n in enumerate(READ)}; OI=idx['u_o']

def corr_pairs():
    P={k:[] for k in WEIGHTS}
    for j in range(1,NHID+1):
        for i in (1,2): P[f"w{i}{j}"].append((f"x{i}",f"a_h{j}"))
        P[f"o{j}"].append((f"a_h{j}","a_o"))          # fwd+back share o_j -> one correlation
        P[f"bh{j}"].append(("HI",f"a_h{j}"))
    P["bo"].append(("HI","a_o"))
    return P
CP=corr_pairs()
def phi(tok,arr,pat):
    if tok=="HI": return 0.8-0.5
    if tok=="x1": return xv(pat[0])-0.5
    if tok=="x2": return xv(pat[1])-0.5
    return arr[idx[tok]]-0.5
def ep_grad(w):
    F,N=run(w); g={k:0.0 for k in WEIGHTS}
    for k,pat in enumerate(PATS):
        for key in WEIGHTS:
            for (s,d) in CP[key]:
                f=phi(s,F[k],pat)*phi(d,F[k],pat); n=phi(s,N[k],pat)*phi(d,N[k],pat)
                g[key]+=((n-f)/BETA)/4.0
    return g,F

def main():
    import sys; mode=sys.argv[1] if len(sys.argv)>1 else "train"
    rng=np.random.default_rng(int(os.environ.get("SEED","0")))
    tgt=np.array([tv(p[2]) for p in PATS])
    if mode=="gradcheck":
        d=0.05; coss=[]
        for t in range(int(os.environ.get("T","4"))):
            w={k:float(0.5*rng.standard_normal()) for k in WEIGHTS}; gfd={}
            for k in WEIGHTS:
                wp=dict(w);wp[k]+=d; wm=dict(w);wm[k]-=d
                Fp,_=run(wp); Fm,_=run(wm)
                gfd[k]=(np.mean((Fp[:,OI]-tgt)**2)-np.mean((Fm[:,OI]-tgt)**2))/(2*d)
            gep,_=ep_grad(w); a=np.array([gep[k] for k in WEIGHTS]); b=np.array([gfd[k] for k in WEIGHTS])
            coss.append(float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-18)))
        print(f"NHID={NHID} MEANCOS {np.mean(coss):+.3f} (min {min(coss):+.3f}) [|FD|max {np.abs(b).max():.1e}]"); return
    if mode=="fwd":
        w={k:float(0.5*rng.standard_normal()) for k in WEIGHTS}
        F,_=run(w); print("u_o per pattern:",np.round(F[:,OI],3),"a_o:",np.round(F[:,idx['a_o']],3)); return
    eta=float(os.environ.get("ETA","0.05")); iters=int(os.environ.get("ITERS","200"))
    mom=float(os.environ.get("MOM","0.85")); sign=float(os.environ.get("USIGN","1"))
    w={k:float(0.4*rng.standard_normal()) for k in WEIGHTS}; vel={k:0.0 for k in WEIGHTS}
    best=9; best_uo=None; best_it=-1; hist=[]
    print(f"TCM-TRAIN NHID={NHID} eta={eta} seed={os.environ.get('SEED','0')} tgt={TLO}/{THI}")
    for it in range(iters):
        g,F=ep_grad(w); o=F[:,OI]; loss=float(np.mean((o-tgt)**2)); hist.append(loss)
        if loss<best: best=loss; best_uo=o.copy(); best_it=it
        for k in WEIGHTS: vel[k]=mom*vel[k]+sign*eta*g[k]; w[k]=float(np.clip(w[k]+vel[k],-3,3))
        if it%20==0 or it==iters-1:
            acc=float(np.mean((o>0.5)==(tgt>0.5)))
            print(f"it{it:3d} loss={loss:.5f} best={best:.5f} acc={acc:.2f} u_o={np.round(o,3)}")
    bacc=float(np.mean((best_uo>0.5)==(tgt>0.5)))
    print(f"FINAL best_loss={round(best,5)} @it{best_it} BEST_acc={bacc:.2f} best_u_o={np.round(best_uo,3)} tgt={tgt}")
    np.savetxt("ep_tcm_loss.dat",np.array(hist))

if __name__=="__main__": main()
