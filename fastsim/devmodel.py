#!/usr/bin/env python3
"""Device-equation model of pc_deep's cells (LEVEL-1, exact piecewise), following the validated
/tmp/fwd methodology: per-cell internal-node solves (damped fixed-point/bisection), decoupled-rails
output-current extraction, vectorized over instances. Each cell validated against fastsim/char.npz.
Models: NNR NMOS VTO=0.2 KP=120u LAMBDA=0.02; PNR PMOS VTO=-0.2 KP=40u LAMBDA=0.02. W/L in multiples of L=100u.
"""
import numpy as np
LAM=0.02
def nI(Vg,Vd,Vs,VTO=0.2,beta=120e-6,lam=LAM):
    Vlo=np.minimum(Vd,Vs); Vhi=np.maximum(Vd,Vs); sgn=np.where(Vd>=Vs,1.0,-1.0)
    Vgs=Vg-Vlo; Vds=Vhi-Vlo; Vov=Vgs-VTO
    return sgn*np.where(Vov<=0,0.0,np.where(Vds<Vov,beta*(Vov*Vds-0.5*Vds**2)*(1+lam*Vds),0.5*beta*Vov**2*(1+lam*Vds)))
def pI(Vg,Vd,Vs,VTO=-0.2,beta=40e-6,lam=LAM):
    # current flowing source->drain (positive when conducting), Vs is source (high side)
    Vsg=Vs-Vg; Vsd=Vs-Vd; Vov=Vsg+VTO
    return np.where(Vov<=0,0.0,np.where(Vsd<Vov,beta*(Vov*Vsd-0.5*Vsd**2)*(1+lam*Vsd),0.5*beta*Vov**2*(1+lam*Vsd)))

# ---------- branch: series NMOS (gate Vg, drain Vd) + resistor R to node Vb (source side) ----------
def branch_I(Vg,Vd,Vb,R,W=2.0,iters=40):
    """current through [drain Vd] -> NMOS -> source s -> R -> [Vb]; solves s by bisection. Vectorized."""
    beta=120e-6*W
    lo=np.broadcast_to(np.asarray(Vb,float),np.broadcast_shapes(np.shape(Vg),np.shape(Vd),np.shape(Vb))).copy()
    hi=np.maximum(np.broadcast_to(np.asarray(Vd,float),lo.shape), lo)+0.0
    for _ in range(iters):
        s=0.5*(lo+hi)
        f=nI(Vg,Vd,s,beta=beta)-(s-Vb)/R   # transistor current vs resistor current
        # if transistor current > resistor current, source voltage rises
        lo=np.where(f>0,s,lo); hi=np.where(f<=0,s,hi)
    s=0.5*(lo+hi)
    return (s-Vb)/R

# ---------- gsyn: degenerated Gilbert. NESTED BISECTION (monotone at every level) ----------
def _solve_mid(inA,outA,inB,outB,wg,nt,RDEG,itm=30):
    """solve mirror node v (na or nb): KCL  branch(inA,outA->v)+branch(inB,outB->v) = branch(wg, v->nt)."""
    shp=np.broadcast_shapes(*[np.shape(a) for a in (inA,outA,inB,outB,wg,nt)])
    lo=np.broadcast_to(nt,shp).astype(float).copy(); hi=np.full(shp,1.0)
    for _ in range(itm):
        v=0.5*(lo+hi)
        f=branch_I(inA,outA,v,RDEG)+branch_I(inB,outB,v,RDEG)-branch_I(wg,v,nt,RDEG)
        lo=np.where(f>0,v,lo); hi=np.where(f<=0,v,hi)   # net inflow>outflow -> v rises
    return 0.5*(lo+hi)
def gsyn_currents(inp,inn,wp,wn,outp,outn,vbn=0.45,RDEG=12e3,ito=28):
    shp=np.broadcast_shapes(*[np.shape(a) for a in (inp,inn,wp,wn,outp,outn)])
    lo=np.zeros(shp); hi=np.full(shp,0.6)
    for _ in range(ito):
        nt=0.5*(lo+hi)
        na=_solve_mid(inp,outp,inn,outn,wp,nt,RDEG)
        nb=_solve_mid(inp,outn,inn,outp,wn,nt,RDEG)
        f=branch_I(wp,na,nt,RDEG)+branch_I(wn,nb,nt,RDEG)-nI(vbn,nt,0.0,beta=120e-6*4)
        lo=np.where(f>0,nt,lo); hi=np.where(f<=0,nt,hi)
    nt=0.5*(lo+hi)
    na=_solve_mid(inp,outp,inn,outn,wp,nt,RDEG)
    nb=_solve_mid(inp,outn,inn,outp,wn,nt,RDEG)
    Iop=branch_I(inp,outp,na,RDEG)+branch_I(inn,outp,nb,RDEG)
    Ion=branch_I(inn,outn,na,RDEG)+branch_I(inp,outn,nb,RDEG)
    return Iop,Ion

# ---------- cmld load + RNODE: solve output pair given total sunk currents ----------
def cmld_solve(Isink_p,Isink_n,WL=10.0,RCS=300e3,RNODE=20e3,iters=40):
    """outputs (p,n): PMOS loads (gate=cmx=(p+n)/2 via RCS divider ~ average), differential RNODE.
    Solve p,n by bisection on each with the other fixed (few passes)."""
    shp=np.broadcast_shapes(np.shape(Isink_p),np.shape(Isink_n))
    p=np.full(shp,0.6); n=np.full(shp,0.6)
    for _ in range(8):
        cmx=0.5*(p+n)
        # node p: PMOS source=vdd gate=cmx drain=p supplies; RNODE carries (p-n)/RNODE; sink Isink_p
        lo=np.zeros(shp); hi=np.full(shp,1.0)
        for _ in range(iters):
            m=0.5*(lo+hi)
            f=pI(cmx,m,1.0,beta=40e-6*WL)-(m-n)/RNODE-Isink_p
            lo=np.where(f>0,m,lo); hi=np.where(f<=0,m,hi)
        p=0.5*(lo+hi)
        lo=np.zeros(shp); hi=np.full(shp,1.0)
        for _ in range(iters):
            m=0.5*(lo+hi)
            f=pI(cmx,m,1.0,beta=40e-6*WL)-(m-p)/RNODE-Isink_n
            lo=np.where(f>0,m,lo); hi=np.where(f<=0,m,hi)
        n=0.5*(lo+hi)
    return p,n


# ---------- dneuron: diff pair + cmld (WITH RCS paths, joint cmx bisection) ----------
def _outpair(sinkA_fn,sinkB_fn,WL,RCS,ito=24,itm=28):
    """solve (p,n,cmx): KCL p: pI(cmx,p,1)=sinkA(p)+(p-cmx)/RCS; same n; cmx=(p+n)/2 via outer bisection."""
    shp=sinkA_fn(np.array(0.5)).shape if hasattr(sinkA_fn(np.array(0.5)),'shape') else ()
    lo=np.zeros(shp); hi=np.full(shp,1.0)
    for _ in range(ito):
        cmx=0.5*(lo+hi)
        l2=np.zeros(shp); h2=np.full(shp,1.0)
        for _ in range(itm):
            m=0.5*(l2+h2); f=pI(cmx,m,1.0,beta=40e-6*WL)-sinkA_fn(m)-(m-cmx)/RCS
            l2=np.where(f>0,m,l2); h2=np.where(f<=0,m,h2)
        p=0.5*(l2+h2)
        l2=np.zeros(shp); h2=np.full(shp,1.0)
        for _ in range(itm):
            m=0.5*(l2+h2); f=pI(cmx,m,1.0,beta=40e-6*WL)-sinkB_fn(m)-(m-cmx)/RCS
            l2=np.where(f>0,m,l2); h2=np.where(f<=0,m,h2)
        n=0.5*(l2+h2)
        f=0.5*(p+n)-cmx
        lo=np.where(f>0,cmx,lo); hi=np.where(f<=0,cmx,hi)
    return p,n
def dneuron(up,un,vbn=0.35,WL=10.0,RCS=300e3,ito=26,itm=30):
    shp=np.broadcast_shapes(np.shape(up),np.shape(un))
    upb=np.broadcast_to(up,shp); unb=np.broadcast_to(un,shp)
    lo=np.zeros(shp); hi=np.full(shp,0.6)
    for _ in range(ito):
        tn=0.5*(lo+hi)
        ap,an=_outpair(lambda m: nI(unb,m,tn,beta=120e-6*2), lambda m: nI(upb,m,tn,beta=120e-6*2), WL, RCS)
        f=nI(unb,ap,tn,beta=120e-6*2)+nI(upb,an,tn,beta=120e-6*2)-nI(vbn,tn,0.0,beta=120e-6*2)
        lo=np.where(f>0,tn,lo); hi=np.where(f<=0,tn,hi)
    tn=0.5*(lo+hi)
    ap,an=_outpair(lambda m: nI(unb,m,tn,beta=120e-6*2), lambda m: nI(upb,m,tn,beta=120e-6*2), WL, RCS)
    return ap,an
# ---------- esub: 4 inputs, one tail, resistive loads (explicit outputs) ----------
def esub(xp,xn,mp,mn,vbn=0.6,RL=50e3,ito=30):
    shp=np.broadcast_shapes(*[np.shape(a) for a in (xp,xn,mp,mn)])
    lo=np.zeros(shp); hi=np.full(shp,0.8)
    def branches(nt,ep,en):
        I1x=nI(xn,ep,nt,beta=120e-6*2); I2x=nI(xp,en,nt,beta=120e-6*2)
        I1m=nI(mp,ep,nt,beta=120e-6*2); I2m=nI(mn,en,nt,beta=120e-6*2)
        return I1x,I2x,I1m,I2m
    def solve_out(gA,gB,nt,itm=30):   # node v: (1-v)/RL = nI(gA,v,nt)+nI(gB,v,nt)  (monotone -> bisection)
        l2=np.zeros(shp); h2=np.full(shp,1.0)
        for _ in range(itm):
            v=0.5*(l2+h2)
            f=(1.0-v)/RL-nI(gA,v,nt,beta=120e-6*2)-nI(gB,v,nt,beta=120e-6*2)
            l2=np.where(f>0,v,l2); h2=np.where(f<=0,v,h2)
        return 0.5*(l2+h2)
    xpb=np.broadcast_to(xp,shp); xnb=np.broadcast_to(xn,shp); mpb=np.broadcast_to(mp,shp); mnb=np.broadcast_to(mn,shp)
    for _ in range(ito):
        nt=0.5*(lo+hi)
        ep=solve_out(xnb,mpb,nt); en=solve_out(xpb,mnb,nt)
        I1x,I2x,I1m,I2m=branches(nt,ep,en)
        f=(I1x+I2x+I1m+I2m)-nI(vbn,nt,0.0,beta=120e-6*4)
        lo=np.where(f>0,nt,lo); hi=np.where(f<=0,nt,hi)
    nt=0.5*(lo+hi)
    ep=solve_out(xnb,mpb,nt); en=solve_out(xpb,mnb,nt)
    return ep,en

if __name__=="__main__":
    # ---- validate gsyn+cmld against fastsim/char.npz ----
    d=np.load(__import__("os").path.join(__import__("os").path.dirname(__file__),"char.npz")); t=d["t"]
    # reproduce: input ramp segments at 5 weight values; load = cmld(WL=1000u/100u=10)+RNODE 20k
    errs=[]
    for seg in range(5):
        m=(t>seg*100+8)&(t<(seg+1)*100-2)
        inp=d["inp"][m]; wpv=d["wp"][m]
        gop_t=d["gop"][m]; gon_t=d["gon"][m]
        inn=1.0-inp; wnv=1.0-wpv
        # iterate cell<->load (decoupled rails, 3 passes)
        op=np.full(inp.shape,0.6); on=np.full(inp.shape,0.6)
        for _ in range(4):
            Ip,In=gsyn_currents(inp,inn,wpv,wnv,op,on,vbn=0.45)
            op,on=cmld_solve(Ip,In)
        pred=op-on; meas=gop_t-gon_t
        e=float(np.abs(pred-meas).max()); errs.append(e)
        print(f"seg{seg} w_diff={2*(wpv[0]-0.5):+.2f}: max|err|={e*1000:.1f}mV  (meas range {meas.min():+.3f}..{meas.max():+.3f}, pred {pred.min():+.3f}..{pred.max():+.3f})")
    print(f"GSYN+CMLD validation: worst {max(errs)*1000:.1f}mV")
    # ---- dneuron validation ----
    m=(t>5)
    up=d["dinp"][m]; un=1.0-up
    ap,an=dneuron(up,un)
    pred=ap-an; meas=d["dap"][m]-d["dan"][m]
    print(f"DNEURON validation: max|err|={np.abs(pred-meas).max()*1000:.1f}mV (meas {meas.min():+.3f}..{meas.max():+.3f})")
    # ---- esub validation ----
    xp=d["exp"][m]; xn=1.0-xp
    ep,en=esub(xp,xn,np.full(xp.shape,0.5),np.full(xp.shape,0.5))
    pred=ep-en; meas=d["eep"][m]-d["een"][m]
    print(f"ESUB validation: max|err|={np.abs(pred-meas).max()*1000:.1f}mV (meas {meas.min():+.3f}..{meas.max():+.3f})")
