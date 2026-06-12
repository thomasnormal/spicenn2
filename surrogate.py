#!/usr/bin/env python3
# Fast circuit-faithful surrogate of the analog trainer.
# Device-level LEVEL-1 MOSFET multiply (with dead-zone), current-difference keystone,
# quadratic ReLU. Calibrated against the ngspice XOR forward.
import numpy as np, json

# ---- device constants (from the SPICE .model cards) ----
VC=1.8; Gs=1.2
VT_SYN=0.2; BETA_SYN=50e-6*10          # NSYN: KP*W/L
VREFH=0.5; RTH=8e3; VG0=0.7; RH=8e3
VT_REL=0.5; BETA_REL=120e-6*12         # NREL: KP*W/L
VREFO=0.5; RTO=7e3
HI=1.3; LO=0.7

def iDS(vg, vd, vs, beta=BETA_SYN, vt=VT_SYN):
    """LEVEL-1 NMOS conventional drain->source current (signed, GAMMA=0), handles reverse."""
    vd=np.asarray(vd,float); vs=np.asarray(vs,float)
    s=np.where(vd>=vs,1.0,-1.0); hi=np.maximum(vd,vs); lo=np.minimum(vd,vs)
    vov=vg-lo-vt; vds=hi-lo
    tri=beta*(vov*vds-0.5*vds*vds); sat=0.5*beta*vov*vov
    return s*np.where(vov<=0,0.0,np.where(vds<vov,tri,sat))

def relu_act(z, clamp=2.6):
    I=0.5*BETA_REL*np.clip(z-VT_REL,0,None)**2
    return np.clip(VG0+RH*I, VG0, clamp)

def diode(col, beta_d=120e-6*20, vt_d=0.5):
    """NMIR diode-connected device current (gate=drain=col)."""
    vov=col-vt_d
    return np.where(vov<=0,0.0,0.5*beta_d*vov*vov)

def solve_col(gp, x, iters=44):
    """Solve column virtual ground: sum_i iDS(gp_i,x_i,col) = diode(col). Vectorized over neurons.
    gp:(Nout,Nin) gate voltages, x:(Nin,). Returns (col, I_col) each (Nout,)."""
    gp=np.atleast_2d(np.asarray(gp,float)); x=np.asarray(x,float)
    lo=np.full(gp.shape[0],0.5+1e-4); hi=np.full(gp.shape[0],3.0)
    for _ in range(iters):
        mid=0.5*(lo+hi)
        f=iDS(gp, x[None,:], mid[:,None]).sum(1)-diode(mid)
        hi=np.where(f<0,mid,hi); lo=np.where(f>=0,mid,lo)
    col=0.5*(lo+hi)
    return col, diode(col)

def keystone(gp, x, kgain, vref, R):
    """gp:(Nout,Nin) gate voltages, x:(Nin,) inputs -> column output voltage (Nout,).
    kgain is the lumped current-mirror ratio (the only fit constant); grounds solved self-consistently."""
    _,Icolp=solve_col(gp, x)
    _,Icoln=solve_col(VC*np.ones((1,len(x))), x)   # shared negative reference
    return vref + kgain*R*(Icolp-Icoln[0])

def xor_forward(W, col0, clamp=2.6):
    gp_h=W[0:18].reshape(6,3); gp_o=W[18:25][None,:]   # output: 1 neuron, 7 inputs (6 hv + bias)
    pats=[(LO,LO),(LO,HI),(HI,LO),(HI,HI)]
    Y=[]; HV=[]
    for a1,a2 in pats:
        xh=np.array([a1,a2,HI])
        z1=keystone(gp_h, xh, col0, VREFH, RTH)
        hv=relu_act(z1, clamp)
        xo=np.concatenate([hv,[HI]])
        y=keystone(gp_o, xo, col0, VREFO, RTO)[0]
        Y.append(y); HV.append(hv)
    return np.array(Y), np.array(HV)

def xor_forward(W, kgain, clamp=2.6):
    gp_h=W[0:18].reshape(6,3); gp_o=W[18:25][None,:]   # output: 1 neuron, 7 inputs (6 hv + bias)
    pats=[(LO,LO),(LO,HI),(HI,LO),(HI,HI)]
    Y=[]; HV=[]
    for a1,a2 in pats:
        xh=np.array([a1,a2,HI])
        z1=keystone(gp_h, xh, kgain, VREFH, RTH)
        hv=relu_act(z1, clamp)
        xo=np.concatenate([hv,[HI]])
        y=keystone(gp_o, xo, kgain, VREFO, RTO)[0]
        Y.append(y); HV.append(hv)
    return np.array(Y), np.array(HV)

if __name__=="__main__":
    W=np.load('xor_W.npy'); gt=json.load(open('xor_gt.json'))
    gtY=np.array([gt[str(k)][0] for k in range(4)]); gtHV=np.array([gt[str(k)][1] for k in range(4)])
    best=None
    for kg in np.arange(0.3,1.01,0.02):
        for clamp in np.arange(2.2,3.1,0.2):
            Y,HV=xor_forward(W,kg,clamp)
            err=np.mean((Y-gtY)**2)+np.mean((HV-gtHV)**2)
            if best is None or err<best[0]: best=(err,kg,clamp,Y,HV)
    err,kg,clamp,Y,HV=best
    print(f"best kgain={kg:.2f} clamp={clamp:.1f} fit-err={err:.4f}")
    print("surrogate y:", np.round(Y,3)); print("ngspice  y:", np.round(gtY,3))
    for k in range(4):
        print(f"  pat{k} hv surrogate {np.round(HV[k],2)}  ngspice {np.round(gtHV[k],2)}")
    np.save('cal.npy', np.array([kg,clamp]))
