#!/usr/bin/env python3
"""Decisive diagnostic: is the two-phase EP gradient correct in the recurrent transistor net?
Compare EP gradient estimate (from free vs nudged phases) to the true finite-difference
gradient of the free-phase loss wrt each weight. High positive correlation => gradient is
right and the failure is optimization; low/none => the EP setup itself is wrong."""
import numpy as np, os, ep_xor_train as E

idx={n:k for k,n in enumerate(E.READ)}; oi=idx[E.OUT]
tgt=np.array([E.tvl(p[2]) for p in E.PATS]); beta=E.BEFF

def loss_of(vc):
    F,_=E.run(vc); o=F[:,oi]; return float(np.mean((o-tgt)**2)), o

DSTACT={f'u_{n}':f'a_{n}' for n in E.NEURONS}   # neuron activation fed by each u-node

def ep_grad(vc, rule="dv2"):
    F,N=E.run(vc); g={w:0.0 for w in E.WEIGHTS}
    for i,(p,q,_) in enumerate(E.PATS):
        x1,x2=E.vl(p),E.vl(q)
        val={'a_x1':x1,'a_x1c':round(1-x1,3),'a_x2':x2,'a_x2c':round(1-x2,3),'vhi':1.0,'vlo':0.0}
        def nv(node,arr): return val[node] if node in val else arr[idx[node]]
        for (na,nb,w) in E.SYN:
            if rule=="dv2":
                df=(nv(na,F[i])-nv(nb,F[i]))**2; dn=(nv(na,N[i])-nv(nb,N[i]))**2
            else:  # Hopfield activation-correlation: src_activation * dst_neuron_activation
                da=DSTACT[nb]
                df=nv(na,F[i])*nv(da,F[i]); dn=nv(na,N[i])*nv(da,N[i])
            g[w]+=((dn-df)/beta)/4.0
    return g

rng=np.random.default_rng(int(os.environ.get("SEED","3")))
delta=0.01; quiet=os.environ.get("QUIET","0")=="1"
coss=[]
for trial in range(int(os.environ.get("T","3"))):
    vc={w:float(np.clip(0.1+0.2*rng.standard_normal(),E.VCMIN,E.VCMAX)) for w in E.WEIGHTS}
    gfd={}
    for w in E.WEIGHTS:
        vp=dict(vc); vp[w]+=delta; vm=dict(vc); vm[w]-=delta
        lp,_=loss_of(vp); lm,_=loss_of(vm); gfd[w]=(lp-lm)/(2*delta)
    b=np.array([gfd[w] for w in E.WEIGHTS])
    out=f"trial{trial}: "
    for rule in ("dv2","corr"):
        gep=ep_grad(vc,rule); a=np.array([gep[w] for w in E.WEIGHTS])
        cos=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-18))
        out+=f"  cos_{rule}={cos:+.3f}"
        if rule=="dv2": coss.append(cos)
    if not quiet: print(out)
print(f"MEANCOS_dv2 {np.mean(coss):+.3f} (min {min(coss):+.3f})")
