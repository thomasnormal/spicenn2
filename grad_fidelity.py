#!/usr/bin/env python3
# DIAGNOSTIC: is the analog deep gradient pointing the right way, per layer?
# Hold the output error e = divis(out)-T FIXED. Then the true gradient the chip's transpose-read
# is approximating is  g_exact_l = d/dW_l  <e, out>  (= J^T e), which we get by finite differences
# through the REAL analog forward. Compare its DIRECTION to the chip's approximate backward
# (transpose W^T + step-mask ReLU'). Cosine per layer = backward fidelity vs depth.
import numpy as np
import deep_fixes as df
from mc_experiment import scale_fit, scale_apply, onehot
from surrogate import VC, Gs, VREFH, RTH, VREFO, RTO, HI, VG0, VT_REL, BETA_REL, RH
clamp=2.6

def relu_g(z):
    I=0.5*BETA_REL*np.clip(z-VT_REL,0,None)**2
    return np.clip(VG0+RH*I, VG0, clamp)
def divis(Y): r=np.clip(Y,0,None); return r/(r.sum(1,keepdims=True)+1e-6)

def forward(Ws,X):
    acts=[X]; zs=[]; A=X; L=len(Ws)
    for l,Wl in enumerate(Ws):
        AB=np.concatenate([A,np.full((len(A),1),HI)],1); gp=VC+Gs*Wl
        last=(l==L-1); vref,R=(VREFO,RTO) if last else (VREFH,RTH)
        Z=df.keystone(gp,AB,vref,R); zs.append(Z); A=Z if last else relu_g(Z); acts.append(A)
    return acts,zs

def approx_grad(Ws,acts,zs,e,relup='step'):
    # chip's transpose-read backward of the fixed output error e.
    # relup: 'step' = comparator (current chip); 'graded' = true ReLU^2 derivative ~ (z-Vt).
    L=len(Ws); gr=[None]*L; d=e; B=len(e)
    for l in reversed(range(L)):
        AB=np.concatenate([acts[l],np.full((B,1),HI)],1); last=(l==L-1)
        if last: dz=d
        elif relup=='graded': dz=d*(RH*BETA_REL*np.clip(zs[l]-VT_REL,0,None))
        else:                  dz=d*np.where(zs[l]>VT_REL,1.0,0.0)
        gr[l]=(dz.T@AB)/B
        if l>0: d=dz@Ws[l][:,:-1]
    return gr

def exact_grad(Ws,X,e,eps=1e-3):
    # g_exact_l[i,j] = d/dW_l[i,j]  mean_b sum_c e[b,c]*out[b,c]   (e held fixed)
    B=len(X)
    def s_of(Ws_):
        out=forward(Ws_,X)[0][-1]; return (e*out).sum(1).mean()
    gr=[np.zeros_like(W) for W in Ws]
    for l in range(len(Ws)):
        W=Ws[l]
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                Wp=[w.copy() for w in Ws]; Wm=[w.copy() for w in Ws]
                Wp[l][i,j]+=eps; Wm[l][i,j]-=eps
                gr[l][i,j]=(s_of(Wp)-s_of(Wm))/(2*eps)
    return gr

def cos(a,b):
    a=a.flatten(); b=b.flatten(); return float(a@b/((np.linalg.norm(a)*np.linalg.norm(b))+1e-12))

if __name__=="__main__":
    C=2
    Xtr,Ytr=df.make_blobs(C,6,40,1,0.9) if hasattr(df,'make_blobs') else (None,None)
    # use checkerboard 2x2 (mildly compositional) so weights are in a real regime
    from deep_super import make_checker
    Xtr,Ytr=make_checker(200,1,2)
    p=scale_fit(Xtr); Xtr=scale_apply(Xtr,p); T=onehot(Ytr,C)
    for depth,sizes in [(1,[6,8,2]),(2,[6,8,8,2]),(3,[6,8,8,8,2]),(4,[6,8,8,8,8,2])]:
        # take partially-trained weights (realistic regime), via deep_fixes
        df.Xtr,df.Ytr,df.Xte,df.Yte=Xtr,Ytr,Xtr,Ytr; df.C=C; df.T=T; df.B=len(Xtr); df.fn=df.NORMS['divisive']
        Ws=df.init_W(sizes,0); Ws=[np.clip(W,-1.5,1.0) for W in Ws]
        # a few real update steps so we're not at a trivial init
        V=[np.zeros_like(W) for W in Ws]
        for _ in range(60):
            acts,zs=forward(Ws,Xtr); e=divis(acts[-1])-T; g=approx_grad(Ws,acts,zs,e)
            for l in range(len(Ws)): V[l]=0.9*V[l]-0.4*g[l]; Ws[l]=np.clip(Ws[l]+np.clip(V[l],-0.1,0.1),-1.5,1.0)
        acts,zs=forward(Ws,Xtr); e=divis(acts[-1])-T
        ge=exact_grad(Ws,Xtr,e)
        ga_s=approx_grad(Ws,acts,zs,e,'step'); ga_g=approx_grad(Ws,acts,zs,e,'graded')
        cs=[round(cos(ga_s[l],ge[l]),3) for l in range(len(Ws))]
        cg=[round(cos(ga_g[l],ge[l]),3) for l in range(len(Ws))]
        print(f"depth {depth} {str(sizes):16s}")
        print(f"   cos vs exact, STEP ReLU' (chip): {cs}")
        print(f"   cos vs exact, GRADED ReLU'     : {cg}")
