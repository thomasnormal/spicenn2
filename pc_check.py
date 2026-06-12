#!/usr/bin/env python3
"""ALGORITHM CHECK (numpy, NOT spice): does supervised Predictive Coding with the LOCAL rule
(weight update = error . presynaptic-activity, hard-clamped output) train XOR?  This only locks the
formulation before I build it in transistors.  No two-phase, no SGD-in-a-loop dressing -- the weight
step IS the local product (a learning-rate = the cap time-constant later).

Layers: x0=input(2, clamped), x1=hidden(H), x2=output(1, clamped to label during learning).
Prediction mu_l = W_l f(x_{l-1}); error eps_l = x_l - mu_l.
Inference: relax x1 to min sum eps^2.  Learning: dW_l ~ eps_l (x) f(x_{l-1}).
"""
import numpy as np, os
H=int(os.environ.get("H","6")); SEEDS=int(os.environ.get("SEEDS","20"))
ITERS=int(os.environ.get("ITERS","400")); NRELAX=int(os.environ.get("NRELAX","30"))
LR=float(os.environ.get("LR","0.05")); LRX=float(os.environ.get("LRX","0.2"))
CONT=int(os.environ.get("CONT","0"))   # 1 = update weights DURING relaxation (continuous), 0 = after settle
f=np.tanh; fp=lambda z:1-np.tanh(z)**2
X=np.array([[-1,-1],[-1,1],[1,-1],[1,1]],float); Y=np.array([-1,1,1,-1.0])  # XOR in +-1
def run(seed):
    rng=np.random.default_rng(seed)
    W1=rng.standard_normal((H,2))*0.5; b1=np.zeros(H)
    W2=rng.standard_normal((1,H))*0.5; b2=np.zeros(1)
    def infer(x,y=None,clamp=True,steps=NRELAX,learn=False):
        x1=f(W1@x+b1)                       # init hidden at feedforward
        for t in range(steps):
            pred1=W1@x+b1; eps1=x1-pred1     # hidden prediction error (bottom-up)
            x2 = np.array([y]) if (clamp and y is not None) else (W2@f(x1)+b2)
            pred2=W2@f(x1)+b2; eps2=x2-pred2 # output error
            dx1 = -eps1 + fp(x1)*(W2.T@eps2)
            x1 = x1 + LRX*dx1
            if learn and CONT:               # continuous: nudge weights a little each relax step
                _upd(eps1,eps2,x,x1,LR/steps)
        return x1,eps1,eps2
    def _upd(eps1,eps2,x,x1,lr):
        nonlocal W1,b1,W2,b2
        W2+=lr*np.outer(eps2,f(x1)); b2+=lr*eps2
        W1+=lr*np.outer(eps1,x);     b1+=lr*eps1
    for it in range(ITERS):
        for k in range(4):
            x1,eps1,eps2=infer(X[k],Y[k],clamp=True,learn=True)
            if not CONT: _upd(eps1,eps2,X[k],x1,LR)   # settle-then-update
    # test: feedforward (output free)
    ok=0
    for k in range(4):
        o=(W2@f(W1@X[k]+b1)+b2)[0]
        ok+=((o>0)==(Y[k]>0))
    return ok
res=[run(s) for s in range(SEEDS)]
print(f"PC numpy XOR (H={H} CONT={CONT}): {sum(r==4 for r in res)}/{SEEDS} solved 4/4   dist={np.bincount(res,minlength=5).tolist()}")
