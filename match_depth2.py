#!/usr/bin/env python3
# Decisive diagnostic: train the EXACT gen_deep depth-2 topology (4x4 -> 2x2x4 -> 1x1x16 -> 3) with
# IDEAL backprop (PyTorch) on the same 3-class 4x4 zscore data the circuit uses. If the ideal ceiling
# is ~66% too, the circuit is capacity-limited; if it's ~85%, the circuit's transpose backward is the gap.
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
tr=np.load("digits_train.npz"); te=np.load("digits_test.npz")
Xtr=torch.tensor(tr["X"],dtype=torch.float32); ytr=torch.tensor(tr["Y"]);
Xte=torch.tensor(te["X"],dtype=torch.float32); yte=torch.tensor(te["Y"]); C=3
def layers(RES,ARCH):
    out=[]; H=W=RES; Cin=1
    for part in ARCH.split(","):
        Cout,ks,st,fi=[int(x) for x in part.split(":")]
        Hout=(H-ks)//st+1; Wout=(W-ks)//st+1; nin=H*W*Cin; nout=Hout*Wout*Cout
        conn=[[] for _ in range(nout)]; r=np.random.default_rng(len(out)+1)
        for io in range(Hout):
          for jo in range(Wout):
            rf=[(io*st+di)*W*Cin+(jo*st+dj)*Cin+c for di in range(ks) for dj in range(ks) for c in range(Cin)]
            rf=np.array(rf); fo=np.zeros(len(rf))
            for co in range(Cout):
                o=(io*Wout+jo)*Cout+co
                cand=sorted(range(len(rf)),key=lambda x:(fo[x],r.random()))[:fi]
                for ci in cand: conn[o].append(int(rf[ci])); fo[ci]+=1
        out.append((conn,nin,nout)); H,W,Cin=Hout,Wout,Cout
    return out,H*W*Cin
class Sparse(nn.Module):
    def __init__(self,conn,nin,nout):
        super().__init__(); m=torch.zeros(nout,nin)
        for o,ins in enumerate(conn):
            for i in ins: m[o,i]=1
        self.register_buffer("m",m); self.w=nn.Parameter(torch.randn(nout,nin)*0.4*m); self.b=nn.Parameter(torch.zeros(nout))
    def forward(self,x): return F.linear(x,self.w*self.m,self.b)
hl,nfeat=layers(4,"4:2:2:4,16:2:2:4")
L1=Sparse(*hl[0]); L2=Sparse(*hl[1])
r=np.random.default_rng(99); oc=[list(r.choice(nfeat,4,replace=False)) for _ in range(C)]
om=torch.zeros(C,nfeat)
for o,ins in enumerate(oc):
    for i in ins: om[o,i]=1
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.L1=L1; s.L2=L2; s.register_buffer("om",om); s.ow=nn.Parameter(torch.randn(C,nfeat)*0.3*om); s.ob=nn.Parameter(torch.zeros(C))
    def forward(s,x): return F.linear(torch.tanh(s.L2(torch.tanh(s.L1(x)))), s.ow*s.om, s.ob)
net=Net(); opt=torch.optim.Adam(net.parameters(),lr=3e-3)
for ep in range(400):
    opt.zero_grad(); F.cross_entropy(net(Xtr),ytr).backward(); opt.step()
acc=(net(Xte).argmax(1)==yte).float().mean().item()
tracc=(net(Xtr).argmax(1)==ytr).float().mean().item()
print(f"IDEAL backprop, exact depth-2 topo (4x4->2x2x4->1x1x16->3): train={tracc*100:.1f}% test={acc*100:.1f}%")
print(f"  (circuit got 66.7% train. If ideal~66 -> capacity; if ideal>>66 -> transpose fidelity is the gap)")
