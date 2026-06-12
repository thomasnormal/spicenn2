#!/usr/bin/env python3
"""Figure: continuous-time END-TO-END EP (weights on caps, physical update cells, one .tran).
A: minimal 1-weight EP regression trains from scratch -> output cap tracks target, both signs.
B: the continuous net REPRESENTS XOR -> per-pattern output (sign=class) with trained weights.
C: the differential-only physical update (gupd2) HOLDS the XOR solution in-circuit over epochs.
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
TH=150.0
fig,ax=plt.subplots(1,3,figsize=(15,4.3))

# A: min regression, both signs
for tsw,c in [(0.12,"C0"),(-0.12,"C3")]:
    d=np.loadtxt(f"e2e_min_{tsw}.dat"); t=d[:,0]*1e6; o=d[:,5]-d[:,7]
    ax[0].plot(t,o*1e3,c,lw=0.8,label=f"output (target {2*tsw*1e3:+.0f}mV)")
    ax[0].axhline(2*tsw*1e3,color=c,ls=":",lw=1)
ax[0].axhline(0,color="k",lw=0.4)
ax[0].set_title("A. minimal EP regression trains from scratch\n(weight on cap, physical update, one .tran)")
ax[0].set_xlabel("time (us)"); ax[0].set_ylabel("output differential (mV)"); ax[0].legend(fontsize=8,loc="center right")

# B: XOR representation (trained weights, forward in .tran)
d=np.loadtxt("e2e_o_fwd.dat"); t=d[:,0]*1e9; o=d[:,1]-d[:,3]; vt=d[:,9]
PAT=["00","01","10","11"]; outs=[]; tgs=[]
for m in range(4):
    g=4+m; i=np.argmin(np.abs(t-(g*2*TH+TH-2))); outs.append((o[i])*1e3); tgs.append(1 if vt[i]>0.5 else -1)
cols=["C2" if (outs[m]>0)==(tgs[m]>0) else "C3" for m in range(4)]
ax[1].bar(range(4),outs,color=cols)
for m in range(4): ax[1].plot(m, tgs[m]*90,"k_",ms=18,mew=2)
ax[1].axhline(0,color="k",lw=0.6); ax[1].set_xticks(range(4)); ax[1].set_xticklabels([f"x={p}" for p in PAT])
ax[1].set_title("B. continuous net REPRESENTS XOR (4/4)\nper-pattern output, sign=class (— = target)")
ax[1].set_ylabel("output differential (mV)")

# C: gupd2 holds the XOR solution
d=np.loadtxt("e2e_o_hold.dat"); t=d[:,0]*1e9; o=d[:,1]-d[:,3]; vt=d[:,9]
NE=40; accs=[]; mars=[]
for e in range(NE):
    ok=0; mar=1e9
    for m in range(4):
        g=e*4+m; i=np.argmin(np.abs(t-(g*2*TH+TH-2))); ok+=((o[i]>0)==(vt[i]>0.5)); mar=min(mar,abs(o[i]*1e3))
    accs.append(ok); mars.append(mar)
axa=ax[2]; axb=axa.twinx()
axa.plot(range(NE),accs,"C0-o",ms=3,label="accuracy (/4)"); axa.set_ylim(0,4.3)
axb.plot(range(NE),mars,"C1-",lw=1,label="min margin (mV)")
axa.set_title("C. physical update HOLDS XOR in-circuit\n(differential-only cell gupd2)")
axa.set_xlabel("epoch"); axa.set_ylabel("accuracy (/4)",color="C0"); axb.set_ylabel("min |output| (mV)",color="C1")
plt.tight_layout(); plt.savefig("ep_e2e.png",dpi=110); print("wrote ep_e2e.png")
print(f"  B: XOR outputs(mV)={np.round(outs,1)} targets={tgs}")
print(f"  C: hold accuracy {accs[0]}/4 -> {accs[-1]}/4 over {NE} epochs")
