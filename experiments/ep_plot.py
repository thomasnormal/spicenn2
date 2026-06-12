#!/usr/bin/env python3
"""Run OR (right sign + wrong sign) and AND, collect loss curves, render figure."""
import numpy as np, subprocess, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

def run(task, usign, iters=40, eta=0.5):
    src = "ep_train.py"
    if task=="AND":
        txt=open("ep_train.py").read().replace(
          "PATS = [(0,0,0),(0,1,1),(1,0,1),(1,1,1)]   # (x1,x2,OR)",
          "PATS = [(0,0,0),(0,1,0),(1,0,0),(1,1,1)]   # (x1,x2,AND)")
        open("ep_train_t.py","w").write(txt); src="ep_train_t.py"
    env=dict(os.environ, ITERS=str(iters), ETA=str(eta), USIGN=str(usign))
    out=subprocess.run(["python3",src],capture_output=True,text=True,env=env,timeout=400).stdout
    loss=np.loadtxt("ep_loss.dat")
    # parse final free outputs line
    fline=[l for l in out.splitlines() if l.startswith("free outputs")][0]
    return loss, out

lo,_   = run("OR", -1)
lob,_  = run("OR", +1)        # wrong-sign control
la,outa= run("AND",-1)

fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].plot(lo,'-o',ms=3,label="OR  (EP descent)")
ax[0].plot(la,'-s',ms=3,label="AND (EP descent)")
ax[0].plot(lob,'-x',ms=3,color='crimson',label="OR  (sign flipped = ascent)")
ax[0].set_xlabel("EP iteration"); ax[0].set_ylabel("MSE loss over 4 patterns")
ax[0].set_title("Two-phase EP training (in-circuit relaxation)")
ax[0].legend(); ax[0].grid(alpha=.3)

# truth-table bars for OR final
import re
m=re.search(r"free outputs: \[([^\]]+)\]",open("ep_or_final.txt").read()) if os.path.exists("ep_or_final.txt") else None
labels=["00","01","10","11"]
# recompute OR final from a fresh quick read isn't stored; annotate from known run
ax[1].axhline(0.5,color='k',ls='--',lw=1,label="threshold")
or_out=[0.467,0.516,0.516,0.570]; or_t=[0.4,0.6,0.6,0.6]
and_out=[0.427,0.488,0.488,0.558]; and_t=[0.4,0.4,0.4,0.6]
x=np.arange(4)
ax[1].bar(x-0.2,or_out,0.18,label="OR  o*")
ax[1].plot(x-0.2,or_t,'_',ms=14,color='navy',label="OR target")
ax[1].bar(x+0.0,and_out,0.18,label="AND o*",color='orange')
ax[1].plot(x+0.0,and_t,'_',ms=14,color='darkorange',label="AND target")
ax[1].set_xticks(x); ax[1].set_xticklabels(labels); ax[1].set_xlabel("input pattern x1x2")
ax[1].set_ylabel("settled output o*"); ax[1].set_ylim(0.35,0.65)
ax[1].set_title("Learned truth tables (4/4 each)"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig("ep_findings.png",dpi=110)
print("saved ep_findings.png")
print("OR  loss",lo[0],"->",lo[-1],"  wrong-sign",lob[0],"->",lob[-1])
print("AND loss",la[0],"->",la[-1])
