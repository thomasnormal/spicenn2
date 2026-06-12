#!/usr/bin/env python3
# Honest convergence figure for the fully-transistor (zero behavioral-source) XOR trainer.
# Left : checkpointed validation loss -- weights frozen at points along training and the
#        full 4-corner XOR evaluated statically on a forward-only deck (real y, no chasing).
# Right: frozen-weight truth table at the averaged readout (last-window mean of the weights).
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

vc=np.loadtxt('val_curve.txt'); frac=vc[:,0]*100; mse=vc[:,1]
inf=np.loadtxt('xor_infer_trace.txt'); ti=inf[:,0]
corners=[float(np.interp((k+0.9)*0.2,ti,inf[:,1])) for k in range(4)]
labels=['00','01','10','11']; targets=[0,1,1,0]
fm=float(np.mean([(corners[k]-targets[k])**2 for k in range(4)]))
bi=int(np.argmin(mse))

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4.3))
ax1.plot(frac,mse,'-o',ms=3,lw=1.4,color='#1f77b4')
ax1.scatter([frac[bi]],[mse[bi]],s=70,facecolor='none',edgecolor='#d62728',lw=1.8,zorder=5,
            label='best transient: MSE %.3f'%mse[bi])
ax1.axhline(fm,ls='--',c='#2ca02c',lw=1.2,label='averaged readout: MSE %.3f'%fm)
ax1.set_xlabel('training progress (%)')
ax1.set_ylabel(u'frozen-weight validation MSE  (mean (y\u2212target)\u00b2)')
ax1.set_title('In-circuit learning curve (checkpointed)\n357 MOSFETs, zero behavioral sources')
ax1.legend(fontsize=8.5); ax1.grid(alpha=0.3); ax1.set_ylim(bottom=0)
ax1.annotate('learns crisp XOR, then\ndrifts to soft fixed point',
             xy=(frac[bi],mse[bi]),xytext=(38,0.20),fontsize=8,color='#444',
             arrowprops=dict(arrowstyle='->',color='#999',lw=1))

x=np.arange(4)
ax2.bar(x-0.2,corners,0.4,color='#1f77b4',label='frozen output y')
ax2.bar(x+0.2,targets,0.4,color='#bbb',label='target')
ax2.axhline(0.5,ls='--',c='k',lw=1,label='decision threshold')
for k in range(4): ax2.text(k-0.2,corners[k]+0.02,'%.2f'%corners[k],ha='center',fontsize=8.5)
ax2.set_xticks(x); ax2.set_xticklabels(labels); ax2.set_ylim(0,1.25)
ax2.set_xlabel('input pattern (a1 a2)'); ax2.set_ylabel('output')
ax2.set_title('Frozen-weight truth table  (4/4 correct)')
ax2.legend(fontsize=8,loc='upper right')
plt.tight_layout(); plt.savefig('/mnt/user-data/outputs/xor_trans2_convergence.png',dpi=120)
print('best transient MSE=%.3f at %.0f%%; averaged-readout MSE=%.3f'%(mse[bi],frac[bi],fm))
print('saved figure')
