import numpy as np, os as _os, sys
name=sys.argv[1]
H=int(_os.environ.get("H","6")); LO=float(_os.environ.get("LO","0.7")); HI=float(_os.environ.get("HI","1.3"))
nhv=min(H,6)
try: raw=np.loadtxt('xor_infer_trace.txt')
except Exception: print(f"{name:13s} NO-DATA (sim failed/timeout)"); sys.exit()
if raw.ndim<2 or raw.shape[0]<4: print(f"{name:13s} NO-DATA"); sys.exit()
t=raw[:,0]; v=raw[:,1::2]; y=v[:,0]; a1=v[:,1]; a2=v[:,2]; hv=v[:,4:4+nhv]
TS=0.2; hold=TS-0.02; slot=np.floor(t/TS).astype(int); frac=t-slot*TS
tgt=[((0,0),0),((0,1),1),((1,0),1),((1,1),0)]; se=[]; ys=[]; allhv=[]
for (b1,b2),tt in tgt:
    va1=HI if b1 else LO; va2=HI if b2 else LO
    m=(np.abs(a1-va1)<.06)&(np.abs(a2-va2)<.06)&(frac<hold-0.01); idx=np.where(m)[0]
    if len(idx)==0: print(f"{name:13s} NO-MATCH"); sys.exit()
    i=idx[-1]; ys.append(y[i]); se.append((y[i]-tt)**2); allhv.append(hv[i])
mse=float(np.mean(se)); nc=sum((ys[i]>0.5)==(tgt[i][1]==1) for i in range(4)); ar=float(np.ptp(np.array(allhv)))
flag="  *** SOLVED" if nc==4 else ""
print(f"{name:13s} MSE={mse:.3f} corners={nc}/4 y=[{ys[0]:.2f},{ys[1]:.2f},{ys[2]:.2f},{ys[3]:.2f}] actrange={ar:.2f}{flag}")
