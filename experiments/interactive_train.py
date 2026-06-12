#!/usr/bin/env python3
# Closed-loop interactive trainer: drive ngspice (-p) from Python, step the in-circuit XOR
# training in chunks, snapshot the weight-cap voltages mid-run, validate each checkpoint with a
# real frozen-inference run, and EARLY-STOP at the crisp minimum (before it drifts to the soft
# (1,0) fixed point). The training PWL sources schedule the inputs; the controller only stops,
# reads weights, evaluates loss, and decides when to stop — learning stays in-circuit.
import subprocess, time, numpy as np, os, re, sys

TOTAL=180.0; DT=0.102; NCK=20; CHUNK=TOTAL/NCK
PATIENCE=3   # stop if validation MSE hasn't improved for this many checkpoints

# --- 1. generate the training deck (env set by caller), parse weight-node list, make circuit-only deck
subprocess.run([sys.executable,"gen_xor_full.py","600","0.3","5e-4"],check=True,
               stdout=subprocess.DEVNULL)
deck=open("xor_full.cir").read().splitlines()
gpline=[l for l in deck if l.startswith("  wrdata weights_full.txt")][0]
GP=re.findall(r"v\(([^)]+)\)", gpline)            # weight-cap nodes in canonical order
ic_idx=max(i for i,l in enumerate(deck) if l.startswith(".ic"))
open("xor_circuit.cir","w").write("\n".join(deck[:ic_idx+1])+"\n.end\n")  # circuit only (keep PWL + .ic)
print(f"interactive XOR: {len(GP)} weights, {NCK} checkpoints over T={TOTAL}")

# --- 2. ngspice interactive pipe
p=subprocess.Popen(['ngspice','-p'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                   stderr=subprocess.STDOUT,text=True,bufsize=1)
def send(s): p.stdin.write(s+"\n"); p.stdin.flush()
def wait(mark,to=60):
    t0=time.time()
    while time.time()-t0<to:
        l=p.stdout.readline()
        if l=="": break
        if mark in l: return True
    return False
send("source xor_circuit.cir"); send("echo LOADED"); wait("LOADED")

def infer_mse(wrow):
    row=np.empty(2*len(wrow)); row[0::2]=np.arange(len(wrow)); row[1::2]=wrow  # wrdata (t,val) layout
    np.savetxt("isnap.txt", row[None,:])
    if os.path.exists("xor_infer_trace.txt"): os.remove("xor_infer_trace.txt")  # no stale reads
    env=dict(os.environ, WFILE="isnap.txt", AVGW="1")
    subprocess.run([sys.executable,"gen_infer.py"],env=env,stdout=subprocess.DEVNULL,check=True)
    subprocess.run(["ngspice","-b","xor_infer.cir"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    # score: read settled y per corner from xor_infer_trace.txt
    raw=np.loadtxt("xor_infer_trace.txt"); t=raw[:,0]; v=raw[:,1::2]
    y=v[:,0]; a1=v[:,1]; a2=v[:,2]; TS=0.2; hold=TS-0.02; frac=t-np.floor(t/TS)*TS
    tgt=[((0,0),0),((0,1),1),((1,0),1),((1,1),0)]; ys=[]
    for (b1,b2),tt in tgt:
        va1=1.3 if b1 else 0.7; va2=1.3 if b2 else 0.7
        m=(np.abs(a1-va1)<.06)&(np.abs(a2-va2)<.06)&(frac<hold-0.01); idx=np.where(m)[0]
        ys.append(y[idx[-1]] if len(idx) else np.nan)
    ys=np.array(ys); mse=np.nanmean((ys-np.array([0,1,1,0]))**2)
    corners=int(np.sum((ys>0.5)==np.array([0,1,1,0])))
    return mse,corners,ys

# --- 3. step + checkpoint + validate + early-stop
hist=[]; best=(1e9,None,None,None); stalls=0; stop_idx=None
started=False
for k in range(1,NCK+1):
    tk=k*CHUNK
    send(f"stop when time > {tk:.4f}")
    send("tran %.4f %.1f uic"%(DT,TOTAL) if not started else "resume"); started=True
    send(f"echo CK{k}"); wait(f"CK{k}")
    send(f"wrdata wsnap.txt "+" ".join(f"v({g})" for g in GP)); send(f"echo WR{k}"); wait(f"WR{k}")
    send(f"delete {k}")                                  # remove fired stop-point
    w=np.loadtxt("wsnap.txt")[-1,1::2]                   # current weight-cap voltages
    mse,corners,ys=infer_mse(w)
    hist.append((tk,mse,corners,ys))
    mark=""
    if mse<best[0]-1e-4: best=(mse,w.copy(),ys,k); stalls=0; mark=" *best"
    else: stalls+=1
    print(f"  ck{k:2d} t={tk:5.1f} ({100*tk/TOTAL:3.0f}%)  MSE={mse:.3f} corners={corners}/4  (1,0)={ys[2]:.2f}{mark}")
    if stalls>=PATIENCE and stop_idx is None:
        stop_idx=k; print(f"  --> EARLY-STOP would fire here (no improvement in {PATIENCE} ckpts); best was ck{best[3]}")
send("quit")
try: p.wait(timeout=5)
except: p.kill()

# --- 4. report: early-stopped(best) vs full-run final
fin=hist[-1]
print("\nRESULT")
print(f"  early-stop (best ck{best[3]}): MSE={best[0]:.3f}  y={np.round(best[2],2)}  (1,0)corner={best[2][2]:.2f}")
print(f"  if-run-to-here  (ck{len(hist)}):   MSE={fin[1]:.3f}  y={np.round(fin[3],2)}  (1,0)corner={fin[3][2]:.2f}")
