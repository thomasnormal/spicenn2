#!/usr/bin/env python3
# Stream an arbitrary multi-class dataset (e.g. sklearn digits) into the live in-circuit trainer.
# No data baked into the netlist: inputs (a_i), update-gate copies (au_i) and one-hot targets
# (tg_c) are DC sources; the Python controller `alter`s them per example and advances the
# continuous, state-preserving simulation one slot at a time. Learning stays in-circuit.
# Usage: env (same as gen_mc) DATAFILE=... stream_mc.py <N_slots>
from pathlib import Path
import subprocess, time, numpy as np, os, re, sys
def ef(k,d): return float(os.environ.get(k,d))
def ei(k,d): return int(os.environ.get(k,d))
D=ei("D",16); C=ei("C",2); Ts=ef("TS","0.3"); DT=ef("DTFRAC","0.34")*Ts
INLO=ef("INLO","0.3"); INHI=ef("INHI","1.7"); AULO=0.3; AUHI=1.0
N=int(sys.argv[1]) if len(sys.argv)>1 else 1500

# 1. generate the (sparse, digit) trainer deck with PWL sources; parse weight nodes; make DC/stream deck
subprocess.run([sys.executable,str(Path(__file__).resolve().with_name("gen_mc.py")),"40","0.3","5e-4"],check=True,stdout=subprocess.DEVNULL)
deck=open("mc.cir").read().splitlines()
GP=re.findall(r"v\(([^)]+)\)", [l for l in deck if l.startswith("  wrdata mc_weights.txt")][0])
ic_idx=max(i for i,l in enumerate(deck) if l.startswith(".ic"))
out=[]
for l in deck[:ic_idx+1]:
    m=re.match(r"(Va\d+|Vau\d+|Vtg\d+) (\S+) 0 PWL", l)
    if m: out.append(f"{m.group(1)} {m.group(2)} 0 dc 0.5")   # alterable DC source
    else: out.append(l)
open("mc_stream.cir","w").write("\n".join(out)+"\n.end\n")

# 2. dataset (controller-side; any source/size)
d=np.load(os.environ["DATAFILE"]); X=d["X"].astype(float); Y=d["Y"].astype(int)
order=np.random.default_rng(3).permutation(len(X))
RAMP=ei("RAMP",1); RFRAC=ef("RFRAC","0.4")   # RAMP sub-steps over first RFRAC of each slot (1=abrupt)
def vals(k):
    i=order[k%len(order)]; x=X[i]; lab=Y[i]; v={}
    for j in range(D):
        v[f"Va{j+1}"]=INLO+x[j]*(INHI-INLO); v[f"Vau{j+1}"]=AULO+x[j]*(AUHI-AULO)
    for c in range(C): v[f"Vtg{c+1}"]=1.0 if lab==c else 0.0
    return v

# 3. interactive ngspice
p=subprocess.Popen(['ngspice','-p'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                   stderr=subprocess.STDOUT,text=True,bufsize=1)
def send(s): p.stdin.write(s+"\n"); p.stdin.flush()
def wait(mark,to=120):
    t0=time.time()
    while time.time()-t0<to:
        l=p.stdout.readline()
        if l=="": return False
        if mark in l: return True
    return False
send("source mc_stream.cir"); send("echo LOADED"); wait("LOADED")
def advance(tstop,mark):  # run/resume the continuous sim up to tstop (fine timesteps), sync
    global started
    if not started: send(f"stop when time > {tstop:.4f}"); send(f"tran {DT:.4f} {N*Ts:.1f} 0 {DT:.4f} uic"); started=True
    else: send("delete all"); send(f"stop when time > {tstop:.4f}"); send("resume")
    send(f"echo {mark}"); return wait(mark)
started=False; prev=vals(0); t0=time.time()
for k in range(N):
    new=vals(k); t_start=k*Ts; ok=True
    for r in range(1,RAMP+1):                          # controller slews inputs prev->new (like a DAC)
        f=r/RAMP
        for s in new: send(f"alter {s} = {prev[s]+(new[s]-prev[s])*f:.4f}")
        if not advance(t_start+RFRAC*Ts*f, f"R{k}_{r}"): ok=False; break
    if not ok or not advance((k+1)*Ts, f"S{k}"): print(f"desync k={k}"); break   # hold for rest of slot
    prev=new
    if k%300==0 or k==N-1: print(f"  streamed {k+1:4d}/{N}  RAMP={RAMP}  [{time.time()-t0:.0f}s]")
send("wrdata wfinal.txt "+" ".join(f"v({g})" for g in GP)); send("echo WF"); wait("WF")
send("quit")
try: p.wait(timeout=5)
except: p.kill()

# 4. freeze + score on the held-out test set
w=np.loadtxt("wfinal.txt")[-1,1::2]
row=np.empty(2*len(w)); row[0::2]=np.arange(len(w)); row[1::2]=w; np.savetxt("isnap.txt",row[None,:])
if os.path.exists("mc_infer_trace.txt"): os.remove("mc_infer_trace.txt")
subprocess.run([sys.executable,str(Path(__file__).resolve().with_name("gen_mc_infer.py"))],
               env=dict(os.environ,WFILE="isnap.txt",AVGW="1",DATAFILE_TE=os.environ.get("DATAFILE_TE","digits_test.npz")),
               stdout=subprocess.DEVNULL,check=True)
subprocess.run(["ngspice","-b","mc_infer.cir"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
print("frozen score of STREAM-trained weights:")
subprocess.run([sys.executable,str(Path(__file__).resolve().with_name("score_mc.py")),"stream-digits"],env=dict(os.environ,C=str(C)))
