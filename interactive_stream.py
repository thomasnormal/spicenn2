#!/usr/bin/env python3
# Stream an ARBITRARY dataset into the live SPICE trainer at runtime — no PWL baked into the deck.
# The inputs/target are plain DC sources; the Python controller `alter`s them per example and
# advances the (continuous, state-preserving) simulation one slot at a time. The dataset is
# generated/served entirely controller-side, so any dataset of any size works without touching
# the netlist. Demo target: cycle the 4 XOR patterns, then freeze + score the truth table.
import subprocess, time, numpy as np, os, re, sys

Ts=0.3; DT=0.102; N=int(sys.argv[1]) if len(sys.argv)>1 else 600

# --- build a STREAMING deck: same trainer, but inputs/target as DC sources (no PWL) ---
subprocess.run([sys.executable,"gen_xor_full.py","600","0.3","5e-4"],check=True,stdout=subprocess.DEVNULL)
deck=open("xor_full.cir").read().splitlines()
gpline=[l for l in deck if l.startswith("  wrdata weights_full.txt")][0]
GP=re.findall(r"v\(([^)]+)\)", gpline)
ic_idx=max(i for i,l in enumerate(deck) if l.startswith(".ic"))
out=[]
for l in deck[:ic_idx+1]:
    if   l.startswith("Va1 "):  out.append("Va1 a1 0 dc 0.7")
    elif l.startswith("Va2 "):  out.append("Va2 a2 0 dc 0.7")
    elif l.startswith("Vau1 "): out.append("Vau1 au1 0 dc 0.3")
    elif l.startswith("Vau2 "): out.append("Vau2 au2 0 dc 0.3")
    elif l.startswith("Vtg "):  out.append("Vtg tg 0 dc 0.0")
    else: out.append(l)
open("xor_stream.cir","w").write("\n".join(out)+"\n.end\n")

# --- the dataset lives HERE in python, not in the netlist (could be anything) ---
PAT=[((0,0),0.0),((0,1),1.0),((1,0),1.0),((1,1),0.0)]
def example(k):
    (b1,b2),t=PAT[k%4]
    return dict(Va1=1.3 if b1 else 0.7, Va2=1.3 if b2 else 0.7,
                Vau1=1.0 if b1 else 0.3, Vau2=1.0 if b2 else 0.3, Vtg=t)

# --- interactive ngspice ---
p=subprocess.Popen(['ngspice','-p'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                   stderr=subprocess.STDOUT,text=True,bufsize=1)
def send(s): p.stdin.write(s+"\n"); p.stdin.flush()
def wait(mark,to=60):
    t0=time.time()
    while time.time()-t0<to:
        l=p.stdout.readline()
        if l=="": return False
        if mark in l: return True
    return False
send("source xor_stream.cir"); send("echo LOADED"); wait("LOADED")

t0=time.time()
for k in range(N):
    ex=example(k)
    for src,val in ex.items(): send(f"alter {src} = {val}")   # <-- stream this example's data in
    tnext=(k+1)*Ts
    if k==0: send(f"stop when time > {tnext:.4f}"); send(f"tran {DT} {N*Ts:.1f} uic")
    else:    send("delete all"); send(f"stop when time > {tnext:.4f}"); send("resume")
    send(f"echo S{k}");
    if not wait(f"S{k}"): print(f"  desync at k={k}"); break
    if k in (0,N//2,N-1) or k%150==0:
        send("wrdata wsnap.txt "+" ".join(f"v({g})" for g in GP)); send("echo W"); wait("W")
        w=np.loadtxt("wsnap.txt")[-1,1::2]
        print(f"  streamed {k+1:4d}/{N} examples  t={tnext:5.1f}  |w-VC|mean={np.abs(w-1.8).mean():.3f}  [{time.time()-t0:.0f}s]")
send("wrdata wfinal.txt "+" ".join(f"v({g})" for g in GP)); send("echo WF"); wait("WF")
send("quit")
try: p.wait(timeout=5)
except: p.kill()

# --- freeze streamed weights, score the XOR truth table with a real inference run ---
w=np.loadtxt("wfinal.txt")[-1,1::2]
row=np.empty(2*len(w)); row[0::2]=np.arange(len(w)); row[1::2]=w; np.savetxt("isnap.txt",row[None,:])
if os.path.exists("xor_infer_trace.txt"): os.remove("xor_infer_trace.txt")
subprocess.run([sys.executable,"gen_infer.py"],env=dict(os.environ,WFILE="isnap.txt",AVGW="1"),
               stdout=subprocess.DEVNULL,check=True)
subprocess.run(["ngspice","-b","xor_infer.cir"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
print("\nfrozen score of the STREAM-trained weights:")
subprocess.run([sys.executable,"score.py","streamed"])
