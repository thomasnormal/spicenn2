#!/usr/bin/env python3
# Step A: validate Python<->ngspice interactive pipe (ngspice -p), with marker-based sync.
import subprocess, sys, time

p = subprocess.Popen(['ngspice','-p'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, bufsize=1)
def send(s): p.stdin.write(s+"\n"); p.stdin.flush()
def wait(marker, timeout=15):
    out=[]; t0=time.time()
    while time.time()-t0<timeout:
        line=p.stdout.readline()
        if line=="" : break
        out.append(line.rstrip())
        if marker in line: return out
    return out

# tiny RC, drive it interactively, change source mid-run, read a node back into Python
deck="""* pipe probe rc
Va a 0 dc 0.2
R1 a b 10k
C1 b 0 200p
.end
"""
open("probe.cir","w").write(deck)
send("source probe.cir"); send("echo LOADED"); print("load:", wait("LOADED")[-2:])
send("stop when time > 30u"); send("tran 0.05u 100u uic"); send("echo RAN1")
wait("RAN1")
send("alter Va = 1.0"); send("delete 1"); send("stop when time > 70u"); send("resume"); send("echo RAN2")
wait("RAN2")
# read v(b) value at current (halted) time back into python
send("print v(b)"); send("echo PRB")
resp=wait("PRB")
print("read-back lines:", [l for l in resp if 'v(b)' in l or '=' in l][:4])
send("quit");
try: p.wait(timeout=5)
except: p.kill()
print("PIPE_OK")
