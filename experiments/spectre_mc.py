#!/usr/bin/env python3
"""Run a gen_mc / gen_mc_infer ngspice deck on SPECTRE (higher-fidelity training dynamics).
Converts the deck: strips .control block, adds SPICE-mode .tran with strobed output, saves only
the nodes the flow needs, runs spectre +spice, parses psfascii, writes the SAME txt files the
ngspice flow writes (mc_weights.txt / mc_infer_trace.txt, wrdata-interleaved) so gen_mc_infer.py
and score_mc.py work unchanged.   Usage: python3 spectre_mc.py train|infer [deckfile]
"""
import sys, os, re, glob, subprocess
import numpy as np
mode=sys.argv[1]; deck=sys.argv[2] if len(sys.argv)>2 else ("mc.cir" if mode=="train" else "mc_infer.cir")
TAG=os.environ.get("TAG","")
SPB="/opt/cadence/installs/SPECTRE231/bin"
txt0=open(deck).read()
# inline .param substitutions ({name} -> value): Spectre spice-mode doesn't expand ngspice {param}
for pl in re.findall(r"^\.param\s+(.+)$",txt0,re.M):
    for name,val in re.findall(r"(\w+)\s*=\s*([0-9.eE+-]+)",pl):
        txt0=txt0.replace("{"+name+"}",val)
lines=txt0.splitlines()
# --- split control block out
body=[]; ctrl=[]; inctrl=False
for l in lines:
    if l.strip().startswith(".control"): inctrl=True; continue
    if l.strip().startswith(".endc"): inctrl=False; continue
    (ctrl if inctrl else body).append(l)
ctxt="\n".join(ctrl)
m=re.search(r"tran\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+uic",ctxt); dt,tstop=float(m.group(1)),float(m.group(2))
def nodes_of(fname):
    mm=re.search(r"wrdata\s+"+re.escape(fname)+r"\s+(.+)",ctxt)
    return re.findall(r"v\(([^)]+)\)",mm.group(1)) if mm else []
if mode=="train":
    nodes=nodes_of(f"mc_weights{TAG}.txt"); out=f"mc_weights{TAG}.txt"
    Ts=float(os.environ.get("TS","0.3"))
else:
    nodes=nodes_of(f"mc_infer_trace{TAG}.txt"); out=f"mc_infer_trace{TAG}.txt"
    Ts=0.2  # gen_mc_infer slot
assert nodes, "could not find node list in control block"
# strobe one settled sample per slot, from t=0 (the outputstart-near-tstop trick yields 0 points)
strobe=f"strobeperiod={Ts} strobedelay={Ts*0.85:.5f} outputstart=0"
scs=["simulator lang=spice","* "+body[0]]+body[1:]
scs=[l for l in scs if not l.strip().startswith(".end")]
scs.append(".save "+" ".join(f"v({n})" for n in nodes))
scs.append(f".tran {dt} {tstop} uic {strobe}")
scs.append(".end")
sf=f"sp_{mode}{TAG}.scs"; open(sf,"w").write("\n".join(scs)+"\n")
raw=f"raw_{mode}{TAG}"
for f in glob.glob(f"{raw}/*.tran.tran"): os.remove(f)
r=subprocess.run([f"{SPB}/spectre","+spice",f"+mt={os.environ.get('MT','4')}",
                  f"+lqtimeout={os.environ.get('LQ','3600')}",sf,"-format","psfascii","-raw",raw,
                  "+log",f"sp_{mode}{TAG}.log"],capture_output=True,text=True,
                 timeout=float(os.environ.get("SPTMO","14000")),
                 env={**os.environ,"PATH":SPB+":"+os.environ.get("PATH","")})
fs=sorted(glob.glob(f"{raw}/*.tran.tran"))
if not fs:
    print("SPECTRE FAIL:",(r.stdout or "")[-600:],(r.stderr or "")[-300:]); sys.exit(1)
txt=open(fs[0]).read()
vi=re.search(r"^VALUE\s*$",txt,re.M).end(); bodytxt=txt[vi:]
lnodes=[n.lower() for n in nodes]; nset=set(lnodes)   # Spectre lowercases node names
times=[]; data={n:[] for n in lnodes}
for line in bodytxt.splitlines():
    line=line.strip()
    if line=="END": break
    if not line or line[0]!='"': continue
    q=line.index('"',1); name=line[1:q].lower(); rest=line[q+1:].strip()
    if name=="time": times.append(float(rest))
    elif name in nset: data[name].append(float(rest))
nodes=lnodes
t=np.array(times); M=np.array([data[n] for n in nodes]).T
if M.ndim<2 or len(t)==0: print("SPECTRE PARSE EMPTY"); sys.exit(1)
rows=np.empty((len(t),2*len(nodes)))
rows[:,0::2]=t[:,None]; rows[:,1::2]=M[:len(t)]
np.savetxt(out,rows,fmt="%.6g")
print(f"SPECTRE {mode} OK: {len(t)} samples x {len(nodes)} nodes -> {out}")
