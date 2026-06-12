#!/usr/bin/env python3
# Convert an ngspice deck (with a .control tran/wrdata block) to a Spectre deck in SPICE-language
# mode. Spectre reads the SPICE circuit; we replace .control with a .tran + .save and let Spectre
# write a nutascii rawfile. Run:  spectre deck.scs -format nutascii -raw out.raw =log spectre.log
import sys, re
src=sys.argv[1]; dst=sys.argv[2] if len(sys.argv)>2 else "deck.scs"
L=open(src).read().splitlines()
ci=next(i for i,l in enumerate(L) if l.strip().lower().startswith(".control"))
ei=next(i for i,l in enumerate(L) if l.strip().lower().startswith(".endc"))
head=L[:ci]; ctrl=L[ci:ei+1]
tran=next(l for l in ctrl if l.strip().lower().startswith("tran"))
m=re.match(r"\s*tran\s+(\S+)\s+(\S+)",tran); dt,tstop=m.group(1),m.group(2)
vecs=[]
for l in ctrl:
    if l.strip().lower().startswith("wrdata"): vecs+=re.findall(r"v\(([^)]+)\)",l)
vecs=list(dict.fromkeys(vecs))
# Spectre SPICE-mode doesn't take ngspice {param} braces -> substitute .param values inline
params={}
for l in head:
    if l.strip().lower().startswith(".param"):
        for kv in re.findall(r"(\w+)\s*=\s*(\S+)", l): params[kv[0]]=kv[1]
def subst(s):
    for k,v in params.items(): s=s.replace("{"+k+"}", v)
    return s
out=["simulator lang=spice"]
for l in head:
    if l.strip().lower().startswith(".param"): continue   # drop; values inlined
    out.append(subst(l) if l.strip() else "*")
out.append(f".tran {dt} {tstop} uic")
out.append(".save "+" ".join(f"v({v})" for v in vecs))
out.append(".end")
open(dst,"w").write("\n".join(out)+"\n")
print(f"wrote {dst}: .tran {dt} {tstop}, {len(vecs)} saved nodes")
