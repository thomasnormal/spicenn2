#!/usr/bin/env python3
"""Forward representability test: can ANY weight setting make this net compute XOR?
Random-search the T-gate controls; evaluate ONLY the free-phase output loss.
If the best loss stays ~linear-floor (~0.06-0.12) the architecture can't represent XOR
(the passive-net gain problem); if it drops near 0 the blocker is the learning dynamics."""
import numpy as np, os, ep_xor2 as E

rng=np.random.default_rng(0)
tgt=np.array([E.vl(p[2]) for p in E.PATS])
XOR_HI=[0,3]; XOR_LO=[1,2]  # {00,11} vs {01,10}
maxsep=0; n=int(os.environ.get("N","250"))
maxhsep=0
for k in range(n):
    vc={w: float(np.clip(rng.uniform(E.VCMIN,E.VCMAX),E.VCMIN,E.VCMAX)) for w in E.WEIGHTS}
    F,_=E.run(vc)
    o=F[:,4]
    sep=abs(o[XOR_HI].mean()-o[XOR_LO].mean())          # output XOR separation
    hsep=max(abs(F[:,0][XOR_HI].mean()-F[:,0][XOR_LO].mean()),
             abs(F[:,2][XOR_HI].mean()-F[:,2][XOR_LO].mean()))  # best hidden XOR sep
    if hsep>maxhsep: maxhsep=hsep
    if sep>maxsep:
        maxsep=sep
        print(f"[{k:3d}] output XOR-sep={sep:.3f} (hidden XOR-sep={hsep:.3f}) o*={np.round(o,3)}")
print(f"\nMAX output XOR-separation over {n} draws: {maxsep:.3f}")
print(f"MAX hidden XOR-separation: {maxhsep:.3f}")
print("output must reach ~0.7 sep to classify XOR with margin; <~0.1 => passive-gain wall")
