"""Parse Spectre nutascii (-format nutascii) raw output robustly."""
import numpy as np, os, glob
def load(raw):
    f=raw if os.path.isfile(raw) else glob.glob(raw+"/*")[0]
    lines=open(f).read().splitlines()
    nvar=[int(l.split(":")[1]) for l in lines if l.startswith("No. Variables")][0]
    vstart=[i for i,l in enumerate(lines) if l.startswith("Variables:")][0]
    vend  =[i for i,l in enumerate(lines) if l.strip()=="Values:"][0]
    block=(" ".join(lines[vstart:vend])).replace("Variables:"," ").split()
    names=[]
    for k in range(nvar):
        j=block.index(str(k)); names.append(block[j+1])
    toks=" ".join(lines[vend+1:]).split()
    vals=np.array([float(x) for x in toks]).reshape(-1,nvar+1)[:,1:]
    return names,vals
if __name__=="__main__":
    import sys; n,v=load(sys.argv[1]); print("vars:",n,"rows:",v.shape[0])
    for i,nm in enumerate(n): print(f"  {nm}: final={v[-1,i]:.5g}")
