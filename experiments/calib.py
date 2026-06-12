import subprocess, numpy as np, itertools, concurrent.futures as cf, os, re
TARGETS={0:0.969,1:0.88,2:0.969,3:0.956}
def run(seed,cfg):
    env={**os.environ,"SEED":str(seed),"SOFTC":"1",**{k:str(v) for k,v in cfg.items()}}
    r=subprocess.run(["python3","fasttrain.py"],capture_output=True,text=True,timeout=400,env=env)
    m=re.search(r"BEST=([0-9.]+)",r.stdout)
    return float(m.group(1)) if m else 0.0
rng=np.random.default_rng(1)
results=[]
for trial in range(24):
    cfg={"ETA":round(10**rng.uniform(-1.8,-0.7),4),"GBK":round(10**rng.uniform(-0.3,0.7),2),
         "KW":round(rng.uniform(1.5,4.0),2),"BETA":round(rng.uniform(0.08,0.35),3)}
    with cf.ThreadPoolExecutor(4) as ex:
        accs={s:f for s,f in zip([0,1,2,3],ex.map(lambda s:run(s,cfg),[0,1,2,3]))}
    err=np.sqrt(np.mean([(accs[s]-TARGETS[s])**2 for s in TARGETS]))
    results.append((err,cfg,accs))
    print(f"t{trial:02d} err={err:.3f} cfg={cfg} accs={[accs[s] for s in [0,1,2,3]]}",flush=True)
results.sort(key=lambda r:r[0])
print("\nBEST 3:")
for err,cfg,accs in results[:3]: print(f"  err={err:.3f} {cfg} -> {[accs[s] for s in [0,1,2,3]]}")
