#!/usr/bin/env python3
"""Persistent interactive ngspice driver (pipe mode) — the load-once / alter / relax / read /
resume loop the controller wants, without regenerating the deck or respawning the process.
Sync: after each command burst send 'echo @@K@@' and read stdout until the marker.
Values come back via wrdata files (robust) rather than parsing stdout."""
import subprocess, numpy as np, os
class Live:
    def __init__(self, deck, fname="live.cir"):
        self.f=fname; open(fname,"w").write(deck)
        self.p=subprocess.Popen(["ngspice","-p"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,text=True,bufsize=1)
        self._burst(["set noaskquit","source "+fname])
    def _burst(self,cmds):
        for c in cmds:
            self.p.stdin.write(c+"\n")
        self.p.stdin.write("echo @@K@@\n"); self.p.stdin.flush()
        out=[]
        while True:
            line=self.p.stdout.readline()
            if not line: break
            if "@@K@@" in line: break
            out.append(line.rstrip())
        return out
    def step(self,alters,analysis,reads,fout=None):
        """alters: list of 'alter ...' ; analysis: 'op' or 'tran 0.2n 60n' ; reads: node names.
        Returns settled values (last row for tran, the single row for op)."""
        fout=fout or f"live_{os.getpid()}.dat"
        cols=" ".join(f"v({n})" for n in reads)
        # 'destroy all' frees the per-op plot vectors (else ngspice memory grows unbounded -> slowdown)
        try: os.remove(fout)
        except OSError: pass
        try:
            self._burst(list(alters)+[analysis,f"wrdata {fout} {cols}","destroy all"])
            rows=np.loadtxt(fout); last=rows[-1] if rows.ndim>1 else rows
            return last[1::2]
        except Exception:
            return None   # unstable solve (no output) -> caller skips this example
    def close(self):
        try: self.p.stdin.write("quit\n"); self.p.stdin.flush()
        except Exception: pass
        self.p.terminate()

if __name__=="__main__":
    # minimal RC test: persistent process, change input, relax, read settled cap voltage
    deck="""rc test
Vin in 0 0.0
R1 in cap 20k
C1 cap 0 1n
.end
"""
    L=Live(deck)
    for vin in [0.3, 0.8, 0.2, 0.6]:
        v=L.step([f"alter Vin = {vin}"], "0.5n 200n", ["cap","in"])
        print(f"Vin={vin}  settled v(cap)={v[0]:.4f}  v(in)={v[1]:.4f}")
    L.close()
