#!/usr/bin/env python3
"""Closed-loop IN-CIRCUIT LEARNING on the oscillator substrate (capstone ONN demo).

Two detuned current-starved ring oscillators, coupled by an NMOS whose gate = the WEIGHT node vw
(a capacitor voltage = K_ij). An in-circuit phase detector (buffered XOR + low-pass) reads the phase
coherence pdf. A 5T OTA integrates (pdf - vtarget) onto the weight cap: when the oscillators are NOT
synchronized (pdf high, beating) the OTA drives vw UP -> stronger coupling -> they lock -> pdf drops
-> vw settles. So the coupling weight LEARNS to synchronize, with NO external optimizer -- the whole
loop (measure phase, update weight) is transistors. vw starts low (unlocked).

Reports vw(t) trajectory and the oscillator frequency spread in early vs late windows
(early unlocked -> late locked  ==>  the weight learned to couple).
"""
import os, subprocess, glob
import numpy as np

SP = "/opt/cadence/installs/SPECTRE231/bin"
TAG = os.environ.get("TAG", "onnl")
TSTOP = float(os.environ.get("TSTOP", "8e-6"))
VBU = os.environ.get("VBU", "0.40")      # OTA tail bias -> learning rate
VTGT = os.environ.get("VTGT", "0.33")    # target phase-detector DC (toward lock)
VW0 = os.environ.get("VW0", "0.25")      # initial weight (unlocked)
DET = float(os.environ.get("DET", "0.12"))

MODELS = (".model NNR NMOS (LEVEL=1 VTO=0.2 KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)\n"
          ".model PNR PMOS (LEVEL=1 VTO=-0.2 KP=40u LAMBDA=0.02 GAMMA=0 PHI=0.7)")
CSINV = (".subckt csinv in out vdd vcp vcn\n"
         "Mps s1 vcp vdd vdd PNR W=8u L=2u\nMp out in s1 vdd PNR W=8u L=2u\n"
         "Mn out in s2 0 NNR W=4u L=2u\nMns s2 vcn 0 0 NNR W=4u L=2u\n.ends")

def build():
    L = ["onn in-circuit learning", "simulator lang=spice", "* title", MODELS, CSINV, "Vdd vdd 0 1.0"]
    for i in range(2):
        d = DET * (i - 0.5)
        L += [f"Vcp{i} vcp{i} 0 {0.45 + 0.10*d:.4f}", f"Vcn{i} vcn{i} 0 {0.55 - 0.10*d:.4f}"]
        for s in range(3):
            L.append(f"X{i}_{s} o{i}_{(s-1)%3} o{i}_{s} vdd vcp{i} vcn{i} csinv")
            L.append(f"C{i}_{s} o{i}_{s} 0 40f")
    # PROGRAMMABLE coupling: NMOS gate = weight node vw (conductance grows with vw)
    L.append("Mcouple o0_0 vw o1_0 0 NNR W=3u L=2u")
    L.append(f"Cvw vw 0 4p")                      # the weight capacitor (analog memory)
    # in-circuit phase detector: buffered XOR(o0_0,o1_0) -> RC low-pass pdf
    L += ["Mp0a x0n o0_0 vdd vdd PNR W=4u L=2u", "Mn0a x0n o0_0 0 0 NNR W=2u L=2u",
          "Mp0b x0 x0n vdd vdd PNR W=4u L=2u",   "Mn0b x0 x0n 0 0 NNR W=2u L=2u",
          "Mp1a x1n o1_0 vdd vdd PNR W=4u L=2u", "Mn1a x1n o1_0 0 0 NNR W=2u L=2u",
          "Mp1b x1 x1n vdd vdd PNR W=4u L=2u",   "Mn1b x1 x1n 0 0 NNR W=2u L=2u",
          "Mx1n pdy x0n x1 0 NNR W=4u L=2u", "Mx1p pdy x0 x1 vdd PNR W=4u L=2u",
          "Mx2n pdy x0 x1n 0 NNR W=4u L=2u", "Mx2p pdy x0n x1n vdd PNR W=4u L=2u",
          "Rpd pdy pdf 50k", "Cpd pdf 0 2p"]
    # 5T OTA update: I(vw) ~ gm*(pdf - vtarget); pdf high (beating) -> charge vw UP -> couple -> lock
    L += [f"Vbu vbu 0 {VBU}", f"Vtgt vtarget 0 {VTGT}",
          "Mtl nt vbu 0 0 NNR W=2u L=8u",
          "M1o da pdf nt 0 NNR W=6u L=8u", "M2o vw vtarget nt 0 NNR W=6u L=8u",
          "M3o da da vdd vdd PNR W=6u L=8u", "M4o vw da vdd vdd PNR W=6u L=8u"]
    L.append(f".ic v(vw)={VW0} v(o0_0)=1 v(o0_1)=0 v(o0_2)=1 v(o1_0)=0 v(o1_1)=1 v(o1_2)=0")
    L.append(f".tran {TSTOP/16000:g} {TSTOP:g}")
    L.append(".end")
    open(f"onl_{TAG}.scs", "w").write("\n".join(L) + "\n")

def parse(globpat, nodes):
    f = sorted(glob.glob(globpat))[0]; nset = set(nodes); t = []; d = {n: [] for n in nodes}
    fh = open(f)
    for ln in fh:
        if ln.strip() == "VALUE": break
    for ln in fh:
        ln = ln.strip()
        if ln == "END": break
        if not ln or ln[0] != '"': continue
        q = ln.find('"', 1)
        if q < 0: continue
        nm = ln[1:q]
        if nm == "time": t.append(float(ln[q+1:]))
        elif nm in nset: d[nm].append(float(ln[q+1:]))
    fh.close()
    n = min([len(t)] + [len(d[k]) for k in nodes])
    return np.array(t[:n]), {k: np.array(d[k][:n]) for k in nodes}

def freq_win(t, v, t0, t1):
    m = (t >= t0) & (t < t1); v = v[m] - 0.5; tt = t[m]
    x = np.where((v[:-1] < 0) & (v[1:] >= 0))[0]
    if len(x) < 2: return 0.0
    return 1.0 / np.mean(np.diff(tt[x]))

if __name__ == "__main__":
    build()
    for f in glob.glob(f"raw_onl_{TAG}/*"): os.remove(f)
    r = subprocess.run([f"{SP}/spectre", "+spice", "+mt=1", f"onl_{TAG}.scs",
                        "-format", "psfascii", "-raw", f"raw_onl_{TAG}", "+log", f"onl_{TAG}.log"],
                       capture_output=True, text=True, timeout=600)
    raws = glob.glob(f"raw_onl_{TAG}/*.tran.tran") or glob.glob(f"raw_onl_{TAG}/*.tran")
    if not raws:
        print("SIM FAILED:", r.stderr[-600:]); raise SystemExit(1)
    t, d = parse(raws[0], ["o0_0", "o1_0", "vw", "pdf"])
    T = t[-1]
    print(f"weight vw(t): start {d['vw'][:50].mean():.3f}V -> end {d['vw'][-200:].mean():.3f}V  (tsim={T*1e6:.1f}us)")
    for lab, t0, t1 in [("early", 0.0, 0.25*T), ("late ", 0.7*T, T)]:
        f0 = freq_win(t, d["o0_0"], t0, t1) / 1e6; f1 = freq_win(t, d["o1_0"], t0, t1) / 1e6
        if f0 > 0 and f1 > 0:
            sp = abs(f0 - f1) / ((f0 + f1) / 2) * 100
            vwm = d["vw"][(t >= t0) & (t < t1)].mean()
            print(f"  {lab}: f0={f0:.1f} f1={f1:.1f} MHz  spread={sp:.1f}%  ({'LOCKED' if sp < 2 else 'unlocked'})  vw={vwm:.3f}V")
