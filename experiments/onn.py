#!/usr/bin/env python3
"""Transistor-level coupled-oscillator (ONN) prototype in Spectre.

N current-starved CMOS ring oscillators (LEVEL-1, 1.0V), each a 3-inverter loop whose frequency is
set by a starve-bias (the natural frequency omega_i). Oscillators are resistively coupled (coupling
conductance ~ K_ij, the programmable 'weight'; gsyn would make it a stored-charge weight). We test the
canonical ONN/Kuramoto behavior: do detuned oscillators frequency-LOCK when coupled?

Usage: RC=<ohms> DET=<detune> N=2 python3 experiments/onn.py
  RC  = coupling resistor (small=strong coupling; 1e9=effectively uncoupled)
  DET = fractional detuning of natural frequencies across oscillators
Prints each oscillator's measured frequency (zero-crossing of v - Vdd/2) and the spread (locked => ~0).
"""
import os, subprocess, glob
import numpy as np

SP = "/opt/cadence/installs/SPECTRE231/bin"
N   = int(os.environ.get("N", "2"))
RC  = os.environ.get("RC", "1e9")            # coupling resistor (ohms)
DET = float(os.environ.get("DET", "0.15"))   # fractional detuning
TAG = os.environ.get("TAG", "onn")
TSTOP = float(os.environ.get("TSTOP", "300e-9"))

MODELS = (".model NNR NMOS (LEVEL=1 VTO=0.2 KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)\n"
          ".model PNR PMOS (LEVEL=1 VTO=-0.2 KP=40u LAMBDA=0.02 GAMMA=0 PHI=0.7)")
# current-starved inverter: pmos-starve gate vcp (low=more current), nmos-starve gate vcn (high=more current)
CSINV = (".subckt csinv in out vdd vcp vcn\n"
         "Mps s1 vcp vdd vdd PNR W=8u L=2u\n"
         "Mp  out in s1 vdd PNR W=8u L=2u\n"
         "Mn  out in s2 0 NNR W=4u L=2u\n"
         "Mns s2 vcn 0 0 NNR W=4u L=2u\n.ends")

def build():
    L = ["coupled ring oscillators (ONN)", "simulator lang=spice", "* title", MODELS, CSINV, "Vdd vdd 0 1.0"]
    ics = []
    for i in range(N):
        # detune: oscillator i gets progressively more starved (slower) -> spread of natural freqs
        d = DET * (i - (N - 1) / 2.0)                 # symmetric detuning around center
        vcp = 0.45 + 0.10 * d                          # higher vcp = less pmos current = slower
        vcn = 0.55 - 0.10 * d                          # lower vcn = less nmos current = slower
        L += [f"Vcp{i} vcp{i} 0 {vcp:.4f}", f"Vcn{i} vcn{i} 0 {vcn:.4f}"]
        nodes = [f"o{i}_{s}" for s in range(3)]
        for s in range(3):
            inn = nodes[(s - 1) % 3]; out = nodes[s]
            L.append(f"X{i}_{s} {inn} {out} vdd vcp{i} vcn{i} csinv")
            L.append(f"C{i}_{s} {out} 0 40f")
        # kick the ring out of metastable DC
        ics += [f"v(o{i}_0)={'1' if i%2==0 else '0'}", f"v(o{i}_1)=0", f"v(o{i}_2)=1"]
    # coupling: chain oscillator i node0 to i+1 node0. COUPLE=res (fixed R) | tgate (PROGRAMMABLE:
    # transmission-gate conductance set by weight gate VW -> K_ij = f(VW), a stored/learnable weight).
    couple = os.environ.get("COUPLE", "res")
    if couple == "tgate":
        VW = float(os.environ.get("VW", "1.0"))      # weight: high=strong coupling, ~0=off
        L += [f"Vw vw 0 {VW:.4f}", f"Vwb vwb 0 {1.0 - VW:.4f}"]
        for i in range(N - 1):
            a, b = f"o{i}_0", f"o{i+1}_0"
            L += [f"Mtn{i} {a} vw {b} 0 NNR W=2u L=2u",      # NMOS pass (on when vw high)
                  f"Mtp{i} {a} vwb {b} vdd PNR W=4u L=2u"]   # PMOS pass (on when vwb low) -> symmetric across rail
    else:
        for i in range(N - 1):
            L.append(f"Rc{i} o{i}_0 o{i+1}_0 {RC}")
    L.append(".ic " + " ".join(ics))
    L.append(f".tran {TSTOP}/6000 {TSTOP}")
    L.append(".end")
    open(f"on_{TAG}.scs", "w").write("\n".join(L) + "\n")

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
        nm = ln[1:q]; rest = ln[q+1:].strip()
        if nm == "time": t.append(float(rest))
        elif nm in nset: d[nm].append(float(rest))
    fh.close()
    n = min([len(t)] + [len(d[k]) for k in nodes])
    return np.array(t[:n]), {k: np.array(d[k][:n]) for k in nodes}

def freq(t, v):
    # count rising zero-crossings of (v - 0.5) over the last 70% (skip startup)
    s = int(0.3 * len(t)); v = v[s:] - 0.5; tt = t[s:]
    xings = np.where((v[:-1] < 0) & (v[1:] >= 0))[0]
    if len(xings) < 2: return 0.0
    periods = np.diff(tt[xings])
    return 1.0 / np.mean(periods) if len(periods) else 0.0

if __name__ == "__main__":
    build()
    for f in glob.glob(f"raw_on_{TAG}/*"): os.remove(f)
    r = subprocess.run([f"{SP}/spectre", "+spice", "+mt=1", f"on_{TAG}.scs",
                        "-format", "psfascii", "-raw", f"raw_on_{TAG}", "+log", f"on_{TAG}.log"],
                       capture_output=True, text=True, timeout=400)
    raws = glob.glob(f"raw_on_{TAG}/*.tran.tran") or glob.glob(f"raw_on_{TAG}/*.tran")
    if not raws:
        print("SIM FAILED:", r.stderr[-500:]); raise SystemExit(1)
    nodes = [f"o{i}_0" for i in range(N)]
    t, d = parse(raws[0], nodes)
    fs = [freq(t, d[n]) / 1e6 for n in nodes]  # MHz
    print(f"N={N} RC={RC} DET={DET}: per-oscillator freq (MHz) = {[round(f,2) for f in fs]}")
    fs = [f for f in fs if f > 0]
    if len(fs) >= 2:
        spread = (max(fs) - min(fs)) / np.mean(fs) * 100
        print(f"  freq spread = {spread:.1f}%  ({'LOCKED' if spread < 2 else 'unlocked'})  tsim={t[-1]*1e9:.0f}ns pts={len(t)}")
