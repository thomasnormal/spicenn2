#!/usr/bin/env python3
"""DC transfer characterizer for a candidate neuron cell (forward-feature fidelity screen).

Input : a file containing ONE subckt `.subckt <NAME> up un ap an vdd vbn ... .ends`
        (may instantiate `cmld`; pins MUST be `up un ap an vdd vbn`, output diff = v(ap)-v(an)).
Method: quasi-static Spectre transient ramp of the differential input around CM=0.5V at Vdd=1.0,
        Vbn=0.35 (matches pc_deep). Slow ramp = DC transfer; reuses the tran PSF parser.
Output: JSON metrics vs an ideal A*tanh(g*x):
        swing (full-scale |out|), gain (small-signal d(out)/d(in) at 0), tanh_fit_g, tanh_rms
        (RMS residual / swing, LOWER=more tanh-faithful), monotonic, sat90 (|in| to reach 90% swing),
        headroom_ok (output stays within rails, no clipping to 0/Vdd).
Usage : python3 char_neuron.py <subckt_file> <NAME> [tag]
"""
import os, sys, subprocess, glob, json
import numpy as np

SP = "/opt/cadence/installs/SPECTRE231/bin"
SUB = (".model NNR NMOS (LEVEL=1 VTO=0.2 KP=120u LAMBDA=0.02 GAMMA=0 PHI=0.7)\n"
       ".model PNR PMOS (LEVEL=1 VTO=-0.2 KP=40u LAMBDA=0.02 GAMMA=0 PHI=0.7)")
CMLD = (".subckt cmld p n vdd\n"
        "Rcs1 p cmx 300k\nRcs2 n cmx 300k\n"
        "MLp p cmx vdd vdd PNR W=1000u L=100u\nMLn n cmx vdd vdd PNR W=1000u L=100u\n.ends")

def parse_tran(globpat, nodes):
    f = sorted(glob.glob(globpat))[0]; nset = set(nodes); t = []; d = {n: [] for n in nodes}
    fh = open(f)
    for ln in fh:
        if ln.strip() == "VALUE": break
    for ln in fh:
        ln = ln.strip()
        if ln == "END": break
        if not ln or ln[0] != '"': continue
        q = ln.index('"', 1); nm = ln[1:q]; rest = ln[q+1:].strip()
        if nm == "time": t.append(float(rest))
        elif nm in nset: d[nm].append(float(rest))
    fh.close()
    n = min([len(t)] + [len(d[k]) for k in nodes])
    return np.array(t[:n]), np.array([d[k][:n] for k in nodes]).T

def characterize(subckt_file, name, tag="charneu", lo=0.15, hi=0.85, tramp=2e-6):
    body = open(subckt_file).read()
    deck = "\n".join([
        "neuron DC characterization", "simulator lang=spice", "* title", SUB, CMLD, body.strip(),
        "Vdd vdd 0 1.0", "Vbn vbn 0 0.35", "Vn un 0 0.5",
        f"Vp up 0 PWL(0 {lo} {tramp} {hi})",
        f"X1 up un ap an vdd vbn {name}",
        f".tran {tramp/4000:g} {tramp:g} maxstep={tramp/4000:g}", ".end"])
    open(f"cn_{tag}.scs", "w").write(deck + "\n")
    for f in glob.glob(f"rawcn_{tag}/*"): os.remove(f)
    r = subprocess.run([f"{SP}/spectre", "+spice", "+mt=1", f"cn_{tag}.scs",
                        "-format", "psfascii", "-raw", f"rawcn_{tag}", "+log", f"cn_{tag}.log"],
                       capture_output=True, text=True, timeout=300)
    raws = glob.glob(f"rawcn_{tag}/*.tran.tran") or glob.glob(f"rawcn_{tag}/*.tran")
    if not raws:
        return {"name": name, "error": "no raw / sim failed", "log_tail": r.stderr[-400:]}
    t, dat = parse_tran(raws[0], ["ap", "an"])
    if len(t) < 50: return {"name": name, "error": f"too few points ({len(t)})"}
    x = lo + (hi - lo) * (t / tramp) - 0.5          # differential input around CM
    out = dat[:, 0] - dat[:, 1]                       # v(ap)-v(an)
    # orient so transfer is increasing
    if np.polyfit(x, out, 1)[0] < 0: out = -out
    swing = float(np.max(np.abs(out)))
    # small-signal gain near x=0
    c = np.abs(x) < 0.03
    gain = float(np.polyfit(x[c], out[c], 1)[0]) if c.sum() >= 3 else float("nan")
    # fit A*tanh(g*x)
    def fit_tanh():
        best = (1e9, 0, 0)
        A0 = swing
        for g in np.linspace(2, 200, 80):
            A = swing  # tanh saturates to A
            pred = A * np.tanh(g * x)
            rms = float(np.sqrt(np.mean((out - pred) ** 2)))
            if rms < best[0]: best = (rms, g, A)
        return best
    rms, gfit, A = fit_tanh()
    tanh_rms = float(rms / swing) if swing > 1e-9 else float("nan")
    dout = np.diff(out)
    monotonic = bool(np.mean(dout >= -0.01 * swing) > 0.98)  # robust to sampling jitter
    # input magnitude to reach 90% of swing
    idx = np.where(np.abs(out) >= 0.9 * swing)[0]
    sat90 = float(np.min(np.abs(x[idx]))) if len(idx) else float("nan")
    np.savetxt(f"cn_{tag}.curve", np.column_stack([x, out]))  # input_diff, output_diff for predict_acc
    # headroom: outputs not pinned at rails
    apmin, apmax = float(dat[:, 0].min()), float(dat[:, 0].max())
    anmin, anmax = float(dat[:, 1].min()), float(dat[:, 1].max())
    headroom_ok = bool(apmin > 0.02 and apmax < 0.98 and anmin > 0.02 and anmax < 0.98)
    return {"name": name, "swing": round(swing, 4), "gain": round(gain, 3),
            "tanh_fit_g": round(gfit, 1), "tanh_rms": round(tanh_rms, 4),
            "monotonic": monotonic, "sat90_in": round(sat90, 4) if sat90 == sat90 else None,
            "headroom_ok": headroom_ok, "npts": int(len(t)),
            "ap_range": [round(apmin, 3), round(apmax, 3)], "an_range": [round(anmin, 3), round(anmax, 3)]}

if __name__ == "__main__":
    sf, nm = sys.argv[1], sys.argv[2]
    tag = sys.argv[3] if len(sys.argv) > 3 else "charneu"
    print(json.dumps(characterize(sf, nm, tag)))
