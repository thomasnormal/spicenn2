#!/usr/bin/env python3
"""TRAINED-net proxy for neuron-cell ranking (fixes the frozen-ridge linearity artifact).

The frozen-feature ridge screen (predict_acc.py) rewards LINEARITY (a linear classifier wins), so it
cannot rank neurons for the real net, which TRAINS its hidden layers. This proxy plugs the measured DC
transfer curve in as the activation of the deep PC trainer (train_faithful) and TRAINS the hidden layers,
so a near-linear cell correctly COLLAPSES (deep-linear = linear) and a useful nonlinearity is rewarded.

Calibration: baseline dneuron fits out ~= swing*tanh(17*x), so a(m)=curve(m/G_REF)/swing with G_REF=17
makes the baseline cell == tanh(m); every other cell differs by its REAL gain/shape at that operating point.

Usage: python3 predict_trained.py <curve_file> [EP]   # prints {"acc_trained":..,"acc_tanh":..,"delta":..}
       python3 predict_trained.py --tanh [EP]
"""
import sys, json
import numpy as np, torch

G_REF = 17.0
_cx = np.array([-1.0, 1.0]); _cout = np.array([-1.0, 1.0]); _cder = np.array([1.0, 1.0])

src = open("experiments/train_faithful.py").read().split('print("=== HIDDEN')[0]
exec(src)                       # defines sat, act, dact, relax, tf, avg, build, ... in module globals
_base_act, _base_dact = act, dact

def act(m, kind):               # noqa: F811  (override to add kind="curve")
    if kind == "curve":
        u = (m / G_REF).detach().cpu().numpy()
        a = np.interp(u, _cx, _cout, left=_cout[0], right=_cout[-1])
        return torch.tensor(a, dtype=m.dtype, device=m.device)
    return _base_act(m, kind)

def dact(x, kind):              # noqa: F811
    if kind == "curve":
        u = (x / G_REF).detach().cpu().numpy()
        da = np.interp(u, _cx, _cder, left=0.0, right=0.0) / G_REF
        return torch.tensor(da, dtype=x.dtype, device=x.device)
    return _base_dact(x, kind)

def load_curve(path):
    global _cx, _cout, _cder
    c = np.loadtxt(path); x, out = c[:, 0], c[:, 1]
    o = np.argsort(x); x, out = x[o], out[o]
    sw = np.max(np.abs(out)) + 1e-9; outn = out / sw
    if np.polyfit(x, outn, 1)[0] < 0: outn = -outn
    _cx, _cout = x, outn
    _cder = np.gradient(outn, x)

if __name__ == "__main__":
    EP = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    tmu, tsd = avg(g=0, bsign="graded", kind="tanh", EP=EP)
    if sys.argv[1] == "--tanh":
        print(json.dumps({"acc_tanh": round(tmu, 2), "sd": round(tsd, 2), "EP": EP}))
    else:
        load_curve(sys.argv[1])
        mu, sd = avg(g=0, bsign="graded", kind="curve", EP=EP)
        print(json.dumps({"acc_trained": round(mu, 2), "sd": round(sd, 2),
                          "acc_tanh": round(tmu, 2), "delta_vs_tanh": round(mu - tmu, 2), "EP": EP}))
