#!/usr/bin/env python3
"""Predict downstream feature quality of a neuron transfer curve (frozen-feature C=10 proxy).

Uses the measured DC transfer curve (cn_<tag>.curve: input_diff, output_diff) AS THE ACTIVATION in a
fixed random deep net (matching the analog topology 64-36-16), then trains a closed-form ridge readout.
Same random features / same pre-activation scaling for every candidate, so differences come purely from
the transfer SHAPE (gain, tanh-fidelity, saturation, asymmetry). This isolates "forward-feature fidelity".

Reports 3-seed test accuracy and the delta vs the ideal-tanh activation run through the identical harness.
NOT a substitute for a real Spectre training run -- it's a fast SCREEN to rank cells; winners get Spectre.

Usage: python3 predict_acc.py <curve_file>        # prints {"acc":..,"acc_tanh":..,"delta":..}
       python3 predict_acc.py --tanh               # baseline reference only
"""
import sys, json
import numpy as np
from sklearn.datasets import load_digits

SIZES = [64, 36, 16]
SCALE = 0.13   # maps unit-std pre-activation into the curve's input domain (~+-0.3); SAME for all cells

def _data():
    d = load_digits(); X = d.data.astype(float); y = d.target
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    return X, y

def _features(X, act, seed):
    rng = np.random.RandomState(seed)
    h = X
    for i in range(len(SIZES) - 1):
        W = rng.randn(SIZES[i] if i else X.shape[1], SIZES[i + 1]) / np.sqrt(SIZES[i] if i else X.shape[1])
        h = act(SCALE * (h @ W))   # pre-activation scaled into activation domain, then transfer
    return h

def _ridge_acc(act, seed):
    X, y = _data()
    rng = np.random.RandomState(seed); idx = rng.permutation(len(X))
    ntr = int(0.7 * len(X)); tr, te = idx[:ntr], idx[ntr:]
    H = _features(X, act, seed)
    Htr, Hte = H[tr], H[te]
    T = -np.ones((len(tr), 10)); T[np.arange(len(tr)), y[tr]] = 1.0
    A = Htr.T @ Htr + 1e-2 * np.eye(Htr.shape[1])
    Wout = np.linalg.solve(A, Htr.T @ T)
    pred = (Hte @ Wout).argmax(1)
    return float((pred == y[te]).mean())

def _act_from_curve(curve_file):
    c = np.loadtxt(curve_file); x, out = c[:, 0], c[:, 1]
    o = np.argsort(x); x, out = x[o], out[o]
    sw = np.max(np.abs(out)) + 1e-9
    outn = out / sw                      # normalize swing to ~+-1
    if np.polyfit(x, outn, 1)[0] < 0: outn = -outn
    return lambda u: np.interp(u, x, outn, left=outn[0], right=outn[-1])

def acc_3seed(act):
    a = [_ridge_acc(act, s) for s in (0, 1, 2)]
    return float(np.mean(a)), float(np.std(a))

if __name__ == "__main__":
    tanh_act = lambda u: np.tanh(16.0 * u)   # ideal high-gain tanh reference (matches dneuron g~16)
    if sys.argv[1] == "--tanh":
        mu, sd = acc_3seed(tanh_act); print(json.dumps({"acc_tanh": round(mu, 4), "sd": round(sd, 4)}))
    else:
        act = _act_from_curve(sys.argv[1])
        mu, sd = acc_3seed(act); tmu, _ = acc_3seed(tanh_act)
        print(json.dumps({"acc": round(mu, 4), "sd": round(sd, 4),
                          "acc_tanh_ref": round(tmu, 4), "delta_vs_tanh": round(mu - tmu, 4)}))
