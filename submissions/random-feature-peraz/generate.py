#!/usr/bin/env python3
"""Physical random-feature/PERAZ family port; no data or trained weights are read."""
import argparse
import math
from pathlib import Path
import random


def circuit(features=96, labels=(0, 1, 7), capacitance=5e-6, seed=1, freeze=False):
    rng = random.Random(seed)
    lines = ["* Physical random features + PERAZ analog delta-rule readout",
             "* MIT; Thomas Dybdahl Ahle and contributors; see generate.py and README.md",
             "* Family adaptation, NOT a reproduction of the historical Python-feature score."]

    def r(tag, a, b, value):
        lines.append(f"R{tag} {a} {b} {value:.12g}")

    def c(tag, a, b, value):
        lines.append(f"C{tag} {a} {b} {value:.12g}")

    def m(tag, d, g, s, width=20, model="nch"):
        body = "vdd" if model == "pch" else "0"
        lines.append(f"M{tag} {d} {g} {s} {body} {model} W={width:.12g}u L=1u")

    def divider(tag, source, voltage, total=100000):
        node = "n_"+tag
        r(tag+"hi", source, node, total*(1-voltage/3))
        r(tag+"lo", node, "0", total*voltage/3)
        return node

    def subtractor(tag, cp, cn, out, resistance):
        mirror = "n_mirror"+tag
        m(tag+"a", cn, cn, "0")
        m(tag+"b", out, cn, "0")
        m(tag+"c", cp, cp, "0")
        m(tag+"d", mirror, cp, "0")
        m(tag+"e", mirror, mirror, "vdd", 60, "pch")
        m(tag+"f", out, mirror, "vdd", 60, "pch")
        r(tag+"sum", out, "vref", resistance)
        return mirror

    tail_bias = divider("tailbias", "vdd", 1.4, 10000)
    for j in range(features):
        cp, cn, z = (f"n_{x}{j}" for x in ("hp", "hn", "z"))
        # Fixed, data-independent electrical projections. Both signs arise from
        # the difference between the random gate voltage and the 1.8-V reference.
        field = rng.sample(range(16), 4)
        for k, signal in enumerate([f"in{i}" for i in field]+["vbias"]):
            weight = rng.gauss(0, .5 if k == 4 else 1)
            voltage = max(.6, min(2.8, 1.8+.5*weight))
            gate = divider(f"fixed{j}_{k}", "vdd", voltage)
            m(f"fp{j}_{k}", signal, gate, cp, 10, "syn")
            m(f"fn{j}_{k}", signal, "vweight", cn, 10, "syn")
        subtractor(f"h{j}", cp, cn, z, 8000)
        dd, dm, tail, hv = (f"n_{x}{j}" for x in ("dd", "dm", "tail", "hv"))
        m(f"actp{j}", dd, z, tail, 12)
        m(f"actn{j}", dm, "vref", tail, 12)
        m(f"acttail{j}", tail, tail_bias, "0", 32)
        m(f"actload{j}", dm, dm, "vdd", 40, "pch")
        m(f"actmirror{j}", dd, dd, "vdd", 40, "pch")
        m(f"actout{j}", hv, dd, "vdd", 40, "pch")
        r(f"hvlo{j}", hv, "vref", 9600)
        r(f"hvhi{j}", hv, "vbias", 48000)

    for label in labels:
        cp, cn, error, az, target = (f"n_{x}{label}" for x in ("op", "on", "error", "az", "target"))
        mirror = subtractor(f"o{label}", cp, cn, f"out{label}", 7000)
        # Passive Thevenin network: target voltage = .467 + target_pin/3.
        # Conductances sum to one; pin loading/energy is included by the harness.
        r(f"targetin{label}", f"target{label}", target, 30000)
        r(f"targetref{label}", "vbias", target, 10000/(.467/1.7))
        r(f"targetzero{label}", target, "0", 10000/(1-1/3-.467/1.7))
        m(f"errorp{label}", error, mirror, "vdd", 60, "pch")
        m(f"errorn{label}", error, cn, "0")
        m(f"offset{label}", error, "voffset", "vdd", 10, "pch")
        m(f"target{label}", error, target, "0", 10)
        r(f"error{label}", error, "verror", 7000)
        r(f"common{label}", error, "n_common", 10000)
        # Historical per-class auto-zero: slow local error mean, not a label-
        # fitted calibration. Gate tracking off at evaluation along with weights.
        r(f"az{label}", error, f"n_azfeed{label}", 6000)
        m(f"aztrack{label}", f"n_azfeed{label}", "0" if freeze else "learn", az, 1000, "syn")
        c(f"az{label}", az, "0", 10e-6)
        m(f"azreset{label}", "verror", "reset", az, 1000, "syn")
        for j, signal in enumerate([f"n_hv{k}" for k in range(features)]+["vbias"]):
            tag = f"{label}_{j}"
            weight, tail, up, enable = (f"n_{x}{tag}" for x in ("w", "utail", "up", "enable"))
            initial = divider("init"+tag, "reset", 1.8+.12*rng.uniform(-1, 1), 1000)
            c("w"+tag, weight, "0", capacitance)
            m("reset"+tag, initial, "reset", weight, 1000, "syn")
            m("pos"+tag, signal, weight, cp, 384/features, "syn")
            m("neg"+tag, signal, "vweight", cn, 384/features, "syn")
            m("enable"+tag, enable, "0" if freeze else "learn", "0", 1000)
            m("utail"+tag, tail, signal, enable)
            m("u1_"+tag, up, az, tail)
            m("u2_"+tag, weight, error, tail)
            m("u3_"+tag, up, up, "vdd", 60, "pch")
            m("u4_"+tag, weight, up, "vdd", 60, "pch")
    c("common", "n_common", "0", 50e-15)
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--features", type=int, choices=range(1, 97), default=96)
    parser.add_argument("--labels", choices=("0,1,7", "0,1,2,3,4,5,6,7,8,9"), default="0,1,7")
    parser.add_argument("--capacitance", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.capacitance) or not 1e-16 <= args.capacitance <= 1e-3:
        parser.error("capacitance must be in [1e-16, 1e-3] farads")
    try:
        with args.output.open("x") as output:
            output.write(circuit(args.features, tuple(map(int, args.labels.split(","))),
                                 args.capacitance, args.seed, args.freeze))
    except OSError as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
