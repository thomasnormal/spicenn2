#!/usr/bin/env python3
"""Data-independent 16 -> 9 -> 3 analog CMSUB backprop circuit generator.

Adapted from experiments/gen_mc.py by Thomas Dybdahl Ahle and contributors.
MIT licensed. Only the standard library is used; no dataset or weight file is read.
"""
import argparse
import math
from pathlib import Path
import random


def circuit(capacitance=1.67e-6, seed=5, freeze_hidden=False, freeze_all=False):
    rng = random.Random(seed)
    lines = ["* CMSUB analog backprop, 16 inputs / 9 trained hidden / labels 0,1,7",
             "* Thomas Dybdahl Ahle and contributors; MIT; see generate.py and README.md",
             "* Adapted from experiments/gen_mc.py; deterministic random reset, no pretrained weights."]
    def r(tag, a, b, value):
        lines.append(f"R{tag} {a} {b} {value:.12g}")
    def c(tag, a, b, value):
        lines.append(f"C{tag} {a} {b} {value:.12g}")
    def m(tag, drain, gate, source, width=20, model="nch"):
        body = "vdd" if model == "pch" else "0"
        lines.append(f"M{tag} {drain} {gate} {source} {body} {model} W={width:.12g}u L=1u")
    def divider(tag, high, low, fraction, resistance=10000):
        node = f"n_{tag}"
        r(tag+"hi", high, node, resistance*(1-fraction))
        r(tag+"lo", node, low, resistance*fraction)
        return node
    vtb = divider("vtb", "vbias", "vref", .75)
    vcbias = divider("vcbias", "verror", "vref", 5/7)
    vulk = divider("vulk", "verror", "vref", 1/14, 1000)
    # Historical update inputs were 0.3..1.0 V while forward inputs were 0.3..1.7 V.
    # These resistor networks implement au = input/2 + 0.15 without another driven pin.
    au = {}
    for i in range(16):
        node = f"n_au{i}"
        r(f"auin{i}", f"in{i}", node, 10000)
        r(f"auref{i}", "vref", node, 10000/.6)
        r(f"augnd{i}", node, "0", 10000/.4)
        au[i] = node
    au["b"] = vcbias

    def synapse(tag, signal, cp, cn, initial):
        weight = f"n_w{tag}"
        c("w"+tag, weight, "0", capacitance)
        # A reset-powered divider consumes energy only during startup. Its fixed
        # ratios encode a random seed, not a learned classifier. The switch then opens.
        init = divider("init"+tag, "reset", "0", initial/3, 1000)
        m("reset"+tag, init, "reset", weight, 1000, "syn")
        m("pos"+tag, signal, weight, cp, 10, "syn")
        m("neg"+tag, signal, "vweight", cn, 10, "syn")
        return weight

    def subtractor(tag, cp, cn, output, resistance):
        mirror = f"n_dpf{tag}"
        m("sa"+tag, cn, cn, "0")
        m("sb"+tag, output, cn, "0")
        m("sc"+tag, cp, cp, "0")
        m("sd"+tag, mirror, cp, "0")
        m("se"+tag, mirror, mirror, "vdd", 60, "pch")
        m("sf"+tag, output, mirror, "vdd", 60, "pch")
        r("sum"+tag, output, "verror" if tag.startswith("bk") else "vref", resistance)
        return mirror

    def update(tag, error, reference, activation, weight, hidden_gate=None):
        tail, mirror, enabled = (f"n_{x}{tag}" for x in ("tail", "up", "enable"))
        control = "0" if freeze_all or (freeze_hidden and hidden_gate) else "learn"
        m("enable"+tag, enabled, control, "0", 1000)
        if hidden_gate:
            middle = f"n_middle{tag}"
            m("htail"+tag, tail, hidden_gate, middle, 48)
            m("hinput"+tag, middle, activation, enabled, 48)
        else:
            m("tail"+tag, tail, activation, enabled)
        m("u1"+tag, mirror, reference, tail)
        m("u2"+tag, weight, error, tail)
        m("u3"+tag, mirror, mirror, "vdd", 60, "pch")
        m("u4"+tag, weight, mirror, "vdd", 60, "pch")

    hidden = {}
    fields = [[4*row+col, 4*row+col+1, 4*row+col+4, 4*row+col+5]
              for row in range(3) for col in range(3)]
    for j, field in enumerate(fields):
        cp, cn, z, hv = (f"n_{x}{j}" for x in ("hp", "hn", "z", "hv"))
        hidden[j] = {}
        for i in field+["b"]:
            initial = 2.16 if i == "b" else 1.8+1.2*round(rng.uniform(-.8, .8), 3)
            hidden[j][i] = synapse(f"h{j}_{i}", "vbias" if i == "b" else f"in{i}", cp, cn, initial)
        subtractor(f"h{j}", cp, cn, z, 8000)
        dd, dm, tail = (f"n_{x}{j}" for x in ("hdd", "dmp", "ntail"))
        m(f"r{j}", dd, z, tail, 12)
        m(f"r2_{j}", dm, "vref", tail, 12)
        m(f"tl{j}", tail, vtb, "0", 32)
        m(f"dm{j}", dm, dm, "vdd", 40, "pch")
        m(f"rp1_{j}", dd, dd, "vdd", 40, "pch")
        m(f"rp2_{j}", hv, dd, "vdd", 40, "pch")
        # Thevenin equivalent of historical 8k to an ideal 0.7-V source.
        r(f"hvlo{j}", hv, "vref", 9600)
        r(f"hvhi{j}", hv, "vbias", 48000)
        ct, ca, on = (f"n_{x}{j}" for x in ("ctail", "cna", "on"))
        m(f"ct{j}", ct, vcbias, "0", 10)
        m(f"ca{j}", ca, z, ct, 20, "syn")
        m(f"cb{j}", on, "vref", ct, 20, "syn")
        m(f"cp1_{j}", ca, ca, "vdd", 40, "pch")
        m(f"cp2_{j}", on, ca, "vdd", 40, "pch")
        r(f"ulk{j}", on, vulk, 60000)

    output = {}
    for label in (0, 1, 7):
        cp, cn, error, negative, ps, ns = (f"n_{x}{label}" for x in ("op", "on", "err", "errn", "eps", "ens"))
        # Avoid collision with the hidden comparator's n_on0/n_on1/n_on7.
        cn = f"n_ocn{label}"
        output[label] = {}
        for j in list(range(9))+["b"]:
            signal = "vbias" if j == "b" else f"n_hv{j}"
            weight = synapse(f"o{label}_{j}", signal, cp, cn, 1.8+1.2*round(rng.uniform(-.3, .3), 3))
            output[label][j] = weight
            update(f"o{label}_{j}", error, "verror", signal, weight)
        mirror = subtractor(f"o{label}", cp, cn, f"out{label}", 7000)
        m(f"ep{label}", error, mirror, "vdd", 60, "pch")
        m(f"en{label}", error, cn, "0")
        m(f"off{label}", error, "voffset", "vdd", 10, "pch")
        m(f"target{label}", error, f"target{label}", "0", 10)
        r(f"err{label}", error, "verror", 7000)
        m(f"qa{label}", ps, cn, "0")
        m(f"qb{label}", ps, f"target{label}", "0", 10)
        m(f"pd{label}", ps, ps, "vdd", 60, "pch")
        m(f"pm{label}", negative, ps, "vdd", 60, "pch")
        m(f"qc{label}", ns, mirror, "vdd", 60, "pch")
        m(f"qd{label}", ns, "voffset", "vdd", 10, "pch")
        m(f"nd{label}", ns, ns, "0")
        m(f"nm{label}", negative, ns, "0")
        r(f"errn{label}", negative, "verror", 7000)
        for suffix, error_node in (("p", error), ("n", negative)):
            node = f"n_buf{suffix}{label}"
            m(f"buf{suffix}{label}", "vdd", error_node, node, 200, "syn")
            r(f"buf{suffix}{label}", node, "0", 20000)

    for j in range(9):
        cp, cn, error = (f"n_{x}{j}" for x in ("bkp", "bkn", "bkerr"))
        for label in (0, 1, 7):
            weight = output[label][j]
            m(f"ta{j}_{label}", f"n_bufp{label}", weight, cp, 10, "syn")
            m(f"tb{j}_{label}", f"n_bufn{label}", "vweight", cp, 10, "syn")
            m(f"tc{j}_{label}", f"n_bufp{label}", "vweight", cn, 10, "syn")
            m(f"td{j}_{label}", f"n_bufn{label}", weight, cn, 10, "syn")
        subtractor(f"bk{j}", cp, cn, error, 16000)
        r(f"cmsub{j}", error, "n_bkmean", 10000)
        for i, weight in hidden[j].items():
            update(f"h{j}_{i}", error, "n_bkmean", au[i], weight, f"n_on{j}")
    c("cmsub", "n_bkmean", "0", 50e-15)
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--capacitance", type=float, default=1.67e-6)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--freeze-hidden", action="store_true", help="untrained-hidden control")
    parser.add_argument("--freeze-all", action="store_true", help="untrained control")
    args = parser.parse_args()
    if not math.isfinite(args.capacitance) or not 1e-16 <= args.capacitance <= 1e-3:
        parser.error("capacitance must be in [1e-16, 1e-3]")
    try:
        with args.output.open("x") as stream:
            stream.write(circuit(args.capacitance, args.seed, args.freeze_hidden, args.freeze_all))
    except OSError as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
