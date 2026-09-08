#!/usr/bin/env python3
"""Generate an electrically initialized continuous-PC learner; reads no dataset.

Cell topology adapted from experiments/gen_pcL.py and pc_deep.py in spicenn2.
MIT License; Copyright (c) 2026 Thomas Dybdahl Ahle and contributors.
"""
import argparse
import random
from pathlib import Path


class Circuit:
    def __init__(self):
        self.lines = []
        self.count = 0

    def node(self, name):
        return "n_" + name

    def r(self, a, b, value):
        self.device("R", [a, b, f"{value:.12g}"])

    def c(self, a, b, value):
        self.device("C", [a, b, f"{value:.12g}"])

    def m(self, d, g, s, kind="nch", ratio=2):
        body = "vdd" if kind == "pch" else "0"
        self.device("M", [d, g, s, body, kind, f"W={ratio * 1e-6:.12g}", "L=1u"])

    def device(self, kind, fields):
        self.count += 1
        self.lines.append(f"{kind}{self.count} " + " ".join(fields))

    def divider(self, name, voltage, rail="vdd", rail_voltage=3, resistance=10000):
        node = self.node(name)
        frac = voltage / rail_voltage
        self.r(rail, node, resistance * (1-frac))
        self.r(node, "0", resistance * frac)
        return node

    def load(self, tag, p, n, shunt=True):
        cm = self.node(tag+"_cm")
        self.r(p, cm, 100000)
        self.r(n, cm, 100000)
        self.m(p, cm, "vdd", "pch", 10)
        self.m(n, cm, "vdd", "pch", 10)
        if shunt:
            self.r(p, n, 6666.66666667)
        self.c(p, "0", 10e-12)
        self.c(n, "0", 10e-12)

    def syn(self, tag, inp, inn, wp, wn, op, on, bias):
        """Degenerated Gilbert synapse; shared cap nodes implement transpose wiring."""
        ns = lambda suffix: self.node(tag+suffix)
        self.m(ns("nt"), bias, "0", ratio=4)
        for gate, branch, source in ((wp,"a","s5"),(wn,"b","s6")):
            self.m(ns(branch),gate,ns(source))
            self.r(ns(source),ns("nt"),4000)
        for k,(out,gate,branch) in enumerate(((op,inp,"a"),(on,inn,"a"),(on,inp,"b"),(op,inn,"b"))):
            source=ns(f"s{k}")
            self.m(out,gate,source)
            self.r(source,ns(branch),4000)

    def neuron(self, tag, xp, xn, ap, an, bias):
        tail=self.node(tag+"tail")
        self.m(an,xp,tail)
        self.m(ap,xn,tail)
        self.m(tail,bias,"0")
        self.load(tag,ap,an,False)

    def error(self, tag, xp, xn, mp, mn, ep, en, bias):
        tail=self.node(tag+"tail")
        self.m(tail,bias,"0",ratio=4)
        for out,gate in ((ep,xn),(en,xp),(ep,mp),(en,mn)):
            self.m(out,gate,tail)
        self.r("vdd",ep,16666.6666667)
        self.r("vdd",en,16666.6666667)
        self.c(ep,"0",10e-12)
        self.c(en,"0",10e-12)

    def product(self, tag, xp, xn, yp, yn, wp, wn, bias):
        """Historical gprod: local four-quadrant multiply and differential integration."""
        ns=lambda suffix:self.node(tag+suffix)
        self.m(ns("nt"),bias,"0",ratio=4)
        self.m(ns("na"),yp,ns("nt"))
        self.m(ns("nb"),yn,ns("nt"))
        for out,gate,source in (("iop",xp,"na"),("ion",xn,"na"),("ion",xp,"nb"),("iop",xn,"nb")):
            self.m(ns(out),gate,ns(source))
        for branch in ("iop","ion"):
            self.m(ns(branch),ns(branch),"vdd","pch")
        for out,positive,negative,key in ((wp,"iop","ion","rn"),(wn,"ion","iop","rp")):
            self.m(out,ns(positive),"vdd","pch")
            self.m(ns(key),ns(negative),"vdd","pch")
            self.m(ns(key),ns(key),"0")
            self.m(out,ns(key),"0")


def generate(frozen=False, hidden=8, seed=1, capacitance=100e-6, hidden_sign=1, output_sign=-1):
    c=Circuit()
    c.lines=["* Continuous predictive coding; 16 inputs, classes 0/1/7 on outputs 0/1/7.",
             "* MIT License; Copyright (c) 2026 Thomas Dybdahl Ahle and contributors.",
             f"* Data-independent seed={seed}; hidden={hidden}; frozen_hidden={int(frozen)}.",
             "* Historical gsyn/dneuron/esub/gprod cells; no external optimizer or readout fit."]
    rng=random.Random(seed)
    bsyn=c.divider("bsyn",1.35)
    bneu=c.divider("bneu",1.05)
    errorbias=c.divider("errorbias",1.8)
    updatebias=c.divider("updatebias",1.8,"learn")
    backbias=c.divider("backbias",1.35,"learn")
    midpoint=c.divider("midpoint",1.5)
    inref=c.divider("inref",2.0)
    targetref=c.divider("targetref",1.4)
    inputs=[]
    for i in range(16):
        p=c.node(f"input{i}")
        c.r(f"in{i}",p,10000)
        c.r("vdd",p,10000)
        inputs.append((p,inref))
    targets=[]
    for j in range(3):
        p=c.node(f"target{j}")
        c.r(f"target{(0,1,7)[j]}",p,4000)
        c.r("vdd",p,6000)
        targets.append((p,targetref))
    weights={}
    def weight(key, fixed=False):
        v=max(-0.45,min(0.45,rng.gauss(0,0.25)))
        p,n=c.node("wp_"+key),c.node("wn_"+key)
        for suffix,node,voltage in (("p",p,1.5+v),("n",n,1.5-v)):
            init=c.divider("init_"+key+suffix,voltage,rail="vdd" if fixed else "reset",resistance=100000 if fixed else 100)
            if fixed:
                c.r(init,node,100)
            else:
                c.c(node,"0",capacitance)
                c.r(node,midpoint,1e12)
                c.m(node,"reset",init,ratio=1000)
        weights[key]=(p,n)
        return p,n
    # Two independent units per 2x2 patch at the default width, all pixels represented.
    patches=[[0,1,4,5],[2,3,6,7],[8,9,12,13],[10,11,14,15]]
    parents=[patches[j%4] for j in range(hidden)]
    for j in range(hidden):
        for i in parents[j]: weight(f"h{j}_{i}",frozen)
    for j in range(3):
        for i in range(hidden): weight(f"o{j}_{i}")
    acts=[]
    hidden_nodes=[]
    for j in range(hidden):
        mp,mn=c.node(f"hm{j}p"),c.node(f"hm{j}n")
        xp,xn=(mp,mn) if frozen else (c.node(f"hx{j}p"),c.node(f"hx{j}n"))
        ap,an=c.node(f"ha{j}p"),c.node(f"ha{j}n")
        ep,en=c.node(f"he{j}p"),c.node(f"he{j}n")
        c.load(f"hml{j}",mp,mn)
        if not frozen: c.load(f"hxl{j}",xp,xn)
        for i in parents[j]:
            wp,wn=weights[f"h{j}_{i}"]
            c.syn(f"hfm{j}_{i}",*inputs[i],wp,wn,mp,mn,bsyn)
            if not frozen: c.syn(f"hfx{j}_{i}",*inputs[i],wp,wn,xp,xn,bsyn)
        c.neuron(f"hn{j}",xp,xn,ap,an,bneu)
        if not frozen:
            c.error(f"he{j}",xp,xn,mp,mn,ep,en,errorbias)
            for i in parents[j]:
                err=(ep,en) if hidden_sign>0 else (en,ep)
                c.product(f"hl{j}_{i}",*err,*inputs[i],*weights[f"h{j}_{i}"],updatebias)
        acts.append((ap,an))
        hidden_nodes.append((xp,xn))
    for j in range(3):
        mp,mn=f"out{(0,1,7)[j]}",c.node(f"om{j}n")
        ep,en=c.node(f"oe{j}p"),c.node(f"oe{j}n")
        c.load(f"oml{j}",mp,mn)
        for i in range(hidden):
            wp,wn=weights[f"o{j}_{i}"]
            c.syn(f"of{j}_{i}",*acts[i],wp,wn,mp,mn,bsyn)
        c.error(f"oe{j}",*targets[j],mp,mn,ep,en,errorbias)
        for i in range(hidden):
            wp,wn=weights[f"o{j}_{i}"]
            err=(ep,en) if output_sign>0 else (en,ep)
            c.product(f"ol{j}_{i}",*err,*acts[i],wp,wn,updatebias)
            if not frozen: c.syn(f"bk{j}_{i}",ep,en,wp,wn,*hidden_nodes[i],backbias)
    for j in range(10):
        if j not in (0,1,7): c.r(f"out{j}","0",1000)
    return "\n".join(c.lines)+"\n"


def main(argv=None, default_frozen=False):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True,help="new circuit file; existing paths are never overwritten")
    parser.add_argument("--frozen",action="store_true",default=default_frozen)
    parser.add_argument("--hidden",type=int,default=8)
    parser.add_argument("--seed",type=int,default=1)
    parser.add_argument("--capacitance",type=float,default=100e-6)
    parser.add_argument("--hidden-sign",type=int,choices=(-1,1),default=1)
    parser.add_argument("--output-sign",type=int,choices=(-1,1),default=-1)
    args=parser.parse_args(argv)
    generated=generate(args.frozen,args.hidden,args.seed,args.capacitance,args.hidden_sign,args.output_sign)
    try:
        with args.output.open("x",encoding="utf-8") as output:
            output.write(generated)
    except FileExistsError:
        parser.error(f"output already exists: {args.output}; choose a new path")
    print(args.output)


if __name__=="__main__": main()
