#!/usr/bin/env python3
"""Unranked internal-weight diagnostic using the unchanged organizer models.

This saves internal nodes in a separate tiny harness; it never modifies a submission
or computes, loads, or updates its weights. This is not a benchmark score.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from competition.runner import build_deck, validate_submission


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset",type=Path)
    parser.add_argument("--circuit",type=Path,default=Path(__file__).with_name("circuit.cir"))
    parser.add_argument("--permute-labels",action="store_true",help="cycle only the three training labels as a diagnostic control")
    args=parser.parse_args()
    original=dict(np.load(args.dataset))
    ids=[int(np.flatnonzero(original["train_y"]==k)[0]) for k in (0,1,7)]
    data={"train_x":original["train_x"][ids],"train_y":original["train_y"][ids],
          "test_x":original["train_x"][ids],"test_y":original["train_y"][ids]}
    if args.permute_labels:
        data["train_y"]=np.array([1,7,0])
    circuit,_,_=validate_submission(args.circuit)
    deck,_,train_end,stop=build_deck(circuit,data,2,0,0.001,0.02,1e-5)
    # xentry is the organizer wrapper's instance name. Only observation is added.
    names=[f"v(xentry.n_{prefix}_{key})" for key in ("h0_0","h1_2","o0_0") for prefix in ("wp","wn")]
    deck=deck.replace("save v(out0)","save "+" ".join(names)+" v(out0)")
    deck=deck.replace("\nquit\n", "\nwrdata weights.txt "+" ".join(names)+"\nquit\n")
    directory=Path(tempfile.mkdtemp(prefix="pc-weight-diagnostic-"))
    (directory/"harness.cir").write_text(deck)
    with (directory/"simulator.log").open("w") as log:
        subprocess.run(["ngspice","-b","harness.cir"],cwd=directory,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    trace=np.loadtxt(directory/"weights.txt",skiprows=1)
    def sample(t): return trace[np.argmin(abs(trace[:,0]-t)),1:]
    before=sample(0.01999); trained=sample(train_end-1e-5); after=sample(stop-1e-5)
    result={"directory":str(directory),"permuted_labels":args.permute_labels,"nodes":names,"startup_volts":before.tolist(),
            "training_change_volts":(trained-before).tolist(),"inference_change_volts":(after-trained).tolist()}
    (directory/"diagnostic.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
