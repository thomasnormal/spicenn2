#!/usr/bin/env python3
# Convert an ngspice trainer deck (gen_mc.py / gen_xor_full.py output) to a Xyce-compatible deck
# that runs the transient FROM the .ic capacitor voltages (the stored weights), exactly like
# ngspice's `uic`, with NO DC operating point moving the cap voltages.
#
# Why this is non-trivial (and how it is solved here):
#   * Xyce's `.TRAN ... UIC` sets every UNSPECIFIED node to ZERO at t=0.  For this MOS-mirror /
#     integrator network that zero guess makes the very first Newton solve at t=0 diverge
#     ("OneStep::rejectStep: Maximum number of failures at time 0  *** Xyce Abort ***").
#     => We give Xyce a fully CONSISTENT t=0 starting point for ALL ~800 nodes, not just the caps,
#        as an enforced .IC block:
#          - cap (weight) nodes  -> the EXACT original .ic strings (HARD REQUIREMENT: v@t=0 == weight,
#            verified bit-for-bit; a DC operating point is never run, so nothing moves them);
#          - all other nodes     -> ngspice's consistent t=0 values, captured by running ngspice for
#            ONE `uic` step and reading every node.  (Seeding the non-cap nodes is what fixes the
#            t=0 abort.  Using the EXACT .ic on the caps -- not ngspice's ~1e-6-drifted readback --
#            also matters: this network is extremely stiff and seed-sensitive, and the exact values
#            let the run reach noticeably deeper into the transient.)
#   * Trapezoidal integration (Xyce default) CANNOT integrate this stiff weight-integrator network:
#     it collapses the time step ("Time step too small ... Exiting transient loop") within a few
#     hundred steps -- regardless of UIC vs DCOP.  GEAR (BDF) is stable and runs deep into the
#     transient -- up to the intrinsically-stiff terminal region (t ~= 43-50s) where ngspice ITSELF
#     also aborts (~t=49.8s).  So in practice both tools cover essentially the same usable window.
#   * NEWLTE must stay at its default (1, global-max reference).  NEWLTE=0/3 make the per-node LTE
#     control so tight on the near-zero cap/output nodes that the step collapses and the run dies early.
#
# The conversion runs ngspice itself (must be on PATH) to capture the t=0 seed.  Override the
# reference netlist/binary or tolerances via env vars if needed.
#
# Usage: python3 to_xyce.py mc.cir mc_xyce.cir
import sys, re, os, subprocess

src = sys.argv[1] if len(sys.argv) > 1 else "mc.cir"
dst = sys.argv[2] if len(sys.argv) > 2 else "mc_xyce.cir"

# --- tunables (env overridable) --------------------------------------------------------------
# These are the loosest tolerances that still track ngspice; they let GEAR push deepest into the
# transient before the genuinely-stiff terminal region (t ~= 43-50s) collapses the time step.
# That terminal region is intrinsically stiff: ngspice ITSELF aborts there (~t=49.8s,
# "timestep too small, node ts_*"), so neither tool integrates much past it.  Tighter tolerances
# (reltol<=7.5e-3 or abstol<=1e-6) just make Xyce die a little earlier, not later.
RELTOL  = os.environ.get("XYCE_RELTOL",  "1e-2")
ABSTOL  = os.environ.get("XYCE_ABSTOL",  "1e-5")
METHOD  = os.environ.get("XYCE_METHOD",  "gear")   # gear (BDF) is MANDATORY; trapezoidal collapses
                                                   # the step within a few hundred steps on these
                                                   # stiff weight-integrator caps.
MAXORD  = os.environ.get("XYCE_MAXORD",  "2")
PRNFILE = os.environ.get("XYCE_PRN",     "xyce_out.prn")
NGSPICE = os.environ.get("NGSPICE_BIN",  "ngspice")
# ---------------------------------------------------------------------------------------------

L = open(src).read().splitlines()
ci = next(i for i, l in enumerate(L) if l.strip().lower().startswith(".control"))
ei = next(i for i, l in enumerate(L) if l.strip().lower().startswith(".endc"))
head = L[:ci]                                   # title, models, devices, sources, .ic  (Xyce-OK)
ctrl = L[ci:ei + 1]

# pull tran params and the printed vectors out of the .control block
tran = next(l for l in ctrl if l.strip().lower().startswith("tran"))
m = re.match(r"\s*tran\s+(\S+)\s+(\S+)", tran)
dt, tstop = m.group(1), m.group(2)
vecs = []
for l in ctrl:
    if l.strip().lower().startswith("wrdata"):
        vecs += re.findall(r"v\(([^)]+)\)", l)
vecs = list(dict.fromkeys(vecs))                # dedup, keep order

# parse the ORIGINAL .ic line(s): these are the cap (weight) voltages and MUST be reproduced
# EXACTLY at t=0 (the HARD REQUIREMENT).  We keep them verbatim and override any ngspice readback.
ic_exact = {}
for l in head:
    if l.strip().lower().startswith(".ic"):
        for mm in re.finditer(r"v\(([^)]+)\)\s*=\s*(\S+)", l, re.I):
            ic_exact[mm.group(1).lower()] = mm.group(2)

# --- 1. enumerate every circuit node (from the device/source lines in the head) --------------
nodes = set()
for l in head:
    s = l.strip()
    if not s or s.startswith("*") or s.startswith("."):
        continue
    t = s.split()
    c = t[0][0].upper()
    if c == "M":                                # M name d g s b model ...
        nodes.update(t[1:5])
    elif c in "RCVILDQ":                         # 2-terminal-ish: name n1 n2 ...
        nodes.update(t[1:3])
nodes.discard("0")
nodes = sorted(nodes)

# --- 2. run ngspice for ONE tiny `uic` step to get a consistent t=0 value for every node ------
# Seed step == the real .TRAN step.  ngspice's first reported point is then its first internal
# substep (~dt/100), the same near-t=0 state ngspice itself uses to launch the full run, so the
# seed lands on ngspice's actual trajectory.  (A much smaller seed step gives slightly different
# values on a few sensitive internal nodes, enough to push the stiff terminal region off-track.)
seed_dt   = dt
seed_dump = os.path.abspath("_xyce_seed_dump.txt")
seed_cir  = os.path.abspath("_xyce_seed.cir")
vecstr = " ".join("v(%s)" % n for n in nodes)
seed_deck = list(head) + [
    ".control",
    "  tran %s %s uic" % (seed_dt, seed_dt),
    "  wrdata %s %s" % (seed_dump, vecstr),
    "  echo XYCE_SEED_OK",
    ".endc",
    ".end",
]
open(seed_cir, "w").write("\n".join(seed_deck) + "\n")
r = subprocess.run([NGSPICE, "-b", seed_cir], capture_output=True, text=True)
if "XYCE_SEED_OK" not in (r.stdout + r.stderr):
    sys.stderr.write(r.stdout + "\n" + r.stderr + "\n")
    raise SystemExit("ngspice seed run failed (needed for a consistent Xyce t=0 start)")

# parse the wrdata: interleaved (time,value) pairs; take the first data row's value columns
seed = {}
with open(seed_dump) as f:
    first = next(l for l in f if l.strip())
    cols = [float(x) for x in first.split()]
    vals = cols[1::2]                           # value columns
    assert len(vals) == len(nodes), (len(vals), len(nodes))
    seed = dict(zip(nodes, vals))

# --- 3. emit the Xyce deck -------------------------------------------------------------------
# Drop the original ngspice `.ic` line: we re-emit ALL nodes (caps + the rest) as enforced .IC.
out = [l for l in head if not l.strip().lower().startswith(".ic")]

# full .IC block.  Cap (weight) nodes get their EXACT original .ic string (hard requirement:
# v(cap)@t=0 == weight, bit-for-bit); every other node gets ngspice's consistent t=0 value.
chunk = []
for n in nodes:
    if n.lower() in ic_exact:
        chunk.append("v(%s)=%s" % (n, ic_exact[n.lower()]))
    else:
        chunk.append("v(%s)=%.8g" % (n, seed[n]))
    if len(chunk) == 12:
        out.append(".IC " + " ".join(chunk)); chunk = []
if chunk:
    out.append(".IC " + " ".join(chunk))

# convergence / integration options:
#   GEAR (BDF) is required -- trapezoidal collapses the step on these stiff weight integrators.
out.append(".options TIMEINT method=%s reltol=%s abstol=%s maxord=%s" % (METHOD, RELTOL, ABSTOL, MAXORD))
# NOTE: do NOT add a `.options NONLIN reltol/abstol` here.  Loosening the Newton (NONLIN)
# tolerance to match the time-integrator tolerance makes the step collapse in the stiff
# terminal region (the full 60s run then dies ~t=43-49).  Xyce's default NONLIN tolerances
# are what let the GEAR run complete the whole transient.
# Xyce: .TRAN <printstep> <tstop> <tstart> <maxstep> UIC.  UIC + the full .IC above => the
# transient starts from the weight voltages with NO bias-point solve moving them.
out.append(".TRAN %s %s 0 %s UIC" % (dt, tstop, dt))
out.append(".PRINT TRAN FORMAT=NOINDEX FILE=%s " % PRNFILE + " ".join("V(%s)" % v for v in vecs))
out.append(".END")
open(dst, "w").write("\n".join(out) + "\n")

for f in (seed_cir, seed_dump):                  # remove scratch files from the ngspice seed run
    try: os.remove(f)
    except OSError: pass

print("wrote %s: .TRAN %s %s (maxstep=%s) UIC, %s GEAR reltol=%s" % (dst, dt, tstop, dt, METHOD, RELTOL))
print("      %d printed nodes, %d seeded .IC nodes, %d MOSFETs"
      % (len(vecs), len(nodes), sum(1 for l in head if l[:1] == "M")))
print("NOTE: Xyce .prn output is a clean column table (TIME, then V(...)), NOT ngspice's")
print("      interleaved (t,val) pairs -> parse columns directly (no [1::2]).")
