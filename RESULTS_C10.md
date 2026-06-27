# In-Circuit C=10 Digit Classification — Result & Lever Map

**Headline:** A fully in-circuit (LEVEL-1 MOSFET, Spectre, 1.0 V supply) analog network reaches
**0.744–0.752** accuracy on 10-class sklearn digits, single net (no ensemble), 3-seed confirmed,
Hungarian-scored (bijective anti-lock). The surrogate (ideal-tanh PyTorch PC) ceiling for the same
frozen-feature architecture is ~0.84; the circuit realizes ~89 % of it.

## The accuracy arc (all real Spectre, 3-seed)

| step | change | C=10 acc | what it fixed |
|------|--------|----------|---------------|
| baseline | local-RF sparse readout (fan-in 4) | **0.448** | spatial features alive (10/10 classes) but readout starved |
| readout fix | dense readout `KOUT=16` | **0.669** (+0.22) | un-starved the readout: each class reads all 16 features |
| frozen-wide | `LAYERS=64,49,36`, dense `KOUT=36`, `NOHID=1` | **0.744** (+0.075) | richer frozen RF features feeding the dense readout |

Best config trains fast: NEP=4 already reaches 0.752 (the frozen readout converges in ~4 epochs).

## Best recipe (the 0.744 net)

```
LAYERS=64,49,36  KOUT=36  NOHID=1            # frozen random RF features + trained dense readout
RFGRID=1 RFK=2 FANIN=4                       # local receptive fields (square pyramid)
CHL=3 HCHOP=4 SYMNUDGE=1 ZEROSUM=1 PERAZ=1   # contrastive readout training (symmetric ±β nudge)
SGNH=-1 SGNO=-1 CWW=300p RWL=5meg            # deep-PC update polarity + weight caps
TH=400 STEP=2n  GBLH=0.6 GBLO=0.6 VBBK=0.6   # per-slot horizon, neuron/backward bias
C=10 NTR=40 NTE=25 NEP=12  TASK=digits  SIM=spectre MT=4
```

## The key insight

The in-circuit **hidden-learning backward path is broken** (see below), so training the hidden layer
makes things worse than freezing it. The win came from **giving up on the broken backward** and instead
extracting maximum value from the **stable forward path**: rich *frozen* random receptive-field features
fed into an *un-starved* dense readout (which trains reliably). Frozen-wide here means more *parallel*
features (less aggressive pooling, 36 vs 16), not a deeper/trained-wide net.

## Lever map — everything tested, what's closed

| lever | result |
|-------|--------|
| forward gain (RCS / DNW / WINIT / cascode / VBNEU) | **null** — confirmed in both sparse and frozen regimes |
| readout sparsity (fan-in) | **FIXED** → 0.669 (dense KOUT) |
| frozen-feature richness (more features read) | **FIXED** → 0.744 (64,49,36) |
| feature-count scaling (49, 64 features) | **saturates** — 49≈36 in-circuit (surrogate keeps climbing) |
| hidden learning (BKSIGN / BKDFA) | **exhausted** — BKDFA chaos (0.42 at any DFAW); BKSIGN sign-loss → frozen-equiv |
| per-slot settling (TH) | **refuted** — TH=800 ≤ TH=400 |
| readout linearity (LINRO degeneration) | **refuted** — more linear monotonically worse (gain loss) |
| neuron sharpness/gain (DNW, VBNEU) | **null** — wider worse, higher-tail tied |
| data (cost-neutral NTR↔NEP swap) | **refuted** — more data needs more epochs, no free lunch |

**The residual gap to the ~0.84 surrogate is analog device-cell fidelity** (the neuron transfer
deviates from ideal tanh, and no parameter closes it). Beyond ~0.745 requires a device-cell redesign,
not a software/parameter/architecture change.

## Why the hidden backward is broken (the diagnosis)

The surrogate proves only a **true signed transpose** backward improves hidden learning (88.3 % vs
73.6 % frozen). In-circuit:
- **BKDFA** (direct feedback alignment, fixed-random feedback) → 0.42 at *every* feedback strength
  (DFAW 0.1/0.2/1.0 all identical) — it corrupts the features and breaks the analog readout lock.
- **BKSIGN** (comparator-regenerated sign) → lands at frozen-equivalent (the regenerated sign isn't
  faithful enough to help).

So the analog current-mode backward can't deliver a faithful signed transpose, and every alternative
either does nothing (frozen-equiv) or destabilizes (chaos).

## Methodology notes (the real deliverable)

- **3-seed means OR a robust invariant** for every accuracy claim (single-seed C=4 Spectre is ±0.2).
- **Hungarian bijective** anti-lock scoring (scipy `linear_sum_assignment` on the confusion matrix);
  greedy-max inflates one-class collapse into false signal.
- Surrogate (PyTorch, ideal-tanh PC) for fast multi-seed mechanism tests; Spectre for invariants and
  multi-seed confirmation of reportable numbers.
- Two mid-investigation hypotheses (readout-sum noise, under-settling) were tested and **refuted**,
  and the record corrected — the surrogate-vs-silicon divergence is itself a finding.

Tooling: `experiments/pc_surrogate.py`, `experiments/hung_eval.py` (Hungarian on a chosen eval block of
a partial raw; takes `TH` from env), `experiments/feat_count.py`, `experiments/ntr_scan.py`.
