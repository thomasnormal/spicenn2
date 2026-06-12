# Ten neurons for depth — which works best

Each neuron = a forward activation f(z) + a backward derivative f'(z) (the chip must supply both).
Tested on the checkerboard-K family (rising K = more compositional/high-frequency = needs depth).

## Tier 1 — isolate the activation (ideal linear layers + Adam + exact backprop)

Test accuracy, deep [6,8,8,8,2], best of 3 seeds (chance 50):

| neuron | K=2 | K=3 | K=4 | analog cell |
|---|---|---|---|---|
| **ReLU² (current chip)** | 98 | 62 | 54 | square-law MOS |
| tanh | 98 | **85** | 55 | differential pair |
| SELU | 98 | 64 | 56 | subthreshold exp |
| abs / fold | 98 | 62 | 55 | full-wave rectifier |
| sine (SIREN) | 72 | 54 | 53 | folding/PLL |
| **Gaussian / RBF** | 96 | **93** | 58 | subthreshold diff-pair bump |
| softplus | 96 | 60 | 54 | subthreshold MOS |
| log-companding | 97 | 67 | 55 | translinear log |
| RMSNorm+tanh | 96 | 73 | 55 | Gilbert normalizer + diff pair |
| quad / σ-π | 98 | 85 | 55 | translinear multiplier |

**Real-depth check** — deep [8,8,8] vs *matched-parameter* wide-shallow [24], K=3 (depth-gain):
ReLU² **−3** (depth *hurts*), SELU +1, abs +4, quad +5, tanh **+13**, **gauss +19**.

**Depth-scaling curves** (test acc vs #hidden layers, K=3 / K=4):
```
            1L  2L  3L  4L  5L
ReLU²(K3)   69  86  80  72  56   <- peaks at 2L then DEGRADES with depth
tanh (K3)   77  83  93  96  96   <- rises with depth, saturates 96
gauss(K3)   80  96  96  96  96
ReLU²(K4)   60  68  58  53  57   <- depth hurts
tanh (K4)   54  60  64  72  76   <- rises every layer to 5L
gauss(K4)   55  60  60  68  79   <- still climbing at 5L
```

**Winner: tanh and Gaussian/RBF.** They convert depth from harmful (ReLU² — accuracy falls with
depth) to monotonically beneficial (deeper is better, to 5 layers). The chip's ReLU² is
**depth-hostile**. tanh is the most buildable (classic diff pair); Gaussian gives the largest
depth-gain (localized bumps compose hierarchical partitions efficiently).

**Why** — signal preservation through stacked *analog* layers (effective rank / activation std):
```
        L1    L2    L3    L4    L5    L6
ReLU²  6.83  4.46  4.42  3.20  1.73  1.53   std 0.228 -> 0.003   COLLAPSES
tanh   6.47  5.80  4.89  3.95  3.11  3.17   std 0.163 -> 0.027   HOLDS
```
Stacked one-sided ReLU² destroys the signal (rank->1.5, magnitude->0); bounded two-sided tanh
(and localized Gaussian) keep it alive. This *is* Finding-5 part-3 made concrete and fixable.

## Tier 2 — the catch (real analog front-end: keystone + approximate transpose-read backward)

| neuron | shallow | deep | depth-gain |
|---|---|---|---|
| ReLU² | 52 | 57 | +4 |
| tanh | 60 | 59 | −1 |
| gauss | 59 | 57 | −3 |

On the actual analog substrate the depth benefit **largely evaporates** for every neuron, and
absolute accuracy is far below ideal (analog tanh deep ~55–59% vs ideal tanh deep ~96% on K=3).
tanh/gauss are still better *shallow* (60 vs 52). So: **a better neuron is necessary but not
sufficient.** The signal-preservation probe shows the tanh forward is fixed (rank ~3, not
collapsed) — therefore the remaining eroder is the **approximate backward** (the transpose-read's
~0.5 cosine, established earlier). Its infidelity was harmless when ReLU² collapsed the forward
(nothing to propagate); once the neuron preserves signal, the backward becomes the binding
constraint. (The earlier "backward is fine" was true only in the ReLU² regime.)

## Bottom line / path to analog depth

1. **Swap ReLU² → tanh (or Gaussian/RBF).** Cheap, buildable, and decisive in the ideal limit:
   it flips depth from harmful to monotonically helpful, and preserves signal through ≥6 layers.
2. **Then the backward must improve** — once the forward preserves signal, the ~0.5-cosine
   transpose-read becomes the limit. This is where a more faithful gradient (calibrated transpose
   transconductance, or Equilibrium Propagation, which sidesteps the Jacobian entirely) finally
   earns its keep — *after* the neuron is fixed, not before.

Net: the neuron question is answered (tanh / Gaussian win, ReLU² is depth-hostile). Analog depth
needs neuron + backward fixed together; fixing either alone is not enough.

## Refinements (simpler / more elegant / more robust / scaling)

**The principle — corrected.** The tidy hypothesis "depth ⟺ the neuron preserves signal rank" is
**false** (corr = −0.56). Counterexamples: **sine** has the highest rank-retention (7.8) but the
worst depth-gain (**−35**); **gauss** has low rank (2.6) but the best depth-gain (+19). What
actually predicts depth benefit is the activation *shape*: **bounded on both sides + smoothly
saturating + non-periodic**. Failure modes are distinct — one-sided/expansive (ReLU², softplus)
collapses signal; periodic (sine) scrambles the composed gradient; a *hard* clip (hardtanh, +5)
underperforms the *smooth* saturator (tanh, +16), so smoothness matters, not just boundedness.

**Simplest / most elegant = plain tanh.** A smooth two-sided saturator is exactly the
differential pair — the most basic analog primitive. And the usual deep-net crutches are **not
needed and even hurt**: orthogonal init gives no gain, and residual skips *reduce* accuracy
(K=3: tanh 93 → tanh+residual 83; the identity path dilutes depth's compositional work). The
elegant answer is just tanh, nothing bolted on.

**Robustness — tanh dominates within capacity.** 10 seeds, K=3, deep:
| neuron | mean | std | min |
|---|---|---|---|
| ReLU² (chip) | 84 | 9.2 | 68 |
| **tanh** | **93** | **4.5** | **81** |
More accurate *and* more reliable on every metric. At the capacity *edge* (K=4, 5 hidden) it
becomes a seed lottery (mean 70, std 14, min→chance) — but that is intrinsic to operating at the
edge, not fixable by init/residual.

**Scaling — depth pays off up to a capacity wall.** Width fixed at 8 (no width): depth helps
monotonically through K=3 and climbs to ~84% at 7 layers on K=4; **K=5 is flat at chance for
every neuron** — a genuine width/capacity wall, not a neuron limit. So the neuron unlocks depth's
benefit *within* the representable regime; pushing the regime itself needs more capacity (the one
lever excluded here) or a more capacity-efficient neuron (Gaussian peaks higher — 95 at K=4 — but
is too high-variance to rely on).

**One-line refined result:** *the depth-enabling neuron is the simplest smooth two-sided
saturator — the differential pair (tanh); it beats the chip's ReLU² on accuracy, robustness, and
depth-scaling simultaneously, with no extra machinery, up to the width-capacity wall.*

## Real-silicon follow-up: a BOUNDED (tanh/diff-pair) neuron that trains XOR in ngspice

The neuron findings above were NumPy-surrogate. Built the bounded neuron as a real SPICE cell
(`gen_xor_full.py ACT=tanh`, `gen_infer.py`): a **differential pair** whose tail current caps the
output, so hv saturates (sigmoid/tanh) instead of ReLU²'s unbounded square-law. Getting it to
actually *train* XOR took finding the right pieces, in order:

1. Forward alone: hv is bounded (validated) but units sit stuck at the diff-pair center -> 2/4.
2. **Weight noise** un-sticks the stuck units (actrange 0.33->1.82) but does NOT make it learn XOR
   — a **red herring** here; the units became active but the net learned a *linear* (OR-ish)
   readout, not XOR.
3. **The real fix = the BACKWARD.** A plain input-gated OTA (no activation-derivative gate) can't
   form the hidden XOR features. Reusing the proven **comparator / ReLU'-style gate** (threshold
   at the tanh center VMID) on the tanh forward trains it: **6/8 seeds 4/4**, and with the right
   backward **noise is unneeded and even hurts**.
4. **Wider output swing** (bigger tail current, `TW=32 VTB=1.4`) crisps the soft corners:
   **7/8 seeds reach 4/4**, MSE 0.035-0.161 (seed 1 y=[0.22,1.02,0.80,0.22]).

Recipe (ngspice): `ACT=tanh TW=32 VTB=1.4 VMID=0.5 OWTW=20` + the comparator backward, no noise.

**Takeaways:** (a) a bounded neuron *is* a buildable, trainable SPICE cell now — the prerequisite
for any real-silicon depth test; (b) for shallow XOR, ReLU² is still slightly more reliable/crisp
(8/8, bigger swings) — tanh's bound only pays off with depth; (c) the binding constraint for a new
neuron is the **backward gate**, not the forward — exactly the Tier-2 inference, now confirmed in
silicon (the plain-OTA backward fails; the comparator gate succeeds).

### RELIABLE: 8/8 seeds train XOR with the bounded tanh neuron
At H=6 it was 7/8 (one init's (1,0) corner structurally stuck — no good checkpoint exists, so
freeze/early-stop can't rescue it). Adding capacity (**H=12**) gives that corner room: **8/8 seeds
reach 4/4** (MSE 0.032-0.125; crisp seeds e.g. y=[0.24,0.96,0.91,0.24]).

**Final reliable recipe (ngspice):**
`ACT=tanh TW=32 VTB=1.4 VMID=0.5 OWTW=20 H=12 BWD=trans2` + comparator backward, no noise, AVGW=40.

Reliability ladder that got here: bounded diff-pair forward (no blow-up) + comparator/ReLU'-style
backward gate (NOT plain OTA — the real fix) + wide output swing (TW) + enough capacity (H). Noise
was a red herring; a single H=12 run trains XOR every time. (`restart_xor.sh` also gives reliability
the orthogonal way — retry-on-failure — even at H=6.)
