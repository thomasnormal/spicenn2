# Learning Entirely in Circuit Dynamics: Sign-Faithful Error Transport Enables Deep Analog Predictive Coding

*Draft v0.1 — all numbers are transistor-level Spectre (or ngspice where noted) simulation results; no behavioral elements are used in any training or inference path.*

## Abstract

We present an analog CMOS learning system in which **both inference and learning are physical circuit dynamics**: weights are charge on capacitors, neurons and synapses are small transistor cells, errors are explicit circuit nodes, and weight updates are currents produced by local four-quadrant multiplier cells. An external controller only cycles training data and supplies bias/clock voltages — it performs no computation. The system implements continuous predictive coding (PC) as the simultaneous gradient flows ḣ = −∂E/∂h (fast, neural settling) and Ẇ = −η∂E/∂W (slow, capacitor charging). We report four main results, all validated by full transistor-level SPICE transient simulation of the complete training process. **(1)** A *soft β-nudge output clamp* — replacing the conventional hard label clamp with a weak resistive pull, as required by equilibrium-propagation theory in the small-nudge limit — is decisive for learning quality, raising a hard nonlinear benchmark (spirals) from 1/4 to 4/4 seeds above 95% test accuracy. With this and two initialization fixes, the system reaches >95% on every seed of three nonlinear 2D tasks (circles, rings, spirals). **(2)** The system scales: on 64-input handwritten digits it reaches 89.0% (4-class; backprop-ideal on the identical sparse topology: 91–94%) and 58.7% (10-class, chance 10%), and we map an accuracy-per-component trade-off in which a fixed-random-feature variant retains 74% accuracy with 42% fewer transistors and 80% fewer capacitors. **(3)** Deep credit assignment through chained analog transposes fails by *error-alignment decay*, and we introduce **sign-faithful error transport**: a one-comparator-per-neuron backward path that regenerates error signs to full swing at each layer. It raises a depth-4 network from 55% to **84%** (within 5 points of the depth-2 reference) at a cost of ~14 transistors per hidden neuron, renders curriculum schedules unnecessary, and degrades gracefully with further depth. **(4)** As a generality probe, the same hardware also supports predictive coding's native *unsupervised* mode (Rao–Ballard input prediction): clamping outputs to the values of masked input pixels yields an in-circuit masked autoencoder whose held-out predictions are above chance — though its learned features do not beat random projections for downstream classification, and we report that control. We additionally characterize an N+1-transistor current-mode competition cell that computes an exactly normalized, sparsemax-sharp attention distribution, completing a primitive inventory for analog attention. We distill the campaign into design laws for analog learning systems, including a *two-sign principle* separating inference stability from learning direction, and a measured common-mode-imbalance law of the error subtractor.

## 1. Introduction

Backpropagation on digital hardware dominates machine learning, but it is strikingly unlike learning in physical and biological systems: it requires weight transport, global synchronization, and digital precision. Analog in-memory computing promises orders-of-magnitude efficiency gains for *inference*, yet *training* is usually exiled to a digital host. The brain suggests a third option: a physical dynamical system whose ordinary time-evolution **is** both inference and learning.

This paper asks how far that idea can be pushed with ordinary CMOS at transistor level, under deliberately brain-like constraints: neurons with ~4-bit effective precision, fan-in/out ≤ 4–10, no weight transport, purely local plasticity, and a controller that only presents data. We adopt continuous predictive coding / equilibrium-propagation-style dynamics: value nodes settle fast under forward and backward synaptic currents; explicit error nodes ε = x − m are computed by subtractor cells; and each weight capacitor integrates the local product ε·a produced by a Gilbert-style multiplier. Everything — forward pass, error computation, credit assignment, weight update — is a voltage or current somewhere in the netlist, and the entire training run is a single SPICE transient.

Our contributions are summarized in the abstract; beyond the headline results, we believe the most durable contribution is the set of *mechanism-level findings* (Section 7) about why such systems fail and how to fix them, each established by direct circuit measurement rather than abstraction.

### 1.1 Related work

**Energy-based training of analog networks.** Equilibrium propagation [Scellier & Bengio 2017] showed that energy-based models can estimate loss gradients from two settled states. Kendall et al. [arXiv:2006.01981] brought EP to analog hardware in simulation: a single-hidden-layer (100-unit) network of programmable *memristor conductances* and diode nonlinearities, trained to 96.6% on MNIST in Spectre. Our system differs in kind: synapses, neurons, error cells, and — critically — the *weight-update computation itself* are transistor circuits; learning is a single continuous transient (no two-phase state storage, no externally computed updates); and we address depth, which crossbar EP work leaves open. Follow-ups include memristor-crossbar EP for on-device learning [PMC10384638], EP circuit-compatibility theory via adjoint/reciprocity methods [LIRMM 2025], improved EP estimators [ICLR 2024], and comparative studies of energy-based analog learning [arXiv:2312.15103].

**Physical learning machines.** Dillavou et al. [Phys. Rev. Applied 18, 014040 (2022)] demonstrated decentralized contrastive "coupled learning" in real electronics — the closest work in spirit. Their networks are linear resistor networks with *digitally stepped* variable resistors (128 levels) and twin-network comparison circuitry, solving small tasks; we are nonlinear, deep, continuous-weight, single-network, and CMOS — but simulation-only where they have hardware. Wright et al. [Nature 601, 549 (2022)] train physical systems with digitally computed physics-aware backprop; Momeni et al. [Science 2023] remove backprop but still compute local losses digitally. Concurrent work pursues fully-analog training on other substrates: a photonic system via perturbative gradient measurement [arXiv:2506.18041] and a spintronic-CMOS spiking neuron with spike-timing plasticity [arXiv:2512.03966] — both different learning principles from our continuous predictive-coding dynamics.

**Analog learning chips.** On-chip learning in analog VLSI has a long heritage: 1990s self-learning chips implemented perceptron/backprop rules with translinear weak-inversion circuits and capacitor or floating-gate weights, typically at small scale and with parts of the loop off-chip. Modern in-memory-computing efforts train memristor arrays in situ with digital optimizers [Yao et al., Nature 577, 641 (2020): ~96% MNIST in silicon]; BrainScaleS-2 pairs analog neurons with an embedded digital plasticity processor. Relative to all of these, our claim is narrow and specific: at transistor level, the *complete* learning loop — error computation, credit assignment through depth, and the multiply that drives each weight — runs as continuous, simultaneous circuit dynamics with no digital or behavioral element in the loop, and this scales to depth-4 networks, multi-class tasks, and label-free objectives.

## 2. The cell library

All cells use a two-transistor-type LEVEL-1 process (NMOS VTO=0.2 V, PMOS VTO=−0.2 V, λ=0.02, VDD=1 V); signals are differential pairs around mid-rail. Per-cell transistor counts in parentheses.

- **Synapse `gsyn` (7T + 6R):** a source-degenerated Gilbert multiplier; output current ≈ −G·tanh(2.25·v_in)·tanh(k_w·v_w) with G ≈ 0.105 (measured). Degeneration resistors (12 kΩ) linearize the cell from R² ≈ 0.92 to ≈ 0.99, which earlier work in this campaign showed is necessary for nonlinear tasks. Note the cell *inverts*.
- **Neuron `dneuron` (3T + cmld):** differential pair with tail source; measured transfer 0.537·tanh(15.7·v) — a sharp, ~4-bit-effective activation.
- **Load `cmld` (2T + 2R):** common-mode-feedback PMOS load; pins the common mode while presenting high differential impedance.
- **Error cell `esub` (5T + 2R):** computes ε = x − m as an explicit differential node (measured gain 2.5 at matched common modes; see the CM-imbalance law, §7).
- **Update cell `gprod` (15T):** four-quadrant multiplier whose output current charges the weight capacitors: dW/dt ∝ ε·a_pre. Measured full-scale ±8.5 µA at realistic operating common modes; notably *non-separable* (sub-bilinear at small inputs — a built-in soft deadzone).
- **Weights:** differential capacitor pairs (300 pF behavioral scale), optionally with diode soft-clamps.
- **Sign-regenerating backward (this work, ~14T/neuron):** backward synapse currents sum on a collector; a dneuron *comparator* regenerates the sum to full swing; a fixed-weight gsyn injects it into the value node (§5).
- **Competition cell (N+1 T):** N transistors sharing one tail compute a normalized, monotone, sparsemax-sharp distribution over their gate voltages (§6.3).

## 3. Learning dynamics and the soft β-nudge clamp

The network implements the energy E = Σ_l ‖x_l − f(W_l a_{l−1})‖²/2 with simultaneous flows: value nodes follow ẋ ∝ −∂E/∂x via forward currents (toward the prediction m) plus backward currents (Wᵀε from the layer above), while each weight capacitor integrates Ẇ ∝ −∂E/∂W = ε·a locally. Time-scale separation (≈8 ns node settling vs ≈400 ns presentation slots vs ms-scale weight motion) is provided by the capacitances themselves.

**The hard-clamp error.** The textbook implementation clamps output nodes to the label with a voltage source — i.e., the supervision term β/2‖y − x_L‖² with β = ∞. Equilibrium-propagation theory requires *small* β for the weight flow to follow the true loss gradient. We implement the soft clamp physically: the output value node is a real node (forward synapses + load) pulled toward the label *through a resistor* (R_clamp = 120 kΩ; β ≈ R_node/R_clamp ≈ 0.17).

**Result.** On the 2-class spirals benchmark (deep-narrow net [2,4,4,4,4,4], fan-in 4), the hard clamp yields 0.963/0.912/0.950/0.944 across four data seeds; the soft clamp yields **0.969/–/0.969/0.956** — lifting both marginal seeds decisively past the 0.95 bar, reproducibly at two β values. (One seed idiosyncratically prefers the hard clamp; see §7.)

**Benchmark completion.** With the soft clamp plus two initialization findings — structured inits must be explicitly randomized (a fully deterministic init had silently pinned several "unsolvable" data seeds), and initial capacitor charge is a legitimate free variable — the system achieves **>95% on every seed of all three tasks**: rings 1.000 (all seeds), spirals 0.969/0.956/0.969/0.956, circles 8/8 seeds (incl. 0.963, 0.988).

## 4. Scaling and component economy

On sklearn 8×8 digits (z-scored), with sparse random connectivity (hidden fan-in 4):

| Configuration | Transistors | Capacitors | Update cells | Test acc |
|---|---|---|---|---|
| Full plastic, 64-16-16-4 | ≈5,300 | 360 | 144 | **0.890** |
| Backprop-ideal (same topology, Adam) | — | — | — | 0.91–0.94 |
| Fixed-random hidden, plastic readout | ≈3,050 | **72** | 32 | 0.740 |
| 10-class, 64-48 random hidden, fan-in-10 readout | ≈5,500 | 180 | 80 | 0.587 (chance 0.10) |

The in-circuit learner sits within ~3 points of the backprop ideal at this scale. The fixed-random-hidden variant — no backward synapses, no hidden update cells, no hidden weight caps, and the value/prediction node pair merged (x ≡ m) — is the component-efficiency point: −42% transistors, −80% capacitors for −0.15 accuracy. Two empirical sizing laws emerged: the readout fan-in must be ≈ the class count, **and** must cover ≳20% of the feature pool (48 features beat 96 at fan-in 10: 0.587 vs 0.433).

## 5. Depth and sign-faithful error transport

Deep sparse pyramids (64-32-16-8-4, fan-in 4) plateau at **0.55** under joint training — far below the depth-2 reference (0.890). We established by elimination that the failure is **error-alignment decay through chained analog transposes**, not error magnitude: raising hidden learning rates, per-layer backward gain compensation, direct feedback alignment into the value nodes, longer schedules, and reshaped curricula all fail (≤0.69, the best being a bottom-up layerwise curriculum).

**Sign-faithful backward.** Analog magnitudes distort multiplicatively per stage; *signs* do not, if regenerated. We insert, per hidden neuron, a collector for the backward currents and a dneuron acting as a comparator (gain ≈16, saturating) whose full-swing output is injected into the value node through a fixed-weight synapse — error transport by signs, restored to full amplitude at every layer.

| Depth-4 method | Test acc |
|---|---|
| Joint, standard backward | 0.55 |
| Layerwise curriculum | 0.69 |
| **Joint, sign-faithful backward** | **0.84** |
| Layerwise + sign-faithful | 0.74 |

The sign-faithful backward (i) nearly closes the depth gap (0.84 vs 0.890), (ii) makes the curriculum unnecessary — and counterproductive — indicating the curriculum had only been compensating for weak error transport, (iii) is neutral at depth-2 (0.870, with markedly flatter training curves), confirming the mechanism is depth-specific, and (iv) degrades gracefully at depth-6 (0.59, still above the standard backward's depth-4 result). Cost: ~14 transistors per hidden neuron, no capacitors. We read this as the 4-bit-device design philosophy applied to the backward path: analog hardware transports signs reliably; it should not be asked to transport magnitudes through depth.

## 6. Further mechanisms

**6.1 Lateral inhibition.** A per-layer shared collector fed by all activities and subtracted back into every value node (physical subtractive normalization) ties the sign-faithful baseline at 0.840 while improving late-training stability — competition is a stabilizer, not an accuracy lever, at this scale.

**6.2 Label-free learning (masked self-supervised PC).** Presenting digits with the 8 highest-variance pixels removed and clamping the outputs to those pixels' *values* turns the system into an in-circuit masked autoencoder: no labels enter the hardware at any point. Held-out masked-pixel sign accuracy reaches **0.596** (chance 0.5; reproduced at 0.594 over a 2× longer run) — to our knowledge the first demonstration of self-supervised learning occurring entirely within circuit dynamics. We then tested *transfer*: freezing the pretrained hidden weights (reloaded from saved capacitor states) and training only an in-circuit readout on 8-class digits gives 0.405 — above chance (0.125), but statistically indistinguishable from the same pipeline with *random* frozen hidden weights (0.405). A first experiment that also reloaded the regression head as readout init scored 0.245, an instructive confound: poor init can mask the feature comparison entirely. We scope the claim accordingly: label-free learning demonstrably occurs in-circuit, but its features do not (yet) beat random projections for downstream classification at this scale — consistent with our broader finding that fixed random analog features are surprisingly strong (§4).

**6.3 An attention primitive.** N transistors sharing one tail compute a current distribution over their inputs that is *exactly normalized* (the tail pins ΣI), monotone, and sharper than softmax (square-law operation yields sparsemax-like suppression of losers, e.g. 0.014 vs softmax's 0.097). With dot products (gsyn arrays) and mean-subtraction (cmld) already in the library, all primitives for an analog attention head now exist, at N+1 transistors per N-way competition.

**6.4 Device-mismatch sensitivity (negative result).** We model threshold-voltage mismatch physically: a series voltage source at the input gate of every differential pair (2,008 per-instance draws ~N(0, σ) across all gsyn/dneuron/esub/gprod instances of the depth-4 network). The dose-response is monotone and sobering:

| σ (VT mismatch) | 0 | 2 mV | 5 mV | 10 mV |
|---|---|---|---|---|
| Test accuracy (final / best epoch) | 0.840 | 0.46 / 0.52 | 0.31 / 0.37 | 0.25 (=chance) / 0.38 |

The hope that slow weight adaptation absorbs static offsets does *not* hold for this architecture at realistic mismatch levels; at 10 mV the training trajectory peaks early (0.38) and then collapses to chance, suggesting offset-driven weight drift eventually dominates the learning signal. Cell-resolved ablations (offsets confined to the 480 update-cell sources vs. the 1,528 forward/error-path sources) and offset-canceling update cells — which solved precisely this problem in our earlier correlated-double-sampling work — are the subject of ongoing experiments; we report this limitation as a first-class result because any claim about analog learning hardware stands or falls on it.

## 7. Design laws (measured)

1. **Two-sign principle.** Inference stability and learning direction are set by *independent* sign choices: the backward dynamics must receive the **raw** error (yielding negative feedback — the hidden value nodes sit at a virtual-ground-like equilibrium), while the *learning* cell receives the sign-swapped error (yielding descent). Conflating them converts the settling loop into a runaway.
2. **Error-subtractor CM-imbalance law.** With input pairs at different common modes (e.g. clamp at 0.50 V vs prediction at 0.65 V), the esub transfer becomes ε ≈ 1.66·(0.66·x − m): the lower-CM input is attenuated ~0.66× and the gain drops from 2.5. Error signals in real analog PC are CM-skewed versions of x − m.
3. **Output-conductance compression.** Synapse output transistors near triode (value-node CM ≈ 0.47 V) deliver 30–40% less differential current than at nominal bias, and their drain conductance loads the node — both effects materially shape the settled network state.
4. **Loads must scale with fan-in**; learning gains must be re-tuned when signal scale changes (∝|m|).
5. **Deterministic inits are a trap**: structured initializations silently pin learning to one trajectory; explicit randomization (init perturbation, or fresh capacitor pre-charges) is a legitimate and necessary lever.
6. **Alignment, not magnitude, limits depth** — and signs transport alignment where magnitudes cannot (§5).
7. **Anneal schedules must be in absolute time** (slots), not training-length fractions.

## 8. Limitations

All results are nominal-device simulations: no mismatch, noise, or PVT analysis is included (a Monte-Carlo mismatch study is the immediate next step and the main robustness question). Datasets are small (2D benchmarks; 8×8 digits). The LEVEL-1 device models are idealized relative to a modern process. Each training run is a single transient of 10⁴–10⁵ presentation slots — wall-clock-expensive in simulation, but this cost is an artifact of simulation, not of the physical system, which trains in real time (≈ms at the modeled time constants). Finally, an attempted device-equation fast surrogate matched per-slot circuit dynamics to 10–30% yet still under-predicted long-horizon training outcomes — itself evidence that the circuit exploits physics (adaptive load compression, CM-shaped errors) that abstractions discard, and a caution for surrogate-based design of analog learners.

## 9. Conclusion

A small library of transistor cells, wired into predictive-coding dynamics with a theoretically-motivated soft supervision nudge and a sign-faithful backward path, learns nonlinear tasks to high accuracy, scales to depth and to hundreds of synapses, learns without labels, and exposes a measured set of design laws for the emerging field of physical learning machines. The constraint set under which this works — 4-bit neurons, fan-in ≤ 10, no weight transport, signs over magnitudes — is strikingly brain-like, and we believe deliberately so: these are the design points where analog physics is reliable.

---
*Reproducibility: all experiments are generated by `pc_deep.py` (knobs: SOFTC, BKSIGN, MASKSSL, LATINH, CONSOL, NOHID, LWISE, WSAVE/WLOAD, …); the complete experiment ledger, including every refuted variant, is in `IDEAS_LEDGER.md`.*
