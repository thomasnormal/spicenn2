Here’s the distilled checklist. The big trend is that serious analog-training papers almost never rely on “raw SGD directly into a sloppy analog weight.” They add **averaging, normalization, common-mode cancellation, reference cancellation, delayed transfer, clipping, quantization-aware training, and freeze/threshold logic** around the analog core.

## First: which papers are actually relevant?

The strictest relevant set is smaller than the broader “analog NN” literature:

| Paper / family                                              |                                                                                                                   How close it is to your criteria | Largest model trained / demonstrated                                                                                                                                                                          |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Equilibrium Propagation in nonlinear resistive networks** |                  Very relevant: SPICE/Spectre analog circuit, local learning, inference and training in the same analog network. Not real silicon. | MNIST with **100 hidden neurons**, 10 epochs, **3.43% test error**; architecture uses doubled inputs/outputs because conductances are positive-only.                                                          |
| **Conventional MOSFET analog NN on-chip learning**          |                                                          Relevant: SPICE, normal MOSFETs, not memristors/PCM/floating gates. Very small benchmark. | Fisher Iris FCNN. The paper emphasizes conventional silicon MOSFET synapses and SPICE-level neuron/update circuits. ([arXiv][1])                                                                              |
| **IBM mixed-precision PCM/capacitor training**              |                                      Important but mixed hardware–software: analog memory participates in training, with digital/software support. | Up to **204,900 synapses**, with results on MNIST, MNIST-backrand, CIFAR-10 and CIFAR-100. ([Nature][2])                                                                                                      |
| **Silicon-synapse memristive DBN / RBM**                    |         Real CMOS-compatible floating-gate memristive arrays. Physical in-situ training of a small RBM; larger DBN mainly device-model simulation. | Hardware RBM: **19 visible × 8 hidden**. Simulated DBN includes a **784×500** first RBM layer and reached **97.05% MNIST**. ([arXiv][3])                                                                      |
| **M-SDC memristor SPICE system**                            | Very practical component style: commercially available silver M-SDC memristor models in SPICE/Proteus plus Arduino-like control. Very small model. | **30 memristors, 4 neurons**, binary classification of 3×3-pixel images. ([arXiv][4])                                                                                                                         |
| **Hybrid FeCAP/memristor memory for training + inference**  |   Real memory/circuit measurements plus hardware-aware NN simulations. Promising, but not a full large NN trained entirely as one fabricated chip. | MNIST **784–200–100–10**, Fashion-MNIST **784–200–100–100–10**, ECG **20–200–1**. ([Nature][5])                                                                                                               |
| **IBM c-TTv2 / AGAD analog in-memory training algorithms**  |                                         Extremely useful algorithmically, but mostly AIHWKit-level device simulations rather than SPICE/full chip. | Small FC/CNN/LSTM plus a **4.3M-parameter ViT on CIFAR-10** as a scaling check. ([Nature][6])                                                                                                                 |
| **Integrated photonic on-chip training papers**             |                                                          Real hardware and on-chip training, but less “normal cheap components” for your purposes. | FICONN: 3-layer photonic DNN with **12 nonlinear optical function units and 3 coherent matrix units**, vowel classification; newer photonic BP papers show small nonlinear classification tasks. ([arXiv][7]) |

## Techniques worth stealing

### 1. Use the **same physical circuit** for forward and backward/error phases

This is the central EqProp trick. Instead of building a separate backward path whose mismatch ruins gradients, the circuit is allowed to settle once in the free phase and once in a weakly nudged phase. The local weight update comes from the difference between the squared voltage drops across the same resistor in the two phases. This is very close to your “timescale separation” idea: settle first, update later. 

Practical version: for each sample, do

[
\Delta g_{ij} \propto -\frac{(\Delta V_{ij}^{\beta})^2 - (\Delta V_{ij}^{0})^2}{\beta}
]

with the smallest nudging current that gives a usable signal. Do not update weights while the network is still moving unless you are explicitly doing continual EP with very slow synapses.

### 2. Randomize or symmetrize the nudging direction

The EP scaling papers found that finite nudging creates a biased gradient estimate; using both positive and negative nudges cancels much of that bias and made EP scale from MNIST-like tasks toward CIFAR-10 convnets. ([arXiv][8])

Practical version: instead of always applying the same sign of output teaching current, alternate or randomize (+\beta) and (-\beta), then combine the two estimates. This is the analog cousin of chopper stabilization.

### 3. Keep nudging small, but not so small that noise dominates

Several papers converge on a similar principle: small perturbations are more gradient-correct, but too-small signals disappear into circuit/device noise. EqProp uses small (\beta); c-TTv2/AGAD normalize update magnitudes; photonic and memristive systems add calibration/averaging around noisy analog signals. 

Practical version: sweep the teaching-current scale and plot three curves: train accuracy peak, collapse time, and gradient signal-to-noise. The best value is often below the fastest-learning value.

### 4. Normalize backward error by output count or activity scale

Your “scale backward error by (1/C)” is strongly aligned with what the better AIMC papers do, even if they do not phrase it as multi-class stabilization. IBM’s AGAD/c-TTv2 work dynamically scales the update-array learning rate by running averages of the input and backpropagated-gradient magnitudes, precisely to remove layer-size and gradient-scale dependence. ([Nature][6])

Practical version: make the error-injection current approximately invariant to class count:

[
I_{\mathrm{err},k} = \eta \frac{t_k - y_k}{C}
]

or normalize by (\sum_k |t_k-y_k|). This directly attacks your C=2 works, C=3 collapses symptom.

### 5. Use **differential outputs**, not absolute class voltages

The SPICE EqProp paper handles positive-only conductances by doubling outputs: each class has (Y_k^+) and (Y_k^-), and the score is (Y_k^+ - Y_k^-). This is a cheap common-mode rejection trick. 

Practical version: do not train absolute output voltages. Train differences. Your lateral inhibition / COMP idea is a stronger version: subtract the output common-mode before generating teaching currents.

### 6. Add output common-mode subtraction or competition

I did not see many hardware papers using an explicit softmax circuit, but the trend is clear: successful systems avoid letting irrelevant common-mode output energy become a learning signal. Differential outputs, reference subtraction, cross-entropy EP, and batch/activation normalization are all versions of this. 

Practical version: for C outputs, inject error from

[
e_k = (t_k - y_k) - \frac{1}{C}\sum_j(t_j-y_j)
]

instead of raw (t_k-y_k). This keeps only the relative class error.

### 7. Avoid “one-vs-rest pulls everything down hard”

Your “milder negative targets” idea is well-motivated. One-vs-rest squared-error targets scale the total negative pull with (C-1), which can push hidden units into saturation. The EP scaling literature moved beyond plain squared-error EP by deriving cross-entropy-compatible variants, and the original SPICE EqProp paper already uses differential class scores rather than a single absolute output per class. ([arXiv][8])

Practical version: target the winning class strongly and the non-targets weakly, or train centered targets such as target (= 1), non-target (= -1/(C-1)). That keeps the total target vector near zero.

### 8. Gate feedback by local activation slope

This is your f′ gating idea. In digital backprop this is mandatory; in analog EP the physical circuit partially provides it “for free” because the perturbed steady state depends on the local I–V slope of each nonlinear element. Newer photonic BP papers explicitly frame on-chip training as needing scalable activation-gradient computation. 

Practical version: add a local transconductance/slope proxy so saturated hidden units receive little or no backward error. For sigmoid/tanh circuits, a simple bump current proportional to activity near midrange is probably worth the area.

### 9. Use bidirectional gain to prevent vanishing signals

The SPICE EqProp architecture added a bidirectional amplifier: voltage gain forward, current gain backward, so activity and error do not decay through layers. 

Practical version: if hidden errors are weak or delayed, add a symmetric-ish gain element rather than increasing global teaching current. Increasing global teaching current risks exactly the runaway you are seeing.

### 10. Clip conductances and enforce hard legal ranges

The SPICE EqProp simulations clip conductances to a positive floor because real conductances cannot go below zero. AIMC papers similarly impose conductance ranges/states. 

Practical version: add hard clamps on weight caps or update currents. Do not let the learning circuit integrate into unreachable or saturated regimes.

### 11. Separate **fast inference weights** from **slow/high-precision learning state**

This is one of the biggest trends. IBM’s 2018 mixed-precision training used PCM for long-term storage, capacitors for near-linear volatile updates, and transfer mechanisms to bridge them. The 2025 FeCAP/memristor paper uses high-precision FeCAP hidden weights updated on every input and low-precision memristor analog weights updated only periodically. ([Nature][2])

Practical version: do not ask the main analog weight to be both the precise optimizer state and the inference weight. Keep a hidden learner state, then periodically project/transfer into the actual analog conductance/cap.

### 12. Transfer weights only every (k) samples

The FeCAP/memristor paper found that updating analog memristor weights every (k=100) inputs preserved MNIST accuracy while cutting programming energy substantially; too-large (k) hurt accuracy, too-small (k) wasted endurance and energy. ([Nature][5])

Practical version: your “freeze-on-convergence” can be generalized to “sample-and-hold learning”: fast hidden state moves every sample, physical weights move every (k) samples, then eventually stop.

### 13. Accumulate update requests and fire only when a threshold is crossed

The silicon-synapse RBM does not write every tiny contrastive-divergence update. It accumulates ternary update requests in counters and sends potentiation/depression pulses only when the counter crosses a threshold. ([arXiv][9])

Practical version: put a leaky or digital-ish accumulator in front of each weight update. This reduces write noise, endurance use, and high-frequency update chatter.

### 14. Blind write when precision is not worth the feedback loop

The silicon-synapse work uses “blind write” for speed: once the accumulated update threshold is crossed, send a pulse without verifying/fine-tuning every update. ([arXiv][9])

Practical version: use write-verify only for initialization or final consolidation, not every SGD micro-step. In a volatile-cap system, this translates to avoiding expensive correction unless a weight is near a boundary.

### 15. Chopper stabilization / polarity inversion

IBM’s 2018 work used polarity inversion to cancel device-to-device variation; the 2024 c-TTv2 algorithm adds chopping to cancel reference offsets during analog gradient accumulation. ([Nature][2])

Practical version: periodically flip signs/polarities of the learning path while keeping the represented weight logically unchanged. This is especially useful if collapse comes from a small DC offset being integrated for many examples.

### 16. Dynamic reference / auto-zero

AGAD’s core idea is to avoid a perfectly programmed static zero reference. It estimates the reference dynamically from recent accumulated-gradient behavior, making reference offsets largely irrelevant. ([Nature][6])

Practical version: add a slow common-mode/reference tracker for each layer or column. Subtract the recent mean update/current before it reaches the weight cap.

### 17. Use reference conductances or differential pairs for signed weights

Real conductances are positive. Papers use separate reference rows/columns, differential devices, or doubled inputs/outputs to represent signs. The silicon-synapse RBM uses a reference conductance to subtract from positive device conductances; EqProp doubles nodes to work around positive-only weights. ([arXiv][9])

Practical version: signed weights should be encoded as (g^+ - g^-), (g - g_{\mathrm{ref}}), or doubled input/output channels. Avoid hiding sign in an offset that can drift.

### 18. Train with moderate noise

The M-SDC SPICE paper explicitly found that adding moderate training noise improved robustness to device variations and noisy inputs; it also found that too much noise degrades generalization. 

Practical version: inject small input noise, device noise, or teaching-current noise during training. Treat it as regularization. But sweep it: in that paper, high noise eventually made the model more fragile.

### 19. Stochastic sampling can be implemented as analog noise, not digital randomness

The silicon-synapse RBM uses stochastic hidden sampling; the paper notes the logistic sampling steepness can be realized by controlling noise-current intensity. ([arXiv][9])

Practical version: noise is not always bad. For RBM-like or probabilistic training, a tunable noise current is a feature.

### 20. Do not reset unless reset cost beats drift cost

The M-SDC SPICE system tested an initial 200 ms reset pulse. Resetting sometimes reduced training cycles, but its time/energy overhead outweighed the benefit; omitting it cut training time and energy substantially. 

Practical version: try “warm start” rather than full reset. Your smaller-init idea is good, but full reset-to-zero may be worse than a controlled midrange initialization.

### 21. Initialize near the device/circuit sweet spot

The M-SDC paper found chromium-type devices trained faster partly because their initial conductance jump placed them closer to most target conductances. 

Practical version: initialize weights near the high-slope, high-controllability region of your device, not necessarily near mathematical zero. For sigmoid hidden units, also initialize biases so hidden activities sit near the nonlinear midpoint.

### 22. Use batch normalization or cheap normalization where possible

The FeCAP/memristor paper reports that mini-batch training with batch normalization improved accuracy and resilience to transfer errors. ([Nature][5])

Practical version: a full batchnorm circuit may be too much, but layer common-mode subtraction, activity RMS limiting, or per-layer gain control can capture most of the stabilizing effect.

### 23. Use mini-batches or accumulated gradients to reduce update noise

The SPICE EqProp paper used mini-batch updates in simulation because updating every sample was costly, and the IBM/FeCAP/silicon-synapse papers all rely on some form of accumulation before physical weight transfer. 

Practical version: your TH/CWW timescale separation is not just a simulation convenience. It is a standard stabilizer.

### 24. Keep the analog weight low precision; keep optimizer state higher precision

The FeCAP/memristor strategy uses 10-bit hidden weights and 4-bit analog weights; during inference the hidden value is truncated/transferred into the analog memristor representation. ([Nature][5])

Practical version: do not fight for high analog precision everywhere. Use high precision only in the learning accumulator; let inference weights be coarse but stable.

### 25. Accept bounded conductance error

The M-SDC SPICE work found little accuracy loss up to about 10% conductance error, especially with moderate noise training, but degradation at 15% error. 

Practical version: define an error budget. Do not spend circuit complexity correcting errors below the model’s actual tolerance.

### 26. Use binary or low-bit neuron states to simplify circuits

The silicon-synapse RBM uses fully binarized neural states, making contrastive divergence ternary and simplifying accumulation/update logic. ([arXiv][9])

Practical version: if your analog hidden states are causing runaway, test a clipped/binary hidden mode. It may reduce representational smoothness but greatly simplify stability.

### 27. Greedy layer-wise training / wake-sleep instead of full global BP

The memristive DBN work trains RBMs layerwise and then uses wake-sleep fine-tuning in simulation. ([arXiv][9])

Practical version: for a fragile analog network, train one layer or block at a time, freeze it, then add the next. This is closely related to your freeze-on-convergence idea.

### 28. Use local adaptive retraining instead of full retraining

The small M-SDC system emphasizes adjustable retraining to restore performance after conductance errors or input-noise shifts. 

Practical version: keep a low-rate adaptation mode after deployment, but gate it behind confidence/error triggers. Continuous always-on learning is a collapse risk.

### 29. Prefer slow leak / weight decay over unconstrained wandering

Few of these papers sell “weight decay” as the headline trick, but nearly all successful designs impose some equivalent: conductance floors/ceilings, finite retention, hidden-state transfer, update thresholds, normalization, or bounded integer state. Your “real weight decay / stronger cap leak” idea fits the trend.

Practical version: a weak leak to a neutral conductance or cap voltage can convert runaway saddles into stable attractors, but tune it slower than class-learning dynamics.

### 30. Use per-layer learning-rate balance

AGAD/c-TTv2 explicitly normalize learning rates to avoid layer-size dependence and gradient-scale drift. EqProp SPICE used different learning rates for the two conductance matrices. ([Nature][6])

Practical version: do not use the same physical update strength for hidden and output layers. Output layers often need faster learning; hidden layers often need more damping.

## Mapping this back to your ten ideas

Your highest-probability fixes, based on the literature, are:

1. **Timescale separation**: longer settle, slower weights, batch/thresholded update. This appears everywhere.
2. **Error normalization by C and common-mode subtraction**: very likely relevant to your C=2 → C=3 failure.
3. **f′ gating / hidden slope gating**: strongly supported by both backprop logic and the way EP naturally attenuates saturated units.
4. **Dynamic LR / annealing / per-layer LR**: supported by AGAD/c-TTv2 and mixed-precision designs.
5. **Freeze or periodic transfer**: supported by RBM thresholding and FeCAP→memristor delayed transfer.
6. **Moderate noise and clipping**: cheap, repeatedly useful, and easy to A/B.
7. **Differential outputs / lateral competition**: not always explicit as softmax, but common-mode rejection is a clear trend.
8. **Weight decay/leak**: less explicitly studied, but consistent with the “bounded, damped, slowly updated state” pattern.

The strongest new suggestion I would add to your list is **chopper/auto-zero learning**: periodically invert the sign/polarity of the teaching path or subtract a slowly tracked per-column/per-layer update mean. If your collapse is caused by a tiny DC teaching-current offset, this can fix a problem that LR annealing only hides.

[1]: https://arxiv.org/abs/1907.00625 "[1907.00625] On-chip learning in a conventional silicon MOSFET based Analog Hardware Neural Network"
[2]: https://www.nature.com/articles/s41586-018-0180-5 "Equivalent-accuracy accelerated neural-network training using analogue memory | Nature"
[3]: https://arxiv.org/abs/2203.09046 "[2203.09046] A memristive deep belief neural network based on silicon synapses"
[4]: https://arxiv.org/abs/2408.14680 "[2408.14680] On-Chip Learning with Memristor-Based Neural Networks: Assessing Accuracy and Efficiency Under Device Variations, Conductance Errors, and Input Noise"
[5]: https://www.nature.com/articles/s41928-025-01454-7 "A ferroelectric–memristor memory for both training and inference | Nature Electronics"
[6]: https://www.nature.com/articles/s41467-024-51221-z "Fast and robust analog in-memory deep neural network training | Nature Communications"
[7]: https://arxiv.org/abs/2208.01623?utm_source=chatgpt.com "Single chip photonic deep neural network with accelerated training"
[8]: https://arxiv.org/abs/2006.03824 "[2006.03824] Scaling Equilibrium Propagation to Deep ConvNets by Drastically Reducing its Gradient Estimator Bias"
[9]: https://arxiv.org/pdf/2203.09046 "memristive_dbn_accepted_version"

