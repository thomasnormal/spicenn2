Here’s the practical inventory I’d steal from the hardware/SPICE MNIST analog-training papers. The big theme is: **do not try to implement textbook backprop literally in analog**. The papers mostly win by changing the training rule, changing the weight representation, delaying analog writes, or making the network tolerate ugly devices.

A caveat: almost none of the strongest papers are “perfectly pure.” Kendall et al. ran the circuit dynamics in Spectre/SPICE, but Python handled netlist generation, loss bookkeeping, and weight updates; Wang et al. trained a small floating-gate RBM in real silicon and extrapolated/hardware-modeled the larger MNIST DBN; Martemucci et al. validated real memory/periphery blocks but used calibrated hardware-aware simulation for the full MNIST network. Those caveats matter, but the tricks are still useful. 

## The highest-value tricks to try first

**1. Use equilibrium propagation instead of backprop.**
This is probably the cleanest trick for a continuous analog circuit. Run the network once in a free/inference phase, then run it again with a small output “nudge” toward the target. The weight update can be local: compare the squared voltage drop across each programmable conductance in the two phases.

[
\Delta g_{ij}\propto -\frac{\eta}{2\beta}
\left((V_i^\beta - V_j^\beta)^2 - (V_i^0 - V_j^0)^2\right)
]

The appealing part is that a resistor only needs to “know” the voltage across itself in two settles. Kendall et al. used nonlinear resistive networks with diode-like nonlinearities and trained MNIST in Spectre; their MNIST model used a single hidden layer with 100 neurons and reached 3.43% test error after 10 epochs. 

**2. Make signed weights out of positive conductances.**
Analog conductances are usually nonnegative, so use one of these encodings:

* differential pair: (w = g^+ - g^-)
* duplicated input nodes: (x) and (-x)
* duplicated output nodes: positive and negative class/output rails
* common-mode shift: store weights around a reference conductance

Kendall et al. discuss differential conductance and also use doubled input/output nodes: MNIST pixels become 1,568 input nodes rather than 784, and 10 classes become 20 output nodes. This is ugly but very practical. 

**3. Add a bias node as a physical voltage rail.**
Instead of treating bias as special software state, add a fixed 1 V node and connect it through trainable conductances. This gives every neuron a learnable offset using exactly the same machinery as other weights. Kendall et al. used this in their SPICE networks. 

**4. Use bidirectional amplifiers to stop passive networks from fading out.**
Pure resistor/diode networks attenuate signals layer by layer. Kendall et al. inserted bidirectional amplifiers: voltage-controlled voltage sources for forward voltage gain and current-controlled current sources for backward current gain. This keeps the “forward” and “backward/nudged” physics compatible without building a separate backprop circuit. 

**5. Prefer binary or stochastic neurons when devices are nonlinear.**
The floating-gate/memristive DBN paper avoided high-precision analog activations by using binary neuron states. Grayscale MNIST pixels were treated as probabilities for sampling binary input states, and hidden/output neurons were sampled stochastically during training. This dodges a lot of DAC/ADC pain and makes non-ohmic device behavior less fatal. ([arXiv][1])

**6. Use contrastive divergence/RBMs when local training is easier than backprop.**
Restricted Boltzmann Machines are much more hardware-friendly than standard MLP backprop because their update is local and outer-product-like:

[
\Delta w \propto v h^\top - v' {h'}^\top
]

With binary visible and hidden units, each synapse only receives ternary update requests: increase, decrease, or do nothing. Wang et al. used this to train RBMs and stack them into a DBN for MNIST. The large DBN used layers of 784–500, 500–500, and 510–2000 units, reaching up to 97.05% MNIST accuracy in their hardware-calibrated workflow. ([arXiv][2])

**7. Accumulate many update requests before touching analog weights.**
This is one of the most reusable tricks. Do not program the analog device after every training example. Instead, keep a small digital or analog accumulator per weight. When the accumulator crosses a threshold, apply one physical write pulse and reset or decrement the accumulator. Wang et al. accumulated ternary update requests in counters and programmed only when the counter reached a threshold such as +5 or −5. This reduces endurance stress and makes nonlinear write pulses less damaging. ([arXiv][1])

**8. Use blind writes instead of expensive write-verify loops.**
Many analog memory papers rely on write-verify, which is accurate but slow and peripheral-heavy. The silicon-synapse DBN paper deliberately used sparse blind writes, relying on the learning algorithm and counters to absorb device imprecision. This is a very useful design philosophy: make the training rule robust enough that every write does not need to be perfect. ([arXiv][1])

**9. Split “training weight” and “inference weight.”**
Martemucci et al. used a clever two-store idea: keep a higher-precision hidden training state in one memory structure and periodically transfer a lower-precision analog version to the array used for forward/backward MACs. In their design, FeCAP-like hidden weights were updated frequently, while memristor analog weights were updated only periodically. This avoids requiring the analog inference weight itself to support perfect, symmetric, tiny SGD updates. ([Nature][3])

**10. Transfer weights periodically, not continuously.**
The same paper varied the transfer period (k): update the hidden training state every sample, but refresh the analog inference weights every (k) inputs. They report MNIST results around 96.7% with (k=100), which is the important lesson: analog weights do not necessarily need to be updated at every SGD step. ([Nature][3])

## Circuit and representation tricks

**Use diode nonlinearities instead of trying to synthesize perfect activations.**
Kendall et al. used antiparallel diodes to create a sigmoid-like neuron nonlinearity. The practical lesson is that the activation function does not need to be exactly ReLU or tanh; it needs to be monotone, stable, differentiable enough for the training rule, and easy to build. 

**Use output current injection for teaching signals.**
In equilibrium propagation, the target is applied by nudging the output nodes with current sources. This is a nice analog trick: the target does not have to be backpropagated digitally; it perturbs the physical equilibrium, and the internal voltages shift accordingly. 

**Clip conductances and enforce a floor.**
Kendall et al. clipped conductances to prevent impossible or numerically unstable weights. For a real analog network, this maps to minimum/maximum device conductance, leakage, and saturation limits. You should train with those limits present from the beginning rather than discovering after layout that half your weights want impossible values. 

**Use differential outputs for classification.**
Instead of one output node per class, use two rails per class and compute a difference. This helps with signed quantities while using only positive physical conductances. Kendall’s MNIST setup used 20 output nodes for 10 classes. 

**Encode labels as visible units for generative models.**
In the DBN/RBM approach, labels can be appended to the visible vector. Training learns joint image-label structure; inference can clamp the image and let the label units compete. This is especially natural for RBMs and other energy-based circuits. ([arXiv][1])

**Use deterministic inference after stochastic training.**
Wang et al. trained with stochastic sampling, but also evaluated deterministic inference by removing injected noise. They also tested repeated sampling at inference for better accuracy. This gives you a knob: deterministic inference for speed/energy, repeated sampling for accuracy. ([arXiv][1])

**Use source switching / floating lines for reverse VMM.**
For floating-gate arrays, Wang et al. implemented forward and backward vector-matrix multiplication by changing which terminals were biased, grounded, or floated. The generic trick is: design the array so the same weights can be read in both directions, even if the device is not mathematically symmetric. ([arXiv][1])

**Add deselection voltages to suppress disturb.**
When programming one device, half-selected devices can drift. Wang et al. used deselection-voltage schemes to reduce unintended programming. For your own arrays, disturb management should be part of the learning algorithm, not just a device afterthought. ([arXiv][1])

## Training-update tricks

**Use ternary update pulses.**
Try to reduce every synapse update to one of three actions: increment, decrement, no-op. This is much easier to implement than arbitrary analog gradient magnitudes. RBM contrastive divergence gives this naturally with binary units. ([arXiv][1])

**Separate “compute the gradient sign” from “program the device.”**
Wang et al.’s modified contrastive-divergence flow separates weighted summation from accumulated update requests. This is a very general trick: let analog hardware compute correlations or local errors, but let a cheap accumulator decide when the physical memory should be changed. ([arXiv][1])

**Use stochastic update masks.**
Martemucci et al. used a probabilistic update mask in their MNIST training setup. Randomly skipping some weight updates reduces write traffic and can act like regularization. They used no momentum, which also simplifies hardware state. ([Nature][3])

**Avoid momentum unless you can afford the state.**
Momentum is great in software but costly in hardware because it requires another state variable per weight. Several hardware-oriented designs either avoid it or move it into hidden/digital state. For a first analog learner, plain SGD-like updates are much easier to make real. ([Nature][3])

**Use pulse coincidence for outer-product updates.**
In analog in-memory training, a common trick is to send stochastic or pulsed representations of activations along rows and errors along columns. Where pulses coincide, the device receives an update. IBM’s analog-training work frames this as in-memory outer-product accumulation. ([Nature][4])

**Use a separate gradient-accumulation array.**
IBM’s Tiki-Taka-style algorithms separate the main weight array from an auxiliary gradient accumulator. The accumulator collects many noisy pulse updates; only occasionally is information transferred to the main weight matrix. This is the same broad idea as “do not let every noisy gradient directly hit your real weight.” ([Nature][4])

**Use chopping / sign flipping to cancel update offsets.**
The c-TTv2 idea flips the sign convention of gradient accumulation so fixed asymmetries and offsets average out. This is analogous to chopper stabilization in analog circuits: periodically invert the thing that has the offset, then invert it back logically. ([Nature][4])

**Use a dynamic reference instead of assuming a perfect zero-update point.**
Real devices often have no stable “do nothing” pulse condition. IBM’s AGAD-style approach estimates a drifting reference point dynamically rather than assuming device symmetry. This is useful whenever potentiation and depression curves do not match. ([Nature][4])

## Precision and robustness tricks

**Train with the actual hardware errors in the loop.**
Do not train an ideal model and then hope the analog circuit works. Include quantization, clipping, saturation, IR drop, programming error, read noise, drift, nonlinear transfer curves, input noise, and output noise in training. IBM’s hardware-aware training work found that input/output noise, especially output noise, can matter more than weight noise. ([Nature][5])

**Optimize input scaling per array.**
Analog arrays have a limited useful voltage/current range. Use a trainable or calibrated scale factor before each array so signals use the dynamic range without saturating. IBM’s hardware-aware model explicitly includes input scaling and column-wise output scaling/bias terms. ([Nature][5])

**Use column-wise output gain and bias calibration.**
Each column/sense path will have different gain and offset. Treat those as calibration parameters, not as failures. Add per-column scale and bias after the analog readout, or physically tune the sense amplifiers. ([Nature][5])

**Prioritize output-node noise reduction.**
A lot of analog-design effort goes into weight precision, but IBM’s hardware-aware experiments suggest output noise is especially damaging. Spend design budget on sense amplifiers, integration capacitors, ADC/no-ADC readout quality, and output filtering. ([Nature][5])

**Use layer norm instead of batch norm for online hardware training.**
Martemucci et al.’s online MNIST setup used layer normalization with ReLU hidden layers. Batch norm can work, but it needs batch statistics and more memory/control. Layer norm is more plausible for online, local-ish hardware training. ([Nature][3])

**Use low-precision analog weights but higher-precision hidden state.**
Their MNIST setup used 10-bit hidden weights and 4-bit analog weights in hardware-aware simulation. This is a very practical split: analog arrays do not need to hold the full optimizer state. ([Nature][3])

**Use multiple devices per weight when precision matters.**
IBM’s optimized programming work uses differential conductance pairs and can add most-significant/least-significant device pairs. This lets you combine coarse and fine devices rather than demanding one perfect analog cell. ([Nature][6])

**Do not map weights linearly by default.**
A software weight (w) does not have to map linearly to conductance. Optimize the translation curve from weight to device conductance, including drift, read noise, programming error, and the device’s usable range. IBM’s programming-strategy work shows that optimized mappings can reduce inference degradation over time. ([Nature][6])

**Use calibration vectors for drift compensation.**
After programming, apply known random vectors and record the outputs. Later, repeat the same vectors and estimate how the array has drifted. IBM’s drift-compensation strategy uses this kind of calibration to rescale outputs over time. ([Nature][6])

## Architecture-level ideas

**Energy-based networks are a better match than ordinary MLPs.**
Equilibrium propagation, RBMs, DBNs, Hopfield-like systems, and oscillator/Ising-style networks all fit analog hardware better than standard feedforward backprop because “settling” is part of the computation. That maps naturally to capacitors, resistors, conductances, oscillators, and current summation. Kendall’s SPICE work and Wang’s RBM/DBN work are the clearest MNIST examples. 

**Use greedy layer-wise training when full end-to-end training is too hard.**
The DBN paper trained stacked RBMs layer by layer. This avoids sending precise gradients through many analog layers. For early hardware, “train one layer/module at a time, then stack” is much more realistic than full deep backprop. ([arXiv][1])

**Use feedback alignment instead of exact backward weights.**
Direct Feedback Alignment sends the output error through fixed random feedback paths rather than through exact transposed forward weights. This removes the weight-transport problem and lets layers update more independently. The photonic DFA paper is not as directly aligned with your “normal components” constraint, but the algorithmic trick is highly reusable in analog electronics. ([ar5iv][7])

**Make derivatives binary when possible.**
Use activations whose derivative is just a gate. ReLU is attractive because the derivative is approximately 0 or 1. The photonic DFA paper used a transimpedance amplifier scheme to combine the feedback signal with the ReLU derivative mask. The electronics analogue is simple: gate the error/update path based on whether the neuron was active. ([ar5iv][7])

**Tile large matrices rather than building one huge perfect array.**
Photonic and AIMC papers both assume matrices are broken into smaller physical blocks. For your own networks, design the training rule so it tolerates tiled arrays, column scaling, inter-tile gain mismatch, and sequential accumulation. ([ar5iv][7])

## The “try this in your own analog NN” recipe

For a **continuous SPICE analog MLP**, I would start with:

1. single hidden layer, not deep
2. equilibrium propagation
3. diode or tanh-like neuron nonlinearity
4. differential or doubled-node signed weights
5. physical bias node
6. conductance clipping and floors
7. free phase plus nudged phase
8. local update from squared voltage-drop difference
9. hardware noise and saturation included during training from day one

For a **messy analog memory / floating-gate / memristive array**, I would start with:

1. binary neurons
2. RBM or shallow DBN
3. stochastic sampling during training
4. ternary update requests
5. per-weight counters
6. thresholded blind writes
7. sparse programming
8. deterministic inference first, repeated sampling only if needed

For a **trainable analog accelerator where writes are expensive**, I would start with:

1. low-precision analog inference weights
2. higher-precision hidden training weights
3. periodic analog refresh every (k) examples
4. stochastic update masks
5. no momentum
6. layer norm rather than batch norm for online training
7. calibration of transfer curves and column gains

The most important practical lesson: **let analog hardware do what it is good at—settling, summing, multiplying by conductance, accumulating charge/current—and move everything that requires exactness into redundancy, calibration, stochasticity, or delayed updates.**

[1]: https://arxiv.org/pdf/2203.09046 "memristive_dbn_accepted_version"
[2]: https://arxiv.org/abs/2203.09046 "[2203.09046] A memristive deep belief neural network based on silicon synapses"
[3]: https://www.nature.com/articles/s41928-025-01454-7 "A ferroelectric–memristor memory for both training and inference | Nature Electronics"
[4]: https://www.nature.com/articles/s41467-024-51221-z "Fast and robust analog in-memory deep neural network training | Nature Communications"
[5]: https://www.nature.com/articles/s41467-023-40770-4 "Hardware-aware training for large-scale and diverse deep learning inference workloads using in-memory computing-based accelerators | Nature Communications"
[6]: https://www.nature.com/articles/s41467-022-31405-1 "Optimised weight programming for analogue memory-based deep neural networks | Nature Communications"
[7]: https://ar5iv.org/pdf/2111.06862 "[2111.06862] Silicon Photonic Architecture for Training Deep Neural Networks with Direct Feedback Alignment"

