# Circuit-learning competition: v0 rules

Design a circuit that **learns during simulation**, then recognizes handwritten
digits with high accuracy and low electrical energy. Submit the circuit as a pull
request; the organizer runs it. There is no entry fee or cash prize. This is a
rolling benchmark with no fixed closing date; submissions can continue after v0
is published. Changes to the task or device models require a new version and a
separate table, not a silent change to existing scores.

## Two separate tracks

| Track | Classes | Training set | Evaluation set | Training passes |
| --- | --- | --- | --- | --- |
| `mnist10-v0` — main | All ten digits | 24 per class (240) | 50 per class (500) | 20 |
| `mnist017-v0` — starter | 0, 1, 7 | 24 per class (72) | 50 per class (150) | 20 |

The two-input blobs task is a tutorial, not a handwriting leaderboard track.
Both MNIST tracks use the original 60,000/10,000 source split, selection seed 0,
4×4 block-average inputs, per-image contrast normalization, and shuffle seed 0.
The exact arrays are identified by the content hashes in the
[task manifests](tasks/README.md). Public MNIST evaluation labels are not secret:
these are **public-development benchmarks**, not unbiased estimates after extensive
test-set tuning. Disclose all data used for design and tuning.

Both tracks fix startup to 20 ms, each example to 1 ms, maximum timestep to 10 µs,
temperature to 27 °C, and all models, bias voltages, ramps, and loads to the
[v0 interface](README.md). Main-track duration is 5.32 simulated seconds; the starter
is 1.61 seconds. The [release manifest](release.json) identifies the reference
ngspice 46 image and numerical verification limits. Xyce is supported for exploration
and cross-checks, but does not determine v0 leaderboard scores.

## Eligible circuits

Submit a flat R/C/M/D netlist using the supplied models and pins. No submitted
voltage/current sources, behavioral expressions, initial conditions, simulator
commands, included files, custom models, or Python readout fitting are allowed.
See the [component grammar and bounds](README.md#circuit-interface).

Task-dependent weights must be learned during the supplied training phase, in the
same continuous simulation as evaluation. All capacitors and sources start at zero;
any reset/preparation energy comes from the metered harness. Test targets are zero.
Do not encode a pretrained classifier, evaluation-image answers, or the known test
order into component values or circuitry. Fixed, data-independent random features
are allowed; document their generation. Architecture and shared learning parameters
(such as weight capacitance) may be tuned offline, with disclosure.

Describe how learning and retention work and what information was used outside
SPICE. The organizer may request an untrained comparison or a class-permutation
control. Format validation cannot prove that a design genuinely learns; eligibility
also requires review. If your design needs a different interface or a pretrained
track, propose it separately rather than comparing incompatible scores.

## Accuracy and energy

The highest of ten output voltages at 90% of each evaluation slot determines the
prediction. A tie within 1 µV is invalid and counts as wrong. Accuracy is correct
predictions divided by the complete evaluation-set size; failures and invalid
predictions do not reduce the denominator.

Inference energy/image is gross energy delivered through **every driven pin**,
including inputs, targets, bias and controls, divided by the number of evaluation
images. Integrate each pin's positive delivered power separately; returned energy
does not cancel delivery elsewhere. Startup and total training energy are also
reported. These are simulated circuit-boundary joules, not the electricity used by
the computer. Driver/readout implementation losses and fabricated-chip parasitics
are outside this simplified model.

The leaderboard marks the **accuracy–inference-energy Pareto frontier**. Circuit A
dominates B if A has at least B's accuracy and no more inference energy, with at
least one strict improvement. Non-dominated entries are frontier entries. There is
no arbitrary weighted sum of accuracy and energy. Rows are sorted by accuracy
descending and energy ascending for readability; that order is not an additional
winner rule. Identical points share frontier status. Startup and training energy
remain visible and are not amortized into an unspecified deployment workload.

## Verification and failures

The organizer uses the trusted release runner and exact reference image on a
disposable machine. The simulator has no network, no repository/dataset/credential
mount, a non-root user, 2 CPUs, 4 GiB memory, a 64-process limit, a 2 GiB file limit,
and a 1,800-second simulation timeout. Resource-limit failures, simulator aborts,
and incomplete or non-monotonic traces produce **no ranked score**. Failure details
are retained and the entrant can submit a corrected version.

**The 30-minute wall-clock budget applies separately to both the 10 µs scoring
run and the 5 µs verification run.** It is a v0 eligibility/resource limit, not
an accuracy or electrical-energy metric. Plan for the finer-step run: the bundled
ten-digit baseline took about 22 minutes at 10 µs and 29 minutes at 5 µs on the
organizer's host, leaving little margin. Slower/larger circuits may exceed this
budget even when their electrical simulation is otherwise valid. The starter
track is a cheaper place to develop them.

CPU quotas do not make different processors equally fast; an entrant's local
timeout is not by itself a rejection. Organizer reference-machine reruns determine
resource eligibility. Raising `--timeout` is useful for local debugging but does
not waive the organizer's limit or yield a ranked score. Increasing the published
budget would need an announced new release policy, not an exception for one entry.

Before recording a result, rerun with maximum timestep 5 µs and otherwise identical
settings. The entire prediction list must match, and each phase's delivered energy
must differ by no more than 1%. A failing numerical check is marked unverified,
not ranked with a favorable coarse-step result. Do not loosen tolerances for one
entry; investigate or propose a new benchmark version. Report the 10 µs result
after this check passes.

The [leaderboard](LEADERBOARD.md) links circuit identities and the paired reports.
Entrants' own reports are useful reproduction evidence but are not automatically
trusted or published as organizer-verified results. CI validates submission format
and report consistency; it is not a substitute for independent simulation.

## Submit and revise

Follow the [submission instructions](../submissions/README.md). Include circuit
source, an explanation, reproduction commands, tuning disclosure, attribution,
and an MIT license. Include generator source if applicable, but the organizer
scores the submitted netlist without running entrant Python.

Use one directory per design revision. A revised circuit gets a new circuit hash
and a new rerun; existing recorded artifacts are retained. The organizer reviews
PRs and may request corrections or decline designs that violate these rules.
Questions and proposed rule changes belong in GitHub issues.
