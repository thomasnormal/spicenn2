# example-delta-rule

Author: Thomas Dybdahl Ahle

Starting point: `competition/examples/mnist_017_tuned.cir`, by Thomas Dybdahl Ahle
and contributors, under MIT. This is a complete example submission for the
`mnist017-v0` starter track; its circuit is identical to the bundled tuned baseline.

## Design

Analog single-layer delta-rule learner. Capacitors store weights, transistor
currents update them during training, and the circuit produces the class scores.
There is no external optimizer or pretrained weight initialization.

The original baseline's 51 weight capacitors are increased from 1.67 µF to 16 µF.
This slows learning because `dV/dt = I/C`, and increases startup energy. The value
was suggested by a capacitance sweep in entrant-trial feedback on this same public
0/1/7 development task. We generated and reran the variant independently; no
pretrained weights are supplied. It is development-set tuning, not evaluation on
an untouched final dataset. No other component values or connections change.

## Reproduce

From the repository root (use new output paths for each experiment):

```bash
python competition/prepare_mnist.py data/mnist-idx /tmp/my-mnist017.npz --download --labels 0,1,7 --normalize zscore
python competition/runner.py submissions/example-delta-rule/circuit.cir /tmp/my-mnist017.npz --task mnist017-v0 --output /tmp/example-delta-rule-score
python competition/validate.py submissions/example-delta-rule
```

## Results

On ngspice 46, protocol `mnist4-circuit-learning-v0`: **134/150 correct (89.33%)**,
zero invalid predictions, 2.76849 mJ startup, 67.9689 mJ training, and
9.86249 µJ per evaluation image. Task settings are 20 training passes, seed 0,
20 ms startup, 1 ms/image, and maximum timestep 10 µs. Dataset content SHA-256:
`7e615774a3a79e96bd5f50bf16ae2eda0b4b7521a47cb154d759ba7ac8f04a3e`.

The [baseline record](../../competition/results/mnist017-v0/delta-rule-16uf/entry.json)
links the circuit identity; its neighboring report and verification files contain
full measurements from the reference image. All predictions matched at 5 µs and
each phase's energy differed by less than 1%. This example is not a second distinct
leaderboard design. Simulated joules exclude physical DAC/ADC and controller losses.

## License

MIT; see LICENSE. Original example attribution is retained.
