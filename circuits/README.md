# Reference circuits

These are source examples and reference decks, retained separately from generated
training runs. Most use ngspice; `xor_infer_xyce.cir` is the Xyce inference example.

- `xor_full.cir`, `xor_infer.cir`: worked in-circuit XOR training and inference.
  See [the XOR findings](../docs/XOR_BACKPROP_README.md).
- `tc_cell.cir`, `dsyn_test.cir`, `neuron2.cir`, `nmos_cell.cir`: synapse, neuron,
  and update-cell characterization. See [the equilibrium-propagation findings](../docs/EP_FINDINGS.md).
- `rec2.cir`, `rec3.cir`, `rec3b.cir`, `relax1.cir`: reciprocal cells and relaxation.
- The remaining decks capture smaller device tests and early proof-of-concept circuits.

For example, from the repository root:

```bash
ngspice -b circuits/xor_infer.cir
```

Run outputs are written into the working directory and ignored by Git. Experiment
generators still write their new `.cir` files there; the copies in this directory
are reference snapshots, not the latest output of every run.
