# Experiment launchers

Run these from the repository root, for example:

```bash
bash scripts/mm_queue.sh 0.010 1
```

- `mm_queue.sh`: launch a transistor-mismatch trial using the main learner.
- `restart_run.sh`: historical restart experiment around the main learner.
- `run_iso.sh`: isolated multiclass MNIST experiment.
- `run_iso_cifar.sh`, `run_iso_cifard.sh`: CIFAR variants.
- `run_iso_spectre.sh`: Spectre variant.

The launchers resolve source paths from the checkout location. Dataset paths remain
relative to that checkout, and datasets must be prepared separately. The isolated
launchers accept a run-directory argument; use a dedicated path for generated output.
Their simulation costs and data requirements vary considerably, so inspect the
configuration before starting a run. These are research experiments, not official
competition scoring commands.
