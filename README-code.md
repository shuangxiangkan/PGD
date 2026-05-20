# GAT-PFD Experiment Code

This folder contains the experiment code for the probabilistic PMC fault diagnosis method. The paper source is kept in a separate folder; this directory is only for dataset generation, method implementation, and experiments.

## Structure

```text
pfd/
  topologies.py      # hypercube, k-ary n-cube, augmented k-ary n-cube
  dataset.py         # probabilistic PMC data generation and .npz IO
  features.py        # node features and bidirectional edge features
  model.py           # reliability-aware GNN
  decision.py        # probability decision and posterior refinement
  metrics.py         # accuracy, F1, Brier, ECE, top-k
  baselines.py       # simple reusable baselines
  train.py           # training helpers

scripts/
  generate_dataset.py
  run_method.py
  run_baseline.py
  train_single_sample.py

configs/
data/
```

## Environment

Create and activate the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Check whether GPU/MPS is available:

```bash
python -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available()); print('mps', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)"
```

`scripts/run_method.py` automatically uses CUDA if available, then Apple MPS if available, otherwise CPU.

## Generate a Dataset

The default number of testing rounds is 3. Use 2 or 3 rounds for the main low-cost probabilistic PMC setting.

```bash
.venv/bin/python scripts/generate_dataset.py \
  --topology hypercube --n 6 \
  --fault-rate 0.3 --rounds 3 \
  --gamma 0.7 --beta 0.3 \
  --num-samples 120 \
  --seed 61 \
  --output data/q6_fr30_t3_noisy_method
```

## Run the Proposed Method

This runs the full method described in the methodology section: feature construction, reliability-aware GNN, probability-based decision, and ambiguous-node posterior refinement.

```bash
.venv/bin/python scripts/run_method.py \
  --dataset data/q6_fr30_t3_noisy_method \
  --epochs 60 \
  --hidden-dim 64 \
  --layers 2 \
  --refine-top-pct 0.1 \
  --log-every 20
```

The script reports both raw GNN posterior results and refined posterior results:

```text
raw_accuracy
raw_f1
raw_brier
raw_ece
refined_accuracy
refined_f1
refined_brier
refined_ece
delta_accuracy
delta_f1
refinement_change_rate
refinement_corrected_rate
refinement_wrong_flip_rate
```

`--refine-top-pct` controls the class-wise least-confident fraction selected for posterior refinement. For example, `0.1` means that the least-confident 10% of preliminarily fault-free nodes and the least-confident 10% of preliminarily faulty nodes are refined, in addition to nodes in the boundary interval.

## Train-Test Reliability Shift

To test probabilistic PMC parameter shift, generate separate train and test datasets:

```bash
.venv/bin/python scripts/generate_dataset.py \
  --topology hypercube --n 6 \
  --fault-rate 0.3 --rounds 2 \
  --gamma 0.9 --beta 0.1 \
  --num-samples 120 \
  --seed 71 \
  --output data/q6_fr30_t2_train_reliable

.venv/bin/python scripts/generate_dataset.py \
  --topology hypercube --n 6 \
  --fault-rate 0.3 --rounds 2 \
  --gamma 0.55 --beta 0.45 \
  --num-samples 60 \
  --seed 81 \
  --output data/q6_fr30_t2_test_shifted

.venv/bin/python scripts/run_method.py \
  --dataset data/q6_fr30_t2_train_reliable \
  --test-dataset data/q6_fr30_t2_test_shifted \
  --epochs 60 \
  --hidden-dim 64 \
  --layers 2 \
  --refine-top-pct 0.1
```

## Run a Simple Baseline

```bash
.venv/bin/python scripts/run_baseline.py --dataset data/q6_fr30_t3_noisy_method
```
