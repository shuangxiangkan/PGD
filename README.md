# PGD

PGD contains experiment code for probabilistic graph-based fault diagnosis under the probabilistic PMC model. The goal is to diagnose faulty processors from noisy mutual-test syndromes on interconnection networks such as hypercubes, k-ary n-cubes, and augmented k-ary n-cubes.

## Method Idea

The method treats fault diagnosis as posterior probability estimation rather than direct hard classification. Given a graph and multi-round mutual-test outcomes, it estimates a node-wise fault probability:

```text
P(node is faulty | graph topology, observed PMC syndromes)
```

The pipeline has four stages:

1. **Diagnostic feature construction**

   Multi-round PMC test outcomes are converted into two complementary representations:

   - node-level statistical features, including average mismatch ratio and neighborhood disagreement dispersion;
   - edge-level syndrome observations, preserved for posterior refinement.

2. **GNN posterior estimation**

   A standard GNN propagates the compact node features over the network and estimates a fault posterior score for each node. The default backbone is GraphSAGE, with GCN and GAT available for architecture comparison.

3. **Probability-based diagnostic decision**

   The model outputs posterior fault scores instead of only binary labels. High-confidence nodes are diagnosed directly, while less reliable predictions are selected for additional checking.

4. **Posterior refinement**

   Selected nodes are refined by checking syndrome consistency with reliable high-confidence neighbors. This step is designed to correct low-confidence or suspicious posterior decisions using local PMC evidence.

## Baselines

The repository currently includes:

- **Clustered-PMC**: a rule-based probabilistic clustered-fault diagnosis algorithm using local factions and threshold `k=3`. Because it expects one PMC syndrome per directed test, it is evaluated on the last collected syndrome of each test instance.
- **Vanilla GNN-Classifier**: a standard node-level GNN classifier that directly predicts binary fault labels without posterior refinement.

## Repository Layout

```text
pfd/
  topologies.py      # hypercube, k-ary n-cube, augmented k-ary n-cube
  dataset.py         # probabilistic PMC data generation and .npz IO
  features.py        # compact node features and edge syndrome summaries
  model.py           # GNN posterior estimators and baselines
  decision.py        # probability decision and posterior refinement
  baselines.py       # rule-based baselines
  metrics.py         # accuracy, F1, Brier, ECE, top-k

scripts/
  generate_dataset.py
  run_method.py
  run_baseline.py
  run_vanilla_gnn.py
  inspect_confidence.py
  analyze_error_ranks.py
```

## Quick Start

Create the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Generate train and test datasets:

```bash
.venv/bin/python scripts/generate_dataset.py \
  --topology hypercube --n 6 \
  --fault-rate 0.1 --rounds 3 \
  --num-samples 120 \
  --seed 1101 \
  --output data/q6_train

.venv/bin/python scripts/generate_dataset.py \
  --topology hypercube --n 6 \
  --fault-rate 0.1 --rounds 3 \
  --num-samples 60 \
  --seed 2101 \
  --output data/q6_test
```

By default, each generated sample uses parameter-randomized PMC reliability:

```text
gamma ~ U(0.8, 1.0)
beta  ~ U(0.0, 0.2)
```

Use `--gamma` and `--beta` only when a fixed-parameter dataset is needed.

Run the proposed method:

```bash
.venv/bin/python scripts/run_method.py \
  --dataset data/q6_train \
  --test-dataset data/q6_test \
  --epochs 30 \
  --hidden-dim 32 \
  --layers 2 \
  --refine-top-pct 0.1
```

Run the baselines:

```bash
.venv/bin/python scripts/run_baseline.py \
  --dataset data/q6_test \
  --method clustered-pmc

.venv/bin/python scripts/run_vanilla_gnn.py \
  --dataset data/q6_train \
  --test-dataset data/q6_test \
  --epochs 30 \
  --hidden-dim 32 \
  --layers 2
```

For more experiment commands, see `README-code.md`.
