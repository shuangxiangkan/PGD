# RQ3 Feature Ablation Runner

This experiment studies the first RQ3 sub-question: the effect of node features.

Variants:

- `full`
- `w/o match ratio`
- `w/o mismatch ratio`
- `w/o dispersion`

## CPU Version

The CPU profile uses the largest topology from each CPU-scale family:

- Hypercube: Q6
- k-ary n-cube: Q4^3
- Augmented k-ary n-cube: AQ(3,4)

Run:

```bash
bash RQs/run_rq3_feature_cpu.sh
```

## GPU Version

The GPU profile uses larger counterparts:

- Hypercube: Q10
- k-ary n-cube: Q4^5
- Augmented k-ary n-cube: AQ(5,4)

Run:

```bash
bash RQs/run_rq3_feature_gpu.sh
```

## Outputs

Each profile runs three independent repetitions and writes:

- `RQs/results/RQ3_feature/<profile>/rq3_feature_raw.csv`
- `RQs/results/RQ3_feature/<profile>/rq3_feature_summary_by_topology.csv`
- `RQs/results/RQ3_feature/<profile>/rq3_feature_summary_by_variant.csv`
- rounded versions of the same CSV files
- per-epoch learning curves under `RQs/results/RQ3_feature/<profile>/curves/`
- logs under `RQs/results/RQ3_feature/<profile>/logs/`

The summaries include standard diagnosis metrics and convergence metrics:

- accuracy, precision, recall, F1, Brier, ECE, top-k localization
- best F1 and best-F1 epoch
- first epoch reaching F1 >= 0.95 and F1 >= 0.99
- mean F1 over evaluated epochs
- final training loss
- training-loss reduction
