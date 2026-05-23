# RQ3 Posterior Refinement and Backbone Runners

These runners cover the second and third RQ3 sub-questions.

## Posterior Refinement Study

Compares:

- `w/o posterior refinement`: raw GNN posterior thresholding
- `full`: posterior refinement via local syndrome consistency

CPU:

```bash
bash RQs/run_rq3_refinement_cpu.sh
```

GPU:

```bash
bash RQs/run_rq3_refinement_gpu.sh
```

The current GPU launcher uses the reduced-data accelerated setting: `100` training samples, `100` test samples, `60` epochs, evaluation every `5` epochs, mini-batch size `16`, and CUDA mixed precision.

Outputs are written under `RQs/results/RQ3_refinement/<profile>/`.

## Backbone Comparison

Compares:

- GCN
- GraphSAGE
- GAT

CPU:

```bash
bash RQs/run_rq3_backbone_cpu.sh
```

GPU:

```bash
bash RQs/run_rq3_backbone_gpu.sh
```

The current GPU launcher uses the reduced-data accelerated setting: `100` training samples, `100` test samples, `60` epochs, evaluation every `5` epochs, mini-batch size `16`, and CUDA mixed precision.

Outputs are written under `RQs/results/RQ3_backbone/<profile>/`.

## Topologies

CPU profile:

- Q6
- Q4^3
- AQ(3,4)

GPU profile:

- Q6
- Q8^2
- AQ(2,8)

## Metrics

Each profile runs three independent repetitions. The CSV summaries include:

- accuracy, precision, recall, F1
- Brier score, ECE, top-k localization
- best F1 and best-F1 epoch
- first epoch reaching F1 >= 0.95 and F1 >= 0.99
- mean F1 over evaluated epochs
- final training loss and training-loss reduction
- for the refinement study: refinement change/correction/wrong-flip rates
