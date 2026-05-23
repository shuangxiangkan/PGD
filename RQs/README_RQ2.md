# RQ2 Experiment Runner

RQ2 studies robustness under different fault rates and compares UP-GDN with Clustered-PMC.

## CPU Version

The CPU profile uses the largest topology from each CPU-scale family:

- Hypercube: Q6
- k-ary n-cube: Q4^3
- Augmented k-ary n-cube: AQ(3,4)

Fault rates:

- 10%, 20%, 30%, 40%, 50%

Run:

```bash
bash RQs/run_rq2_cpu.sh
```

## GPU Version

The GPU profile uses larger counterparts:

- Hypercube: Q10
- k-ary n-cube: Q4^5
- Augmented k-ary n-cube: AQ(5,4)

Fault rates:

- 10%, 15%, 20%, 25%, 30%, 35%, 40%, 45%, 50%

Run:

```bash
bash RQs/run_rq2_gpu.sh
```

## Outputs

Each profile runs three independent repetitions and writes:

- `RQs/results/RQ2/<profile>/rq2_raw.csv`
- `RQs/results/RQ2/<profile>/rq2_summary_by_topology.csv`
- `RQs/results/RQ2/<profile>/rq2_summary_by_rate.csv`
- rounded versions of the summary CSV files
- logs under `RQs/results/RQ2/<profile>/logs/`

Default setting:

- rounds: 2
- gamma ~ U(0.7, 1.0)
- beta ~ U(0.0, 0.3)
- backbone: GraphSAGE
