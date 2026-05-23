# RQ1 Experiment Runner

RQ1 compares UP-GDN with Clustered-PMC.

## CPU Version

The CPU profile uses smaller topologies:

- Hypercube: Q4, Q5, Q6
- k-ary n-cube: K(2,4), K(2,6), K(3,4)
- Augmented k-ary n-cube: AQ(2,4), AQ(2,6), AQ(3,4)

Run:

```bash
bash RQs/run_rq1_cpu.sh
```

## GPU Version

The GPU profile uses larger topologies:

- Hypercube: Q6, Q8, Q10
- k-ary n-cube: K(8,2), K(6,3), K(5,4)
- Augmented k-ary n-cube: AQ(8,2), AQ(6,3), AQ(5,4)

Run:

```bash
bash RQs/run_rq1_gpu.sh
```

## Outputs

Each profile runs three independent repetitions and writes:

- `RQs/results/RQ1/<profile>/rq1_raw.csv`
- `RQs/results/RQ1/<profile>/rq1_summary.csv`
- logs under `RQs/results/RQ1/<profile>/logs/`

Default setting:

- fault rate: 30%
- rounds: 2
- gamma ~ U(0.7, 1.0)
- beta ~ U(0.0, 0.3)
- backbone: GraphSAGE
