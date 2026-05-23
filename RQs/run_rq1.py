#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"


@dataclass(frozen=True)
class TopologyCase:
    family: str
    label: str
    topology: str
    n: int
    k: int | None = None

    @property
    def dataset_id(self) -> str:
        if self.k is None:
            return f"{self.family}_n{self.n}"
        return f"{self.family}_n{self.n}_k{self.k}"


CPU_CASES = [
    TopologyCase("hypercube", "Q4", "hypercube", 4),
    TopologyCase("hypercube", "Q5", "hypercube", 5),
    TopologyCase("hypercube", "Q6", "hypercube", 6),
    TopologyCase("kary", "K(2,4)", "kary", 2, 4),
    TopologyCase("kary", "K(2,6)", "kary", 2, 6),
    TopologyCase("kary", "K(3,4)", "kary", 3, 4),
    TopologyCase("augmented_kary", "AQ(2,4)", "augmented_kary", 2, 4),
    TopologyCase("augmented_kary", "AQ(2,6)", "augmented_kary", 2, 6),
    TopologyCase("augmented_kary", "AQ(3,4)", "augmented_kary", 3, 4),
]

GPU_CASES = [
    TopologyCase("hypercube", "Q6", "hypercube", 6),
    TopologyCase("hypercube", "Q8", "hypercube", 8),
    TopologyCase("hypercube", "Q10", "hypercube", 10),
    TopologyCase("kary", "K(8,2)", "kary", 2, 8),
    TopologyCase("kary", "K(6,3)", "kary", 3, 6),
    TopologyCase("kary", "K(5,4)", "kary", 4, 5),
    TopologyCase("augmented_kary", "AQ(8,2)", "augmented_kary", 2, 8),
    TopologyCase("augmented_kary", "AQ(6,3)", "augmented_kary", 3, 6),
    TopologyCase("augmented_kary", "AQ(5,4)", "augmented_kary", 4, 5),
]


METRICS = ["accuracy", "precision", "recall", "f1", "brier"]


def run_command(cmd: list[str], log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed; see {log_path}\n{' '.join(cmd)}")
    return proc.stdout


def parse_key_values(output: str, prefix: str = "") -> dict[str, float]:
    values: dict[str, float] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        key = key.strip()
        if prefix and not key.startswith(prefix):
            continue
        name = key[len(prefix) :] if prefix else key
        try:
            values[name] = float(raw.strip())
        except ValueError:
            continue
    return values


def dataset_command(case: TopologyCase, output: Path, seed: int, samples: int, args) -> list[str]:
    cmd = [
        str(PYTHON),
        "scripts/generate_dataset.py",
        "--topology",
        case.topology,
        "--n",
        str(case.n),
        "--fault-rate",
        str(args.fault_rate),
        "--rounds",
        str(args.rounds),
        "--gamma-min",
        str(args.gamma_min),
        "--gamma-max",
        str(args.gamma_max),
        "--beta-min",
        str(args.beta_min),
        "--beta-max",
        str(args.beta_max),
        "--num-samples",
        str(samples),
        "--seed",
        str(seed),
        "--output",
        str(output),
    ]
    if case.k is not None:
        cmd.extend(["--k", str(case.k)])
    return cmd


def method_command(train_data: Path, test_data: Path, seed: int, args) -> list[str]:
    cmd = [
        str(PYTHON),
        "scripts/run_method.py",
        "--dataset",
        str(train_data),
        "--test-dataset",
        str(test_data),
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--hidden-dim",
        str(args.hidden_dim),
        "--layers",
        str(args.layers),
        "--backbone",
        args.backbone,
        "--refine-top-pct",
        str(args.refine_top_pct),
        "--device",
        args.device,
        "--log-every",
        str(args.log_every),
    ]
    if args.batch_size > 1:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.amp:
        cmd.append("--amp")
    return cmd


def baseline_command(test_data: Path) -> list[str]:
    return [
        str(PYTHON),
        "scripts/run_baseline.py",
        "--dataset",
        str(test_data),
        "--method",
        "clustered-pmc",
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["profile"]), str(row["topology"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)

    summary = []
    for (profile, topology, method), group in sorted(grouped.items()):
        out: dict[str, object] = {
            "profile": profile,
            "topology": topology,
            "method": method,
            "runs": len(group),
        }
        for metric in METRICS:
            vals = [float(row[metric]) for row in group]
            out[f"{metric}_mean"] = statistics.mean(vals)
            out[f"{metric}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summary.append(out)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RQ1 effectiveness experiments.")
    parser.add_argument("--profile", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--fault-rate", type=float, default=0.3)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--gamma-min", type=float, default=0.7)
    parser.add_argument("--gamma-max", type=float, default=1.0)
    parser.add_argument("--beta-min", type=float, default=0.0)
    parser.add_argument("--beta-max", type=float, default=0.3)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--backbone", choices=["gcn", "graphsage", "gat"], default="graphsage")
    parser.add_argument("--refine-top-pct", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed-base", type=int, default=3100)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "RQs" / "results" / "RQ1")
    parser.add_argument("--force", action="store_true", help="Regenerate datasets even if they exist.")
    args = parser.parse_args()

    if not PYTHON.exists():
        raise FileNotFoundError(f"virtualenv python not found: {PYTHON}")

    cases = CPU_CASES if args.profile == "cpu" else GPU_CASES
    train_samples = args.train_samples or (120 if args.profile == "cpu" else 200)
    test_samples = args.test_samples or (60 if args.profile == "cpu" else 100)
    epochs = args.epochs or (80 if args.profile == "cpu" else 100)
    args.epochs = epochs

    data_dir = args.output_dir / args.profile / "data"
    log_dir = args.output_dir / args.profile / "logs"
    rows: list[dict[str, object]] = []

    for case in cases:
        for run_id in range(1, args.runs + 1):
            seed = args.seed_base + 1000 * run_id + len(rows)
            stem = f"{case.dataset_id}_run{run_id}"
            train_data = data_dir / f"{stem}_train"
            test_data = data_dir / f"{stem}_test"

            if args.force or not (train_data.with_suffix(".npz")).exists():
                print(f"[generate] {case.label} run={run_id} train")
                run_command(
                    dataset_command(case, train_data, seed, train_samples, args),
                    log_dir / f"{stem}_generate_train.log",
                )
            if args.force or not (test_data.with_suffix(".npz")).exists():
                print(f"[generate] {case.label} run={run_id} test")
                run_command(
                    dataset_command(case, test_data, seed + 97, test_samples, args),
                    log_dir / f"{stem}_generate_test.log",
                )

            print(f"[method] {case.label} run={run_id}")
            method_out = run_command(
                method_command(train_data, test_data, seed, args),
                log_dir / f"{stem}_up_gdn.log",
            )
            method_metrics = parse_key_values(method_out, prefix="refined_")
            rows.append(
                {
                    "profile": args.profile,
                    "family": case.family,
                    "topology": case.label,
                    "method": "UP-GDN",
                    "run": run_id,
                    "seed": seed,
                    **{metric: method_metrics[metric] for metric in METRICS},
                }
            )

            print(f"[baseline] {case.label} run={run_id}")
            baseline_out = run_command(
                baseline_command(test_data),
                log_dir / f"{stem}_clustered_pmc.log",
            )
            baseline_metrics = parse_key_values(baseline_out)
            rows.append(
                {
                    "profile": args.profile,
                    "family": case.family,
                    "topology": case.label,
                    "method": "Clustered-PMC",
                    "run": run_id,
                    "seed": seed,
                    **{metric: baseline_metrics[metric] for metric in METRICS},
                }
            )

    raw_path = args.output_dir / args.profile / "rq1_raw.csv"
    summary_path = args.output_dir / args.profile / "rq1_summary.csv"
    write_csv(raw_path, rows)
    write_csv(summary_path, summarize(rows))
    print(f"wrote {raw_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
