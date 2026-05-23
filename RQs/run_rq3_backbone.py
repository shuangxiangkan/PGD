#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import statistics
import subprocess


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
    TopologyCase("hypercube", "Q6", "hypercube", 6),
    TopologyCase("kary", "Q4^3", "kary", 3, 4),
    TopologyCase("augmented_kary", "AQ(3,4)", "augmented_kary", 3, 4),
]

GPU_CASES = [
    TopologyCase("hypercube", "Q10", "hypercube", 10),
    TopologyCase("kary", "Q4^5", "kary", 5, 4),
    TopologyCase("augmented_kary", "AQ(5,4)", "augmented_kary", 5, 4),
]

BACKBONES = ["gcn", "graphsage", "gat"]
FINAL_METRICS = ["accuracy", "precision", "recall", "f1", "brier", "ece", "topk"]
CONVERGENCE_METRICS = [
    "best_f1",
    "best_f1_epoch",
    "epoch_to_f1_095",
    "epoch_to_f1_099",
    "mean_f1_over_epochs",
    "final_train_loss",
    "loss_reduction",
]


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


def method_command(train_data: Path, test_data: Path, curve_path: Path, backbone: str, seed: int, args) -> list[str]:
    return [
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
        backbone,
        "--refine-top-pct",
        str(args.refine_top_pct),
        "--device",
        args.device,
        "--eval-every",
        str(args.eval_every),
        "--curve-output",
        str(curve_path),
        "--log-every",
        str(args.log_every),
    ]


def read_curve(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def first_epoch_at(rows: list[dict[str, str]], metric: str, threshold: float) -> int:
    for row in rows:
        if float(row[metric]) >= threshold:
            return int(float(row["epoch"]))
    return -1


def collect_metrics(curve_path: Path) -> dict[str, float | int]:
    rows = read_curve(curve_path)
    if not rows:
        raise ValueError(f"empty curve file: {curve_path}")
    final = rows[-1]
    best = max(rows, key=lambda row: float(row["refined_f1"]))
    first_loss = float(rows[0]["train_loss"])
    final_loss = float(final["train_loss"])
    out: dict[str, float | int] = {
        metric: float(final[f"refined_{metric}"]) for metric in FINAL_METRICS
    }
    out.update(
        {
            "best_f1": float(best["refined_f1"]),
            "best_f1_epoch": int(float(best["epoch"])),
            "epoch_to_f1_095": first_epoch_at(rows, "refined_f1", 0.95),
            "epoch_to_f1_099": first_epoch_at(rows, "refined_f1", 0.99),
            "mean_f1_over_epochs": statistics.mean(float(row["refined_f1"]) for row in rows),
            "final_train_loss": final_loss,
            "loss_reduction": first_loss - final_loss,
        }
    )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]], group_keys: list[str]) -> list[dict[str, object]]:
    metrics = FINAL_METRICS + CONVERGENCE_METRICS
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(row)

    summary = []
    for key, group in sorted(grouped.items()):
        out: dict[str, object] = {k: v for k, v in zip(group_keys, key)}
        out["runs"] = len(group)
        for metric in metrics:
            vals = [float(row[metric]) for row in group]
            out[f"{metric}_mean"] = statistics.mean(vals)
            out[f"{metric}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summary.append(out)
    return summary


def round_csv(src: Path, dst: Path) -> None:
    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    if not rows:
        return
    with dst.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            out = {}
            for key, value in row.items():
                try:
                    if key in {"runs", "run", "seed"} or key.endswith("_epoch") or key.startswith("epoch_to_"):
                        out[key] = str(int(round(float(value))))
                    else:
                        out[key] = f"{float(value):.4f}"
                except ValueError:
                    out[key] = value
            writer.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RQ3 GNN-backbone comparison experiments.")
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
    parser.add_argument("--refine-top-pct", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=9300)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "RQs" / "results" / "RQ3_backbone")
    parser.add_argument("--force", action="store_true", help="Regenerate datasets and rerun curves.")
    args = parser.parse_args()

    if not PYTHON.exists():
        raise FileNotFoundError(f"virtualenv python not found: {PYTHON}")

    cases = CPU_CASES if args.profile == "cpu" else GPU_CASES
    train_samples = args.train_samples or (120 if args.profile == "cpu" else 200)
    test_samples = args.test_samples or (60 if args.profile == "cpu" else 100)
    args.epochs = args.epochs or (80 if args.profile == "cpu" else 100)

    out_dir = args.output_dir / args.profile
    data_dir = out_dir / "data"
    log_dir = out_dir / "logs"
    curve_dir = out_dir / "curves"
    rows: list[dict[str, object]] = []

    for case in cases:
        for run_id in range(1, args.runs + 1):
            seed = args.seed_base + 1000 * run_id + len(rows)
            data_stem = f"{case.dataset_id}_run{run_id}"
            train_data = data_dir / f"{data_stem}_train"
            test_data = data_dir / f"{data_stem}_test"

            if args.force or not train_data.with_suffix(".npz").exists():
                print(f"[generate] {case.label} run={run_id} train")
                run_command(dataset_command(case, train_data, seed, train_samples, args), log_dir / f"{data_stem}_generate_train.log")
            if args.force or not test_data.with_suffix(".npz").exists():
                print(f"[generate] {case.label} run={run_id} test")
                run_command(dataset_command(case, test_data, seed + 97, test_samples, args), log_dir / f"{data_stem}_generate_test.log")

            for backbone in BACKBONES:
                stem = f"{data_stem}_{backbone}"
                curve_path = curve_dir / f"{stem}.csv"
                if args.force and curve_path.exists():
                    curve_path.unlink()
                if not curve_path.exists():
                    print(f"[method] {case.label} run={run_id} backbone={backbone}")
                    run_command(method_command(train_data, test_data, curve_path, backbone, seed, args), log_dir / f"{stem}.log")
                rows.append(
                    {
                        "profile": args.profile,
                        "family": case.family,
                        "topology": case.label,
                        "backbone": backbone,
                        "run": run_id,
                        "seed": seed,
                        **collect_metrics(curve_path),
                    }
                )

    raw_path = out_dir / "rq3_backbone_raw.csv"
    topology_path = out_dir / "rq3_backbone_summary_by_topology.csv"
    backbone_path = out_dir / "rq3_backbone_summary_by_backbone.csv"
    write_csv(raw_path, rows)
    write_csv(topology_path, summarize(rows, ["profile", "topology", "backbone"]))
    write_csv(backbone_path, summarize(rows, ["profile", "backbone"]))
    round_csv(raw_path, out_dir / "rq3_backbone_raw_rounded.csv")
    round_csv(topology_path, out_dir / "rq3_backbone_summary_by_topology_rounded.csv")
    round_csv(backbone_path, out_dir / "rq3_backbone_summary_by_backbone_rounded.csv")
    print(f"wrote {raw_path}")
    print(f"wrote {topology_path}")
    print(f"wrote {backbone_path}")


if __name__ == "__main__":
    main()
