import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


METRIC_RE = re.compile(
    r"Final test metrics (?:after fixed-epoch training|from best validation checkpoint) \| "
    r"Precision: (?P<precision>\d+\.\d+) \| "
    r"Recall: (?P<recall>\d+\.\d+) \| "
    r"F1: (?P<f1>\d+\.\d+) \| "
    r"Threshold: (?P<threshold>-?\d+\.\d+) \| "
    r"Hits@1: (?P<h1>\d+\.\d+) \| "
    r"Hits@5: (?P<h5>\d+\.\d+) \| "
    r"Hits@10: (?P<h10>\d+\.\d+) \| "
    r"MRR: (?P<mrr>\d+\.\d+)"
)

CSLS_RE = re.compile(
    r"Supplementary CSLS ranking metrics \| "
    r"Hits@1: (?P<h1>\d+\.\d+) \| "
    r"Hits@5: (?P<h5>\d+\.\d+) \| "
    r"Hits@10: (?P<h10>\d+\.\d+) \| "
    r"MRR: (?P<mrr>\d+\.\d+)"
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    env: Dict[str, str]


@dataclass(frozen=True)
class VariantSpec:
    name: str
    env: Dict[str, str]


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    variant: str
    seed: int
    env: Dict[str, str]

    @property
    def name(self) -> str:
        return f"{self.dataset}_{self.variant}_seed{self.seed}"


def parse_seed_list(seed_text: str) -> List[int]:
    seeds: List[int] = []
    for part in seed_text.split(","):
        token = part.strip()
        if token:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def build_datasets() -> List[DatasetSpec]:
    return [
        DatasetSpec(
            name="dbp15k_zh_en",
            env={
                "KG_ALIGN_DATASET_FAMILY": "dbp15k_raw",
                "KG_ALIGN_ROOT": "data/dbp15k",
                "KG_ALIGN_PAIR": "zh_en",
                "KG_ALIGN_RAW_SPLIT": "0_3",
                "KG_ALIGN_PROTOCOL": "simplified_core",
            },
        ),
        DatasetSpec(
            name="openea_en_fr",
            env={
                "KG_ALIGN_DATASET_FAMILY": "openea",
                "KG_ALIGN_ROOT": "data/openea",
                "KG_ALIGN_PAIR": "EN_FR_15K_V2",
                "KG_ALIGN_OPENEA_SPLIT": "721_5fold/1",
                "KG_ALIGN_PROTOCOL": "simplified_core",
            },
        ),
    ]


def build_variants(profile: str) -> List[VariantSpec]:
    mainline = [VariantSpec("base", {})]

    core_ablation = [
        VariantSpec("base", {}),
        VariantSpec("w_o_mst", {"KG_ALIGN_USE_MST": "0"}),
        VariantSpec("w_o_ce", {"KG_ALIGN_USE_CE": "0"}),
        VariantSpec("neighbors6", {"KG_ALIGN_NUM_NEIGHBORS": "6"}),
        VariantSpec("neighbors10", {"KG_ALIGN_NUM_NEIGHBORS": "10"}),
    ]

    if profile == "mainline":
        return mainline
    if profile == "core_ablation":
        return core_ablation
    raise ValueError(f"Unknown profile: {profile}")


def build_runs(seeds: List[int], profile: str) -> List[RunSpec]:
    runs: List[RunSpec] = []
    for dataset in build_datasets():
        for variant in build_variants(profile):
            for seed in seeds:
                runs.append(
                    RunSpec(
                        dataset=dataset.name,
                        variant=variant.name,
                        seed=seed,
                        env={
                            **dataset.env,
                            **variant.env,
                            "KG_ALIGN_SEED": str(seed),
                        },
                    )
                )
    return runs


def parse_metrics(log_path: Path) -> Optional[Dict[str, float]]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    match = METRIC_RE.search(text)
    if not match:
        return None
    metrics = {
        "Precision": float(match.group("precision")),
        "Recall": float(match.group("recall")),
        "F1": float(match.group("f1")),
        "Threshold": float(match.group("threshold")),
        "Hits@1": float(match.group("h1")),
        "Hits@5": float(match.group("h5")),
        "Hits@10": float(match.group("h10")),
        "MRR": float(match.group("mrr")),
    }
    csls_match = CSLS_RE.search(text)
    if csls_match:
        metrics.update(
            {
                "CSLS_Hits@1": float(csls_match.group("h1")),
                "CSLS_Hits@5": float(csls_match.group("h5")),
                "CSLS_Hits@10": float(csls_match.group("h10")),
                "CSLS_MRR": float(csls_match.group("mrr")),
            }
        )
    return metrics


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def aggregate_results(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    metric_keys = [
        "Precision",
        "Recall",
        "F1",
        "Hits@1",
        "Hits@5",
        "Hits@10",
        "MRR",
        "CSLS_Hits@1",
        "CSLS_Hits@5",
        "CSLS_Hits@10",
        "CSLS_MRR",
    ]
    groups: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for item in results:
        if item.get("status") != "ok":
            continue
        groups.setdefault((str(item["dataset"]), str(item["variant"])), []).append(item)

    rows: List[Dict[str, object]] = []
    for (dataset, variant), items in groups.items():
        row: Dict[str, object] = {
            "dataset": dataset,
            "variant": variant,
            "runs": len(items),
            "seeds": ",".join(str(item["seed"]) for item in sorted(items, key=lambda x: int(x["seed"]))),
        }
        for key in metric_keys:
            values = [float(item[key]) for item in items if key in item]
            if values:
                avg, std = mean_std(values)
                row[f"{key}_mean"] = avg
                row[f"{key}_std"] = std
        rows.append(row)

    return sorted(
        rows,
        key=lambda item: (
            str(item["dataset"]),
            -float(item.get("Hits@1_mean", -1.0)),
            -float(item.get("MRR_mean", -1.0)),
        ),
    )


def write_outputs(output_dir: Path, results: List[Dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_results(results)

    (output_dir / "runs_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "aggregate_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    run_headers = [
        "dataset",
        "variant",
        "seed",
        "status",
        "Precision",
        "Recall",
        "F1",
        "Threshold",
        "Hits@1",
        "Hits@5",
        "Hits@10",
        "MRR",
        "CSLS_Hits@1",
        "CSLS_Hits@5",
        "CSLS_Hits@10",
        "CSLS_MRR",
        "log_path",
    ]
    run_lines = ["\t".join(run_headers)]
    for item in results:
        run_lines.append("\t".join(str(item.get(key, "")) for key in run_headers))
    (output_dir / "runs_summary.tsv").write_text("\n".join(run_lines) + "\n", encoding="utf-8")

    agg_headers = [
        "dataset",
        "variant",
        "runs",
        "seeds",
        "Hits@1_mean",
        "Hits@1_std",
        "Hits@10_mean",
        "Hits@10_std",
        "MRR_mean",
        "MRR_std",
        "CSLS_Hits@1_mean",
        "CSLS_Hits@1_std",
        "CSLS_MRR_mean",
        "CSLS_MRR_std",
        "F1_mean",
        "F1_std",
    ]
    agg_lines = ["\t".join(agg_headers)]
    for item in aggregate:
        agg_lines.append("\t".join(str(item.get(key, "")) for key in agg_headers))
    (output_dir / "aggregate_summary.tsv").write_text("\n".join(agg_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run 3-seed experiments on DBP15K zh_en and OpenEA EN_FR_15K_V2."
    )
    parser.add_argument(
        "--profile",
        choices=("mainline", "core_ablation"),
        default="core_ablation",
        help="mainline runs the retained model; core_ablation runs the compact architecture suite.",
    )
    parser.add_argument(
        "--seeds",
        default="42,43,44",
        help="Comma-separated seeds. Default: 42,43,44.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/two_dataset_3seed_ablation",
        help="Directory for logs and summary files.",
    )
    parser.add_argument("--list", action="store_true", help="Print planned runs and exit.")
    args = parser.parse_args()

    seeds = parse_seed_list(args.seeds)
    runs = build_runs(seeds=seeds, profile=args.profile)
    if args.list:
        print(json.dumps([{"name": run.name, "env": run.env} for run in runs], ensure_ascii=False, indent=2))
        return 0

    output_dir = Path(args.output_dir)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(__file__).with_name("train.py")
    python_bin = sys.executable
    results: List[Dict[str, object]] = []

    for index, run in enumerate(runs, start=1):
        log_path = log_dir / f"{run.name}.log"
        env = os.environ.copy()
        env.update(run.env)
        env["PYTHONUNBUFFERED"] = "1"

        print(f"\n[{index}/{len(runs)}] Running {run.name}")
        print(f"Log: {log_path}")
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.run(
                [python_bin, str(train_path)],
                cwd=str(Path(__file__).resolve().parent),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )

        item: Dict[str, object] = {
            "dataset": run.dataset,
            "variant": run.variant,
            "seed": run.seed,
            "status": "failed",
            "log_path": str(log_path),
        }
        metrics = parse_metrics(log_path) if log_path.exists() else None
        if process.returncode == 0 and metrics:
            item["status"] = "ok"
            item.update(metrics)
            print(
                f"Finished {run.name}: "
                f"Hits@1={metrics['Hits@1']:.4f}, "
                f"Hits@10={metrics['Hits@10']:.4f}, "
                f"MRR={metrics['MRR']:.4f}"
            )
        else:
            item["status"] = f"failed({process.returncode})"
            print(f"Finished {run.name}: failed or metrics missing; inspect {log_path}")

        results.append(item)
        write_outputs(output_dir, results)

    print(f"\nRun summary: {output_dir / 'runs_summary.tsv'}")
    print(f"Aggregate summary: {output_dir / 'aggregate_summary.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
