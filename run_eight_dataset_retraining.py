import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


FINAL_RE = re.compile(
    r"Final test metrics (?:after fixed-epoch training|from best validation checkpoint) \| "
    r"Precision: (?P<precision>\d+\.\d+) \| Recall: (?P<recall>\d+\.\d+) \| "
    r"F1: (?P<f1>\d+\.\d+) \| Threshold: (?P<threshold>-?\d+\.\d+) \| "
    r"Hits@1: (?P<h1>\d+\.\d+) \| Hits@5: (?P<h5>\d+\.\d+) \| "
    r"Hits@10: (?P<h10>\d+\.\d+) \| MRR: (?P<mrr>\d+\.\d+)"
)
CSLS_RE = re.compile(
    r"Supplementary CSLS ranking metrics \| Hits@1: (?P<h1>\d+\.\d+) \| "
    r"Hits@5: (?P<h5>\d+\.\d+) \| Hits@10: (?P<h10>\d+\.\d+) \| "
    r"MRR: (?P<mrr>\d+\.\d+)"
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    env: Dict[str, str]


@dataclass(frozen=True)
class RunSpec:
    dataset: DatasetSpec
    seed: int

    @property
    def name(self) -> str:
        return f"{self.dataset.name}_retained_global_seed{self.seed}"


def dataset_specs() -> List[DatasetSpec]:
    common = {
        "KG_ALIGN_PROTOCOL": "simplified_core_csls",
        "KG_ALIGN_SEM_USE_GLOBAL_VIEW": "1",
        "KG_ALIGN_SEM_RESIDUAL_MODE": "gated",
        "KG_ALIGN_SELECT_CSLS_ON_VALIDATION": "1",
        "KG_ALIGN_CSLS_K_CANDIDATES": "3,5,7,10,15,20",
        "KG_ALIGN_CSLS_BLEND_CANDIDATES": "1.00",
        "KG_ALIGN_CSLS_SELECTION_METRIC": "MRR",
        "KG_ALIGN_VALIDATION_EVERY_EPOCHS": "5",
        "KG_ALIGN_USE_WEIGHT_AVG": "0",
    }

    def dbp(name: str, pair: str) -> DatasetSpec:
        return DatasetSpec(name, {
            **common,
            "KG_ALIGN_DATASET_FAMILY": "dbp15k_raw",
            "KG_ALIGN_ROOT": "data/dbp15k",
            "KG_ALIGN_PAIR": pair,
            "KG_ALIGN_RAW_SPLIT": "0_3",
        })

    def openea(name: str, pair: str) -> DatasetSpec:
        return DatasetSpec(name, {
            **common,
            "KG_ALIGN_DATASET_FAMILY": "openea",
            "KG_ALIGN_ROOT": "data/openea",
            "KG_ALIGN_PAIR": pair,
            "KG_ALIGN_OPENEA_SPLIT": "721_5fold/1",
        })

    return [
        dbp("dbp15k_zh_en", "zh_en"),
        dbp("dbp15k_ja_en", "ja_en"),
        dbp("dbp15k_fr_en", "fr_en"),
        openea("openea_en_fr", "EN_FR_15K_V2"),
        openea("openea_en_de", "EN_DE_15K_V2"),
        openea("openea_d_w", "D_W_15K_V2"),
        openea("openea_d_y", "D_Y_15K_V2"),
        DatasetSpec("eventea_en_en", {
            **common,
            "KG_ALIGN_DATASET_FAMILY": "eventea",
            "KG_ALIGN_ROOT": "data/eventea/EventEA",
            "KG_ALIGN_PAIR": "EN_EN_20K",
            "KG_ALIGN_OPENEA_SPLIT": ".",
        }),
    ]


def parse_list(value: str, cast) -> List:
    result = [cast(part.strip()) for part in value.split(",") if part.strip()]
    if not result:
        raise ValueError("Expected at least one value")
    return result


def parse_metrics(log_path: Path) -> Optional[Dict[str, float]]:
    if not log_path.exists():
        return None
    content = log_path.read_text(encoding="utf-8", errors="ignore")
    final_match = FINAL_RE.search(content)
    if final_match is None:
        return None
    metrics = {
        "Precision": float(final_match.group("precision")),
        "Recall": float(final_match.group("recall")),
        "F1": float(final_match.group("f1")),
        "Threshold": float(final_match.group("threshold")),
        "Hits@1": float(final_match.group("h1")),
        "Hits@5": float(final_match.group("h5")),
        "Hits@10": float(final_match.group("h10")),
        "MRR": float(final_match.group("mrr")),
    }
    csls_match = CSLS_RE.search(content)
    if csls_match is not None:
        metrics.update({
            "CSLS_Hits@1": float(csls_match.group("h1")),
            "CSLS_Hits@5": float(csls_match.group("h5")),
            "CSLS_Hits@10": float(csls_match.group("h10")),
            "CSLS_MRR": float(csls_match.group("mrr")),
        })
    return metrics


def checkpoint_exists(run_dir: Path) -> bool:
    return len(list(run_dir.glob("best_model_*.pt"))) == 1


def run_one(
    run: RunSpec,
    root: Path,
    output_dir: Path,
    batch_size: int,
    epochs: Optional[int],
    threads_per_job: int,
    resume: bool,
    discard_checkpoints: bool,
) -> Dict[str, object]:
    log_dir = output_dir / "logs"
    run_dir = output_dir / "checkpoints" / run.name
    log_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run.name}.log"

    existing = parse_metrics(log_path)
    if resume and existing is not None and (
        discard_checkpoints or checkpoint_exists(run_dir)
    ):
        return {
            "dataset": run.dataset.name,
            "seed": run.seed,
            "status": "ok",
            "resumed": True,
            "checkpoint_retained": not discard_checkpoints,
            "log_path": str(log_path),
            **existing,
        }

    env = os.environ.copy()
    env.update(run.dataset.env)
    env.update({
        "KG_ALIGN_SEED": str(run.seed),
        "KG_ALIGN_SAVE_DIR": str(run_dir),
        "KG_ALIGN_BATCH_SIZE": str(batch_size),
        "KG_ALIGN_EVAL_BATCH_SIZE": str(batch_size),
        "KG_ALIGN_USE_EARLY_STOPPING": "1",
        "KG_ALIGN_EARLY_STOP_PATIENCE": "4",
        "OMP_NUM_THREADS": str(threads_per_job),
        "MKL_NUM_THREADS": str(threads_per_job),
        "PYTHONUNBUFFERED": "1",
    })
    if epochs is not None:
        env["KG_ALIGN_JOINT_EPOCHS"] = str(epochs)

    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            [sys.executable, str(root / "train.py")],
            cwd=str(root),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )

    metrics = parse_metrics(log_path)
    result: Dict[str, object] = {
        "dataset": run.dataset.name,
        "seed": run.seed,
        "status": "ok" if completed.returncode == 0 and metrics else f"failed({completed.returncode})",
        "resumed": False,
        "log_path": str(log_path),
    }
    if metrics:
        result.update(metrics)
    if result["status"] == "ok" and discard_checkpoints:
        shutil.rmtree(run_dir)
        result["checkpoint_retained"] = False
    else:
        result["checkpoint_retained"] = checkpoint_exists(run_dir)
    return result


def aggregate(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for dataset in sorted({str(row["dataset"]) for row in results}):
        group = [row for row in results if row["dataset"] == dataset and row["status"] == "ok"]
        if not group:
            continue
        summary: Dict[str, object] = {
            "dataset": dataset,
            "runs": len(group),
            "seeds": ",".join(str(row["seed"]) for row in sorted(group, key=lambda x: int(x["seed"]))),
        }
        for metric in ("Hits@1", "Hits@5", "Hits@10", "MRR", "CSLS_Hits@1", "CSLS_Hits@10", "CSLS_MRR"):
            values = [float(row[metric]) for row in group if metric in row]
            if values:
                summary[f"{metric}_mean"] = statistics.mean(values)
                summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append(summary)
    return rows


def write_outputs(output_dir: Path, results: List[Dict[str, object]]) -> None:
    ordered = sorted(results, key=lambda row: (str(row["dataset"]), int(row["seed"])))
    summaries = aggregate(ordered)
    (output_dir / "runs_summary.json").write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "aggregate_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrain the retained model on eight datasets.")
    parser.add_argument("--datasets", default=",".join(spec.name for spec in dataset_specs()))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--output-dir", default="outputs/eight_dataset_retrained_20260819")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--threads-per-job", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--discard-checkpoints", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    all_specs = {spec.name: spec for spec in dataset_specs()}
    selected_names = parse_list(args.datasets, str)
    unknown = sorted(set(selected_names) - set(all_specs))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    seeds = parse_list(args.seeds, int)
    runs = [RunSpec(all_specs[name], seed) for name in selected_names for seed in seeds]
    if args.list:
        print(json.dumps([{"name": run.name, "env": run.dataset.env} for run in runs], ensure_ascii=False, indent=2))
        return 0

    root = Path(__file__).resolve().parent
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = {
            executor.submit(
                run_one,
                run,
                root,
                output_dir,
                args.batch_size,
                args.epochs,
                max(1, args.threads_per_job),
                not args.no_resume,
                args.discard_checkpoints,
            ): run
            for run in runs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            run = futures[future]
            result = future.result()
            results.append(result)
            write_outputs(output_dir, results)
            if result["status"] == "ok":
                print(
                    f"[{index}/{len(runs)}] {run.name} | "
                    f"Hits@1={result['Hits@1']:.4f} MRR={result['MRR']:.4f}",
                    flush=True,
                )
            else:
                print(f"[{index}/{len(runs)}] {run.name} | {result['status']}", flush=True)

    failed = [row for row in results if row["status"] != "ok"]
    print(f"WROTE {output_dir}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
