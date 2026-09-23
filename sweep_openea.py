import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


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


@dataclass
class SweepRun:
    name: str
    env: Dict[str, str]


def parse_seed_list(seed_text: str) -> List[int]:
    seeds = []
    for part in seed_text.split(","):
        token = part.strip()
        if not token:
            continue
        seeds.append(int(token))
    if not seeds:
        raise ValueError("At least one seed must be provided.")
    return seeds


def build_seeded_runs(base_name: str, seeds: List[int], env: Dict[str, str]) -> List[SweepRun]:
    return [
        SweepRun(
            name=f"{base_name}_seed{seed}",
            env={**env, "KG_ALIGN_SEED": str(seed)},
        )
        for seed in seeds
    ]


def build_minimal_sweep(pair: str, split: str, seeds: List[int]) -> List[SweepRun]:
    shared = {
        "KG_ALIGN_DATASET_FAMILY": "openea",
        "KG_ALIGN_PAIR": pair,
        "KG_ALIGN_OPENEA_SPLIT": split,
    }
    specs = [
        (
            "baseline",
            {},
        ),
        (
            "lr3e4_long50",
            {
                "KG_ALIGN_LR": "3e-4",
                "KG_ALIGN_JOINT_EPOCHS": "50",
            },
        ),
        (
            "temp005",
            {
                "KG_ALIGN_TEMPERATURE": "0.05",
                "KG_ALIGN_LR": "3e-4",
                "KG_ALIGN_JOINT_EPOCHS": "50",
            },
        ),
        (
            "combined_stable",
            {
                "KG_ALIGN_LR": "3e-4",
                "KG_ALIGN_JOINT_EPOCHS": "50",
                "KG_ALIGN_TEMPERATURE": "0.05",
            },
        ),
    ]

    runs: List[SweepRun] = []
    for name, env in specs:
        runs.extend(build_seeded_runs(name, seeds, {**shared, **env}))
    return runs


def build_refine_sweep(pair: str, split: str, seeds: List[int]) -> List[SweepRun]:
    shared = {
        "KG_ALIGN_DATASET_FAMILY": "openea",
        "KG_ALIGN_PAIR": pair,
        "KG_ALIGN_OPENEA_SPLIT": split,
        "KG_ALIGN_LR": "3e-4",
        "KG_ALIGN_JOINT_EPOCHS": "50",
    }
    specs = [
        (
            "full_ranking",
            {},
        ),
        (
            "full_ranking_csls15",
            {
                "KG_ALIGN_ALIGNMENT_CSLS_K": "15",
            },
        ),
        (
            "full_ranking_neighbors12",
            {
                "KG_ALIGN_NUM_NEIGHBORS": "12",
            },
        ),
        (
            "full_ranking_gnn2",
            {
                "KG_ALIGN_GNN_LAYERS": "2",
            },
        ),
    ]

    runs: List[SweepRun] = []
    for name, env in specs:
        runs.extend(build_seeded_runs(name, seeds, {**shared, **env}))
    return runs


def build_refine2_sweep(pair: str, split: str, seeds: List[int]) -> List[SweepRun]:
    shared = {
        "KG_ALIGN_DATASET_FAMILY": "openea",
        "KG_ALIGN_PAIR": pair,
        "KG_ALIGN_OPENEA_SPLIT": split,
        "KG_ALIGN_LR": "3e-4",
        "KG_ALIGN_JOINT_EPOCHS": "50",
    }
    specs = [
        (
            "full_ranking_blend05",
            {
                "KG_ALIGN_ALIGNMENT_CSLS_BLEND": "0.5",
            },
        ),
        (
            "full_ranking_blend075",
            {
                "KG_ALIGN_ALIGNMENT_CSLS_BLEND": "0.75",
            },
        ),
        (
            "full_ranking_neighbors10",
            {
                "KG_ALIGN_NUM_NEIGHBORS": "10",
            },
        ),
        (
            "full_ranking_neighbors14",
            {
                "KG_ALIGN_NUM_NEIGHBORS": "14",
            },
        ),
    ]

    runs: List[SweepRun] = []
    for name, env in specs:
        runs.extend(build_seeded_runs(name, seeds, {**shared, **env}))
    return runs


def build_fullrank_next_sweep(pair: str, split: str, seeds: List[int]) -> List[SweepRun]:
    """
    Next full-ranking sweep after removing blocking.

    Focus:
    - whether CSLS still helps under full ranking
    - whether CSLS should be softened rather than disabled
    - whether neighbor budget near the current default improves recall
    - whether a slower/longer optimizer schedule is more stable
    """
    shared = {
        "KG_ALIGN_DATASET_FAMILY": "openea",
        "KG_ALIGN_PAIR": pair,
        "KG_ALIGN_OPENEA_SPLIT": split,
        "KG_ALIGN_LR": "3e-4",
        "KG_ALIGN_JOINT_EPOCHS": "50",
    }
    specs = [
        (
            "fullrank_base",
            {},
        ),
        (
            "fullrank_csls_off",
            {
                "KG_ALIGN_ALIGNMENT_CSLS_K": "0",
            },
        ),
        (
            "fullrank_csls_blend075",
            {
                "KG_ALIGN_ALIGNMENT_CSLS_BLEND": "0.75",
            },
        ),
        (
            "fullrank_csls_blend05",
            {
                "KG_ALIGN_ALIGNMENT_CSLS_BLEND": "0.5",
            },
        ),
        (
            "fullrank_neighbors10",
            {
                "KG_ALIGN_NUM_NEIGHBORS": "10",
            },
        ),
        (
            "fullrank_neighbors12",
            {
                "KG_ALIGN_NUM_NEIGHBORS": "12",
            },
        ),
        (
            "fullrank_temp005",
            {
                "KG_ALIGN_TEMPERATURE": "0.05",
            },
        ),
        (
            "fullrank_lr2e4_e60",
            {
                "KG_ALIGN_LR": "2e-4",
                "KG_ALIGN_JOINT_EPOCHS": "60",
            },
        ),
    ]

    runs: List[SweepRun] = []
    for name, env in specs:
        runs.extend(build_seeded_runs(name, seeds, {**shared, **env}))
    return runs


def build_simplified_core_next_sweep(pair: str, split: str, seeds: List[int]) -> List[SweepRun]:
    """
    Focused sweep for the current OpenEA mainline:
    - protocol=simplified_core
    - CE-lite on
    - auxiliary losses off
    - bidirectional InfoNCE only

    The four variants are the highest-signal next checks:
    - lower lr + longer training
    - fewer neighbors
    - more neighbors
    - smaller CE residual ratio
    """
    shared = {
        "KG_ALIGN_DATASET_FAMILY": "openea",
        "KG_ALIGN_PAIR": pair,
        "KG_ALIGN_OPENEA_SPLIT": split,
        "KG_ALIGN_PROTOCOL": "simplified_core",
    }
    specs = [
        (
            "simplified_core_base",
            {},
        ),
        (
            "simplified_core_lr2e4_e60",
            {
                "KG_ALIGN_LR": "2e-4",
                "KG_ALIGN_JOINT_EPOCHS": "60",
            },
        ),
        (
            "simplified_core_neighbors6",
            {
                "KG_ALIGN_NUM_NEIGHBORS": "6",
            },
        ),
        (
            "simplified_core_neighbors10",
            {
                "KG_ALIGN_NUM_NEIGHBORS": "10",
            },
        ),
        (
            "simplified_core_ce_res005",
            {
                "KG_ALIGN_CE_RESIDUAL_RATIO": "0.05",
            },
        ),
    ]

    runs: List[SweepRun] = []
    for name, env in specs:
        runs.extend(build_seeded_runs(name, seeds, {**shared, **env}))
    return runs


def parse_metrics(log_path: Path) -> Optional[Dict[str, float]]:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    match = METRIC_RE.search(text)
    if not match:
        return None
    return {
        "Precision": float(match.group("precision")),
        "Recall": float(match.group("recall")),
        "F1": float(match.group("f1")),
        "Threshold": float(match.group("threshold")),
        "Hits@1": float(match.group("h1")),
        "Hits@5": float(match.group("h5")),
        "Hits@10": float(match.group("h10")),
        "MRR": float(match.group("mrr")),
    }


def write_summary(summary_path: Path, results: List[Dict[str, object]]) -> None:
    ranked = sorted(
        results,
        key=lambda item: (
            item.get("Hits@1", -1.0),
            item.get("MRR", -1.0),
            item.get("Hits@10", -1.0),
            item.get("F1", -1.0),
        ),
        reverse=True,
    )
    summary_path.write_text(json.dumps(ranked, ensure_ascii=True, indent=2), encoding="utf-8")

    tsv_path = summary_path.with_suffix(".tsv")
    lines = ["name\tstatus\tPrecision\tRecall\tF1\tThreshold\tHits@1\tHits@5\tHits@10\tMRR\tlog_path"]
    for item in ranked:
        lines.append(
            "\t".join(
                [
                    str(item["name"]),
                    str(item["status"]),
                    str(item.get("Precision", "")),
                    str(item.get("Recall", "")),
                    str(item.get("F1", "")),
                    str(item.get("Threshold", "")),
                    str(item.get("Hits@1", "")),
                    str(item.get("Hits@5", "")),
                    str(item.get("Hits@10", "")),
                    str(item.get("MRR", "")),
                    str(item["log_path"]),
                ]
            )
        )
    tsv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OpenEA official-split tuning sweeps.")
    parser.add_argument("--pair", default="EN_FR_15K_V2", help="OpenEA pair name.")
    parser.add_argument("--split", default="721_5fold/1", help="OpenEA official split.")
    parser.add_argument(
        "--profile",
        choices=("minimal", "refine", "refine2", "fullrank_next", "simplified_core_next"),
        default="minimal",
        help="Which sweep profile to run.",
    )
    parser.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated seed list. Default is a single-seed tuning pass.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/sweeps_openea",
        help="Directory for logs and summaries.",
    )
    parser.add_argument("--list", action="store_true", help="Print planned runs and exit.")
    args = parser.parse_args()

    seeds = parse_seed_list(args.seeds)
    if args.profile == "refine":
        runs = build_refine_sweep(args.pair, args.split, seeds)
    elif args.profile == "refine2":
        runs = build_refine2_sweep(args.pair, args.split, seeds)
    elif args.profile == "fullrank_next":
        runs = build_fullrank_next_sweep(args.pair, args.split, seeds)
    elif args.profile == "simplified_core_next":
        runs = build_simplified_core_next_sweep(args.pair, args.split, seeds)
    else:
        runs = build_minimal_sweep(args.pair, args.split, seeds)
    if args.list:
        print(json.dumps([{"name": run.name, "env": run.env} for run in runs], ensure_ascii=True, indent=2))
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_tag = args.split.replace("/", "_")
    seed_tag = "single_seed" if len(seeds) == 1 else "multi_seed"
    summary_path = output_dir / f"openea_{args.pair}_{split_tag}_{args.profile}_{seed_tag}_summary.json"

    results: List[Dict[str, object]] = []
    python_bin = sys.executable
    train_path = Path(__file__).with_name("train.py")

    for run in runs:
        log_path = output_dir / f"{run.name}.log"
        env = os.environ.copy()
        env.update(run.env)
        env["PYTHONUNBUFFERED"] = "1"

        print(f"\n=== Running {run.name} ===")
        print(f"Log: {log_path}")
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.run(
                [python_bin, str(train_path)],
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).resolve().parent),
                check=False,
            )

        metrics = parse_metrics(log_path)
        result = {
            "name": run.name,
            "status": "ok" if process.returncode == 0 and metrics else f"failed({process.returncode})",
            "log_path": str(log_path),
        }
        if metrics:
            result.update(metrics)
            print(
                f"Finished {run.name}: "
                f"P={metrics['Precision']:.4f}, "
                f"R={metrics['Recall']:.4f}, "
                f"F1={metrics['F1']:.4f}, "
                f"Thr={metrics['Threshold']:.4f}, "
                f"Hits@1={metrics['Hits@1']:.4f}, "
                f"Hits@5={metrics['Hits@5']:.4f}, "
                f"Hits@10={metrics['Hits@10']:.4f}, "
                f"MRR={metrics['MRR']:.4f}"
            )
        else:
            print(f"Finished {run.name}: metrics not found, check log.")
        results.append(result)
        write_summary(summary_path, results)

    print(f"\nSummary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
