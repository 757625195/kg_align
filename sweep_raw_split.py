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


def build_seeded_runs(name_prefix: str, seeds: List[int], env: Dict[str, str]) -> List[SweepRun]:
    return [
        SweepRun(
            name=f"{name_prefix}_seed{seed}",
            env={
                **env,
                "KG_ALIGN_SEED": str(seed),
            },
        )
        for seed in seeds
    ]


def build_minimal_sweep(seeds: List[int]) -> List[SweepRun]:
    return build_seeded_runs("baseline", seeds, {})


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
            item.get("F1", -1.0),
            item.get("Precision", -1.0),
            item.get("Recall", -1.0),
            item.get("Hits@1", -1.0),
            item.get("MRR", -1.0),
            item.get("Hits@10", -1.0),
        ),
        reverse=True,
    )
    summary_path.write_text(
        json.dumps(ranked, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

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
    parser = argparse.ArgumentParser(description="Run the locked raw-split baseline sweep.")
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the sweep configuration and exit.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/sweeps",
        help="Directory for logs and summaries.",
    )
    parser.add_argument(
        "--seeds",
        default="42,43,44",
        help="Comma-separated seed list. Use '--seeds 42' for single-seed tuning.",
    )
    args = parser.parse_args()

    seeds = parse_seed_list(args.seeds)
    runs = build_minimal_sweep(seeds)
    if args.list:
        payload = [{"name": run.name, "env": run.env} for run in runs]
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_tag = "single_seed" if len(seeds) == 1 else "multi_seed"
    summary_path = output_dir / f"raw_split_baseline_{seed_tag}_summary.json"

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
