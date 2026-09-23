# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ("Hits@1", "Hits@10", "MRR")


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def discover_runs(roots: list[Path]) -> dict[str, list[dict]]:
    variants: dict[str, list[dict]] = {}
    for root in roots:
        for path in sorted(root.glob("**/selection/test_runs.json")):
            variant = path.parent.parent.name
            if variant in variants:
                raise RuntimeError(f"Duplicate completed variant {variant}: {path}")
            variants[variant] = json.loads(path.read_text(encoding="utf-8"))
    return variants


def aggregate(variants: dict[str, list[dict]]) -> list[dict]:
    rows: list[dict] = []
    for variant, runs in sorted(variants.items()):
        by_dataset: dict[str, list[dict]] = defaultdict(list)
        for run in runs:
            by_dataset[run["dataset"]].append(run)
        for dataset, dataset_runs in sorted(by_dataset.items()):
            row = {
                "variant": variant,
                "dataset": dataset,
                "runs": len(dataset_runs),
                "depth": dataset_runs[0]["depth"],
                "semantic_weight": dataset_runs[0]["semantic_weight"],
                "csls_k": dataset_runs[0]["csls_k"],
            }
            for metric in METRICS:
                values = [float(run[metric]) for run in dataset_runs]
                row[f"{metric}_mean"] = mean(values)
                row[f"{metric}_std"] = sample_std(values)
            rows.append(row)
    return rows


def paired_deltas(variants: dict[str, list[dict]]) -> list[dict]:
    if "main" not in variants:
        raise RuntimeError("The completed main variant is required for paired deltas")
    main = {
        (run["dataset"], int(run["seed"])): run
        for run in variants["main"]
    }
    rows: list[dict] = []
    for variant, runs in sorted(variants.items()):
        if variant == "main":
            continue
        by_dataset: dict[str, list[dict]] = defaultdict(list)
        for run in runs:
            by_dataset[run["dataset"]].append(run)
        for dataset, dataset_runs in sorted(by_dataset.items()):
            row = {"variant": variant, "dataset": dataset, "pairs": 0}
            deltas: dict[str, list[float]] = defaultdict(list)
            for run in dataset_runs:
                reference = main.get((dataset, int(run["seed"])))
                if reference is None:
                    continue
                row["pairs"] += 1
                for metric in METRICS:
                    deltas[metric].append(float(run[metric]) - float(reference[metric]))
            for metric in METRICS:
                row[f"delta_{metric}_mean"] = mean(deltas[metric]) if deltas[metric] else None
                row[f"delta_{metric}_std"] = sample_std(deltas[metric]) if deltas[metric] else None
            rows.append(row)
    return rows


def write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-variants", default="")
    args = parser.parse_args()

    variants = discover_runs(args.roots)
    expected = {item for item in args.expected_variants.split(",") if item}
    missing = sorted(expected - variants.keys())
    summary = aggregate(variants)
    deltas = paired_deltas(variants)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed_variants": sorted(variants),
        "missing_variants": missing,
        "summary": summary,
        "paired_deltas_vs_main": deltas,
    }
    (args.output_dir / "ablation_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_tsv(args.output_dir / "ablation_summary.tsv", summary)
    write_tsv(args.output_dir / "paired_deltas_vs_main.tsv", deltas)
    print(f"Completed variants: {len(variants)}")
    print(f"Missing variants: {', '.join(missing) if missing else 'none'}")
    print(f"WROTE {args.output_dir}")


if __name__ == "__main__":
    main()
