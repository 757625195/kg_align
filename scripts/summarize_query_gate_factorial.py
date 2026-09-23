from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "outputs" / "reviewer_query_gate_2x2_20260831"
VARIANTS = {
    "query_sem_gate_sem": ("semantic", "semantic", "SS"),
    "query_struct_gate_sem": ("structural", "semantic", "TS"),
    "query_sem_gate_struct": ("semantic", "structural", "ST"),
    "query_struct_gate_struct": ("structural", "structural", "TT"),
}
DATASETS = {
    "dbp15k_zh_en": "DBP15K ZH–EN",
    "openea_en_fr": "OpenEA EN–FR-15K-V2",
}
PRIMARY_RE = re.compile(
    r"Primary CSLS ranking metrics \| Hits@1: ([0-9.]+) \| Hits@5: ([0-9.]+) "
    r"\| Hits@10: ([0-9.]+) \| MRR: ([0-9.]+)"
)
K_RE = re.compile(r"Selected CSLS parameters on validation \| k: (\d+)")
NAME_RE = re.compile(r"^(dbp15k_zh_en|openea_en_fr)_(.+)_seed(\d+)\.log$")


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_runs() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for log_path in sorted(RESULT_ROOT.glob("*/logs/*.log")):
        match = NAME_RE.match(log_path.name)
        if match is None:
            continue
        dataset, variant, seed_text = match.groups()
        if variant not in VARIANTS:
            continue
        text = log_path.read_text(encoding="utf-8", errors="replace")
        metrics = list(PRIMARY_RE.finditer(text))
        selected = list(K_RE.finditer(text))
        if not metrics or not selected:
            continue
        query_source, gate_source, cell = VARIANTS[variant]
        h1, h5, h10, mrr = (float(value) for value in metrics[-1].groups())
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": DATASETS[dataset],
                "variant": variant,
                "cell": cell,
                "query_source": query_source,
                "gate_source": gate_source,
                "seed": int(seed_text),
                "csls_k": int(selected[-1].group(1)),
                "Hits@1": h1,
                "Hits@5": h5,
                "Hits@10": h10,
                "MRR": mrr,
                "log_path": str(log_path),
            }
        )
    return rows


def aggregate_runs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for dataset in DATASETS:
        for variant, (query_source, gate_source, cell) in VARIANTS.items():
            selected = [
                row for row in rows
                if row["dataset"] == dataset and row["variant"] == variant
            ]
            if not selected:
                continue
            record: dict[str, object] = {
                "dataset": dataset,
                "dataset_label": DATASETS[dataset],
                "variant": variant,
                "cell": cell,
                "query_source": query_source,
                "gate_source": gate_source,
                "n": len(selected),
            }
            for metric in ("Hits@1", "Hits@5", "Hits@10", "MRR"):
                mean, std = mean_std([float(row[metric]) for row in selected])
                record[f"{metric}_mean"] = mean
                record[f"{metric}_std"] = std
            output.append(record)
    return output


def paired_effects(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    effects = {
        "query_effect_sem_gate": ("SS", "TS"),
        "query_effect_struct_gate": ("ST", "TT"),
        "gate_effect_sem_query": ("SS", "ST"),
        "gate_effect_struct_query": ("TS", "TT"),
    }
    output: list[dict[str, object]] = []
    for dataset in DATASETS:
        by_key = {
            (str(row["cell"]), int(row["seed"])): row
            for row in rows if row["dataset"] == dataset
        }
        for metric in ("Hits@1", "MRR"):
            seed_differences: dict[str, list[float]] = {}
            for name, (positive, negative) in effects.items():
                differences = [
                    float(by_key[(positive, seed)][metric])
                    - float(by_key[(negative, seed)][metric])
                    for seed in (42, 43, 44)
                    if (positive, seed) in by_key and (negative, seed) in by_key
                ]
                seed_differences[name] = differences
                if differences:
                    mean, std = mean_std(differences)
                    output.append(
                        {
                            "dataset": dataset,
                            "dataset_label": DATASETS[dataset],
                            "metric": metric,
                            "effect": name,
                            "n": len(differences),
                            "mean": mean,
                            "std": std,
                            "points_mean": 100 * mean,
                            "points_std": 100 * std,
                        }
                    )
            if all(seed_differences.get(name) for name in effects):
                query_main = [
                    0.5 * (a + b)
                    for a, b in zip(
                        seed_differences["query_effect_sem_gate"],
                        seed_differences["query_effect_struct_gate"],
                    )
                ]
                gate_main = [
                    0.5 * (a + b)
                    for a, b in zip(
                        seed_differences["gate_effect_sem_query"],
                        seed_differences["gate_effect_struct_query"],
                    )
                ]
                interaction = [
                    a - b
                    for a, b in zip(
                        seed_differences["query_effect_sem_gate"],
                        seed_differences["query_effect_struct_gate"],
                    )
                ]
                for name, values in (
                    ("query_main_effect", query_main),
                    ("gate_main_effect", gate_main),
                    ("query_gate_interaction", interaction),
                ):
                    mean, std = mean_std(values)
                    output.append(
                        {
                            "dataset": dataset,
                            "dataset_label": DATASETS[dataset],
                            "metric": metric,
                            "effect": name,
                            "n": len(values),
                            "mean": mean,
                            "std": std,
                            "points_mean": 100 * mean,
                            "points_std": 100 * std,
                        }
                    )
    return output


def main() -> None:
    rows = load_runs()
    aggregate = aggregate_runs(rows)
    effects = paired_effects(rows)
    summary = RESULT_ROOT / "summary"
    write_tsv(summary / "runs.tsv", rows, list(rows[0]) if rows else [])
    write_tsv(summary / "aggregate.tsv", aggregate, list(aggregate[0]) if aggregate else [])
    write_tsv(summary / "paired_effects.tsv", effects, list(effects[0]) if effects else [])
    print(f"runs={len(rows)} aggregate={len(aggregate)} effects={len(effects)}")


if __name__ == "__main__":
    main()
