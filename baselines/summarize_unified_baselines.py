import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHOD_ORDER = ("mtranse", "jape", "bootea", "rdgcn", "rrea_basic")
DATASET_ORDER = ("dbp15k_zh_en", "openea_en_fr")
METRICS = ("Hits@1", "Hits@5", "Hits@10", "MRR")


def format_metric(mean, std, count):
    if count == 1:
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {std:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", default="outputs/unified_existing_baselines_20260826"
    )
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    input_dir = ROOT / args.input_dir
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    records = []
    missing = []
    for method in METHOD_ORDER:
        for dataset in DATASET_ORDER:
            for seed in seeds:
                path = input_dir / method / f"{dataset}_seed{seed}.json"
                if not path.exists():
                    missing.append(str(path.relative_to(ROOT)))
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                records.append(record)

    if missing and not args.allow_incomplete:
        raise SystemExit("Missing result files:\n" + "\n".join(missing))
    if not records:
        raise SystemExit("No result files found.")

    rows = []
    for method in METHOD_ORDER:
        for dataset in DATASET_ORDER:
            group = [
                record
                for record in records
                if record["method"] == method and record["dataset"] == dataset
            ]
            if not group:
                continue
            row = {
                "method": method,
                "method_variant": group[0].get("method_variant", method),
                "dataset": dataset,
                "completed_seeds": len(group),
                "seeds": ",".join(str(record["seed"]) for record in group),
            }
            for metric in METRICS:
                values = np.asarray(
                    [record["test"][metric] for record in group], dtype=np.float64
                )
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=0))
            k_counts = Counter(int(record["selected_csls_k"]) for record in group)
            row["selected_csls_k"] = ", ".join(
                f"{k} ({count})" for k, count in sorted(k_counts.items())
            )
            row["elapsed_seconds_mean"] = float(
                np.mean([record["elapsed_seconds"] for record in group])
            )
            rows.append(row)

    input_dir.mkdir(parents=True, exist_ok=True)
    csv_path = input_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "| Dataset | Method | Seeds | Hits@1 | Hits@10 | MRR | Selected CSLS k |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        markdown.append(
            "| {dataset} | {method} | {completed_seeds} | {hits1} | {hits10} | "
            "{mrr} | {csls} |".format(
                dataset=row["dataset"],
                method=row["method_variant"],
                completed_seeds=row["completed_seeds"],
                hits1=format_metric(
                    row["Hits@1_mean"], row["Hits@1_std"], row["completed_seeds"]
                ),
                hits10=format_metric(
                    row["Hits@10_mean"], row["Hits@10_std"], row["completed_seeds"]
                ),
                mrr=format_metric(
                    row["MRR_mean"], row["MRR_std"], row["completed_seeds"]
                ),
                csls=row["selected_csls_k"],
            )
        )
    if missing:
        markdown.extend(["", "Incomplete runs:", *[f"- `{path}`" for path in missing]])
    markdown_path = input_dir / "summary.md"
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(csv_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
