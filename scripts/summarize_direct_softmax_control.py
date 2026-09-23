from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VARIANT_ROOT = (
    ROOT
    / "outputs"
    / "revision_all_neighbor_softmax_20260831"
    / "all_neighbor_softmax"
)
LOG_ROOT = VARIANT_ROOT / "logs"
OUTPUT_ROOT = VARIANT_ROOT / "summary_direct"

SELECTED_RE = re.compile(
    r"Selected CSLS parameters on validation \| k: (?P<k>\d+)"
)
METRIC_RE = re.compile(
    r"Primary CSLS ranking metrics \| Hits@1: (?P<h1>[0-9.]+) \| "
    r"Hits@5: (?P<h5>[0-9.]+) \| Hits@10: (?P<h10>[0-9.]+) "
    r"\| MRR: (?P<mrr>[0-9.]+)"
)


def parse_log(path: Path) -> dict[str, str]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    selected = list(SELECTED_RE.finditer(content))
    metrics = list(METRIC_RE.finditer(content))
    if not selected or not metrics:
        raise RuntimeError(f"Incomplete CSLS output: {path}")
    match = re.match(
        r"(?P<dataset>.+)_all_neighbor_softmax_seed(?P<seed>\d+)\.log$",
        path.name,
    )
    if match is None:
        raise RuntimeError(f"Unexpected log name: {path.name}")
    metric_values = metrics[-1].groupdict()
    return {
        "dataset": match.group("dataset"),
        "seed": match.group("seed"),
        "csls_k": selected[-1].group("k"),
        "Hits@1": metric_values["h1"],
        "Hits@5": metric_values["h5"],
        "Hits@10": metric_values["h10"],
        "MRR": metric_values["mrr"],
    }


def write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = [parse_log(path) for path in sorted(LOG_ROOT.glob("*.log"))]
    expected = {
        (dataset, str(seed))
        for dataset in ("dbp15k_zh_en", "openea_en_fr")
        for seed in (42, 43, 44)
    }
    observed = {(row["dataset"], row["seed"]) for row in rows}
    if observed != expected:
        raise RuntimeError(f"Expected {sorted(expected)}, found {sorted(observed)}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "seed", "csls_k", "Hits@1", "Hits@5", "Hits@10", "MRR"]
    write_tsv(OUTPUT_ROOT / "test_runs.tsv", rows, fields)

    aggregate_rows: list[dict[str, str]] = []
    for dataset in ("dbp15k_zh_en", "openea_en_fr"):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        aggregate: dict[str, str] = {
            "dataset": dataset,
            "runs": str(len(dataset_rows)),
            "csls_k_by_seed": ",".join(row["csls_k"] for row in dataset_rows),
        }
        for metric in ("Hits@1", "Hits@5", "Hits@10", "MRR"):
            values = [float(row[metric]) for row in dataset_rows]
            aggregate[f"{metric}_mean"] = str(statistics.mean(values))
            aggregate[f"{metric}_std"] = str(statistics.stdev(values))
        aggregate_rows.append(aggregate)
    aggregate_fields = [
        "dataset",
        "runs",
        "csls_k_by_seed",
        "Hits@1_mean",
        "Hits@1_std",
        "Hits@5_mean",
        "Hits@5_std",
        "Hits@10_mean",
        "Hits@10_std",
        "MRR_mean",
        "MRR_std",
    ]
    write_tsv(OUTPUT_ROOT / "test_aggregate.tsv", aggregate_rows, aggregate_fields)
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()
