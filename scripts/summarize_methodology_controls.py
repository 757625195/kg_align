import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
CSLS_RE = re.compile(
    r"(?:Primary|Supplementary) CSLS ranking metrics \| Hits@1: (?P<h1>\d+\.\d+) \| "
    r"Hits@5: (?P<h5>\d+\.\d+) \| Hits@10: (?P<h10>\d+\.\d+) \| "
    r"MRR: (?P<mrr>\d+\.\d+)"
)
SELECTED_K_RE = re.compile(r"Selected CSLS parameters on validation \| k: (?P<k>\d+)")
SEED_RE = re.compile(r"_seed(?P<seed>\d+)\.log$")


EXISTING_LOG_ROOTS = {
    "main": ROOT / "outputs/new_main_full_ablation_20260825/main/logs",
    "learned_structure_init": (
        ROOT / "outputs/new_main_ablation_parallel_20260825/learned_structure_init/logs"
    ),
    "fixed_k8_softmax": (
        ROOT / "outputs/new_main_ablation_parallel_20260825/fixed_k8_softmax/logs"
    ),
    "no_relation_types": (
        ROOT / "outputs/new_main_ablation_parallel_20260825/no_relation_types/logs"
    ),
    "no_structure_supervision": (
        ROOT / "outputs/new_main_ablation_parallel_20260825/no_structure_supervision/logs"
    ),
    "no_layer_selector": (
        ROOT / "outputs/new_main_ablation_primary_20260825/no_layer_selector/logs"
    ),
    "no_token_view": ROOT / "outputs/new_main_ablation_primary_20260825/no_token_view/logs",
    "no_phrase_view": ROOT / "outputs/new_main_ablation_primary_20260825/no_phrase_view/logs",
    "no_global_view": ROOT / "outputs/new_main_ablation_primary_20260825/no_global_view/logs",
    "structure_only": ROOT / "outputs/new_main_ablation_primary_20260825/structure_only/logs",
    "semantic_only": ROOT / "outputs/new_main_ablation_primary_20260825/semantic_only/logs",
    "structural_neighbor_query": (
        ROOT / "outputs/new_main_ablation_primary_20260825/structural_neighbor_query/logs"
    ),
    "mean_fusion": ROOT / "outputs/new_main_ablation_parallel_20260825/mean_fusion/logs",
    "late_concat_matched": (
        ROOT / "outputs/new_main_ablation_parallel_20260825/late_concat_matched/logs"
    ),
}


def dataset_from_name(name: str) -> str:
    for dataset in (
        "dbp15k_zh_en",
        "dbp15k_ja_en",
        "dbp15k_fr_en",
        "openea_en_fr",
        "eventea_en_en",
    ):
        if name.startswith(dataset + "_"):
            return dataset
    raise ValueError(f"Cannot identify dataset from {name}")


def parse_log(path: Path, variant: str) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    metrics = CSLS_RE.search(text)
    selected_k = SELECTED_K_RE.search(text)
    seed = SEED_RE.search(path.name)
    if metrics is None or selected_k is None or seed is None:
        raise ValueError(f"Incomplete direct-joint evaluation log: {path}")
    return {
        "variant": variant,
        "dataset": dataset_from_name(path.name),
        "seed": int(seed.group("seed")),
        "csls_k": int(selected_k.group("k")),
        "Hits@1": float(metrics.group("h1")),
        "Hits@5": float(metrics.group("h5")),
        "Hits@10": float(metrics.group("h10")),
        "MRR": float(metrics.group("mrr")),
    }


def collect_logs(log_root: Path, variant: str) -> List[Dict[str, object]]:
    return [parse_log(path, variant) for path in sorted(log_root.glob("*.log"))]


def aggregate(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((row["variant"], row["dataset"]), []).append(row)

    summary = []
    for (variant, dataset), group in sorted(grouped.items()):
        result: Dict[str, object] = {
            "variant": variant,
            "dataset": dataset,
            "runs": len(group),
            "selected_csls_k": ",".join(str(row["csls_k"]) for row in group),
        }
        for metric in ("Hits@1", "Hits@5", "Hits@10", "MRR"):
            values = [float(row[metric]) for row in group]
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(result)
    return summary


def write_tsv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--new-control-root",
        default="outputs/methodology_controls_20260829",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/methodology_controls_20260829/summary",
    )
    args = parser.parse_args()

    rows: List[Dict[str, object]] = []
    for variant, log_root in EXISTING_LOG_ROOTS.items():
        rows.extend(collect_logs(log_root, variant))

    new_root = ROOT / args.new_control_root
    for variant in ("disjoint_relation_vocab", "bidirectional_same_relation"):
        log_root = new_root / variant / "logs"
        if log_root.exists():
            rows.extend(collect_logs(log_root, variant))

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate(rows)
    write_tsv(output_dir / "direct_joint_runs.tsv", rows)
    write_tsv(output_dir / "direct_joint_aggregate.tsv", summary)
    (output_dir / "direct_joint_runs.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "direct_joint_aggregate.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"WROTE {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
