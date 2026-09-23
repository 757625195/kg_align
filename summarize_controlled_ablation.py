import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs" / "controlled_ablation_20260820"
BASE_ROOT = ROOT / "outputs" / "eight_dataset_retrained_20260819" / "dataset_specific_selection"
DATASETS = ("dbp15k_zh_en", "openea_en_fr")
VARIANTS = {
    "main": "完整模型",
    "no_relation_types": "去除关系类型",
    "no_layer_selector": "去除层选择器",
    "no_token_view": "去除 token 视图",
    "no_phrase_view": "去除 phrase 视图",
    "no_global_view": "去除 global 视图",
    "late_concat_matched": "参数匹配晚期融合",
    "structural_neighbor_query": "无语义查询邻居",
    "token_only": "仅 token 视图",
    "structure_only": "仅结构分支",
    "semantic_only": "仅语义分支",
    "mean_fusion": "无可训练融合（加权平均）",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def index_rows(rows: List[Dict[str, Any]], *keys: str) -> Dict[tuple, Dict[str, Any]]:
    return {tuple(row[key] for key in keys): row for row in rows}


def write_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    base_aggregate = index_rows(load_json(BASE_ROOT / "test_aggregate.json"), "dataset")
    base_runs = index_rows(load_json(BASE_ROOT / "test_runs.json"), "dataset", "seed")
    base_config = index_rows(load_json(BASE_ROOT / "selected_config.json"), "dataset")

    aggregate_by_variant = {"main": base_aggregate}
    runs_by_variant = {"main": base_runs}
    config_by_variant = {"main": base_config}
    for variant in VARIANTS:
        if variant == "main":
            continue
        selection = OUTPUT_ROOT / variant / "selection"
        aggregate_by_variant[variant] = index_rows(
            load_json(selection / "test_aggregate.json"), "dataset"
        )
        runs_by_variant[variant] = index_rows(
            load_json(selection / "test_runs.json"), "dataset", "seed"
        )
        config_by_variant[variant] = index_rows(
            load_json(selection / "selected_config.json"), "dataset"
        )

    rows: List[Dict[str, Any]] = []
    paired_rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        base = aggregate_by_variant["main"][(dataset,)]
        for variant, label in VARIANTS.items():
            aggregate = aggregate_by_variant[variant][(dataset,)]
            config = config_by_variant[variant][(dataset,)]
            row = {
                "dataset": dataset,
                "variant": variant,
                "label_zh": label,
                "semantic_weight": float(config["semantic_weight"]),
                "depth": int(config["depth"]),
                "csls_k": int(config["csls_k"]),
                "Hits@1_mean": float(aggregate["test_Hits@1_mean"]),
                "Hits@1_std": float(aggregate["test_Hits@1_std"]),
                "delta_Hits@1": float(aggregate["test_Hits@1_mean"])
                - float(base["test_Hits@1_mean"]),
                "Hits@10_mean": float(aggregate["test_Hits@10_mean"]),
                "Hits@10_std": float(aggregate["test_Hits@10_std"]),
                "delta_Hits@10": float(aggregate["test_Hits@10_mean"])
                - float(base["test_Hits@10_mean"]),
                "MRR_mean": float(aggregate["test_MRR_mean"]),
                "MRR_std": float(aggregate["test_MRR_std"]),
                "delta_MRR": float(aggregate["test_MRR_mean"])
                - float(base["test_MRR_mean"]),
            }
            rows.append(row)

            if variant == "main":
                continue
            seed_h1 = []
            seed_mrr = []
            for seed in (42, 43, 44):
                variant_run = runs_by_variant[variant][(dataset, seed)]
                base_run = base_runs[(dataset, seed)]
                h1_delta = float(variant_run["Hits@1"]) - float(base_run["Hits@1"])
                mrr_delta = float(variant_run["MRR"]) - float(base_run["MRR"])
                seed_h1.append(h1_delta)
                seed_mrr.append(mrr_delta)
                paired_rows.append({
                    "dataset": dataset,
                    "variant": variant,
                    "seed": seed,
                    "delta_Hits@1": h1_delta,
                    "delta_MRR": mrr_delta,
                })
            row["paired_delta_Hits@1_std"] = statistics.stdev(seed_h1)
            row["paired_delta_MRR_std"] = statistics.stdev(seed_mrr)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_tsv(OUTPUT_ROOT / "ablation_summary.tsv", rows)
    write_tsv(OUTPUT_ROOT / "paired_seed_deltas.tsv", paired_rows)
    (OUTPUT_ROOT / "ablation_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 统一协议多随机种子消融汇总",
        "",
        "所有变体均使用随机种子 42、43、44，并分别依据三个种子的平均验证 MRR 选择 α、L 和 CSLS k。",
        "Δ 为相对同数据集完整模型的测试集绝对变化。",
        "",
    ]
    for dataset in DATASETS:
        lines.extend([
            f"## {dataset}",
            "",
            "| 变体 | α/L/k | Hits@1 | ΔH@1 | Hits@10 | MRR | ΔMRR |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in [item for item in rows if item["dataset"] == dataset]:
            lines.append(
                f"| {row['label_zh']} | {row['semantic_weight']:.2f}/{row['depth']}/{row['csls_k']} | "
                f"{row['Hits@1_mean']:.4f}±{row['Hits@1_std']:.4f} | "
                f"{row['delta_Hits@1']:+.4f} | "
                f"{row['Hits@10_mean']:.4f}±{row['Hits@10_std']:.4f} | "
                f"{row['MRR_mean']:.4f}±{row['MRR_std']:.4f} | "
                f"{row['delta_MRR']:+.4f} |"
            )
        lines.append("")

    (OUTPUT_ROOT / "ablation_report_zh.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"WROTE {OUTPUT_ROOT / 'ablation_summary.json'}")
    print(f"WROTE {OUTPUT_ROOT / 'ablation_report_zh.md'}")


if __name__ == "__main__":
    main()
