import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


LABELS = {
    "main": "完整模型",
    "learned_structure_init": "可学习实体初始化",
    "no_structure_supervision": "去除结构辅助监督",
    "fixed_k8_softmax": "固定 8 邻居 + softmax",
    "no_relation_types": "去除关系类型",
    "no_layer_selector": "去除层选择器",
    "no_token_view": "去除 token 视图",
    "no_phrase_view": "去除 phrase 视图",
    "no_global_view": "去除 global 视图",
    "structure_only": "仅结构分支",
    "semantic_only": "仅语义分支",
    "mean_fusion": "无可训练融合",
    "late_concat_matched": "参数匹配晚期融合",
    "structural_neighbor_query": "结构查询邻居",
}


def parse_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_tsv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def index(rows, *keys: str) -> Dict[Tuple[object, ...], Dict[str, object]]:
    return {tuple(row[key] for key in keys): row for row in rows}


def formatted(row: Dict[str, object], metric: str) -> str:
    return f"{row[f'test_{metric}_mean']:.4f}+-{row[f'test_{metric}_std']:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", default="dbp15k_zh_en,openea_en_fr")
    parser.add_argument("--variants", default=",".join(LABELS))
    args = parser.parse_args()

    root = Path(__file__).resolve().parent / args.output_dir
    datasets = parse_list(args.datasets)
    variants = parse_list(args.variants)
    aggregate_by_variant = {}
    runs_by_variant = {}
    for variant in variants:
        selection = root / variant / "selection"
        aggregate_by_variant[variant] = index(
            load_json(selection / "test_aggregate.json"), "dataset"
        )
        runs_by_variant[variant] = index(
            load_json(selection / "test_runs.json"), "dataset", "seed"
        )

    rows: List[Dict[str, object]] = []
    paired_rows: List[Dict[str, object]] = []
    for dataset in datasets:
        base = aggregate_by_variant["main"][(dataset,)]
        for variant in variants:
            current = aggregate_by_variant[variant][(dataset,)]
            row: Dict[str, object] = {
                "dataset": dataset,
                "variant": variant,
                "label_zh": LABELS.get(variant, variant),
                "semantic_weight": current["semantic_weight"],
                "depth": current["depth"],
                "csls_k": current["csls_k"],
            }
            for metric in ("Hits@1", "Hits@5", "Hits@10", "MRR"):
                mean_key = f"test_{metric}_mean"
                std_key = f"test_{metric}_std"
                row[f"{metric}_mean"] = current[mean_key]
                row[f"{metric}_std"] = current[std_key]
                row[f"delta_{metric}"] = current[mean_key] - base[mean_key]
            rows.append(row)

            if variant == "main":
                continue
            for seed in (42, 43, 44):
                value = runs_by_variant[variant][(dataset, seed)]
                base_value = runs_by_variant["main"][(dataset, seed)]
                paired_rows.append({
                    "dataset": dataset,
                    "variant": variant,
                    "seed": seed,
                    "delta_Hits@1": value["Hits@1"] - base_value["Hits@1"],
                    "delta_MRR": value["MRR"] - base_value["MRR"],
                })

    for row in rows:
        if row["variant"] == "main":
            row["paired_delta_Hits@1_std"] = 0.0
            row["paired_delta_MRR_std"] = 0.0
            continue
        paired = [
            value for value in paired_rows
            if value["dataset"] == row["dataset"] and value["variant"] == row["variant"]
        ]
        row["paired_delta_Hits@1_std"] = statistics.stdev(
            value["delta_Hits@1"] for value in paired
        )
        row["paired_delta_MRR_std"] = statistics.stdev(
            value["delta_MRR"] for value in paired
        )

    write_tsv(root / "validated_ablation_summary.tsv", rows)
    write_tsv(root / "validated_ablation_paired_deltas.tsv", paired_rows)
    (root / "validated_ablation_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 新主协议验证集选择后的消融结果",
        "",
        "每个数据集和变体使用三个随机种子的平均验证 MRR 选择融合权重、传播深度和 CSLS k。",
        "",
    ]
    for dataset in datasets:
        lines.extend([
            f"## {dataset}",
            "",
            "| 变体 | alpha/L/k | Hits@1 | Delta H@1 | Hits@10 | MRR | Delta MRR |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in (value for value in rows if value["dataset"] == dataset):
            lines.append(
                f"| {row['label_zh']} | {float(row['semantic_weight']):.2f}/{row['depth']}/{row['csls_k']} | "
                f"{float(row['Hits@1_mean']):.4f}+-{float(row['Hits@1_std']):.4f} | "
                f"{float(row['delta_Hits@1']):+.4f} | "
                f"{float(row['Hits@10_mean']):.4f}+-{float(row['Hits@10_std']):.4f} | "
                f"{float(row['MRR_mean']):.4f}+-{float(row['MRR_std']):.4f} | "
                f"{float(row['delta_MRR']):+.4f} |"
            )
        lines.append("")
    (root / "validated_ablation_report_zh.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"WROTE {root / 'validated_ablation_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
