import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from evaluate import (
    _compute_similarity_statistics,
    encode_entity_outputs,
)
from graph_utils import build_adj_list, build_topology_features
from train import (
    Config,
    build_candidate_right_ids,
    build_model,
    build_protocol_splits,
    load_dataset,
)


DATASETS = {
    "dbp15k_zh_en",
    "dbp15k_ja_en",
    "dbp15k_fr_en",
    "openea_en_fr",
    "openea_en_de",
    "openea_d_w",
    "openea_d_y",
    "eventea_en_en",
}


def parse_candidates(value: str, cast) -> List:
    candidates = [cast(token.strip()) for token in value.split(",") if token.strip()]
    if not candidates:
        raise ValueError("Candidate list must not be empty")
    return candidates


def write_tsv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def load_config(checkpoint: Dict[str, Any]) -> Config:
    cfg = Config()
    for key, value in checkpoint["config"].items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.device = "cpu"
    return cfg


def find_checkpoint(checkpoint_root: Path, dataset: str, seed: int) -> Path:
    base_dir = checkpoint_root
    run_dirs = sorted(base_dir.glob(f"{dataset}_*seed{seed}"))
    if len(run_dirs) != 1:
        raise FileNotFoundError(
            f"Expected one seed-{seed} run directory for {dataset} in {base_dir}"
        )
    run_dir = run_dirs[0]
    model_paths = sorted(run_dir.glob("best_model_*.pt"))
    if len(model_paths) != 1:
        raise FileNotFoundError(f"Expected one model checkpoint in {run_dir}")
    return model_paths[0]


def prepare_dataset(cfg: Config) -> Dict[str, Any]:
    data = load_dataset(cfg)
    adj_list = build_adj_list(data["edge_index"].cpu(), data["total_nodes"])
    return {"data": data, "adj_list": adj_list}


def prepare_pairs(cfg: Config, data: Dict[str, Any]) -> Dict[str, Any]:
    splits = build_protocol_splits(
        cfg,
        list(data["train_pairs"]),
        list(data["test_pairs"]),
        explicit_val_pairs=list(data.get("val_pairs", [])),
    )
    right_ids = build_candidate_right_ids(data["n1"], data["n2"])
    right_index = {int(node_id): idx for idx, node_id in enumerate(right_ids.tolist())}

    def split_tensors(pairs: Sequence[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        left_ids = torch.tensor([left for left, _ in pairs], dtype=torch.long)
        gt_index = torch.tensor([right_index[right] for _, right in pairs], dtype=torch.long)
        return left_ids, gt_index

    val_left_ids, val_gt = split_tensors(splits["val_pairs"])
    test_left_ids, test_gt = split_tensors(splits["test_pairs"])
    return {
        "right_ids": right_ids,
        "val_left_ids": val_left_ids,
        "val_gt": val_gt,
        "test_left_ids": test_left_ids,
        "test_gt": test_gt,
    }


def load_model(
    cfg: Config,
    checkpoint: Dict[str, Any],
    data: Dict[str, Any],
):
    model = build_model(
        cfg,
        total_nodes=data["total_nodes"],
        num_relations=int(data.get("num_relations", 0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    if cfg.structure_initialization == "topology":
        topology_features = build_topology_features(
            edge_index=data["edge_index"],
            edge_type=data["edge_type"],
            num_nodes=data["total_nodes"],
            graph_sizes=(data["n1"], data["n2"]),
        )
        model.set_structure_topology_features(topology_features)
    model.eval()
    return model


def set_effective_depth(model, depth: int) -> None:
    encoder = model.gnn_encoder
    if encoder is None:
        raise ValueError("Effective-depth selection requires a graph encoder")
    trained_depth = len(encoder.layer_dims)
    if depth < 1 or depth > trained_depth:
        raise ValueError(f"Depth {depth} is outside the trained range [1, {trained_depth}]")
    encoder.num_layers = depth


def encode_split(
    model,
    data: Dict[str, Any],
    adj_list,
    node_ids: torch.Tensor,
    cfg: Config,
    z_struct_all: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    return encode_entity_outputs(
        model=model,
        node_ids=node_ids,
        edge_index=data["edge_index"],
        edge_type=data.get("edge_type"),
        seq_features=data["seq_features"],
        adj_list=adj_list,
        num_neighbors=cfg.num_neighbors,
        batch_size=cfg.eval_batch_size,
        device=torch.device("cpu"),
        z_struct_all=z_struct_all,
    )


def fused_embeddings(outputs: Dict[str, torch.Tensor], semantic_weight: float) -> torch.Tensor:
    if not 0.0 <= semantic_weight <= 1.0:
        raise ValueError("semantic_weight must be within [0, 1]")
    semantic = F.normalize(outputs["z_sem_enhanced"], p=2, dim=-1)
    structural = F.normalize(outputs["z_struct_enhanced"], p=2, dim=-1)
    return F.normalize(
        semantic_weight * semantic + (1.0 - semantic_weight) * structural,
        p=2,
        dim=-1,
    )


def retrieval_embeddings(
    outputs: Dict[str, torch.Tensor],
    semantic_weight: float,
    representation: str,
) -> torch.Tensor:
    if representation == "joint":
        return F.normalize(outputs["z_joint"], p=2, dim=-1)
    if representation == "branch_weighted":
        return fused_embeddings(outputs, semantic_weight)
    raise ValueError(f"Unknown retrieval representation: {representation!r}")


def ranking_metrics(similarity: torch.Tensor, gt_index: torch.Tensor) -> Dict[str, float]:
    gt_score = similarity[torch.arange(similarity.size(0)), gt_index].unsqueeze(1)
    ranks = (similarity > gt_score).sum(dim=1) + 1
    return {
        "Hits@1": float((ranks <= 1).float().mean().item()),
        "Hits@5": float((ranks <= 5).float().mean().item()),
        "Hits@10": float((ranks <= 10).float().mean().item()),
        "MRR": float((1.0 / ranks.float()).mean().item()),
    }


@torch.no_grad()
def evaluate_validation_grid(
    left_outputs: Dict[str, torch.Tensor],
    right_outputs: Dict[str, torch.Tensor],
    gt_index: torch.Tensor,
    semantic_weights: Sequence[float],
    csls_ks: Sequence[int],
    csls_blends: Sequence[float],
    retrieval_representation: str = "joint",
) -> List[Dict[str, float]]:
    max_k = min(max(csls_ks), left_outputs["z_joint"].size(0), right_outputs["z_joint"].size(0))
    rows = []

    effective_semantic_weights = semantic_weights if retrieval_representation == "branch_weighted" else [0.0]
    for semantic_weight in effective_semantic_weights:
        left_emb = retrieval_embeddings(left_outputs, semantic_weight, retrieval_representation)
        right_emb = retrieval_embeddings(right_outputs, semantic_weight, retrieval_representation)
        cosine = torch.matmul(left_emb, right_emb.t())
        row_top = cosine.topk(k=max_k, dim=1).values.cumsum(dim=1)
        col_top = cosine.topk(k=max_k, dim=0).values.cumsum(dim=0)

        for csls_k in csls_ks:
            effective_k = min(csls_k, max_k)
            row_correction = row_top[:, effective_k - 1] / float(effective_k)
            col_correction = col_top[effective_k - 1] / float(effective_k)
            csls_delta = cosine - row_correction.unsqueeze(1) - col_correction.unsqueeze(0)

            for csls_blend in csls_blends:
                similarity = cosine + float(csls_blend) * csls_delta
                rows.append({
                    "retrieval_representation": retrieval_representation,
                    "semantic_weight": float(semantic_weight),
                    "csls_k": int(csls_k),
                    "csls_blend": float(csls_blend),
                    **ranking_metrics(similarity, gt_index),
                })
    return rows


@torch.no_grad()
def evaluate_test_configuration(
    left_outputs: Dict[str, torch.Tensor],
    right_outputs: Dict[str, torch.Tensor],
    gt_index: torch.Tensor,
    selected: Dict[str, Any],
    cfg: Config,
) -> Dict[str, float]:
    representation = str(selected.get("retrieval_representation", "joint"))
    left_emb = retrieval_embeddings(
        left_outputs,
        float(selected["semantic_weight"]),
        representation,
    )
    right_emb = retrieval_embeddings(
        right_outputs,
        float(selected["semantic_weight"]),
        representation,
    )
    metrics, _, _ = _compute_similarity_statistics(
        left_emb=left_emb,
        right_emb=right_emb,
        gt_index=gt_index,
        batch_size=cfg.eval_batch_size,
        csls_k=int(selected["csls_k"]),
        csls_blend=float(selected["csls_blend"]),
    )
    return {key: float(value) for key, value in metrics.items()}


def aggregate_validation(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["retrieval_representation"],
            row["depth"],
            row["semantic_weight"],
            row["csls_k"],
            row["csls_blend"],
        )
        grouped[key].append(row)

    aggregates = []
    for key, group in grouped.items():
        dataset, retrieval_representation, depth, semantic_weight, csls_k, csls_blend = key
        aggregate = {
            "dataset": dataset,
            "retrieval_representation": retrieval_representation,
            "depth": depth,
            "semantic_weight": semantic_weight,
            "csls_k": csls_k,
            "csls_blend": csls_blend,
            "runs": len(group),
        }
        for metric in ("Hits@1", "Hits@5", "Hits@10", "MRR"):
            values = [float(row[metric]) for row in group]
            aggregate[f"val_{metric}_mean"] = statistics.mean(values)
            aggregate[f"val_{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregates.append(aggregate)
    return aggregates


def select_dataset_configs(aggregates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected = []
    datasets = sorted({row["dataset"] for row in aggregates})
    for dataset in datasets:
        candidates = [row for row in aggregates if row["dataset"] == dataset]
        best = max(
            candidates,
            key=lambda row: (
                row["val_MRR_mean"],
                row["val_Hits@1_mean"],
                row["val_Hits@10_mean"],
                -row["depth"],
                -row["csls_k"],
                -abs(row["semantic_weight"] - 0.5),
            ),
        )
        selected.append(dict(best))
    return selected


def aggregate_test(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    for dataset in sorted({row["dataset"] for row in rows}):
        group = [row for row in rows if row["dataset"] == dataset]
        first = group[0]
        summary = {
            "dataset": dataset,
            "retrieval_representation": first["retrieval_representation"],
            "depth": first["depth"],
            "semantic_weight": first["semantic_weight"],
            "csls_k": first["csls_k"],
            "csls_blend": first["csls_blend"],
            "runs": len(group),
        }
        for metric in ("Hits@1", "Hits@5", "Hits@10", "MRR"):
            values = [float(row[metric]) for row in group]
            summary[f"test_{metric}_mean"] = statistics.mean(values)
            summary[f"test_{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(path: Path, selected: Sequence[Dict[str, Any]], test_summary: Sequence[Dict[str, Any]]) -> None:
    selected_map = {row["dataset"]: row for row in selected}
    lines = [
        "# 数据集独立超参数选择结果",
        "",
        "选择准则：三个随机种子的验证集平均 MRR；测试集不参与超参数选择。",
        "主协议直接检索训练所用的联合表示 z_joint；仅 branch_weighted 兼容模式选择融合权重 α。",
        "传播层数表示三层检查点中启用的最大有效传播深度。",
        "",
        "| 数据集 | 检索表示 | α | 传播层数 | CSLS k | CSLS混合 | 验证MRR | 测试Hits@1 | 测试Hits@10 | 测试MRR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in test_summary:
        val = selected_map[row["dataset"]]
        alpha_display = (
            "—"
            if row["retrieval_representation"] == "joint"
            else f"{row['semantic_weight']:.2f}"
        )
        lines.append(
            f"| {row['dataset']} | {row['retrieval_representation']} | "
            f"{alpha_display} | {row['depth']} | "
            f"{row['csls_k']} | {row['csls_blend']:.2f} | "
            f"{val['val_MRR_mean']:.4f} | "
            f"{row['test_Hits@1_mean']:.4f}±{row['test_Hits@1_std']:.4f} | "
            f"{row['test_Hits@10_mean']:.4f}±{row['test_Hits@10_std']:.4f} | "
            f"{row['test_MRR_mean']:.4f}±{row['test_MRR_std']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(sorted(DATASETS)))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--depths", default="1,2,3")
    parser.add_argument("--semantic-weights", default="0.20,0.30,0.40,0.50,0.60,0.70,0.80")
    parser.add_argument(
        "--retrieval-representation",
        choices=("joint", "branch_weighted"),
        default="joint",
        help="Use the trained joint vector by default; branch_weighted reproduces the legacy post-hoc fusion grid.",
    )
    parser.add_argument("--csls-k", default="3,5,7,10,15,20")
    parser.add_argument("--csls-blends", default="1.00")
    parser.add_argument(
        "--checkpoint-root",
        default="outputs/eight_dataset_retrained_20260819/checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/eight_dataset_retrained_20260819/dataset_specific_selection",
    )
    args = parser.parse_args()

    dataset_names = parse_candidates(args.datasets, str)
    unknown = sorted(set(dataset_names) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    seeds = parse_candidates(args.seeds, int)
    depths = sorted(set(parse_candidates(args.depths, int)))
    semantic_weights = sorted(set(parse_candidates(args.semantic_weights, float)))
    csls_ks = sorted(set(parse_candidates(args.csls_k, int)))
    csls_blends = sorted(set(parse_candidates(args.csls_blends, float)))

    root = Path(__file__).resolve().parent
    checkpoint_root = root / args.checkpoint_root
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_rows = []

    for dataset_name in dataset_names:
        prepared = None
        for seed in seeds:
            model_path = find_checkpoint(checkpoint_root, dataset_name, seed)
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            cfg = load_config(checkpoint)
            if prepared is None:
                prepared = prepare_dataset(cfg)
            data = prepared["data"]
            pair_data = prepare_pairs(cfg, data)
            model = load_model(cfg, checkpoint, data)

            for depth in depths:
                set_effective_depth(model, depth)
                z_struct_all = model.encode_structure_all(
                    edge_index=data["edge_index"],
                    edge_type=data.get("edge_type"),
                )
                left_outputs = encode_split(
                    model, data, prepared["adj_list"], pair_data["val_left_ids"], cfg, z_struct_all
                )
                right_outputs = encode_split(
                    model, data, prepared["adj_list"], pair_data["right_ids"], cfg, z_struct_all
                )
                grid = evaluate_validation_grid(
                    left_outputs=left_outputs,
                    right_outputs=right_outputs,
                    gt_index=pair_data["val_gt"],
                    semantic_weights=semantic_weights,
                    csls_ks=csls_ks,
                    csls_blends=csls_blends,
                    retrieval_representation=args.retrieval_representation,
                )
                validation_rows.extend({
                    "dataset": dataset_name,
                    "seed": seed,
                    "depth": depth,
                    **row,
                } for row in grid)
                best_depth_row = max(grid, key=lambda row: (row["MRR"], row["Hits@1"]))
                print(
                    f"VALIDATION {dataset_name} seed={seed} depth={depth} | "
                    f"repr={best_depth_row['retrieval_representation']} "
                    + (
                        f"alpha={best_depth_row['semantic_weight']:.2f} "
                        if best_depth_row["retrieval_representation"] == "branch_weighted"
                        else ""
                    )
                    +
                    f"k={best_depth_row['csls_k']} beta={best_depth_row['csls_blend']:.2f} | "
                    f"Hits@1={best_depth_row['Hits@1']:.4f} MRR={best_depth_row['MRR']:.4f}",
                    flush=True,
                )

    validation_aggregates = aggregate_validation(validation_rows)
    selected = select_dataset_configs(validation_aggregates)
    selected_map = {row["dataset"]: row for row in selected}
    write_tsv(output_dir / "validation_grid.tsv", validation_rows)
    write_tsv(output_dir / "validation_grid_aggregate.tsv", validation_aggregates)
    write_tsv(output_dir / "selected_config.tsv", selected)
    save_json(output_dir / "selected_config.json", selected)

    test_rows = []
    for dataset_name in dataset_names:
        chosen = selected_map[dataset_name]
        prepared = None
        for seed in seeds:
            model_path = find_checkpoint(checkpoint_root, dataset_name, seed)
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            cfg = load_config(checkpoint)
            if prepared is None:
                prepared = prepare_dataset(cfg)
            data = prepared["data"]
            pair_data = prepare_pairs(cfg, data)
            model = load_model(cfg, checkpoint, data)
            set_effective_depth(model, int(chosen["depth"]))
            z_struct_all = model.encode_structure_all(
                edge_index=data["edge_index"],
                edge_type=data.get("edge_type"),
            )
            left_outputs = encode_split(
                model, data, prepared["adj_list"], pair_data["test_left_ids"], cfg, z_struct_all
            )
            right_outputs = encode_split(
                model, data, prepared["adj_list"], pair_data["right_ids"], cfg, z_struct_all
            )
            metrics = evaluate_test_configuration(
                left_outputs=left_outputs,
                right_outputs=right_outputs,
                gt_index=pair_data["test_gt"],
                selected=chosen,
                cfg=cfg,
            )
            row = {
                "dataset": dataset_name,
                "seed": seed,
                "retrieval_representation": chosen["retrieval_representation"],
                "depth": int(chosen["depth"]),
                "semantic_weight": float(chosen["semantic_weight"]),
                "csls_k": int(chosen["csls_k"]),
                "csls_blend": float(chosen["csls_blend"]),
                **metrics,
            }
            test_rows.append(row)
            print(
                f"TEST {dataset_name} seed={seed} | "
                f"Hits@1={metrics['Hits@1']:.4f} Hits@10={metrics['Hits@10']:.4f} "
                f"MRR={metrics['MRR']:.4f}",
                flush=True,
            )

    test_summary = aggregate_test(test_rows)
    write_tsv(output_dir / "test_runs.tsv", test_rows)
    write_tsv(output_dir / "test_aggregate.tsv", test_summary)
    save_json(output_dir / "test_runs.json", test_rows)
    save_json(output_dir / "test_aggregate.json", test_summary)
    write_report(output_dir / "report_zh.md", selected, test_summary)
    print(f"WROTE {output_dir}", flush=True)


if __name__ == "__main__":
    main()
