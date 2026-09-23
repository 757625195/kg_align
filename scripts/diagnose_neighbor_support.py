import argparse
import glob
import json
from pathlib import Path

import torch

from evaluate import encode_entity_outputs
from graph_utils import build_adj_list
from train import Config, build_model, load_dataset


def resolve_checkpoint(path_pattern: str) -> Path:
    matches = [Path(path) for path in glob.glob(path_pattern)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one checkpoint for {path_pattern!r}, found {len(matches)}"
        )
    return matches[0]


def summarize(checkpoint_path: Path, batch_size: int) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = Config(**checkpoint["config"])
    cfg.device = "cpu"

    data = load_dataset(cfg)
    edge_index = data["edge_index"].cpu()
    edge_type = data.get("edge_type")
    if edge_type is not None:
        edge_type = edge_type.cpu()

    model = build_model(
        cfg,
        total_nodes=data["total_nodes"],
        num_relations=int(data.get("num_relations", 0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    outputs = encode_entity_outputs(
        model=model,
        node_ids=torch.arange(data["total_nodes"], dtype=torch.long),
        edge_index=edge_index,
        edge_type=edge_type,
        seq_features=data["seq_features"],
        adj_list=build_adj_list(edge_index, data["total_nodes"]),
        num_neighbors=cfg.num_neighbors,
        batch_size=batch_size,
        device=torch.device("cpu"),
    )
    valid = outputs["neighbor_valid_count"].to(dtype=torch.float32)
    support = outputs["neighbor_support_size"].to(dtype=torch.float32)
    active = valid > 0
    active_valid = valid[active]
    active_support = support[active]
    retention = active_support / active_valid

    quantiles = torch.tensor([0.5, 0.9, 0.99], dtype=torch.float32)
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "dataset_family": cfg.dataset_family,
        "pair": cfg.pair,
        "neighbor_attention": cfg.neighbor_attention,
        "neighbor_attention_temperature": cfg.neighbor_attention_temperature,
        "entities": int(valid.numel()),
        "entities_with_neighbors": int(active.sum().item()),
        "valid_neighbors_mean": float(active_valid.mean().item()),
        "selected_neighbors_mean": float(active_support.mean().item()),
        "selected_to_valid_ratio": float(retention.mean().item()),
        "entities_pruning_any_neighbor_ratio": float(
            (active_support < active_valid).to(dtype=torch.float32).mean().item()
        ),
        "selected_neighbors_quantiles": {
            name: float(value)
            for name, value in zip(
                ("p50", "p90", "p99"),
                torch.quantile(active_support, quantiles).tolist(),
            )
        },
        "selected_neighbors_max": int(active_support.max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure non-zero neighbor support from trained checkpoints."
    )
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    reports = [
        summarize(resolve_checkpoint(pattern), args.batch_size)
        for pattern in args.checkpoints
    ]
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
