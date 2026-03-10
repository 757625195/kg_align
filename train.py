import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Any

import torch
from torch.utils.data import DataLoader

from data_utils import load_dbp15k_from_pyg
from dataset import AlignmentTrainDataset, collate_alignment_batch
from evaluate import evaluate_alignment
from models.full_model import JointEAModel
from models.losses import total_loss
from sampler import random_negative_sampling, hard_negative_sampling
from active_learning import run_active_learning_round
from graph_utils import build_adj_list, sample_neighbors


@dataclass
class Config:
    # =========================
    # Data
    # =========================
    root: str = "data/DBP15K"
    pair: str = "zh_en"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================
    # Training
    # =========================
    batch_size: int = 128
    eval_batch_size: int = 256

    warmup_epochs: int = 5
    joint_epochs: int = 15

    lr: float = 1e-3
    weight_decay: float = 1e-5

    # =========================
    # Model
    # =========================
    text_input_dim: int = 300
    node_input_dim: int = 128
    gnn_hidden_dim: int = 128
    text_hidden_dim: int = 128
    fusion_dim: int = 128
    gnn_layers: int = 2
    text_heads: int = 4
    text_layers: int = 2
    dropout: float = 0.1

    # neighbor-aware fusion
    num_neighbors: int = 8

    # =========================
    # Loss
    # =========================
    lambda_struct: float = 0.2
    lambda_sem: float = 0.2
    lambda_neg: float = 0.2

    temperature: float = 0.07
    margin: float = 0.2
    hard_negative_weight: float = 1.5

    # =========================
    # Negative sampling
    # =========================
    num_random_neg: int = 1
    num_hard_neg: int = 1
    hard_topk: int = 5

    # =========================
    # Active Learning
    # =========================
    do_active_learning: bool = False
    al_rounds: int = 3
    al_budget: int = 100
    al_every_epochs: int = 5

    # =========================
    # Misc
    # =========================
    seed: int = 42
    save_dir: str = "outputs"


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: Config, total_nodes: int) -> JointEAModel:
    return JointEAModel(
        num_nodes=total_nodes,
        text_input_dim=cfg.text_input_dim,
        node_input_dim=cfg.node_input_dim,
        gnn_hidden_dim=cfg.gnn_hidden_dim,
        text_hidden_dim=cfg.text_hidden_dim,
        fusion_dim=cfg.fusion_dim,
        gnn_layers=cfg.gnn_layers,
        text_heads=cfg.text_heads,
        text_layers=cfg.text_layers,
        dropout=cfg.dropout,
    )


def make_loader(train_pairs: List[Tuple[int, int]], batch_size: int, shuffle: bool = True):
    ds = AlignmentTrainDataset(train_pairs=train_pairs)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_alignment_batch,
    )


def fetch_seq(
    seq_features: torch.Tensor,
    node_ids: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    seq_features: [N, L, D]，通常保存在 CPU
    node_ids: [B]，通常在 GPU
    """
    return seq_features[node_ids.cpu()].to(device)


def forward_entities(
    model: JointEAModel,
    node_ids: torch.Tensor,
    edge_index: torch.Tensor,
    seq_features: torch.Tensor,
    adj_list: Dict[int, List[int]],
    cfg: Config,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    对一批实体执行：
    1) 取语义序列特征
    2) 采样固定数量邻居
    3) 调用邻居感知的 fusion 模型
    """
    seq_x = fetch_seq(seq_features, node_ids, device)

    neighbor_ids, neighbor_mask = sample_neighbors(
        node_ids=node_ids,
        adj_list=adj_list,
        num_neighbors=cfg.num_neighbors,
        device=device,
    )

    out = model(
        node_ids=node_ids,
        edge_index=edge_index,
        seq_features=seq_x,
        neighbor_ids=neighbor_ids,
        neighbor_mask=neighbor_mask,
    )
    return out


def train_one_epoch(
    model: JointEAModel,
    loader: DataLoader,
    edge_index: torch.Tensor,
    seq_features: torch.Tensor,
    adj_list: Dict[int, List[int]],
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    device: torch.device,
    all_train_pairs: List[Tuple[int, int]],
    warmup_mode: bool = False,
) -> Dict[str, float]:
    model.train()

    known_pair_set: Set[Tuple[int, int]] = set(all_train_pairs)
    all_left_entities = sorted({l for l, _ in all_train_pairs})
    all_right_entities = sorted({r for _, r in all_train_pairs})

    total = 0.0
    total_align = 0.0
    total_struct = 0.0
    total_sem = 0.0
    total_neg = 0.0
    num_batches = 0

    for batch in loader:
        left_id = batch["left_id"].to(device)
        right_id = batch["right_id"].to(device)

        # 正样本前向
        left_out = forward_entities(
            model=model,
            node_ids=left_id,
            edge_index=edge_index,
            seq_features=seq_features,
            adj_list=adj_list,
            cfg=cfg,
            device=device,
        )

        right_out = forward_entities(
            model=model,
            node_ids=right_id,
            edge_index=edge_index,
            seq_features=seq_features,
            adj_list=adj_list,
            cfg=cfg,
            device=device,
        )

        neg_left_joint = None
        neg_right_joint = None

        # 非 warmup 阶段才做对齐损失和负样本
        if not warmup_mode:
            batch_pairs = list(zip(
                left_id.detach().cpu().tolist(),
                right_id.detach().cpu().tolist()
            ))

            # 随机负样本
            rand_negs = random_negative_sampling(
                batch_pairs=batch_pairs,
                all_left_entities=all_left_entities,
                all_right_entities=all_right_entities,
                known_pairs=known_pair_set,
                num_random_neg=cfg.num_random_neg,
            )

            # 难负样本（基于当前 batch joint embedding）
            hard_negs = hard_negative_sampling(
                left_emb=left_out["z_joint"].detach(),
                right_emb=right_out["z_joint"].detach(),
                batch_pairs=batch_pairs,
                batch_left_ids=left_id,
                batch_right_ids=right_id,
                all_left_ids=left_id,
                all_right_ids=right_id,
                known_pairs=known_pair_set,
                topk=cfg.hard_topk,
                num_hard_neg=cfg.num_hard_neg,
            )

            all_negs = rand_negs + hard_negs

            if len(all_negs) > 0:
                neg_left_ids = torch.tensor(
                    [l for l, _ in all_negs],
                    dtype=torch.long,
                    device=device
                )
                neg_right_ids = torch.tensor(
                    [r for _, r in all_negs],
                    dtype=torch.long,
                    device=device
                )

                neg_left_out = forward_entities(
                    model=model,
                    node_ids=neg_left_ids,
                    edge_index=edge_index,
                    seq_features=seq_features,
                    adj_list=adj_list,
                    cfg=cfg,
                    device=device,
                )

                neg_right_out = forward_entities(
                    model=model,
                    node_ids=neg_right_ids,
                    edge_index=edge_index,
                    seq_features=seq_features,
                    adj_list=adj_list,
                    cfg=cfg,
                    device=device,
                )

                neg_left_joint = neg_left_out["z_joint"]
                neg_right_joint = neg_right_out["z_joint"]

        # 计算总损失
        loss_dict = total_loss(
            left_outputs=left_out,
            right_outputs=right_out,
            lambda_struct=cfg.lambda_struct,
            lambda_sem=cfg.lambda_sem,
            lambda_neg=cfg.lambda_neg,
            temperature=cfg.temperature,
            neg_left_joint=neg_left_joint,
            neg_right_joint=neg_right_joint,
            margin=cfg.margin,
            hard_negative_weight=cfg.hard_negative_weight,
            warmup_mode=warmup_mode,
        )

        optimizer.zero_grad()
        loss_dict["loss"].backward()
        optimizer.step()

        total += loss_dict["loss"].item()
        total_align += loss_dict["align_loss"].item()
        total_struct += loss_dict["struct_loss"].item()
        total_sem += loss_dict["sem_loss"].item()
        total_neg += loss_dict["neg_loss"].item()
        num_batches += 1

    return {
        "loss": total / max(1, num_batches),
        "align_loss": total_align / max(1, num_batches),
        "struct_loss": total_struct / max(1, num_batches),
        "sem_loss": total_sem / max(1, num_batches),
        "neg_loss": total_neg / max(1, num_batches),
    }


def main():
    cfg = Config()
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print("Loading PyG DBP15K...")
    data = load_dbp15k_from_pyg(root=cfg.root, pair=cfg.pair)

    total_nodes = data["total_nodes"]
    edge_index = data["edge_index"].to(device)
    seq_features = data["seq_features"]   # 通常放 CPU
    train_pairs = list(data["train_pairs"])
    test_pairs = list(data["test_pairs"])

    # 用 CPU 上的 edge_index 建邻接表，避免 GPU tensor 转 list 问题
    adj_list = build_adj_list(data["edge_index"].cpu(), total_nodes)

    print(f"Pair: {cfg.pair}")
    print(f"Total nodes: {total_nodes}")
    print(f"Edges total: {edge_index.size(1)}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")
    print(f"Sequence features: {tuple(seq_features.shape)}")

    model = build_model(cfg, total_nodes=total_nodes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    best_hits1 = -1.0
    best_path = os.path.join(cfg.save_dir, f"best_model_{cfg.pair}.pt")

    # =========================
    # Stage 1: Warmup
    # =========================
    print("\n===== Stage 1: Warmup =====")
    warmup_loader = make_loader(train_pairs, cfg.batch_size, shuffle=True)

    for epoch in range(1, cfg.warmup_epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=warmup_loader,
            edge_index=edge_index,
            seq_features=seq_features,
            adj_list=adj_list,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            all_train_pairs=train_pairs,
            warmup_mode=True,
        )

        metrics = evaluate_alignment(
            model=model,
            edge_index=edge_index,
            seq_features=seq_features,
            adj_list=adj_list,
            num_neighbors=cfg.num_neighbors,
            test_pairs=test_pairs,
            batch_size=cfg.eval_batch_size,
            device=device,
        )

        print(
            f"[Warmup] Epoch {epoch:03d} | "
            f"Loss: {train_stats['loss']:.4f} | "
            f"Struct: {train_stats['struct_loss']:.4f} | "
            f"Hits@1: {metrics['Hits@1']:.4f} | "
            f"Hits@10: {metrics['Hits@10']:.4f} | "
            f"MRR: {metrics['MRR']:.4f}"
        )

    # =========================
    # Stage 2: Joint Training
    # =========================
    print("\n===== Stage 2: Joint Training =====")
    al_round_done = 0

    for epoch in range(1, cfg.joint_epochs + 1):
        train_loader = make_loader(train_pairs, cfg.batch_size, shuffle=True)

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            edge_index=edge_index,
            seq_features=seq_features,
            adj_list=adj_list,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            all_train_pairs=train_pairs,
            warmup_mode=False,
        )

        metrics = evaluate_alignment(
            model=model,
            edge_index=edge_index,
            seq_features=seq_features,
            adj_list=adj_list,
            num_neighbors=cfg.num_neighbors,
            test_pairs=test_pairs,
            batch_size=cfg.eval_batch_size,
            device=device,
        )

        print(
            f"[Joint ] Epoch {epoch:03d} | "
            f"Loss: {train_stats['loss']:.4f} | "
            f"Align: {train_stats['align_loss']:.4f} | "
            f"Struct: {train_stats['struct_loss']:.4f} | "
            f"Sem: {train_stats['sem_loss']:.4f} | "
            f"Neg: {train_stats['neg_loss']:.4f} | "
            f"Hits@1: {metrics['Hits@1']:.4f} | "
            f"Hits@10: {metrics['Hits@10']:.4f} | "
            f"MRR: {metrics['MRR']:.4f}"
        )

        if metrics["Hits@1"] > best_hits1:
            best_hits1 = metrics["Hits@1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.__dict__,
                },
                best_path,
            )
            print(f"Saved best model to {best_path}")

        # =========================
        # Optional Active Learning
        # =========================
        if (
            cfg.do_active_learning
            and epoch % cfg.al_every_epochs == 0
            and al_round_done < cfg.al_rounds
        ):
            al_stats = run_active_learning_round(
                model=model,
                cfg=cfg,
                edge_index=edge_index,
                seq_features=seq_features,
                adj_list=adj_list,
                train_pairs=train_pairs,
                test_pairs=test_pairs,
                device=device,
            )
            al_round_done += 1

            print(
                f"[AL    ] Round {al_round_done:02d} | "
                f"Candidates: {al_stats['candidates']} | "
                f"NewPos: {al_stats['new_positive']} | "
                f"NewNeg: {al_stats['new_negative']} | "
                f"Added: {al_stats['added_to_train']}"
            )

    print("\nTraining finished.")
    print(f"Best Hits@1: {best_hits1:.4f}")


if __name__ == "__main__":
    main()