import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Any

import torch
from torch.utils.data import DataLoader

from data_utils import load_dbp15k_raw_split
from dataset import AlignmentTrainDataset, collate_alignment_batch
from evaluate import evaluate_alignment, encode_entity_outputs
from models.full_model import JointEAModel
from models.losses import total_loss
from sampler import random_negative_sampling, hard_negative_sampling, queue_hard_negative_sampling
from graph_utils import build_adj_list, sample_neighbors


@dataclass
class Config:
    # =========================
    # Data
    # =========================
    root: str = "data/dbp15k"
    pair: str = "zh_en"
    # Default to the strongest verified formal setting: DBP15K raw 0_3.
    data_source: str = "raw_split"
    raw_split: str = "0_3"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================
    # Training
    # =========================
    # Training hyperparameters are chosen with reference to prior entity alignment
    # work and adjusted to fit the current model architecture.
    batch_size: int = 128
    eval_batch_size: int = 256

    warmup_epochs: int = 3
    joint_epochs: int = 36

    lr: float = 5e-4
    weight_decay: float = 1e-5
    use_final_weight_averaging: bool = True
    weight_average_last_k: int = 10

    # =========================
    # Model
    # =========================
    text_input_dim: int = 300
    node_input_dim: int = 128
    gnn_hidden_dim: int = 128
    text_hidden_dim: int = 128
    fusion_dim: int = 128
    ce_residual_ratio: float = 0.1
    gnn_layers: int = 3
    use_relation_gnn: bool = True
    relation_layer_fusion: bool = True
    gnn_share_parameters: bool = False
    gnn_use_depthwise_separable: bool = False
    alignment_csls_k: int = 10
    alignment_csls_blend: float = 1.0
    text_heads: int = 4
    text_layers: int = 2
    dropout: float = 0.1

    # neighbor-aware fusion
    num_neighbors: int = 8
    use_relation_aware_neighbor_sampling: bool = True
    neighbor_relation_score_alpha: float = 1.0
    neighbor_degree_score_alpha: float = 0.25

    # =========================
    # Ablation
    # =========================
    use_mst: bool = True
    use_light_gnn: bool = True
    use_cross_modal_enhancement: bool = True

    # =========================
    # Loss
    # =========================
    lambda_branch_align: float = 0.15
    lambda_cross_modal: float = 0.005
    lambda_joint_branch: float = 0.01
    lambda_struct: float = 0.05
    lambda_sem: float = 0.05
    lambda_neg: float = 0.3
    lambda_ranking: float = 0.05
    lambda_topology: float = 0.00
    branch_align_start: float = 0.25
    branch_align_end: float = 0.65
    cross_modal_start: float = 0.40
    cross_modal_end: float = 0.80
    joint_branch_start: float = 0.55
    joint_branch_end: float = 0.90
    topology_start: float = 0.70
    topology_end: float = 0.98

    temperature: float = 0.07
    margin: float = 0.2
    hard_negative_weight: float = 2.0
    per_pair_hard_neg: bool = True
    stable_neg_topk: int = 1
    stable_ranking_topk: int = 1

    # =========================
    # Negative sampling
    # =========================
    num_random_neg: int = 1
    num_hard_neg: int = 2
    hard_topk: int = 10
    num_global_hard_neg: int = 2
    global_hard_topk: int = 20
    global_hard_topk_max: int = 50
    use_global_hard_neg: bool = True
    global_hard_start_epoch: int = 1
    global_hard_reciprocal_guard_topk: int = 0
    num_conflict_neg: int = 0
    memory_bank_epochs: int = 2

    seed: int = 42
    save_dir: str = "outputs"

    @property
    def experiment_tag(self) -> str:
        return "best_baseline"


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_raw_split_best_defaults(cfg: Config) -> None:
    """
    Strongest verified raw-split baseline.

    This preset intentionally keeps:
    - joint-only final scoring
    - active learning off
    - topology matching off
    while baking in the confirmed improvements from the text encoder, fusion,
    relation-aware structural encoder, and per-pair hardest negatives.
    """
    cfg.warmup_epochs = 3
    cfg.joint_epochs = 36
    cfg.lr = 5e-4
    cfg.use_final_weight_averaging = True
    cfg.weight_average_last_k = 10

    cfg.lambda_branch_align = 0.15
    cfg.branch_align_start = 0.25
    cfg.branch_align_end = 0.65
    cfg.cross_modal_start = 0.40
    cfg.cross_modal_end = 0.80
    cfg.joint_branch_start = 0.55
    cfg.joint_branch_end = 0.90
    cfg.topology_start = 0.70
    cfg.topology_end = 0.98
    cfg.lambda_cross_modal = 0.005
    cfg.lambda_joint_branch = 0.01
    cfg.lambda_struct = 0.05
    cfg.lambda_sem = 0.05
    cfg.lambda_topology = 0.00

    cfg.ce_residual_ratio = 0.10
    cfg.num_hard_neg = 2
    cfg.hard_topk = 10
    cfg.num_global_hard_neg = 2
    cfg.global_hard_topk = 20
    cfg.global_hard_topk_max = 50
    cfg.global_hard_start_epoch = 1
    cfg.global_hard_reciprocal_guard_topk = 0
    cfg.memory_bank_epochs = 2
    cfg.hard_negative_weight = 2.0
    cfg.lambda_ranking = 0.05
    cfg.per_pair_hard_neg = True
    cfg.stable_neg_topk = 1
    cfg.stable_ranking_topk = 1

    cfg.gnn_layers = 3
    cfg.relation_layer_fusion = True
    cfg.use_relation_aware_neighbor_sampling = True
    cfg.neighbor_relation_score_alpha = 1.0
    cfg.neighbor_degree_score_alpha = 0.25

def apply_runtime_overrides(cfg: Config) -> None:
    # Lock training to the strongest verified formal baseline.
    cfg.data_source = "raw_split"
    apply_raw_split_best_defaults(cfg)

    seed_env = os.environ.get("KG_ALIGN_SEED")
    if seed_env is not None:
        try:
            cfg.seed = int(seed_env)
        except ValueError:
            raise ValueError(f"KG_ALIGN_SEED must be an int, got: {seed_env}")


def snapshot_model_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def average_state_dicts(state_history: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not state_history:
        raise ValueError("state_history must not be empty")

    averaged_state = {}
    for key in state_history[0].keys():
        values = [state[key] for state in state_history]
        first = values[0]
        if torch.is_floating_point(first):
            averaged_state[key] = torch.stack(values, dim=0).mean(dim=0)
        else:
            averaged_state[key] = first.clone()
    return averaged_state


def build_protocol_splits(train_pairs: List[Tuple[int, int]], test_pairs: List[Tuple[int, int]]) -> Dict[str, List[Tuple[int, int]]]:
    return {
        "train_pairs": list(train_pairs),
        "test_pairs": list(test_pairs),
    }


def load_dataset(cfg: Config) -> Dict[str, Any]:
    return load_dbp15k_raw_split(root=cfg.root, pair=cfg.pair, split=cfg.raw_split)


def build_candidate_right_ids(
    eval_pairs: List[Tuple[int, int]],
) -> torch.Tensor:
    # Match the common DBP15K raw-split protocol: rank each left entity against
    # the right-side reference pool of the current evaluation split.
    return torch.tensor(
        sorted({r for _, r in eval_pairs}),
        dtype=torch.long,
    )


def build_model(cfg: Config, total_nodes: int, num_relations: int) -> JointEAModel:
    return JointEAModel(
        num_nodes=total_nodes,
        text_input_dim=cfg.text_input_dim,
        node_input_dim=cfg.node_input_dim,
        gnn_hidden_dim=cfg.gnn_hidden_dim,
        text_hidden_dim=cfg.text_hidden_dim,
        fusion_dim=cfg.fusion_dim,
        gnn_layers=cfg.gnn_layers,
        use_relation_gnn=cfg.use_relation_gnn,
        num_relations=num_relations,
        relation_layer_fusion=cfg.relation_layer_fusion,
        alignment_csls_k=cfg.alignment_csls_k,
        alignment_csls_blend=cfg.alignment_csls_blend,
        gnn_share_parameters=cfg.gnn_share_parameters,
        gnn_use_depthwise_separable=cfg.gnn_use_depthwise_separable,
        text_heads=cfg.text_heads,
        text_layers=cfg.text_layers,
        dropout=cfg.dropout,
        ce_residual_ratio=cfg.ce_residual_ratio,
        use_mst=cfg.use_mst,
        use_light_gnn=cfg.use_light_gnn,
        use_cross_modal_enhancement=cfg.use_cross_modal_enhancement,
        use_explicit_topology_matching=True,
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
    edge_type: torch.Tensor,
    seq_features: torch.Tensor,
    adj_list: Dict[int, List[int]],
    cfg: Config,
    device: torch.device,
    z_struct_all: torch.Tensor = None,
) -> Dict[str, torch.Tensor]:
    """
    对一批实体执行：
    1) 取语义序列特征
    2) 采样固定数量邻居
    3) 调用邻居感知的 fusion 模型
    """
    unique_node_ids, inverse = torch.unique(node_ids, sorted=True, return_inverse=True)

    seq_x = fetch_seq(seq_features, unique_node_ids, device)

    neighbor_ids, neighbor_mask = sample_neighbors(
        node_ids=unique_node_ids,
        adj_list=adj_list,
        num_neighbors=cfg.num_neighbors,
        device=device,
    )

    out = model(
        node_ids=unique_node_ids,
        edge_index=edge_index,
        seq_features=seq_x,
        neighbor_ids=neighbor_ids,
        neighbor_mask=neighbor_mask,
        edge_type=edge_type,
        z_struct_all=z_struct_all,
    )
    return {
        key: value[inverse] if torch.is_tensor(value) else value
        for key, value in out.items()
    }


@torch.no_grad()
def build_global_negative_cache(
    model: JointEAModel,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    seq_features: torch.Tensor,
    adj_list: Dict[int, List[int]],
    train_pairs: List[Tuple[int, int]],
    cfg: Config,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    left_ids = torch.tensor(sorted({l for l, _ in train_pairs}), dtype=torch.long)
    right_ids = torch.tensor(sorted({r for _, r in train_pairs}), dtype=torch.long)
    z_struct_all = model.encode_structure_all(
        edge_index=edge_index,
        edge_type=edge_type,
    )

    left_outputs = encode_entity_outputs(
        model=model,
        node_ids=left_ids,
        edge_index=edge_index,
        edge_type=edge_type,
        seq_features=seq_features,
        adj_list=adj_list,
        num_neighbors=cfg.num_neighbors,
        batch_size=cfg.eval_batch_size,
        device=device,
        z_struct_all=z_struct_all,
    )
    right_outputs = encode_entity_outputs(
        model=model,
        node_ids=right_ids,
        edge_index=edge_index,
        edge_type=edge_type,
        seq_features=seq_features,
        adj_list=adj_list,
        num_neighbors=cfg.num_neighbors,
        batch_size=cfg.eval_batch_size,
        device=device,
        z_struct_all=z_struct_all,
    )

    return {
        "left_ids": left_ids,
        "right_ids": right_ids,
        "left_joint": left_outputs["z_joint"],
        "right_joint": right_outputs["z_joint"],
        "left_struct": left_outputs["z_struct"],
        "right_struct": right_outputs["z_struct"],
        "left_sem": left_outputs["z_sem"],
        "right_sem": right_outputs["z_sem"],
    }


def merge_negative_banks(history: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    merged = {}
    for key in history[0]:
        merged[key] = torch.cat([entry[key] for entry in history], dim=0)
    return merged


def get_dynamic_topk(epoch: int, total_epochs: int, base_topk: int, max_topk: int) -> int:
    if total_epochs <= 1:
        return base_topk
    ratio = (epoch - 1) / max(1, total_epochs - 1)
    return int(round(base_topk + (max_topk - base_topk) * ratio))


def merge_negative_groups(*group_sources: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    merged = None
    for source in group_sources:
        if source is None or len(source) == 0:
            continue
        if merged is None:
            merged = [list(group) for group in source]
            continue
        if len(source) != len(merged):
            raise ValueError("Negative group sources must have the same batch length")
        for target_group, extra_group in zip(merged, source):
            target_group.extend(extra_group)
    return merged or []


def flatten_negative_groups(
    groups: List[List[Tuple[int, int]]]
) -> List[Tuple[int, int]]:
    if not groups:
        return []

    max_group_size = max(len(group) for group in groups)
    if max_group_size <= 0:
        return []

    flat_pairs: List[Tuple[int, int]] = []
    for group in groups:
        if not group:
            raise ValueError(
                "per_pair_hard_neg requires at least one negative for every positive pair"
            )
        padded_group = list(group)
        while len(padded_group) < max_group_size:
            padded_group.append(padded_group[-1])
        flat_pairs.extend(padded_group)
    return flat_pairs


def train_one_epoch(
    model: JointEAModel,
    loader: DataLoader,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    seq_features: torch.Tensor,
    adj_list: Dict[int, List[int]],
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    device: torch.device,
    all_train_pairs: List[Tuple[int, int]],
    negative_bank: Dict[str, torch.Tensor] = None,
    dynamic_global_topk: int = None,
    current_joint_epoch: int = 1,
    total_joint_epochs: int = 1,
    warmup_mode: bool = False,
) -> Dict[str, float]:
    model.train()

    known_pair_set: Set[Tuple[int, int]] = set(all_train_pairs)
    all_left_entities = sorted({l for l, _ in all_train_pairs})
    all_right_entities = sorted({r for _, r in all_train_pairs})

    total = 0.0
    total_align = 0.0
    total_branch_align = 0.0
    total_cross_modal = 0.0
    total_joint_branch = 0.0
    total_branch_scale = 0.0
    total_cross_scale = 0.0
    total_joint_scale = 0.0
    total_topology_scale = 0.0
    total_struct = 0.0
    total_sem = 0.0
    total_neg = 0.0
    total_ranking = 0.0
    total_topology = 0.0
    num_batches = 0

    for batch in loader:
        left_id = batch["left_id"].to(device)
        right_id = batch["right_id"].to(device)
        z_struct_all = model.encode_structure_all(
            edge_index=edge_index,
            edge_type=edge_type,
        )

        # 正样本前向
        left_out = forward_entities(
            model=model,
            node_ids=left_id,
            edge_index=edge_index,
            edge_type=edge_type,
            seq_features=seq_features,
            adj_list=adj_list,
            cfg=cfg,
            device=device,
            z_struct_all=z_struct_all,
        )

        right_out = forward_entities(
            model=model,
            node_ids=right_id,
            edge_index=edge_index,
            edge_type=edge_type,
            seq_features=seq_features,
            adj_list=adj_list,
            cfg=cfg,
            device=device,
            z_struct_all=z_struct_all,
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
            use_grouped_negs = cfg.per_pair_hard_neg

            rand_negs = random_negative_sampling(
                batch_pairs=batch_pairs,
                all_left_entities=all_left_entities,
                all_right_entities=all_right_entities,
                known_pairs=known_pair_set,
                num_random_neg=cfg.num_random_neg,
                return_grouped=use_grouped_negs,
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
                return_grouped=use_grouped_negs,
            )

            global_hard_negs = []
            if cfg.use_global_hard_neg and negative_bank is not None:
                global_hard_negs = queue_hard_negative_sampling(
                    batch_pairs=batch_pairs,
                    batch_left_ids=left_id,
                    batch_right_ids=right_id,
                    batch_left_outputs=left_out,
                    batch_right_outputs=right_out,
                    bank=negative_bank,
                    known_pairs=known_pair_set,
                    joint_topk=dynamic_global_topk or cfg.global_hard_topk,
                    num_global_hard_neg=cfg.num_global_hard_neg,
                    num_conflict_neg=cfg.num_conflict_neg,
                    reciprocal_guard_topk=cfg.global_hard_reciprocal_guard_topk,
                    return_grouped=use_grouped_negs,
                )

            if use_grouped_negs:
                grouped_negs = merge_negative_groups(rand_negs, hard_negs, global_hard_negs)
                all_negs = flatten_negative_groups(grouped_negs)
            else:
                all_negs = rand_negs + hard_negs + global_hard_negs

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
                    edge_type=edge_type,
                    seq_features=seq_features,
                    adj_list=adj_list,
                    cfg=cfg,
                    device=device,
                    z_struct_all=z_struct_all,
                )

                neg_right_out = forward_entities(
                    model=model,
                    node_ids=neg_right_ids,
                    edge_index=edge_index,
                    edge_type=edge_type,
                    seq_features=seq_features,
                    adj_list=adj_list,
                    cfg=cfg,
                    device=device,
                    z_struct_all=z_struct_all,
                )

                neg_left_joint = neg_left_out["z_joint"]
                neg_right_joint = neg_right_out["z_joint"]

        # 计算总损失
        loss_dict = total_loss(
            left_outputs=left_out,
            right_outputs=right_out,
            lambda_branch_align=cfg.lambda_branch_align,
            lambda_cross_modal=cfg.lambda_cross_modal,
            lambda_joint_branch=cfg.lambda_joint_branch,
            lambda_struct=cfg.lambda_struct,
            lambda_sem=cfg.lambda_sem,
            lambda_neg=cfg.lambda_neg,
            lambda_ranking=cfg.lambda_ranking,
            lambda_topology=cfg.lambda_topology,
            current_joint_epoch=current_joint_epoch,
            total_joint_epochs=total_joint_epochs,
            branch_align_start=cfg.branch_align_start,
            branch_align_end=cfg.branch_align_end,
            cross_modal_start=cfg.cross_modal_start,
            cross_modal_end=cfg.cross_modal_end,
            joint_branch_start=cfg.joint_branch_start,
            joint_branch_end=cfg.joint_branch_end,
            topology_start=cfg.topology_start,
            topology_end=cfg.topology_end,
            temperature=cfg.temperature,
            neg_left_joint=neg_left_joint,
            neg_right_joint=neg_right_joint,
            margin=cfg.margin,
            hard_negative_weight=cfg.hard_negative_weight,
            per_pair_hard_neg=cfg.per_pair_hard_neg,
            stable_neg_topk=cfg.stable_neg_topk,
            stable_ranking_topk=cfg.stable_ranking_topk,
            warmup_mode=warmup_mode,
        )

        optimizer.zero_grad()
        loss_dict["loss"].backward()
        optimizer.step()

        total += loss_dict["loss"].item()
        total_align += loss_dict["align_loss"].item()
        total_branch_align += loss_dict["branch_align_loss"].item()
        total_cross_modal += loss_dict["cross_modal_loss"].item()
        total_joint_branch += loss_dict["joint_branch_loss"].item()
        total_branch_scale += loss_dict["branch_align_scale"].item()
        total_cross_scale += loss_dict["cross_modal_scale"].item()
        total_joint_scale += loss_dict["joint_branch_scale"].item()
        total_topology_scale += loss_dict["topology_scale"].item()
        total_struct += loss_dict["struct_loss"].item()
        total_sem += loss_dict["sem_loss"].item()
        total_neg += loss_dict["neg_loss"].item()
        total_ranking += loss_dict["ranking_loss"].item()
        total_topology += loss_dict["topology_loss"].item()
        num_batches += 1

    return {
        "loss": total / max(1, num_batches),
        "align_loss": total_align / max(1, num_batches),
        "branch_align_loss": total_branch_align / max(1, num_batches),
        "cross_modal_loss": total_cross_modal / max(1, num_batches),
        "joint_branch_loss": total_joint_branch / max(1, num_batches),
        "branch_align_scale": total_branch_scale / max(1, num_batches),
        "cross_modal_scale": total_cross_scale / max(1, num_batches),
        "joint_branch_scale": total_joint_scale / max(1, num_batches),
        "topology_scale": total_topology_scale / max(1, num_batches),
        "struct_loss": total_struct / max(1, num_batches),
        "sem_loss": total_sem / max(1, num_batches),
        "neg_loss": total_neg / max(1, num_batches),
        "ranking_loss": total_ranking / max(1, num_batches),
        "topology_loss": total_topology / max(1, num_batches),
    }


def main():
    cfg = Config()
    apply_runtime_overrides(cfg)
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print("Loading DBP15K...")
    data = load_dataset(cfg)

    total_nodes = data["total_nodes"]
    edge_index = data["edge_index"].to(device)
    edge_type = data.get("edge_type")
    if edge_type is not None:
        edge_type = edge_type.to(device)
    num_relations = int(data.get("num_relations", 0))
    seq_features = data["seq_features"]   # 通常放 CPU
    raw_train_pairs = list(data["train_pairs"])
    raw_test_pairs = list(data["test_pairs"])
    splits = build_protocol_splits(raw_train_pairs, raw_test_pairs)
    train_pairs = splits["train_pairs"]
    test_pairs = splits["test_pairs"]
    test_candidate_right_ids = build_candidate_right_ids(test_pairs)

    # 用 CPU 上的 edge_index 建邻接表，避免 GPU tensor 转 list 问题
    adj_list = build_adj_list(
        data["edge_index"].cpu(),
        total_nodes,
        edge_type=data.get("edge_type").cpu() if data.get("edge_type") is not None else None,
        use_relation_aware=cfg.use_relation_aware_neighbor_sampling,
        relation_score_alpha=cfg.neighbor_relation_score_alpha,
        neighbor_degree_alpha=cfg.neighbor_degree_score_alpha,
    )

    print(f"Pair: {cfg.pair}")
    print(f"Data source: {cfg.data_source}")
    print(f"Raw split: {cfg.raw_split}")
    print(f"Experiment: {cfg.experiment_tag}")
    print(f"Seed: {cfg.seed}")
    print(f"Total nodes: {total_nodes}")
    print(f"Edges total: {edge_index.size(1)}")
    if num_relations > 0:
        print(f"Relations total: {num_relations}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")
    print(f"Sequence features: {tuple(seq_features.shape)}")
    print(
        "Ablation switches: "
        f"MST={cfg.use_mst}, "
        f"LightGNN={cfg.use_light_gnn}, "
        f"CE={cfg.use_cross_modal_enhancement}"
    )
    print(
        "Negative sampling: "
        f"random={cfg.num_random_neg}, "
        f"batch_hard={cfg.num_hard_neg}, "
        f"global_hard={cfg.num_global_hard_neg}, "
        f"conflict={cfg.num_conflict_neg}, "
        f"global_topk={cfg.global_hard_topk}->{cfg.global_hard_topk_max}, "
        f"global_start={cfg.global_hard_start_epoch}, "
        f"recip_guard_topk={cfg.global_hard_reciprocal_guard_topk}, "
        f"bank_epochs={cfg.memory_bank_epochs}, "
        f"use_global_hard={cfg.use_global_hard_neg}, "
        f"per_pair_hard={cfg.per_pair_hard_neg}, "
        f"stable_neg_topk={cfg.stable_neg_topk}, "
        f"stable_rank_topk={cfg.stable_ranking_topk}"
    )
    print(
        "Collaborative loss schedule: "
        f"branch={cfg.lambda_branch_align} [{cfg.branch_align_start:.2f},{cfg.branch_align_end:.2f}], "
        f"cross={cfg.lambda_cross_modal} [{cfg.cross_modal_start:.2f},{cfg.cross_modal_end:.2f}], "
        f"joint_branch={cfg.lambda_joint_branch} [{cfg.joint_branch_start:.2f},{cfg.joint_branch_end:.2f}], "
        f"ranking={cfg.lambda_ranking:.2f}, "
        f"topology={cfg.lambda_topology:.2f} [{cfg.topology_start:.2f},{cfg.topology_end:.2f}]"
    )
    print(
        "Structural encoder switches: "
        f"relation_gnn={cfg.use_relation_gnn and num_relations > 0}, "
        f"layer_fusion={cfg.relation_layer_fusion and cfg.use_relation_gnn and num_relations > 0}, "
        f"shared_gnn={cfg.gnn_share_parameters}, "
        f"depthwise_separable={cfg.gnn_use_depthwise_separable}"
    )
    print(
        "Neighbor selector: "
        f"relation_aware={cfg.use_relation_aware_neighbor_sampling and edge_type is not None}, "
        f"rel_alpha={cfg.neighbor_relation_score_alpha:.2f}, "
        f"deg_alpha={cfg.neighbor_degree_score_alpha:.2f}"
    )
    print(
        "Alignment scorer: "
        f"temperature={cfg.temperature:.2f}, "
        f"csls_k={cfg.alignment_csls_k}, "
        f"csls_blend={cfg.alignment_csls_blend:.2f}"
    )
    print(f"CE residual ratio: {cfg.ce_residual_ratio:.2f}")
    if cfg.use_final_weight_averaging:
        print(
            f"Final weight strategy: average_last_k "
            f"(k={cfg.weight_average_last_k})"
        )
    model = build_model(cfg, total_nodes=total_nodes, num_relations=num_relations).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    joint_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.joint_epochs,
        eta_min=cfg.lr * 0.2,
    )

    best_path = os.path.join(cfg.save_dir, f"best_model_{cfg.pair}_{cfg.experiment_tag}.pt")

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
            edge_type=edge_type,
            seq_features=seq_features,
            adj_list=adj_list,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            all_train_pairs=train_pairs,
            warmup_mode=True,
        )
        print(
            f"[Warmup] Epoch {epoch:03d} | "
            f"Loss: {train_stats['loss']:.4f} | "
            f"Struct: {train_stats['struct_loss']:.4f} | "
            f"Topo: {train_stats['topology_loss']:.4f}"
        )

    # =========================
    # Stage 2: Joint Training
    # =========================
    print("\n===== Stage 2: Joint Training =====")
    negative_bank_history = []
    weight_average_history = []
    for epoch in range(1, cfg.joint_epochs + 1):
        train_loader = make_loader(train_pairs, cfg.batch_size, shuffle=True)
        negative_bank = None
        dynamic_global_topk = 0
        global_hard_active = cfg.use_global_hard_neg and epoch >= cfg.global_hard_start_epoch
        if global_hard_active:
            dynamic_global_topk = get_dynamic_topk(
                epoch=epoch,
                total_epochs=cfg.joint_epochs,
                base_topk=cfg.global_hard_topk,
                max_topk=cfg.global_hard_topk_max,
            )
            current_bank = build_global_negative_cache(
                model=model,
                edge_index=edge_index,
                edge_type=edge_type,
                seq_features=seq_features,
                adj_list=adj_list,
                train_pairs=train_pairs,
                cfg=cfg,
                device=device,
            )
            negative_bank_history.append(current_bank)
            negative_bank_history = negative_bank_history[-cfg.memory_bank_epochs:]
            negative_bank = merge_negative_banks(negative_bank_history)

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            edge_index=edge_index,
            edge_type=edge_type,
            seq_features=seq_features,
            adj_list=adj_list,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            all_train_pairs=train_pairs,
            negative_bank=negative_bank,
            dynamic_global_topk=dynamic_global_topk,
            current_joint_epoch=epoch,
            total_joint_epochs=cfg.joint_epochs,
            warmup_mode=False,
        )

        print(
            f"[Joint ] Epoch {epoch:03d} | "
            f"Loss: {train_stats['loss']:.4f} | "
            f"Align: {train_stats['align_loss']:.4f} | "
            f"Branch: {train_stats['branch_align_loss']:.4f}({train_stats['branch_align_scale']:.2f}) | "
            f"Cross: {train_stats['cross_modal_loss']:.4f}({train_stats['cross_modal_scale']:.2f}) | "
            f"JointBr: {train_stats['joint_branch_loss']:.4f}({train_stats['joint_branch_scale']:.2f}) | "
            f"Struct: {train_stats['struct_loss']:.4f} | "
            f"Sem: {train_stats['sem_loss']:.4f} | "
            f"Neg: {train_stats['neg_loss']:.4f} | "
            f"Rank: {train_stats['ranking_loss']:.4f} | "
            f"Topo: {train_stats['topology_loss']:.4f}({train_stats['topology_scale']:.2f}) | "
            f"TopK: {dynamic_global_topk} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if (
            cfg.use_final_weight_averaging
            and epoch >= max(1, cfg.joint_epochs - cfg.weight_average_last_k + 1)
        ):
            weight_average_history.append(snapshot_model_state_dict(model))
            weight_average_history = weight_average_history[-cfg.weight_average_last_k:]

        if joint_scheduler is not None:
            joint_scheduler.step()

    print("\nTraining finished.")
    if cfg.use_final_weight_averaging and len(weight_average_history) > 1:
        averaged_state = average_state_dicts(weight_average_history)
        model.load_state_dict(averaged_state)
        print(
            f"Applied final weight averaging over the last "
            f"{len(weight_average_history)} joint epochs"
        )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.__dict__,
        },
        best_path,
    )
    print(f"Saved final model to {best_path}")
    final_test_metrics = evaluate_alignment(
        model=model,
        edge_index=edge_index,
        edge_type=edge_type,
        seq_features=seq_features,
        adj_list=adj_list,
        num_neighbors=cfg.num_neighbors,
        test_pairs=test_pairs,
        candidate_right_ids=test_candidate_right_ids,
        batch_size=cfg.eval_batch_size,
        device=device,
        fusion_weights=None,
    )
    print(
        "Final test metrics after fixed-epoch training | "
        f"Hits@1: {final_test_metrics['Hits@1']:.4f} | "
        f"Hits@10: {final_test_metrics['Hits@10']:.4f} | "
        f"MRR: {final_test_metrics['MRR']:.4f}"
    )


if __name__ == "__main__":
    main()
