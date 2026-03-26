import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Any

import torch
from torch.utils.data import DataLoader

from data_utils import load_dbp15k_from_pyg, load_dbp15k_raw_split, load_dbp15k_fixed_eval_split
from dataset import AlignmentTrainDataset, collate_alignment_batch
from evaluate import evaluate_alignment, encode_entity_outputs
from models.full_model import JointEAModel
from models.losses import total_loss
from sampler import random_negative_sampling, hard_negative_sampling, queue_hard_negative_sampling
from active_learning import run_active_learning_round
from graph_utils import build_adj_list, sample_neighbors


@dataclass
class Config:
    # =========================
    # Data
    # =========================
    root: str = "data/dbp15k"
    pair: str = "zh_en"
    # Follow the commonly used DBP15K 0_3 supervision protocol for formal experiments.
    data_source: str = "fixed_eval"
    raw_split: str = "0_3"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    val_ratio: float = 0.1
    al_pool_ratio: float = 0.15

    # =========================
    # Training
    # =========================
    # Training hyperparameters are chosen with reference to prior entity alignment
    # work and adjusted to fit the current model architecture.
    batch_size: int = 128
    eval_batch_size: int = 256

    warmup_epochs: int = 5
    joint_epochs: int = 15

    lr: float = 1e-3
    weight_decay: float = 1e-5
    use_final_weight_averaging: bool = False
    weight_average_last_k: int = 5
    use_ema_eval: bool = False
    ema_decay: float = 0.90
    ema_start_epoch: int = 20
    eval_fusion: bool = False
    eval_fusion_weights: Tuple[float, float, float] = (0.7, 0.15, 0.15)

    # =========================
    # Model
    # =========================
    text_input_dim: int = 300
    node_input_dim: int = 128
    gnn_hidden_dim: int = 128
    text_hidden_dim: int = 128
    fusion_dim: int = 128
    ce_residual_ratio: float = 0.1
    gnn_layers: int = 2
    gnn_share_parameters: bool = False
    gnn_use_depthwise_separable: bool = False
    text_heads: int = 4
    text_layers: int = 2
    dropout: float = 0.1

    # neighbor-aware fusion
    num_neighbors: int = 8

    # =========================
    # Ablation
    # =========================
    use_mst: bool = True
    use_light_gnn: bool = True
    use_cross_modal_enhancement: bool = True
    ablation_name: str = "full"

    # =========================
    # Loss
    # =========================
    lambda_branch_align: float = 0.15
    lambda_cross_modal: float = 0.03
    lambda_joint_branch: float = 0.05
    lambda_struct: float = 0.2
    lambda_sem: float = 0.2
    lambda_neg: float = 0.3
    lambda_ranking: float = 0.05
    lambda_topology: float = 0.03
    branch_align_start: float = 0.15
    branch_align_end: float = 0.50
    cross_modal_start: float = 0.25
    cross_modal_end: float = 0.65
    joint_branch_start: float = 0.40
    joint_branch_end: float = 0.80
    topology_start: float = 0.55
    topology_end: float = 0.90

    temperature: float = 0.07
    margin: float = 0.2
    hard_negative_weight: float = 2.0
    use_pairwise_hard_neg: bool = False
    per_pair_hard_neg: bool = False

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
    num_conflict_neg: int = 0
    memory_bank_epochs: int = 2

    # =========================
    # Active Learning
    # =========================
    do_active_learning: bool = True
    al_rounds: int = 2
    al_budget: int = 120
    al_every_epochs: int = 8
    al_require_bidirectional: bool = True
    al_min_confidence: float = 0.40
    al_min_margin: float = 0.03
    al_min_uncertainty: float = 0.00
    al_max_uncertainty: float = 0.85
    al_negative_batch_size: int = 32
    al_negative_weight: float = 1.0
    al_pool_negatives_per_left: int = 5
    al_pool_use_full_candidates: bool = True
    al_block_size: int = 1024
    al_alpha_uncertainty: float = 0.35
    al_alpha_repr: float = 0.65
    al_use_diversity: bool = True
    al_max_filter_stage: str = "score_only"

    # =========================
    # Misc
    # =========================
    seed: int = 42
    save_dir: str = "outputs"
    early_stop_patience: int = 4
    early_stop_min_delta: float = 1e-3
    use_early_stopping: bool = True

    @property
    def experiment_tag(self) -> str:
        disabled = []
        if not self.use_mst:
            disabled.append("w_o_MST")
        if not self.use_light_gnn:
            disabled.append("w_o_LightGNN")
        if not self.use_cross_modal_enhancement:
            disabled.append("w_o_CE")
        if not self.do_active_learning:
            disabled.append("w_o_AL")
        if self.gnn_share_parameters:
            disabled.append("shared_gnn")
        if self.gnn_use_depthwise_separable:
            disabled.append("dwsep_gnn")
        return "full_model" if not disabled else "_".join(disabled)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_bool_env(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def apply_optional_env(cfg: Config, env_name: str, attr: str, cast) -> None:
    env_value = os.environ.get(env_name)
    if env_value is None:
        return
    setattr(cfg, attr, cast(env_value))


def apply_runtime_overrides(cfg: Config) -> None:
    seed_env = os.environ.get("KG_ALIGN_SEED")
    if seed_env is not None:
        cfg.seed = int(seed_env)

    ablation_env = os.environ.get("KG_ALIGN_ABLATION")
    if ablation_env:
        cfg.ablation_name = ablation_env

    al_env = os.environ.get("KG_ALIGN_AL")
    if al_env is not None:
        cfg.do_active_learning = parse_bool_env(al_env)

    data_source_env = os.environ.get("KG_ALIGN_DATA_SOURCE")
    if data_source_env:
        cfg.data_source = data_source_env

    ce_ratio_env = os.environ.get("KG_ALIGN_CE_RATIO")
    if ce_ratio_env is not None:
        cfg.ce_residual_ratio = float(ce_ratio_env)

    eval_fusion_env = os.environ.get("KG_ALIGN_EVAL_FUSION")
    if eval_fusion_env is not None:
        cfg.eval_fusion = parse_bool_env(eval_fusion_env)

    eval_weights_env = os.environ.get("KG_ALIGN_EVAL_WEIGHTS")
    if eval_weights_env:
        parts = [p.strip() for p in eval_weights_env.split(",") if p.strip()]
        if len(parts) == 3:
            cfg.eval_fusion_weights = (float(parts[0]), float(parts[1]), float(parts[2]))

    seed_env = os.environ.get("KG_ALIGN_SEED")
    if seed_env is not None:
        try:
            cfg.seed = int(seed_env)
        except ValueError:
            raise ValueError(f"KG_ALIGN_SEED must be an int, got: {seed_env}")

    per_pair_env = os.environ.get("KG_ALIGN_PER_PAIR_HARD_NEG")
    if per_pair_env is not None:
        cfg.per_pair_hard_neg = parse_bool_env(per_pair_env)

    if cfg.data_source == "raw_split":
        # Treat the official DBP15K 0_3 protocol as a separate experimental
        # regime. It has much less supervision than the fixed split setting, so
        # we train longer, lower the learning rate, and make the collaborative
        # objectives ramp up later.
        cfg.warmup_epochs = 3
        cfg.joint_epochs = 36
        cfg.lr = 5e-4
        cfg.use_final_weight_averaging = True
        cfg.weight_average_last_k = 10
        cfg.use_ema_eval = False
        cfg.ema_decay = 0.90
        cfg.ema_start_epoch = 20
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
        # Raw-split experiments are sensitive to extra structural regularization.
        # Keep topology matching available for manual sweeps, but do not enable
        # it by default in the low-supervision setting.
        cfg.lambda_topology = 0.00
        # Baseline-first protocol: keep active learning off by default while we
        # strengthen the main supervised alignment path. It can still be
        # re-enabled explicitly via KG_ALIGN_AL=1 for follow-up experiments.
        cfg.do_active_learning = False
        cfg.ce_residual_ratio = 0.10
        cfg.num_global_hard_neg = 2
        cfg.global_hard_topk_max = 50
        cfg.use_pairwise_hard_neg = False
        cfg.per_pair_hard_neg = False

        # Use weaker branch-to-joint consistency so the structural/semantic
        # branches keep more complementarity instead of collapsing too
        # aggressively toward the joint space.
        cfg.lambda_struct = 0.05
        cfg.lambda_sem = 0.05

        # In raw-split the model is already quite stable, so repeated runs with
        # the same seed mostly collapse to the same optimum. We therefore make
        # active learning more conservative by delaying it, shrinking the
        # budget, and disallowing low-quality non-reciprocal fallback stages.
        #
        # Important protocol choice: for raw-split experiments we no longer use
        # the official test split as an oracle annotation pool. Instead, we hold
        # out a small slice of the training supervision as the AL pool so the
        # active-learning loop stays within the training regime.
        cfg.al_pool_ratio = 0.05
        cfg.al_rounds = 2
        cfg.al_budget = 80
        cfg.al_every_epochs = 12
        cfg.al_min_confidence = 0.50
        cfg.al_min_margin = 0.05
        cfg.al_negative_weight = 0.60
        cfg.al_alpha_uncertainty = 0.25
        cfg.al_alpha_repr = 0.75
        cfg.al_max_filter_stage = "reciprocal_relaxed_margin"

    apply_optional_env(cfg, "KG_ALIGN_BATCH_SIZE", "batch_size", int)
    apply_optional_env(cfg, "KG_ALIGN_EVAL_BATCH_SIZE", "eval_batch_size", int)
    apply_optional_env(cfg, "KG_ALIGN_WARMUP_EPOCHS", "warmup_epochs", int)
    apply_optional_env(cfg, "KG_ALIGN_JOINT_EPOCHS", "joint_epochs", int)
    apply_optional_env(cfg, "KG_ALIGN_LR", "lr", float)
    apply_optional_env(cfg, "KG_ALIGN_WEIGHT_DECAY", "weight_decay", float)
    apply_optional_env(cfg, "KG_ALIGN_TEMPERATURE", "temperature", float)
    apply_optional_env(cfg, "KG_ALIGN_MARGIN", "margin", float)
    apply_optional_env(cfg, "KG_ALIGN_WEIGHT_AVG_LAST_K", "weight_average_last_k", int)
    apply_optional_env(cfg, "KG_ALIGN_NUM_RANDOM_NEG", "num_random_neg", int)
    apply_optional_env(cfg, "KG_ALIGN_NUM_HARD_NEG", "num_hard_neg", int)
    apply_optional_env(cfg, "KG_ALIGN_HARD_TOPK", "hard_topk", int)
    apply_optional_env(cfg, "KG_ALIGN_NUM_GLOBAL_HARD_NEG", "num_global_hard_neg", int)
    apply_optional_env(cfg, "KG_ALIGN_GLOBAL_HARD_TOPK", "global_hard_topk", int)
    apply_optional_env(cfg, "KG_ALIGN_GLOBAL_HARD_TOPK_MAX", "global_hard_topk_max", int)
    apply_optional_env(cfg, "KG_ALIGN_GLOBAL_HARD_START", "global_hard_start_epoch", int)
    apply_optional_env(cfg, "KG_ALIGN_MEMORY_BANK_EPOCHS", "memory_bank_epochs", int)
    apply_optional_env(cfg, "KG_ALIGN_NUM_CONFLICT_NEG", "num_conflict_neg", int)
    apply_optional_env(cfg, "KG_ALIGN_LAMBDA_BRANCH", "lambda_branch_align", float)
    apply_optional_env(cfg, "KG_ALIGN_LAMBDA_CROSS", "lambda_cross_modal", float)
    apply_optional_env(cfg, "KG_ALIGN_LAMBDA_JOINT_BRANCH", "lambda_joint_branch", float)
    apply_optional_env(cfg, "KG_ALIGN_LAMBDA_STRUCT", "lambda_struct", float)
    apply_optional_env(cfg, "KG_ALIGN_LAMBDA_SEM", "lambda_sem", float)
    apply_optional_env(cfg, "KG_ALIGN_LAMBDA_NEG", "lambda_neg", float)
    apply_optional_env(cfg, "KG_ALIGN_LAMBDA_RANKING", "lambda_ranking", float)
    apply_optional_env(cfg, "KG_ALIGN_LAMBDA_TOPOLOGY", "lambda_topology", float)
    apply_optional_env(cfg, "KG_ALIGN_AL_ROUNDS", "al_rounds", int)
    apply_optional_env(cfg, "KG_ALIGN_AL_BUDGET", "al_budget", int)
    apply_optional_env(cfg, "KG_ALIGN_AL_EVERY_EPOCHS", "al_every_epochs", int)
    apply_optional_env(cfg, "KG_ALIGN_AL_POOL_RATIO", "al_pool_ratio", float)
    apply_optional_env(cfg, "KG_ALIGN_AL_POOL_NEG_PER_LEFT", "al_pool_negatives_per_left", int)
    apply_optional_env(cfg, "KG_ALIGN_AL_MIN_CONF", "al_min_confidence", float)
    apply_optional_env(cfg, "KG_ALIGN_AL_MIN_MARGIN", "al_min_margin", float)
    apply_optional_env(cfg, "KG_ALIGN_AL_MIN_UNCERTAINTY", "al_min_uncertainty", float)
    apply_optional_env(cfg, "KG_ALIGN_AL_MAX_UNCERTAINTY", "al_max_uncertainty", float)
    apply_optional_env(cfg, "KG_ALIGN_AL_NEG_BATCH", "al_negative_batch_size", int)
    apply_optional_env(cfg, "KG_ALIGN_AL_NEG_WEIGHT", "al_negative_weight", float)
    apply_optional_env(cfg, "KG_ALIGN_AL_ALPHA_UNCERTAINTY", "al_alpha_uncertainty", float)
    apply_optional_env(cfg, "KG_ALIGN_AL_ALPHA_REPR", "al_alpha_repr", float)
    apply_optional_env(cfg, "KG_ALIGN_AL_MAX_STAGE", "al_max_filter_stage", str)

    use_global_hard_env = os.environ.get("KG_ALIGN_USE_GLOBAL_HARD_NEG")
    if use_global_hard_env is not None:
        cfg.use_global_hard_neg = parse_bool_env(use_global_hard_env)

    use_weight_avg_env = os.environ.get("KG_ALIGN_USE_WEIGHT_AVG")
    if use_weight_avg_env is not None:
        cfg.use_final_weight_averaging = parse_bool_env(use_weight_avg_env)

    use_ema_env = os.environ.get("KG_ALIGN_USE_EMA")
    if use_ema_env is not None:
        cfg.use_ema_eval = parse_bool_env(use_ema_env)

    al_diversity_env = os.environ.get("KG_ALIGN_AL_USE_DIVERSITY")
    if al_diversity_env is not None:
        cfg.al_use_diversity = parse_bool_env(al_diversity_env)

    full_candidates_env = os.environ.get("KG_ALIGN_AL_FULL_CANDIDATES")
    if full_candidates_env is not None:
        cfg.al_pool_use_full_candidates = parse_bool_env(full_candidates_env)

    require_bidir_env = os.environ.get("KG_ALIGN_AL_REQUIRE_BIDIRECTIONAL")
    if require_bidir_env is not None:
        cfg.al_require_bidirectional = parse_bool_env(require_bidir_env)

    # Re-apply the top-level AL switch after raw-split profile defaults so
    # baseline-first defaults do not block explicit follow-up AL experiments.
    al_env = os.environ.get("KG_ALIGN_AL")
    if al_env is not None:
        cfg.do_active_learning = parse_bool_env(al_env)


def split_pairs(
    pairs: List[Tuple[int, int]],
    holdout_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    if not 0.0 < holdout_ratio < 1.0:
        raise ValueError(f"holdout_ratio must be in (0, 1), got {holdout_ratio}")
    if len(pairs) < 2:
        raise ValueError("Need at least 2 pairs to create a holdout split")

    shuffled = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    holdout_size = max(1, int(round(len(shuffled) * holdout_ratio)))
    holdout_size = min(holdout_size, len(shuffled) - 1)
    holdout_pairs = shuffled[:holdout_size]
    remain_pairs = shuffled[holdout_size:]
    return remain_pairs, holdout_pairs


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


def update_ema_state_dict(
    ema_state: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    decay: float,
) -> Dict[str, torch.Tensor]:
    current_state = model.state_dict()
    if ema_state is None:
        return {
            key: value.detach().cpu().clone()
            for key, value in current_state.items()
        }

    updated_state = {}
    for key, value in current_state.items():
        value_cpu = value.detach().cpu()
        if torch.is_floating_point(value_cpu):
            updated_state[key] = ema_state[key] * decay + value_cpu * (1.0 - decay)
        else:
            updated_state[key] = value_cpu.clone()
    return updated_state


def build_experiment_splits(
    train_pairs: List[Tuple[int, int]],
    test_pairs: List[Tuple[int, int]],
    val_ratio: float,
    al_pool_ratio: float,
    seed: int,
    do_active_learning: bool,
) -> Dict[str, List[Tuple[int, int]]]:
    train_after_val, val_pairs = split_pairs(train_pairs, val_ratio, seed)

    if do_active_learning and al_pool_ratio > 0.0:
        train_pairs, al_pool_pairs = split_pairs(train_after_val, al_pool_ratio, seed + 1)
    else:
        train_pairs = train_after_val
        al_pool_pairs = []

    return {
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "al_pool_pairs": al_pool_pairs,
        "al_pool_gold_pairs": list(al_pool_pairs),
        "al_pool_left_ids": [],
        "al_pool_right_ids": [],
        "test_pairs": list(test_pairs),
    }


def build_protocol_splits(cfg: Config, train_pairs: List[Tuple[int, int]], test_pairs: List[Tuple[int, int]]) -> Dict[str, List[Tuple[int, int]]]:
    if cfg.data_source == "raw_split":
        # Keep the official reference pairs exclusively for final testing.
        # When active learning is enabled, reserve a small subset of the
        # provided training supervision as the oracle annotation pool.
        train_pairs = list(train_pairs)
        if cfg.do_active_learning and cfg.al_pool_ratio > 0.0:
            train_pairs, al_pool_gold_pairs = split_pairs(train_pairs, cfg.al_pool_ratio, cfg.seed + 1)
            if cfg.al_pool_use_full_candidates:
                al_pool_left_ids = sorted({l for l, _ in al_pool_gold_pairs})
                al_pool_right_ids = sorted({r for _, r in al_pool_gold_pairs})
                al_pool_pairs = []
            else:
                right_id_pool = sorted({r for _, r in al_pool_gold_pairs})
                rng = random.Random(cfg.seed + 2)
                pool_pairs = set()
                for l, r_true in al_pool_gold_pairs:
                    pool_pairs.add((l, r_true))
                    for _ in range(cfg.al_pool_negatives_per_left):
                        r_neg = rng.choice(right_id_pool)
                        while r_neg == r_true:
                            r_neg = rng.choice(right_id_pool)
                        pool_pairs.add((l, r_neg))
                al_pool_pairs = list(pool_pairs)
                al_pool_left_ids = []
                al_pool_right_ids = []
        else:
            al_pool_pairs = []
            al_pool_gold_pairs = []
            al_pool_left_ids = []
            al_pool_right_ids = []
        return {
            "train_pairs": list(train_pairs),
            "val_pairs": [],
            "al_pool_pairs": list(al_pool_pairs),
            "al_pool_gold_pairs": list(al_pool_gold_pairs),
            "al_pool_left_ids": list(al_pool_left_ids),
            "al_pool_right_ids": list(al_pool_right_ids),
            "test_pairs": list(test_pairs),
        }

    return build_experiment_splits(
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        val_ratio=cfg.val_ratio,
        al_pool_ratio=cfg.al_pool_ratio,
        seed=cfg.seed,
        do_active_learning=cfg.do_active_learning,
    )


def build_fixed_eval_splits(
    cfg: Config,
    train_pairs: List[Tuple[int, int]],
    val_pairs: List[Tuple[int, int]],
    test_pairs: List[Tuple[int, int]],
) -> Dict[str, List[Tuple[int, int]]]:
    if cfg.do_active_learning and cfg.al_pool_ratio > 0.0:
        train_pairs, al_pool_pairs = split_pairs(train_pairs, cfg.al_pool_ratio, cfg.seed + 1)
    else:
        al_pool_pairs = []

    return {
        "train_pairs": list(train_pairs),
        "val_pairs": list(val_pairs),
        "al_pool_pairs": list(al_pool_pairs),
        "al_pool_gold_pairs": list(al_pool_pairs),
        "al_pool_left_ids": [],
        "al_pool_right_ids": [],
        "test_pairs": list(test_pairs),
    }


def should_use_early_stopping(cfg: Config) -> bool:
    if cfg.data_source == "raw_split":
        return False
    return cfg.use_early_stopping


def apply_ablation_config(cfg: Config) -> None:
    if cfg.ablation_name == "full":
        return
    if cfg.ablation_name == "w_o_MST":
        cfg.use_mst = False
        return
    if cfg.ablation_name == "w_o_LightGNN":
        cfg.use_light_gnn = False
        return
    if cfg.ablation_name == "w_o_CE":
        cfg.use_cross_modal_enhancement = False
        # Keep the ablation semantically clean: once the fusion module is
        # removed, the CE-specific consistency objectives should also be off.
        cfg.lambda_cross_modal = 0.0
        cfg.lambda_joint_branch = 0.0
        return
    if cfg.ablation_name == "w_o_AL":
        cfg.do_active_learning = False
        return
    raise ValueError(f"Unsupported ablation_name: {cfg.ablation_name}")


def load_dataset(cfg: Config) -> Dict[str, Any]:
    if cfg.data_source == "raw_split":
        return load_dbp15k_raw_split(root=cfg.root, pair=cfg.pair, split=cfg.raw_split)
    if cfg.data_source == "fixed_eval":
        return load_dbp15k_fixed_eval_split(root=cfg.root, pair=cfg.pair)
    raise ValueError(
        f"Unsupported data_source: {cfg.data_source}. "
        "This project now uses paper-style raw DBP15K splits for formal experiments."
    )


def build_candidate_right_ids(
    cfg: Config,
    n1: int,
    n2: int,
    eval_pairs: List[Tuple[int, int]],
) -> torch.Tensor:
    if cfg.data_source == "raw_split":
        # Match the common DBP15K protocol: rank each left entity against the
        # right-side reference pool of the current evaluation split.
        return torch.tensor(
            sorted({r for _, r in eval_pairs}),
            dtype=torch.long,
        )
    return torch.arange(n1, n1 + n2, dtype=torch.long)


def build_model(cfg: Config, total_nodes: int) -> JointEAModel:
    return JointEAModel(
        num_nodes=total_nodes,
        text_input_dim=cfg.text_input_dim,
        node_input_dim=cfg.node_input_dim,
        gnn_hidden_dim=cfg.gnn_hidden_dim,
        text_hidden_dim=cfg.text_hidden_dim,
        fusion_dim=cfg.fusion_dim,
        gnn_layers=cfg.gnn_layers,
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


@torch.no_grad()
def build_global_negative_cache(
    model: JointEAModel,
    edge_index: torch.Tensor,
    seq_features: torch.Tensor,
    adj_list: Dict[int, List[int]],
    train_pairs: List[Tuple[int, int]],
    cfg: Config,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    left_ids = torch.tensor(sorted({l for l, _ in train_pairs}), dtype=torch.long)
    right_ids = torch.tensor(sorted({r for _, r in train_pairs}), dtype=torch.long)

    left_outputs = encode_entity_outputs(
        model=model,
        node_ids=left_ids,
        edge_index=edge_index,
        seq_features=seq_features,
        adj_list=adj_list,
        num_neighbors=cfg.num_neighbors,
        batch_size=cfg.eval_batch_size,
        device=device,
    )
    right_outputs = encode_entity_outputs(
        model=model,
        node_ids=right_ids,
        edge_index=edge_index,
        seq_features=seq_features,
        adj_list=adj_list,
        num_neighbors=cfg.num_neighbors,
        batch_size=cfg.eval_batch_size,
        device=device,
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
    al_negative_pairs: List[Tuple[int, int]] = None,
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
    total_al_neg = 0.0
    total_ranking = 0.0
    total_topology = 0.0
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
        al_neg_left_joint = None
        al_neg_right_joint = None

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
                )

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

            if al_negative_pairs:
                sample_size = min(cfg.al_negative_batch_size, len(al_negative_pairs))
                sampled_al_negs = random.sample(al_negative_pairs, sample_size)
                al_neg_left_ids = torch.tensor(
                    [l for l, _ in sampled_al_negs],
                    dtype=torch.long,
                    device=device,
                )
                al_neg_right_ids = torch.tensor(
                    [r for _, r in sampled_al_negs],
                    dtype=torch.long,
                    device=device,
                )

                al_neg_left_out = forward_entities(
                    model=model,
                    node_ids=al_neg_left_ids,
                    edge_index=edge_index,
                    seq_features=seq_features,
                    adj_list=adj_list,
                    cfg=cfg,
                    device=device,
                )

                al_neg_right_out = forward_entities(
                    model=model,
                    node_ids=al_neg_right_ids,
                    edge_index=edge_index,
                    seq_features=seq_features,
                    adj_list=adj_list,
                    cfg=cfg,
                    device=device,
                )

                al_neg_left_joint = al_neg_left_out["z_joint"]
                al_neg_right_joint = al_neg_right_out["z_joint"]

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
            al_neg_left_joint=al_neg_left_joint,
            al_neg_right_joint=al_neg_right_joint,
            margin=cfg.margin,
            hard_negative_weight=cfg.hard_negative_weight,
            al_negative_weight=cfg.al_negative_weight,
            use_pairwise_hard_neg=cfg.use_pairwise_hard_neg,
            per_pair_hard_neg=cfg.per_pair_hard_neg,
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
        total_al_neg += loss_dict["al_neg_loss"].item()
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
        "al_neg_loss": total_al_neg / max(1, num_batches),
        "ranking_loss": total_ranking / max(1, num_batches),
        "topology_loss": total_topology / max(1, num_batches),
    }


def main():
    cfg = Config()
    apply_runtime_overrides(cfg)
    apply_ablation_config(cfg)
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print("Loading DBP15K...")
    data = load_dataset(cfg)

    total_nodes = data["total_nodes"]
    n1 = data["n1"]
    n2 = data["n2"]
    edge_index = data["edge_index"].to(device)
    seq_features = data["seq_features"]   # 通常放 CPU
    raw_train_pairs = list(data["train_pairs"])
    raw_test_pairs = list(data["test_pairs"])
    if cfg.data_source == "fixed_eval":
        raw_val_pairs = list(data["val_pairs"])
        splits = build_fixed_eval_splits(cfg, raw_train_pairs, raw_val_pairs, raw_test_pairs)
    else:
        splits = build_protocol_splits(cfg, raw_train_pairs, raw_test_pairs)
    train_pairs = splits["train_pairs"]
    val_pairs = splits["val_pairs"]
    al_pool_pairs = splits["al_pool_pairs"]
    al_pool_gold_pairs = splits.get("al_pool_gold_pairs", al_pool_pairs)
    al_pool_left_ids = splits.get("al_pool_left_ids", [])
    al_pool_right_ids = splits.get("al_pool_right_ids", [])
    test_pairs = splits["test_pairs"]
    val_candidate_right_ids = None
    if val_pairs:
        val_candidate_right_ids = build_candidate_right_ids(cfg, n1, n2, val_pairs)
    test_candidate_right_ids = build_candidate_right_ids(cfg, n1, n2, test_pairs)

    # 用 CPU 上的 edge_index 建邻接表，避免 GPU tensor 转 list 问题
    adj_list = build_adj_list(data["edge_index"].cpu(), total_nodes)

    print(f"Pair: {cfg.pair}")
    print(f"Data source: {cfg.data_source}")
    if cfg.data_source == "raw_split":
        print(f"Raw split: {cfg.raw_split}")
    print(f"Experiment: {cfg.experiment_tag}")
    print(f"Seed: {cfg.seed}")
    print(f"Total nodes: {total_nodes}")
    print(f"Edges total: {edge_index.size(1)}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")
    if al_pool_left_ids and al_pool_right_ids:
        print(
            f"AL pool pairs: {len(al_pool_left_ids)} x {len(al_pool_right_ids)} "
            "(full candidates)"
        )
    else:
        print(f"AL pool pairs: {len(al_pool_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")
    if cfg.data_source == "raw_split" and cfg.do_active_learning:
        print("AL pool source: held-out training supervision")
    print(f"Sequence features: {tuple(seq_features.shape)}")
    print(
        "Ablation switches: "
        f"MST={cfg.use_mst}, "
        f"LightGNN={cfg.use_light_gnn}, "
        f"CE={cfg.use_cross_modal_enhancement}, "
        f"AL={cfg.do_active_learning}"
    )
    print(
        "Negative sampling: "
        f"random={cfg.num_random_neg}, "
        f"batch_hard={cfg.num_hard_neg}, "
        f"global_hard={cfg.num_global_hard_neg}, "
        f"conflict={cfg.num_conflict_neg}, "
        f"global_topk={cfg.global_hard_topk}->{cfg.global_hard_topk_max}, "
        f"global_start={cfg.global_hard_start_epoch}, "
        f"bank_epochs={cfg.memory_bank_epochs}, "
        f"use_global_hard={cfg.use_global_hard_neg}"
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
        f"shared_gnn={cfg.gnn_share_parameters}, "
        f"depthwise_separable={cfg.gnn_use_depthwise_separable}"
    )
    print(f"CE residual ratio: {cfg.ce_residual_ratio:.2f}")
    print(
        "Active learning filters: "
        f"bidirectional={cfg.al_require_bidirectional}, "
        f"min_conf={cfg.al_min_confidence:.2f}, "
        f"min_margin={cfg.al_min_margin:.2f}, "
        f"min_unc={cfg.al_min_uncertainty:.2f}, "
        f"max_unc={cfg.al_max_uncertainty:.2f}, "
        f"max_stage={cfg.al_max_filter_stage}"
    )
    print(
        "Active learning negatives: "
        f"batch_size={cfg.al_negative_batch_size}, "
        f"weight={cfg.al_negative_weight:.2f}"
    )
    if cfg.eval_fusion:
        w_joint, w_struct, w_sem = cfg.eval_fusion_weights
        print(
            f"Eval fusion: joint={w_joint:.2f}, "
            f"struct={w_struct:.2f}, sem={w_sem:.2f}"
        )
    if cfg.use_ema_eval:
        print(
            f"Final weight strategy: EMA "
            f"(decay={cfg.ema_decay:.2f}, start_epoch={cfg.ema_start_epoch})"
        )
    elif cfg.use_final_weight_averaging:
        print(
            f"Final weight strategy: average_last_k "
            f"(k={cfg.weight_average_last_k})"
        )
    print(f"Early stopping enabled: {should_use_early_stopping(cfg)}")

    model = build_model(cfg, total_nodes=total_nodes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    joint_scheduler = None
    if cfg.data_source == "raw_split":
        joint_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.joint_epochs,
            eta_min=cfg.lr * 0.2,
        )

    best_hits1 = -1.0
    best_path = os.path.join(cfg.save_dir, f"best_model_{cfg.pair}_{cfg.experiment_tag}.pt")
    early_stopping_enabled = should_use_early_stopping(cfg)

    # =========================
    # Stage 1: Warmup
    # =========================
    print("\n===== Stage 1: Warmup =====")
    warmup_loader = make_loader(train_pairs, cfg.batch_size, shuffle=True)
    fusion_weights = cfg.eval_fusion_weights if cfg.eval_fusion else None

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

        if val_pairs:
            metrics = evaluate_alignment(
                model=model,
                edge_index=edge_index,
                seq_features=seq_features,
                adj_list=adj_list,
                num_neighbors=cfg.num_neighbors,
                test_pairs=val_pairs,
                candidate_right_ids=val_candidate_right_ids,
                batch_size=cfg.eval_batch_size,
                device=device,
                fusion_weights=fusion_weights,
            )

            print(
                f"[Warmup] Epoch {epoch:03d} | "
                f"Loss: {train_stats['loss']:.4f} | "
                f"Struct: {train_stats['struct_loss']:.4f} | "
                f"Topo: {train_stats['topology_loss']:.4f} | "
                f"ValHits@1: {metrics['Hits@1']:.4f} | "
                f"ValHits@10: {metrics['Hits@10']:.4f} | "
                f"ValMRR: {metrics['MRR']:.4f}"
            )
        else:
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
    al_round_done = 0
    queried_al_pairs = set()
    al_negative_pairs = []
    negative_bank_history = []
    weight_average_history = []
    ema_state = None
    ema_updates = 0
    early_stop_counter = 0

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
            seq_features=seq_features,
            adj_list=adj_list,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            all_train_pairs=train_pairs,
            al_negative_pairs=al_negative_pairs,
            negative_bank=negative_bank,
            dynamic_global_topk=dynamic_global_topk,
            current_joint_epoch=epoch,
            total_joint_epochs=cfg.joint_epochs,
            warmup_mode=False,
        )

        if val_pairs:
            metrics = evaluate_alignment(
                model=model,
                edge_index=edge_index,
                seq_features=seq_features,
                adj_list=adj_list,
                num_neighbors=cfg.num_neighbors,
                test_pairs=val_pairs,
                candidate_right_ids=val_candidate_right_ids,
                batch_size=cfg.eval_batch_size,
                device=device,
                fusion_weights=fusion_weights,
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
                f"ALNeg: {train_stats['al_neg_loss']:.4f} | "
                f"Rank: {train_stats['ranking_loss']:.4f} | "
                f"Topo: {train_stats['topology_loss']:.4f}({train_stats['topology_scale']:.2f}) | "
                f"TopK: {dynamic_global_topk} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                f"ValHits@1: {metrics['Hits@1']:.4f} | "
                f"ValHits@10: {metrics['Hits@10']:.4f} | "
                f"ValMRR: {metrics['MRR']:.4f}"
            )

            if metrics["Hits@1"] > best_hits1 + cfg.early_stop_min_delta:
                best_hits1 = metrics["Hits@1"]
                early_stop_counter = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": cfg.__dict__,
                    },
                    best_path,
                )
                print(f"Saved best model to {best_path}")
            else:
                early_stop_counter += 1

            if early_stopping_enabled and early_stop_counter >= cfg.early_stop_patience:
                print(
                    f"Early stopping triggered at joint epoch {epoch:03d} "
                    f"(best Hits@1={best_hits1:.4f})"
                )
                break
        else:
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
                f"ALNeg: {train_stats['al_neg_loss']:.4f} | "
                f"Rank: {train_stats['ranking_loss']:.4f} | "
                f"Topo: {train_stats['topology_loss']:.4f}({train_stats['topology_scale']:.2f}) | "
                f"TopK: {dynamic_global_topk} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

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
                pool_pairs=al_pool_pairs,
                gold_pairs=al_pool_gold_pairs,
                pool_left_ids=al_pool_left_ids,
                pool_right_ids=al_pool_right_ids,
                queried_pairs=queried_al_pairs,
                device=device,
            )
            al_round_done += 1
            for pair in al_stats["negative_pairs"]:
                if pair not in al_negative_pairs:
                    al_negative_pairs.append(pair)

            print(
                f"[AL    ] Round {al_round_done:02d} | "
                f"Candidates: {al_stats['candidates']} | "
                f"Strict: {al_stats['strict_filtered_candidates']} | "
                f"Filtered: {al_stats['filtered_candidates']} | "
                f"Stage: {al_stats['filter_stage']} | "
                f"NewPos: {al_stats['new_positive']} | "
                f"NewNeg: {al_stats['new_negative']} | "
                f"Added: {al_stats['added_to_train']} | "
                f"RejectBi: {al_stats['rejected_bidirectional']} | "
                f"RejectConf: {al_stats['rejected_confidence']} | "
                f"RejectMargin: {al_stats['rejected_margin']} | "
                f"RejectLowUnc: {al_stats['rejected_low_uncertainty']} | "
                f"RejectUnc: {al_stats['rejected_uncertainty']} | "
                f"PoolLeft: {al_stats['remaining_pool']} | "
                f"ALNegPool: {len(al_negative_pairs)}"
            )

        if (
            cfg.use_final_weight_averaging
            and epoch >= max(1, cfg.joint_epochs - cfg.weight_average_last_k + 1)
        ):
            weight_average_history.append(snapshot_model_state_dict(model))
            weight_average_history = weight_average_history[-cfg.weight_average_last_k:]

        if cfg.use_ema_eval and epoch >= cfg.ema_start_epoch:
            ema_state = update_ema_state_dict(ema_state, model, cfg.ema_decay)
            ema_updates += 1

        if joint_scheduler is not None:
            joint_scheduler.step()

    print("\nTraining finished.")
    if val_pairs:
        print(f"Best Hits@1: {best_hits1:.4f}")

        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            final_test_metrics = evaluate_alignment(
                model=model,
                edge_index=edge_index,
                seq_features=seq_features,
                adj_list=adj_list,
                num_neighbors=cfg.num_neighbors,
                test_pairs=test_pairs,
                candidate_right_ids=test_candidate_right_ids,
                batch_size=cfg.eval_batch_size,
                device=device,
                fusion_weights=fusion_weights,
            )
            print(
                "Final test metrics from best validation checkpoint | "
                f"Hits@1: {final_test_metrics['Hits@1']:.4f} | "
                f"Hits@10: {final_test_metrics['Hits@10']:.4f} | "
                f"MRR: {final_test_metrics['MRR']:.4f}"
            )
    else:
        if cfg.use_ema_eval and ema_state is not None:
            model.load_state_dict(ema_state)
            print(
                f"Applied EMA weights with decay={cfg.ema_decay:.2f} "
                f"from joint epoch {cfg.ema_start_epoch} "
                f"(updates={ema_updates})"
            )
        elif cfg.use_final_weight_averaging and len(weight_average_history) > 1:
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
            seq_features=seq_features,
            adj_list=adj_list,
            num_neighbors=cfg.num_neighbors,
            test_pairs=test_pairs,
            candidate_right_ids=test_candidate_right_ids,
            batch_size=cfg.eval_batch_size,
            device=device,
            fusion_weights=fusion_weights,
        )
        print(
            "Final test metrics after fixed-epoch training | "
            f"Hits@1: {final_test_metrics['Hits@1']:.4f} | "
            f"Hits@10: {final_test_metrics['Hits@10']:.4f} | "
            f"MRR: {final_test_metrics['MRR']:.4f}"
        )


if __name__ == "__main__":
    main()
