import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import torch
from torch.utils.data import DataLoader

from data_utils import build_entity_name_features, load_dbp15k_raw_split, load_openea_dataset
from dataset import AlignmentTrainDataset, collate_alignment_batch
from evaluate import (
    evaluate_alignment,
    select_csls_parameters,
)
from models.full_model import JointEAModel
from models.losses import total_loss
from graph_utils import build_adj_list, build_topology_features, sample_neighbors


@dataclass
class Config:
    # =========================
    # Data
    # =========================
    root: str = "data/dbp15k"
    pair: str = "zh_en"
    dataset_family: str = "dbp15k_raw"
    # Default to the strongest verified formal setting: DBP15K raw 0_3.
    data_source: str = "raw_split"
    raw_split: str = "0_3"
    openea_split: str = "721_5fold/1"
    glove_path: str = "data/dbp15k/raw/sub.glove.300d"
    text_max_len: int = 32
    raw_split_protocol: str = "best_baseline"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    val_ratio: float = 0.10
    eval_acceptance_threshold: float = 0.0

    # =========================
    # Training
    # =========================
    # Training hyperparameters are chosen with reference to prior entity alignment
    # work and adjusted to fit the current model architecture.
    batch_size: int = 128
    eval_batch_size: int = 256

    joint_epochs: int = 36

    lr: float = 5e-4
    weight_decay: float = 1e-5
    use_final_weight_averaging: bool = True
    weight_average_last_k: int = 10
    use_early_stopping: bool = False
    early_stop_patience: int = 6
    early_stop_min_delta: float = 1e-3
    validation_every_epochs: int = 1

    # =========================
    # Model
    # =========================
    text_input_dim: int = 300
    node_input_dim: int = 128
    gnn_hidden_dim: int = 128
    text_hidden_dim: int = 128
    fusion_dim: int = 128
    ce_residual_ratio: float = 0.1
    fusion_mode: str = "early_interaction"
    gnn_layers: int = 3
    use_relation_gnn: bool = True
    relation_layer_fusion: bool = True
    use_relation_types: bool = True
    gnn_share_parameters: bool = False
    gnn_use_depthwise_separable: bool = False
    alignment_csls_k: int = 10
    alignment_csls_blend: float = 1.0
    select_csls_on_validation: bool = False
    csls_k_candidates: str = "5,10,20"
    csls_blend_candidates: str = "0.50,0.75,1.00"
    csls_selection_metric: str = "MRR"
    text_heads: int = 4
    text_layers: int = 2
    dropout: float = 0.1

    # Main protocol: expose the complete one-hop outgoing neighborhood and let
    # entmax assign data-dependent weights. A value of 0 enables batch-local
    # padding instead of a fixed neighbor cap.
    num_neighbors: int = 0
    neighbor_attention: str = "entmax15"
    neighbor_attention_temperature: float = 0.25

    # =========================
    # Ablation
    # =========================
    use_mst: bool = True
    use_light_gnn: bool = True
    use_cross_modal_enhancement: bool = True
    sem_use_token_view: bool = True
    sem_use_phrase_view: bool = True
    sem_use_global_view: bool = True
    sem_residual_mode: str = "gated"
    neighbor_query_mode: str = "semantic"
    neighbor_gate_mode: str = "semantic"
    neighbor_query_semantic_weight: float = 0.5
    structure_encoder: str = "relation_gnn"
    structure_initialization: str = "learned"
    share_seed_structure_embeddings: bool = False
    topology_entity_residual_ratio: float = 0.1
    share_cross_graph_relations: bool = True
    add_reverse_edges: bool = False

    # =========================
    # Loss
    # =========================
    temperature: float = 0.07
    structure_loss_weight: float = 0.0
    decay_structure_loss: bool = False

    seed: int = 42
    save_dir: str = "outputs"

    @property
    def experiment_tag(self) -> str:
        return self.raw_split_protocol


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_validation_selection_metric_name(cfg: Config) -> str:
    # Entity alignment is a retrieval task. Use the same ranking criterion for
    # checkpoint selection on every dataset family so threshold calibration
    # cannot influence which model is reported.
    return "MRR"


def parse_bool_env(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_int_candidates(value: str) -> List[int]:
    candidates = sorted({int(token.strip()) for token in value.split(",") if token.strip()})
    if not candidates or any(candidate <= 0 for candidate in candidates):
        raise ValueError(f"Expected positive integer candidates, got: {value!r}")
    return candidates


def parse_float_candidates(value: str) -> List[float]:
    candidates = sorted({float(token.strip()) for token in value.split(",") if token.strip()})
    if not candidates or any(candidate < 0.0 or candidate > 1.0 for candidate in candidates):
        raise ValueError(f"Expected candidates in [0, 1], got: {value!r}")
    return candidates


def apply_optional_env(cfg: Config, env_name: str, attr: str, cast) -> None:
    env_value = os.environ.get(env_name)
    if env_value is None:
        return
    setattr(cfg, attr, cast(env_value))


def apply_raw_split_best_defaults(cfg: Config) -> None:
    """
    Strongest verified raw-split baseline.

    This preset keeps joint-only final scoring and the confirmed text, fusion,
    and relation-aware structural components.
    """
    cfg.joint_epochs = 36
    cfg.lr = 5e-4
    cfg.use_final_weight_averaging = True
    cfg.weight_average_last_k = 10

    cfg.ce_residual_ratio = 0.10

    cfg.gnn_layers = 3
    cfg.relation_layer_fusion = True
    cfg.raw_split_protocol = "best_baseline"
    cfg.use_early_stopping = False


def apply_openea_official_defaults(cfg: Config) -> None:
    """
    Current OpenEA official-split full-ranking default.

    Compared with the raw DBP15K baseline, the OpenEA mainline keeps the same
    full-ranking evaluation protocol while using the full official training set.

    These defaults are only applied when dataset_family=openea and can still
    be overridden explicitly through environment variables.
    """
    cfg.lr = 3e-4
    cfg.joint_epochs = 50


def apply_simplified_core_defaults(cfg: Config) -> None:
    cfg.raw_split_protocol = "simplified_core"
    cfg.use_cross_modal_enhancement = True
    # Residual blending did not improve the three-seed selection criterion.
    cfg.ce_residual_ratio = 0.0
    # Keep the complete token/phrase/global semantic encoder in the retained
    # model. The global Transformer view is part of the paper architecture.
    cfg.sem_use_global_view = True


def apply_simplified_core_csls_defaults(cfg: Config) -> None:
    apply_simplified_core_defaults(cfg)
    cfg.raw_split_protocol = "simplified_core_csls"
    cfg.select_csls_on_validation = True
    cfg.validation_every_epochs = 5


def apply_runtime_overrides(cfg: Config) -> None:
    cfg.data_source = "raw_split"
    protocol = os.environ.get("KG_ALIGN_PROTOCOL", cfg.raw_split_protocol).strip().lower()
    apply_raw_split_best_defaults(cfg)

    seed_env = os.environ.get("KG_ALIGN_SEED")
    if seed_env is not None:
        try:
            cfg.seed = int(seed_env)
        except ValueError:
            raise ValueError(f"KG_ALIGN_SEED must be an int, got: {seed_env}")

    apply_optional_env(cfg, "KG_ALIGN_DATASET_FAMILY", "dataset_family", str)
    apply_optional_env(cfg, "KG_ALIGN_ROOT", "root", str)
    apply_optional_env(cfg, "KG_ALIGN_PAIR", "pair", str)
    apply_optional_env(cfg, "KG_ALIGN_RAW_SPLIT", "raw_split", str)
    apply_optional_env(cfg, "KG_ALIGN_OPENEA_SPLIT", "openea_split", str)
    apply_optional_env(cfg, "KG_ALIGN_SAVE_DIR", "save_dir", str)

    family = cfg.dataset_family.strip().lower()
    if family in {"openea", "eventea"}:
        if os.environ.get("KG_ALIGN_ROOT") is None and cfg.root == "data/dbp15k":
            cfg.root = "data/openea" if family == "openea" else "data/eventea/EventEA"
        cfg.data_source = "official_split"
        apply_openea_official_defaults(cfg)
    if protocol == "simplified_core":
        apply_simplified_core_defaults(cfg)
    elif protocol == "simplified_core_csls":
        apply_simplified_core_csls_defaults(cfg)

    apply_optional_env(cfg, "KG_ALIGN_GLOVE_PATH", "glove_path", str)
    apply_optional_env(cfg, "KG_ALIGN_TEXT_MAX_LEN", "text_max_len", int)
    apply_optional_env(cfg, "KG_ALIGN_EVAL_THRESHOLD", "eval_acceptance_threshold", float)
    apply_optional_env(cfg, "KG_ALIGN_GNN_LAYERS", "gnn_layers", int)

    apply_optional_env(cfg, "KG_ALIGN_BATCH_SIZE", "batch_size", int)
    apply_optional_env(cfg, "KG_ALIGN_EVAL_BATCH_SIZE", "eval_batch_size", int)
    apply_optional_env(cfg, "KG_ALIGN_JOINT_EPOCHS", "joint_epochs", int)
    apply_optional_env(cfg, "KG_ALIGN_LR", "lr", float)
    apply_optional_env(cfg, "KG_ALIGN_WEIGHT_DECAY", "weight_decay", float)
    apply_optional_env(cfg, "KG_ALIGN_WEIGHT_AVG_LAST_K", "weight_average_last_k", int)

    use_weight_avg_env = os.environ.get("KG_ALIGN_USE_WEIGHT_AVG")
    if use_weight_avg_env is not None:
        cfg.use_final_weight_averaging = parse_bool_env(use_weight_avg_env)

    use_early_stop_env = os.environ.get("KG_ALIGN_USE_EARLY_STOPPING")
    if use_early_stop_env is not None:
        cfg.use_early_stopping = parse_bool_env(use_early_stop_env)

    apply_optional_env(cfg, "KG_ALIGN_EARLY_STOP_PATIENCE", "early_stop_patience", int)
    apply_optional_env(cfg, "KG_ALIGN_EARLY_STOP_MIN_DELTA", "early_stop_min_delta", float)
    apply_optional_env(cfg, "KG_ALIGN_VALIDATION_EVERY_EPOCHS", "validation_every_epochs", int)

    apply_optional_env(cfg, "KG_ALIGN_ALIGNMENT_CSLS_K", "alignment_csls_k", int)
    apply_optional_env(cfg, "KG_ALIGN_ALIGNMENT_CSLS_BLEND", "alignment_csls_blend", float)
    apply_optional_env(cfg, "KG_ALIGN_CSLS_K_CANDIDATES", "csls_k_candidates", str)
    apply_optional_env(cfg, "KG_ALIGN_CSLS_BLEND_CANDIDATES", "csls_blend_candidates", str)
    apply_optional_env(cfg, "KG_ALIGN_CSLS_SELECTION_METRIC", "csls_selection_metric", str)
    select_csls_env = os.environ.get("KG_ALIGN_SELECT_CSLS_ON_VALIDATION")
    if select_csls_env is not None:
        cfg.select_csls_on_validation = parse_bool_env(select_csls_env)
    apply_optional_env(cfg, "KG_ALIGN_DROPOUT", "dropout", float)
    apply_optional_env(cfg, "KG_ALIGN_NUM_NEIGHBORS", "num_neighbors", int)
    apply_optional_env(cfg, "KG_ALIGN_NEIGHBOR_ATTENTION", "neighbor_attention", str)
    apply_optional_env(
        cfg,
        "KG_ALIGN_NEIGHBOR_ATTENTION_TEMPERATURE",
        "neighbor_attention_temperature",
        float,
    )
    apply_optional_env(cfg, "KG_ALIGN_CE_RESIDUAL_RATIO", "ce_residual_ratio", float)
    apply_optional_env(cfg, "KG_ALIGN_VAL_RATIO", "val_ratio", float)
    apply_optional_env(cfg, "KG_ALIGN_TEMPERATURE", "temperature", float)

    mst_env = os.environ.get("KG_ALIGN_USE_MST")
    if mst_env is not None:
        cfg.use_mst = parse_bool_env(mst_env)

    light_gnn_env = os.environ.get("KG_ALIGN_USE_LIGHT_GNN")
    if light_gnn_env is not None:
        cfg.use_light_gnn = parse_bool_env(light_gnn_env)

    ce_env = os.environ.get("KG_ALIGN_USE_CE")
    if ce_env is not None:
        cfg.use_cross_modal_enhancement = parse_bool_env(ce_env)

    apply_optional_env(cfg, "KG_ALIGN_FUSION_MODE", "fusion_mode", str)
    cfg.fusion_mode = cfg.fusion_mode.strip().lower()
    if not cfg.use_cross_modal_enhancement and cfg.fusion_mode == "early_interaction":
        cfg.fusion_mode = "mean"
    valid_fusion_modes = {
        "early_interaction",
        "reciprocal_add",
        "reciprocal_residual_add",
        "late_concat_mlp",
        "late_concat_matched",
        "mean",
        "structure_only",
        "semantic_only",
    }
    if cfg.fusion_mode not in valid_fusion_modes:
        raise ValueError(
            f"Unsupported KG_ALIGN_FUSION_MODE={cfg.fusion_mode!r}; "
            f"expected one of {sorted(valid_fusion_modes)}"
        )
    cfg.use_cross_modal_enhancement = cfg.fusion_mode in {
        "early_interaction",
        "reciprocal_add",
        "reciprocal_residual_add",
    }

    relation_gnn_env = os.environ.get("KG_ALIGN_USE_RELATION_GNN")
    if relation_gnn_env is not None:
        cfg.use_relation_gnn = parse_bool_env(relation_gnn_env)

    relation_layer_fusion_env = os.environ.get("KG_ALIGN_RELATION_LAYER_FUSION")
    if relation_layer_fusion_env is not None:
        cfg.relation_layer_fusion = parse_bool_env(relation_layer_fusion_env)

    relation_types_env = os.environ.get("KG_ALIGN_USE_RELATION_TYPES")
    if relation_types_env is not None:
        cfg.use_relation_types = parse_bool_env(relation_types_env)

    sem_token_env = os.environ.get("KG_ALIGN_SEM_USE_TOKEN_VIEW")
    if sem_token_env is not None:
        cfg.sem_use_token_view = parse_bool_env(sem_token_env)

    sem_phrase_env = os.environ.get("KG_ALIGN_SEM_USE_PHRASE_VIEW")
    if sem_phrase_env is not None:
        cfg.sem_use_phrase_view = parse_bool_env(sem_phrase_env)

    sem_global_env = os.environ.get("KG_ALIGN_SEM_USE_GLOBAL_VIEW")
    if sem_global_env is not None:
        cfg.sem_use_global_view = parse_bool_env(sem_global_env)

    apply_optional_env(cfg, "KG_ALIGN_SEM_RESIDUAL_MODE", "sem_residual_mode", str)
    cfg.sem_residual_mode = cfg.sem_residual_mode.strip().lower()

    apply_optional_env(cfg, "KG_ALIGN_NEIGHBOR_QUERY_MODE", "neighbor_query_mode", str)
    apply_optional_env(cfg, "KG_ALIGN_NEIGHBOR_GATE_MODE", "neighbor_gate_mode", str)
    apply_optional_env(
        cfg,
        "KG_ALIGN_NEIGHBOR_QUERY_SEMANTIC_WEIGHT",
        "neighbor_query_semantic_weight",
        float,
    )
    apply_optional_env(cfg, "KG_ALIGN_STRUCTURE_ENCODER", "structure_encoder", str)
    cfg.structure_encoder = cfg.structure_encoder.strip().lower()
    if cfg.structure_encoder not in {"relation_gnn", "rdgcn", "rrea"}:
        raise ValueError(
            "KG_ALIGN_STRUCTURE_ENCODER must be 'relation_gnn', 'rdgcn', or 'rrea'"
        )
    apply_optional_env(
        cfg,
        "KG_ALIGN_STRUCTURE_INITIALIZATION",
        "structure_initialization",
        str,
    )
    cfg.structure_initialization = cfg.structure_initialization.strip().lower()
    if cfg.structure_initialization not in {"learned", "name", "topology"}:
        raise ValueError(
            "KG_ALIGN_STRUCTURE_INITIALIZATION must be 'learned', 'name', or 'topology'"
        )
    apply_optional_env(
        cfg,
        "KG_ALIGN_STRUCTURE_LOSS_WEIGHT",
        "structure_loss_weight",
        float,
    )
    if cfg.structure_loss_weight < 0.0:
        raise ValueError("KG_ALIGN_STRUCTURE_LOSS_WEIGHT must be non-negative")
    decay_structure_env = os.environ.get("KG_ALIGN_DECAY_STRUCTURE_LOSS")
    if decay_structure_env is not None:
        cfg.decay_structure_loss = parse_bool_env(decay_structure_env)
    apply_optional_env(
        cfg,
        "KG_ALIGN_TOPOLOGY_ENTITY_RESIDUAL_RATIO",
        "topology_entity_residual_ratio",
        float,
    )
    seed_share_env = os.environ.get("KG_ALIGN_SHARE_SEED_STRUCTURE_EMBEDDINGS")
    if seed_share_env is not None:
        cfg.share_seed_structure_embeddings = parse_bool_env(seed_share_env)
    relation_vocab_env = os.environ.get("KG_ALIGN_SHARE_CROSS_GRAPH_RELATIONS")
    if relation_vocab_env is not None:
        cfg.share_cross_graph_relations = parse_bool_env(relation_vocab_env)
    reverse_edges_env = os.environ.get("KG_ALIGN_ADD_REVERSE_EDGES")
    if reverse_edges_env is not None:
        cfg.add_reverse_edges = parse_bool_env(reverse_edges_env)
    cfg.neighbor_query_mode = cfg.neighbor_query_mode.strip().lower()
    if cfg.neighbor_query_mode not in {"semantic", "structural", "hybrid"}:
        raise ValueError(
            "KG_ALIGN_NEIGHBOR_QUERY_MODE must be 'semantic', 'structural', or "
            "'hybrid', got "
            f"{cfg.neighbor_query_mode!r}"
        )
    cfg.neighbor_gate_mode = cfg.neighbor_gate_mode.strip().lower()
    if cfg.neighbor_gate_mode not in {"semantic", "structural", "hybrid"}:
        raise ValueError(
            "KG_ALIGN_NEIGHBOR_GATE_MODE must be 'semantic', 'structural', or "
            "'hybrid', got "
            f"{cfg.neighbor_gate_mode!r}"
        )
    if not 0.0 <= cfg.neighbor_query_semantic_weight <= 1.0:
        raise ValueError(
            "KG_ALIGN_NEIGHBOR_QUERY_SEMANTIC_WEIGHT must be within [0, 1]"
        )
    cfg.neighbor_attention = cfg.neighbor_attention.strip().lower()
    if cfg.neighbor_attention not in {"softmax", "entmax15"}:
        raise ValueError(
            "KG_ALIGN_NEIGHBOR_ATTENTION must be 'softmax' or 'entmax15', got "
            f"{cfg.neighbor_attention!r}"
        )
    if cfg.neighbor_attention_temperature <= 0.0:
        raise ValueError("KG_ALIGN_NEIGHBOR_ATTENTION_TEMPERATURE must be positive")



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


def get_checkpoint_path(cfg: Config) -> str:
    split_token = (
        cfg.openea_split
        if cfg.dataset_family.strip().lower() in {"openea", "eventea"}
        else cfg.raw_split
    )
    split_token = split_token.replace("/", "_")
    filename = (
        f"best_model_{cfg.dataset_family}_{cfg.pair}_{split_token}_{cfg.experiment_tag}.pt"
    )
    return os.path.join(cfg.save_dir, filename)


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


def build_protocol_splits(
    cfg: Config,
    train_pairs: List[Tuple[int, int]],
    test_pairs: List[Tuple[int, int]],
    explicit_val_pairs: List[Tuple[int, int]] = None,
) -> Dict[str, List[Tuple[int, int]]]:
    if explicit_val_pairs:
        validation_pairs = list(explicit_val_pairs)
        final_train_pairs = list(train_pairs)
    elif cfg.select_csls_on_validation:
        final_train_pairs, validation_pairs = split_pairs(
            pairs=train_pairs,
            holdout_ratio=cfg.val_ratio,
            seed=cfg.seed,
        )
    else:
        validation_pairs = []
        final_train_pairs = list(train_pairs)

    return {
        "train_pairs": final_train_pairs,
        "val_pairs": validation_pairs,
        "test_pairs": list(test_pairs),
    }


def load_dataset(cfg: Config) -> Dict[str, Any]:
    family = cfg.dataset_family.strip().lower()
    if family == "dbp15k_raw":
        data = load_dbp15k_raw_split(
            root=cfg.root,
            pair=cfg.pair,
            split=cfg.raw_split,
            share_cross_graph_relations=cfg.share_cross_graph_relations,
        )
    elif family == "openea":
        data = load_openea_dataset(
            root=cfg.root,
            pair=cfg.pair,
            split=cfg.openea_split,
            glove_path=cfg.glove_path,
            text_max_len=cfg.text_max_len,
            share_cross_graph_relations=cfg.share_cross_graph_relations,
        )
    elif family == "eventea":
        data = load_openea_dataset(
            root=cfg.root,
            pair=cfg.pair,
            split=cfg.openea_split,
            glove_path=cfg.glove_path,
            text_max_len=cfg.text_max_len,
            share_cross_graph_relations=cfg.share_cross_graph_relations,
        )
    else:
        raise ValueError(f"Unsupported dataset_family: {cfg.dataset_family}")

    if cfg.add_reverse_edges:
        data = dict(data)
        edge_index = data["edge_index"]
        data["edge_index"] = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        if data.get("edge_type") is not None:
            data["edge_type"] = torch.cat([data["edge_type"], data["edge_type"]], dim=0)
    return data


def build_candidate_right_ids(
    right_start_id: int,
    num_right_nodes: int,
) -> torch.Tensor:
    # Evaluate each left entity against the full target KG, rather than only the
    # gold/reference subset. This matches the paper-style retrieval setting more
    # closely and avoids optimistic scores from a restricted candidate pool.
    return torch.arange(
        right_start_id,
        right_start_id + num_right_nodes,
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
        text_heads=cfg.text_heads,
        text_layers=cfg.text_layers,
        use_relation_gnn=cfg.use_relation_gnn,
        num_relations=num_relations,
        relation_layer_fusion=cfg.relation_layer_fusion,
        use_relation_types=cfg.use_relation_types,
        alignment_csls_k=cfg.alignment_csls_k,
        alignment_csls_blend=cfg.alignment_csls_blend,
        gnn_share_parameters=cfg.gnn_share_parameters,
        gnn_use_depthwise_separable=cfg.gnn_use_depthwise_separable,
        dropout=cfg.dropout,
        ce_residual_ratio=cfg.ce_residual_ratio,
        use_mst=cfg.use_mst,
        use_light_gnn=cfg.use_light_gnn,
        use_cross_modal_enhancement=cfg.use_cross_modal_enhancement,
        sem_use_token_view=cfg.sem_use_token_view,
        sem_use_phrase_view=cfg.sem_use_phrase_view,
        sem_use_global_view=cfg.sem_use_global_view,
        sem_residual_mode=cfg.sem_residual_mode,
        fusion_mode=cfg.fusion_mode,
        neighbor_query_mode=cfg.neighbor_query_mode,
        neighbor_gate_mode=cfg.neighbor_gate_mode,
        neighbor_query_semantic_weight=cfg.neighbor_query_semantic_weight,
        neighbor_attention=cfg.neighbor_attention,
        neighbor_attention_temperature=cfg.neighbor_attention_temperature,
        structure_encoder=cfg.structure_encoder,
        structure_initialization=cfg.structure_initialization,
        share_seed_structure_embeddings=cfg.share_seed_structure_embeddings,
        topology_entity_residual_ratio=cfg.topology_entity_residual_ratio,
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
    2) 按配置提取固定预算或变长邻居
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
    epoch: int,
) -> Dict[str, float]:
    model.train()

    total = 0.0
    total_align = 0.0
    total_structure_align = 0.0
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

        structure_weight = cfg.structure_loss_weight
        if cfg.decay_structure_loss and cfg.joint_epochs > 1:
            structure_weight *= 1.0 - (epoch - 1) / (cfg.joint_epochs - 1)
        loss_dict = total_loss(
            left_outputs=left_out,
            right_outputs=right_out,
            temperature=cfg.temperature,
            structure_loss_weight=structure_weight,
        )

        optimizer.zero_grad()
        loss_dict["loss"].backward()
        optimizer.step()

        total += loss_dict["loss"].item()
        total_align += loss_dict["align_loss"].item()
        total_structure_align += loss_dict["structure_align_loss"].item()
        num_batches += 1

    return {
        "loss": total / max(1, num_batches),
        "align_loss": total_align / max(1, num_batches),
        "structure_align_loss": total_structure_align / max(1, num_batches),
    }


def main():
    cfg = Config()
    apply_runtime_overrides(cfg)
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print(f"Loading dataset family={cfg.dataset_family}...")
    data = load_dataset(cfg)

    total_nodes = data["total_nodes"]
    edge_index = data["edge_index"].to(device)
    edge_type = data.get("edge_type")
    if edge_type is not None:
        edge_type = edge_type.to(device)
    num_relations = int(data.get("num_relations", 0))
    seq_features = data["seq_features"]   # 通常放 CPU
    raw_train_pairs = list(data["train_pairs"])
    raw_val_pairs = list(data.get("val_pairs", []))
    raw_test_pairs = list(data["test_pairs"])
    splits = build_protocol_splits(
        cfg,
        raw_train_pairs,
        raw_test_pairs,
        explicit_val_pairs=raw_val_pairs,
    )
    train_pairs = splits["train_pairs"]
    val_pairs = splits["val_pairs"]
    test_pairs = splits["test_pairs"]
    val_candidate_right_ids = None
    full_right_ids = build_candidate_right_ids(
        right_start_id=data["n1"],
        num_right_nodes=data["n2"],
    )
    if val_pairs:
        val_candidate_right_ids = full_right_ids
    test_candidate_right_ids = full_right_ids

    # 用 CPU 上的 edge_index 建邻接表，避免 GPU tensor 转 list 问题
    adj_list = build_adj_list(data["edge_index"].cpu(), total_nodes)

    print(f"Pair: {cfg.pair}")
    print(f"Dataset family: {cfg.dataset_family}")
    print(f"Root: {cfg.root}")
    print(f"Data source: {cfg.data_source}")
    print(f"Raw split: {cfg.raw_split}")
    if cfg.dataset_family.strip().lower() in {"openea", "eventea"}:
        print(f"Official split: {cfg.openea_split}")
    print(f"Raw protocol: {cfg.raw_split_protocol}")
    print(f"Experiment: {cfg.experiment_tag}")
    print(f"Seed: {cfg.seed}")
    print(f"Total nodes: {total_nodes}")
    print(f"Edges total: {edge_index.size(1)}")
    if num_relations > 0:
        print(f"Relations total: {num_relations}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")
    print(f"Sequence features: {tuple(seq_features.shape)}")
    print(
        "Ablation switches: "
        f"MST={cfg.use_mst}, "
        f"LightGNN={cfg.use_light_gnn}, "
        f"CE={cfg.use_cross_modal_enhancement}, "
        f"fusion_mode={cfg.fusion_mode}"
    )
    print(
        "Training objective: "
        f"bidirectional InfoNCE, temperature={cfg.temperature:.2f}, "
        "in-batch non-matching entities as negatives, "
        f"structure_loss_weight={cfg.structure_loss_weight:.2f}"
    )
    print(
        "Structural encoder switches: "
        f"encoder={cfg.structure_encoder}, "
        f"initialization={cfg.structure_initialization}, "
        f"relation_gnn={cfg.use_relation_gnn and num_relations > 0}, "
        f"relation_types={cfg.use_relation_types and cfg.use_relation_gnn and num_relations > 0}, "
        f"layer_fusion={cfg.relation_layer_fusion and cfg.use_relation_gnn and num_relations > 0}, "
        f"shared_gnn={cfg.gnn_share_parameters}, "
        f"depthwise_separable={cfg.gnn_use_depthwise_separable}"
    )
    print(
        "Graph protocol switches: "
        f"shared_cross_graph_relations={cfg.share_cross_graph_relations}, "
        f"reverse_edges={cfg.add_reverse_edges}"
    )
    print(
        "Semantic encoder switches: "
        f"token_view={cfg.sem_use_token_view}, "
        f"phrase_view={cfg.sem_use_phrase_view}, "
        f"global_view={cfg.sem_use_global_view}, "
        f"residual_mode={cfg.sem_residual_mode}"
    )
    if cfg.num_neighbors > 0:
        print(
            "Neighbor collection: "
            f"deterministic fixed budget, k={cfg.num_neighbors}"
        )
    else:
        print("Neighbor collection: all outgoing neighbors, batch-local padding")
    print(
        "Alignment scorer: "
        f"temperature={cfg.temperature:.2f}, "
        f"csls_k={cfg.alignment_csls_k}, "
        f"csls_blend={cfg.alignment_csls_blend:.2f}"
    )
    print(
        "Evaluation protocol: "
        f"full_ranking=True, "
        f"threshold_selection={'val_f1' if val_pairs else 'fixed_default'}, "
        f"checkpoint_selection="
        f"{('val_' + get_validation_selection_metric_name(cfg).lower()) if val_pairs else 'fixed_epoch'}, "
        f"validation_every={cfg.validation_every_epochs}, "
        f"ranking_metric=raw_cosine, "
        f"csls_report={'on' if cfg.alignment_csls_k > 0 and cfg.alignment_csls_blend > 0.0 else 'off'}"
    )
    if cfg.select_csls_on_validation:
        print(
            "CSLS validation selection: "
            f"metric={cfg.csls_selection_metric}, "
            f"k_candidates={parse_int_candidates(cfg.csls_k_candidates)}, "
            f"blend_candidates={parse_float_candidates(cfg.csls_blend_candidates)}"
        )
    if cfg.fusion_mode in {
        "early_interaction",
        "reciprocal_add",
        "reciprocal_residual_add",
    }:
        print(f"CE residual ratio: {cfg.ce_residual_ratio:.2f}")
    print(
        f"Neighbor query mode: {cfg.neighbor_query_mode}, "
        f"gate mode={cfg.neighbor_gate_mode}, "
        f"semantic_weight={cfg.neighbor_query_semantic_weight:.2f}"
    )
    print(
        "Neighbor attention: "
        f"{cfg.neighbor_attention}, temperature={cfg.neighbor_attention_temperature:.2f}"
    )
    if cfg.use_final_weight_averaging:
        print(
            f"Final weight strategy: average_last_k "
            f"(k={cfg.weight_average_last_k})"
        )
    model = build_model(cfg, total_nodes=total_nodes, num_relations=num_relations).to(device)
    if cfg.share_seed_structure_embeddings:
        model.set_structure_seed_pairs(train_pairs)
    if cfg.structure_initialization == "topology":
        topology_features = build_topology_features(
            edge_index=data["edge_index"],
            edge_type=data["edge_type"],
            num_nodes=total_nodes,
            graph_sizes=(data["n1"], data["n2"]),
        )
        model.set_structure_topology_features(topology_features.to(device))
    if cfg.structure_initialization == "name":
        if cfg.dataset_family.strip().lower() == "dbp15k_raw":
            valid = (seq_features.abs().sum(dim=-1) > 0).float().unsqueeze(-1)
            name_features = (seq_features * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
            name_features = torch.nn.functional.normalize(name_features, p=2, dim=-1)
        else:
            name_features = build_entity_name_features(
                entity_surface_map=data.get("entity_surface_map", {}),
                total_nodes=total_nodes,
                glove_path=cfg.glove_path,
                dim=cfg.text_input_dim,
            )
        model.set_structure_name_features(name_features.to(device))
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

    best_path = get_checkpoint_path(cfg)
    validation_selection_metric_name = get_validation_selection_metric_name(cfg)
    best_validation_metric = float("-inf")
    best_acceptance_threshold = cfg.eval_acceptance_threshold
    early_stop_counter = 0
    weight_average_history = []

    print("\n===== Training =====")
    for epoch in range(1, cfg.joint_epochs + 1):
        train_loader = make_loader(train_pairs, cfg.batch_size, shuffle=True)
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
            epoch=epoch,
        )

        should_validate = (
            bool(val_pairs)
            and (
                epoch % max(1, cfg.validation_every_epochs) == 0
                or epoch == cfg.joint_epochs
            )
        )
        if should_validate:
            metrics = evaluate_alignment(
                model=model,
                edge_index=edge_index,
                edge_type=edge_type,
                seq_features=seq_features,
                adj_list=adj_list,
                num_neighbors=cfg.num_neighbors,
                test_pairs=val_pairs,
                candidate_right_ids=val_candidate_right_ids,
                batch_size=cfg.eval_batch_size,
                device=device,
                calibrate_threshold=True,
            )

            print(
                f"[Joint ] Epoch {epoch:03d} | "
                f"Loss: {train_stats['loss']:.4f} | "
                f"Align: {train_stats['align_loss']:.4f} | "
                f"StructAlign: {train_stats['structure_align_loss']:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                f"ValP: {metrics['Precision']:.4f} | "
                f"ValR: {metrics['Recall']:.4f} | "
                f"ValF1: {metrics['F1']:.4f} | "
                f"ValThr: {metrics['Threshold']:.4f} | "
                f"ValHits@1: {metrics['Hits@1']:.4f} | "
                f"ValHits@5: {metrics['Hits@5']:.4f} | "
                f"ValHits@10: {metrics['Hits@10']:.4f} | "
                f"ValMRR: {metrics['MRR']:.4f}"
                + (
                    f" | ValCSLSMRR: {metrics['CSLSMRR']:.4f}"
                    if "CSLSMRR" in metrics
                    else ""
                )
            )

            current_validation_metric = float(metrics[validation_selection_metric_name])
            if current_validation_metric > best_validation_metric + cfg.early_stop_min_delta:
                best_validation_metric = current_validation_metric
                best_acceptance_threshold = metrics["Threshold"]
                early_stop_counter = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": cfg.__dict__,
                        "best_validation_metric": best_validation_metric,
                        "best_validation_metric_name": validation_selection_metric_name,
                        "best_acceptance_threshold": best_acceptance_threshold,
                    },
                    best_path,
                )
                print(f"Saved best model to {best_path}")
            else:
                early_stop_counter += 1

            if cfg.use_early_stopping and early_stop_counter >= cfg.early_stop_patience:
                print(
                    f"Early stopping triggered at joint epoch {epoch:03d} "
                    f"(best {validation_selection_metric_name}={best_validation_metric:.4f}, "
                    f"threshold={best_acceptance_threshold:.4f})"
                )
                break
        else:
            print(
                f"[Joint ] Epoch {epoch:03d} | "
                f"Loss: {train_stats['loss']:.4f} | "
                f"Align: {train_stats['align_loss']:.4f} | "
                f"StructAlign: {train_stats['structure_align_loss']:.4f} | "
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
    elif not val_pairs:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": cfg.__dict__,
                "best_validation_metric": best_validation_metric,
                "best_validation_metric_name": validation_selection_metric_name,
                "best_acceptance_threshold": best_acceptance_threshold,
            },
            best_path,
        )
        print(f"Saved final model to {best_path}")

    if val_pairs and os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_acceptance_threshold = float(
            checkpoint.get("best_acceptance_threshold", best_acceptance_threshold)
        )
        print(f"Loaded best validation checkpoint from {best_path}")

    if cfg.select_csls_on_validation:
        if not val_pairs:
            raise ValueError("CSLS validation selection requires non-empty validation pairs")
        csls_selection = select_csls_parameters(
            model=model,
            edge_index=edge_index,
            edge_type=edge_type,
            seq_features=seq_features,
            adj_list=adj_list,
            num_neighbors=cfg.num_neighbors,
            validation_pairs=val_pairs,
            candidate_right_ids=val_candidate_right_ids,
            batch_size=cfg.eval_batch_size,
            device=device,
            csls_k_candidates=parse_int_candidates(cfg.csls_k_candidates),
            csls_blend_candidates=parse_float_candidates(cfg.csls_blend_candidates),
            selection_metric=cfg.csls_selection_metric,
        )
        for row in csls_selection["grid"]:
            print(
                "[CSLS  ] "
                f"k={row['k']:02d} | "
                f"blend={row['blend']:.2f} | "
                f"ValHits@1={row['Hits@1']:.4f} | "
                f"ValHits@10={row['Hits@10']:.4f} | "
                f"ValMRR={row['MRR']:.4f}"
            )
        cfg.alignment_csls_k = csls_selection["best_k"]
        cfg.alignment_csls_blend = csls_selection["best_blend"]
        model.align_head.csls_k = cfg.alignment_csls_k
        model.align_head.csls_blend = cfg.alignment_csls_blend
        selected_metrics = csls_selection["best_metrics"]
        print(
            "Selected CSLS parameters on validation | "
            f"k: {cfg.alignment_csls_k} | "
            f"blend: {cfg.alignment_csls_blend:.2f} | "
            f"metric: {csls_selection['selection_metric']} | "
            f"ValHits@1: {selected_metrics['Hits@1']:.4f} | "
            f"ValHits@10: {selected_metrics['Hits@10']:.4f} | "
            f"ValMRR: {selected_metrics['MRR']:.4f}"
        )

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
        acceptance_threshold=best_acceptance_threshold,
    )
    print(
        (
            "Final test metrics from best validation checkpoint | "
            if val_pairs else
            "Final test metrics after fixed-epoch training | "
        ) +
        f"Precision: {final_test_metrics['Precision']:.4f} | "
        f"Recall: {final_test_metrics['Recall']:.4f} | "
        f"F1: {final_test_metrics['F1']:.4f} | "
        f"Threshold: {final_test_metrics['Threshold']:.4f} | "
        f"Hits@1: {final_test_metrics['Hits@1']:.4f} | "
        f"Hits@5: {final_test_metrics['Hits@5']:.4f} | "
        f"Hits@10: {final_test_metrics['Hits@10']:.4f} | "
        f"MRR: {final_test_metrics['MRR']:.4f}"
    )
    if "CSLSMRR" in final_test_metrics:
        print(
            "Primary CSLS ranking metrics | "
            f"Hits@1: {final_test_metrics['CSLSHits@1']:.4f} | "
            f"Hits@5: {final_test_metrics['CSLSHits@5']:.4f} | "
            f"Hits@10: {final_test_metrics['CSLSHits@10']:.4f} | "
            f"MRR: {final_test_metrics['CSLSMRR']:.4f}"
        )
if __name__ == "__main__":
    main()
