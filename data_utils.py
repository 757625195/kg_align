import os
import re
import hashlib
from urllib.parse import unquote
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.datasets import DBP15K


def load_dbp15k_from_pyg(root: str = "data/dbp15k", pair: str = "zh_en") -> Dict:
    dataset = DBP15K(root=root, pair=pair)
    data = dataset[0]

    x1 = data.x1.float()                         # [N1, L1, D]
    x2 = data.x2.float()                         # [N2, L2, D]
    edge_index1 = data.edge_index1.long()       # [2, E1]
    edge_index2 = data.edge_index2.long()       # [2, E2]
    train_y = data.train_y.long()               # [2, T]
    test_y = data.test_y.long()                 # [2, U]

    n1 = x1.size(0)
    n2 = x2.size(0)
    total_nodes = n1 + n2

    # 统一图 id：KG2 整体偏移 n1
    edge_index2_global = edge_index2 + n1
    edge_index = torch.cat([edge_index1, edge_index2_global], dim=1)

    # 统一语义特征表 [N1+N2, L, D]
    # PyG 中两个图的序列长度通常一致；这里稳妥起见按最大长度 pad
    l1, d1 = x1.size(1), x1.size(2)
    l2, d2 = x2.size(1), x2.size(2)
    if d1 != d2:
        raise ValueError(f"x1 dim={d1} and x2 dim={d2} mismatch")

    max_len = max(l1, l2)

    if l1 < max_len:
        pad = torch.zeros(n1, max_len - l1, d1, dtype=x1.dtype)
        x1 = torch.cat([x1, pad], dim=1)

    if l2 < max_len:
        pad = torch.zeros(n2, max_len - l2, d2, dtype=x2.dtype)
        x2 = torch.cat([x2, pad], dim=1)

    seq_features = torch.cat([x1, x2], dim=0)   # [N, L, D]

    train_pairs = [
        (int(train_y[0, i].item()), int(train_y[1, i].item() + n1))
        for i in range(train_y.size(1))
    ]
    test_pairs = [
        (int(test_y[0, i].item()), int(test_y[1, i].item() + n1))
        for i in range(test_y.size(1))
    ]

    # 为了兼容旧代码接口，提供全局 id -> 语义序列
    seq_feature_map = {i: seq_features[i] for i in range(total_nodes)}

    return {
        "total_nodes": total_nodes,
        "n1": n1,
        "n2": n2,
        "edge_index": edge_index,
        "edge_type": None,
        "num_relations": 0,
        "seq_features": seq_features,
        "seq_feature_map": seq_feature_map,
        "entity_uri_map": {},
        "entity_surface_map": {},
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "raw_data": data,
    }


def _read_ent_id_mapping(path: str) -> Dict[int, int]:
    raw_to_local = {}
    with open(path, "r", encoding="utf-8") as f:
        for local_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            raw_id_str, _ = line.split("\t", 1)
            raw_to_local[int(raw_id_str)] = local_idx
    return raw_to_local


def _read_ent_id_records(path: str) -> Tuple[Dict[int, int], Dict[int, str], Dict[int, str]]:
    raw_to_local: Dict[int, int] = {}
    local_to_uri: Dict[int, str] = {}
    local_to_surface: Dict[int, str] = {}
    local_idx = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_id_str, entity_uri = line.split("\t", 1)
            raw_id = int(raw_id_str)
            raw_to_local[raw_id] = local_idx
            local_to_uri[local_idx] = entity_uri
            local_to_surface[local_idx] = _surface_from_identifier(entity_uri)
            local_idx += 1
    return raw_to_local, local_to_uri, local_to_surface


def _read_relation_vocab(path: str) -> Dict[int, str]:
    rel_vocab = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel_id_str, rel_name = line.split("\t", 1)
            rel_vocab[int(rel_id_str)] = rel_name
    return rel_vocab


def _canonicalize_relation_name(name: str) -> str:
    if "dbpedia.org" in name:
        return name.split("dbpedia.org", 1)[1]
    return name


def _read_alignment_pairs(path: str) -> List[Tuple[int, int]]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            left_id_str, right_id_str = line.split("\t")
            pairs.append((int(left_id_str), int(right_id_str)))
    return pairs


def _find(parent: Dict[Tuple[str, int], Tuple[str, int]], node: Tuple[str, int]) -> Tuple[str, int]:
    root = parent[node]
    if root != node:
        parent[node] = _find(parent, root)
    return parent[node]


def _union(
    parent: Dict[Tuple[str, int], Tuple[str, int]],
    a: Tuple[str, int],
    b: Tuple[str, int],
) -> None:
    root_a = _find(parent, a)
    root_b = _find(parent, b)
    if root_a != root_b:
        parent[root_b] = root_a


def _build_relation_id_maps(
    left_rel_path: str,
    right_rel_path: str,
    sup_rel_path: Optional[str] = None,
    share_cross_graph_relations: bool = True,
) -> Tuple[Dict[int, int], Dict[int, int], int]:
    left_vocab = _read_relation_vocab(left_rel_path)
    right_vocab = _read_relation_vocab(right_rel_path)

    parent = {
        ("left", rel_id): ("left", rel_id)
        for rel_id in left_vocab
    }
    parent.update({
        ("right", rel_id): ("right", rel_id)
        for rel_id in right_vocab
    })

    if share_cross_graph_relations:
        canonical_representatives: Dict[str, Tuple[str, int]] = {}
        for side, vocab in (("left", left_vocab), ("right", right_vocab)):
            for rel_id, rel_name in vocab.items():
                node = (side, rel_id)
                canonical_name = _canonicalize_relation_name(rel_name)
                if canonical_name in canonical_representatives:
                    _union(parent, node, canonical_representatives[canonical_name])
                else:
                    canonical_representatives[canonical_name] = node

    if (
        share_cross_graph_relations
        and sup_rel_path is not None
        and os.path.exists(sup_rel_path)
    ):
        for left_rel_id, right_rel_id in _read_alignment_pairs(sup_rel_path):
            left_node = ("left", left_rel_id)
            right_node = ("right", right_rel_id)
            if left_node in parent and right_node in parent:
                _union(parent, left_node, right_node)

    root_to_global = {}
    left_rel_to_global = {}
    right_rel_to_global = {}

    for rel_id in left_vocab:
        root = _find(parent, ("left", rel_id))
        if root not in root_to_global:
            root_to_global[root] = len(root_to_global)
        left_rel_to_global[rel_id] = root_to_global[root]

    for rel_id in right_vocab:
        root = _find(parent, ("right", rel_id))
        if root not in root_to_global:
            root_to_global[root] = len(root_to_global)
        right_rel_to_global[rel_id] = root_to_global[root]

    return left_rel_to_global, right_rel_to_global, len(root_to_global)


def _read_relation_aware_triples(
    path: str,
    ent_raw_to_local: Dict[int, int],
    rel_raw_to_global: Dict[int, int],
    node_offset: int,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    edges = []
    edge_types = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            head_raw_str, rel_raw_str, tail_raw_str = line.split("\t")
            head_raw = int(head_raw_str)
            rel_raw = int(rel_raw_str)
            tail_raw = int(tail_raw_str)
            if head_raw not in ent_raw_to_local or tail_raw not in ent_raw_to_local:
                continue
            if rel_raw not in rel_raw_to_global:
                continue
            edges.append((
                ent_raw_to_local[head_raw] + node_offset,
                ent_raw_to_local[tail_raw] + node_offset,
            ))
            edge_types.append(rel_raw_to_global[rel_raw])
    return edges, edge_types


def _load_relation_aware_graph(
    left_ent_path: str,
    right_ent_path: str,
    triples1_path: str,
    triples2_path: str,
    left_rel_path: Optional[str],
    right_rel_path: Optional[str],
    sup_rel_path: Optional[str],
    right_global_offset: int,
    share_cross_graph_relations: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    required_paths = [
        left_ent_path,
        right_ent_path,
        triples1_path,
        triples2_path,
        left_rel_path,
        right_rel_path,
    ]
    if any(path is None or not os.path.exists(path) for path in required_paths):
        return None, None, 0

    left_ent_map = _read_ent_id_mapping(left_ent_path)
    right_ent_map = _read_ent_id_mapping(right_ent_path)
    left_rel_map, right_rel_map, num_relations = _build_relation_id_maps(
        left_rel_path=left_rel_path,
        right_rel_path=right_rel_path,
        sup_rel_path=sup_rel_path,
        share_cross_graph_relations=share_cross_graph_relations,
    )

    edges1, edge_types1 = _read_relation_aware_triples(
        path=triples1_path,
        ent_raw_to_local=left_ent_map,
        rel_raw_to_global=left_rel_map,
        node_offset=0,
    )
    edges2, edge_types2 = _read_relation_aware_triples(
        path=triples2_path,
        ent_raw_to_local=right_ent_map,
        rel_raw_to_global=right_rel_map,
        node_offset=right_global_offset,
    )

    all_edges = edges1 + edges2
    all_edge_types = edge_types1 + edge_types2
    if not all_edges:
        return None, None, 0

    edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(all_edge_types, dtype=torch.long)
    return edge_index, edge_type, num_relations


def _read_pair_ids(
    path: str,
    left_raw_to_local: Dict[int, int],
    right_raw_to_local: Dict[int, int],
    right_global_offset: int,
) -> List[Tuple[int, int]]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            left_raw_str, right_raw_str = line.split("\t")
            left_raw = int(left_raw_str)
            right_raw = int(right_raw_str)
            left_local = left_raw_to_local[left_raw]
            right_local = right_raw_to_local[right_raw]
            pairs.append((left_local, right_local + right_global_offset))
    return pairs


def _read_positive_examples(
    path: str,
    left_raw_to_local: Dict[int, int],
    right_raw_to_local: Dict[int, int],
    right_global_offset: int,
) -> List[Tuple[int, int]]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            left_raw_str, right_raw_str, label_str = line.split("\t")
            if int(label_str) != 1:
                continue
            left_raw = int(left_raw_str)
            right_raw = int(right_raw_str)
            left_local = left_raw_to_local[left_raw]
            right_local = right_raw_to_local[right_raw]
            pairs.append((left_local, right_local + right_global_offset))
    return pairs


def _read_string_triples(path: str) -> List[Tuple[str, str, str]]:
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            head, rel, tail = line.split("\t", 2)
            triples.append((head, rel, tail))
    return triples


def _read_string_alignment_pairs(path: str) -> List[Tuple[str, str]]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            left_id, right_id = line.split("\t", 1)
            pairs.append((left_id, right_id))
    return pairs


def _surface_from_identifier(value: str) -> str:
    text = unquote(value.strip())
    if text.startswith("http://") or text.startswith("https://"):
        text = text.rsplit("/", 1)[-1]
        text = text.rsplit("#", 1)[-1]
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"['\"`]", " ", text)
    text = re.sub(r"[^0-9A-Za-z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _surface_from_literal(value: str) -> str:
    text = value.strip()
    if text.startswith('"') and '"' in text[1:]:
        end = text.rfind('"')
        if end > 0:
            text = text[1:end]
    text = unquote(text)
    text = text.replace("_", " ")
    text = re.sub(r"[^0-9A-Za-z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _tokenize_text(text: str) -> List[str]:
    if not text:
        return []
    return [token for token in text.split() if token]


def _append_tokens(
    token_store: Dict[str, List[str]],
    entity_id: str,
    text: str,
    limit: int,
) -> None:
    if limit <= 0:
        return
    tokens = _tokenize_text(text)
    if not tokens:
        return
    bucket = token_store.setdefault(entity_id, [])
    remaining = limit - len(bucket)
    if remaining > 0:
        bucket.extend(tokens[:remaining])


def _add_entity(
    entity_id: str,
    mapping: Dict[str, int],
    token_store: Dict[str, List[str]],
    token_limit: int,
) -> None:
    if entity_id not in mapping:
        mapping[entity_id] = len(mapping)
        _append_tokens(
            token_store=token_store,
            entity_id=entity_id,
            text=_surface_from_identifier(entity_id),
            limit=token_limit,
        )


def _hashed_oov_vector(token: str, dim: int = 300) -> torch.Tensor:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    generator = torch.Generator()
    generator.manual_seed(seed)
    vec = torch.randn(dim, generator=generator, dtype=torch.float32)
    return torch.nn.functional.normalize(vec, p=2, dim=0)


def _load_glove_subset(glove_path: str, needed_tokens: set) -> Dict[str, torch.Tensor]:
    token_to_vec: Dict[str, torch.Tensor] = {}
    if not needed_tokens or not os.path.exists(glove_path):
        return token_to_vec

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            token = parts[0]
            if token not in needed_tokens:
                continue
            vector = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
            token_to_vec[token] = vector
    return token_to_vec


def _build_seq_features_from_tokens(
    left_entities: Dict[str, int],
    right_entities: Dict[str, int],
    left_tokens: Dict[str, List[str]],
    right_tokens: Dict[str, List[str]],
    text_max_len: int,
    glove_path: str,
) -> torch.Tensor:
    total_nodes = len(left_entities) + len(right_entities)
    seq_features = torch.zeros(total_nodes, text_max_len, 300, dtype=torch.float32)

    needed_tokens = set()
    for token_list in left_tokens.values():
        needed_tokens.update(token_list[:text_max_len])
    for token_list in right_tokens.values():
        needed_tokens.update(token_list[:text_max_len])

    glove = _load_glove_subset(glove_path=glove_path, needed_tokens=needed_tokens)

    for entity_id, local_id in left_entities.items():
        tokens = left_tokens.get(entity_id, [])[:text_max_len]
        if not tokens:
            tokens = _tokenize_text(_surface_from_identifier(entity_id))[:text_max_len]
        for pos, token in enumerate(tokens):
            seq_features[local_id, pos] = glove.get(token, _hashed_oov_vector(token))

    offset = len(left_entities)
    for entity_id, local_id in right_entities.items():
        tokens = right_tokens.get(entity_id, [])[:text_max_len]
        if not tokens:
            tokens = _tokenize_text(_surface_from_identifier(entity_id))[:text_max_len]
        for pos, token in enumerate(tokens):
            seq_features[offset + local_id, pos] = glove.get(token, _hashed_oov_vector(token))

    return seq_features


def build_entity_name_features(
    entity_surface_map: Dict[int, str],
    total_nodes: int,
    glove_path: str,
    dim: int = 300,
) -> torch.Tensor:
    """Build fixed mean-pooled name vectors without relation/attribute tokens."""
    token_map = {
        entity_id: _tokenize_text(surface)
        for entity_id, surface in entity_surface_map.items()
    }
    needed_tokens = {token for tokens in token_map.values() for token in tokens}
    glove = _load_glove_subset(glove_path, needed_tokens)
    features = torch.zeros(total_nodes, dim, dtype=torch.float32)
    for entity_id in range(total_nodes):
        tokens = token_map.get(entity_id, [])
        if not tokens:
            continue
        vectors = [glove.get(token, _hashed_oov_vector(token, dim)) for token in tokens]
        features[entity_id] = torch.stack(vectors).mean(dim=0)
    return torch.nn.functional.normalize(features, p=2, dim=-1)


def _pairs_from_string_links(
    pairs: List[Tuple[str, str]],
    left_entities: Dict[str, int],
    right_entities: Dict[str, int],
    right_global_offset: int,
) -> List[Tuple[int, int]]:
    mapped = []
    for left_id, right_id in pairs:
        if left_id not in left_entities or right_id not in right_entities:
            continue
        mapped.append((left_entities[left_id], right_entities[right_id] + right_global_offset))
    return mapped


def _format_dir_preview(path: str, limit: int = 12) -> str:
    if not os.path.exists(path):
        return "(missing)"
    if not os.path.isdir(path):
        return "(not a directory)"
    try:
        entries = sorted(os.listdir(path))
    except OSError as exc:
        return f"(unreadable: {exc})"
    if not entries:
        return "(empty)"
    preview = entries[:limit]
    if len(entries) > limit:
        preview.append(f"... (+{len(entries) - limit} more)")
    return ", ".join(preview)


def _resolve_openea_dataset_dir(root: str, pair: str) -> str:
    normalized_root = os.path.normpath(root)
    candidates = []
    if os.path.basename(normalized_root) == pair:
        candidates.append(root)
    candidates.append(os.path.join(root, pair))

    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "rel_triples_1")):
            return candidate
    return candidates[0]


def load_openea_dataset(
    root: str = "data/openea",
    pair: str = "EN_FR_15K_V2",
    split: str = "721_5fold/1",
    glove_path: Optional[str] = None,
    text_max_len: int = 32,
    share_cross_graph_relations: bool = True,
) -> Dict:
    dataset_dir = _resolve_openea_dataset_dir(root, pair)
    split_dir = os.path.join(dataset_dir, split)
    if glove_path is None:
        glove_path = os.path.join("data", "dbp15k", "raw", "sub.glove.300d")

    processed_dir = os.path.join(root, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    cache_stem = f"{pair}_{split.replace('/', '_')}_len{text_max_len}"
    cache_name = (
        f"{cache_stem}.pt"
        if share_cross_graph_relations
        else f"{cache_stem}_reldisjoint.pt"
    )
    cache_path = os.path.join(processed_dir, cache_name)
    disable_cache = (
        os.environ.get("KG_ALIGN_DISABLE_DATA_CACHE", "0").strip().lower()
        in {"1", "true", "yes", "on"}
        or "eventea" in os.path.normcase(root)
        or not share_cross_graph_relations
    )
    if not disable_cache and os.path.exists(cache_path):
        cached = torch.load(cache_path, map_location="cpu")
        if "entity_surface_map" in cached:
            return cached

    rel_triples_1_path = os.path.join(dataset_dir, "rel_triples_1")
    rel_triples_2_path = os.path.join(dataset_dir, "rel_triples_2")
    attr_triples_1_path = os.path.join(dataset_dir, "attr_triples_1")
    attr_triples_2_path = os.path.join(dataset_dir, "attr_triples_2")
    name_list_1_path = os.path.join(dataset_dir, "name_list_1")
    name_list_2_path = os.path.join(dataset_dir, "name_list_2")
    ent_links_path = os.path.join(dataset_dir, "ent_links")
    train_links_path = os.path.join(split_dir, "train_links")
    valid_links_path = os.path.join(split_dir, "valid_links")
    test_links_path = os.path.join(split_dir, "test_links")

    required_paths = [
        rel_triples_1_path,
        rel_triples_2_path,
        train_links_path,
        valid_links_path,
        test_links_path,
    ]
    missing_paths = [path for path in required_paths if not os.path.exists(path)]
    if missing_paths:
        missing_names = ", ".join(os.path.relpath(path, dataset_dir) for path in missing_paths)
        raise FileNotFoundError(
            "OpenEA dataset files not found.\n"
            f"  root: {root}\n"
            f"  pair: {pair}\n"
            f"  resolved dataset_dir: {dataset_dir}\n"
            f"  split: {split}\n"
            f"  missing: {missing_names}\n"
            f"  root entries: {_format_dir_preview(root)}\n"
            f"  dataset_dir entries: {_format_dir_preview(dataset_dir)}\n"
            f"  split_dir entries: {_format_dir_preview(split_dir)}\n"
            "Expected layout:\n"
            f"  {dataset_dir}/rel_triples_1\n"
            f"  {dataset_dir}/rel_triples_2\n"
            f"  {dataset_dir}/attr_triples_1  (optional)\n"
            f"  {dataset_dir}/attr_triples_2  (optional)\n"
            f"  {dataset_dir}/name_list_1     (optional)\n"
            f"  {dataset_dir}/name_list_2     (optional)\n"
            f"  {dataset_dir}/ent_links       (optional)\n"
            f"  {split_dir}/train_links\n"
            f"  {split_dir}/valid_links\n"
            f"  {split_dir}/test_links\n"
            "If KG_ALIGN_ROOT already points at the pair directory itself, keep KG_ALIGN_PAIR unchanged; "
            "the loader will accept either root=<...>/openea with pair=EN_FR_15K_V2 or "
            "root=<...>/EN_FR_15K_V2 with pair=EN_FR_15K_V2."
        )

    rel_triples_1 = _read_string_triples(rel_triples_1_path)
    rel_triples_2 = _read_string_triples(rel_triples_2_path)
    attr_triples_1 = _read_string_triples(attr_triples_1_path) if os.path.exists(attr_triples_1_path) else []
    attr_triples_2 = _read_string_triples(attr_triples_2_path) if os.path.exists(attr_triples_2_path) else []
    name_list_1 = _read_string_alignment_pairs(name_list_1_path) if os.path.exists(name_list_1_path) else []
    name_list_2 = _read_string_alignment_pairs(name_list_2_path) if os.path.exists(name_list_2_path) else []
    left_name_map = dict(name_list_1)
    right_name_map = dict(name_list_2)
    train_links = _read_string_alignment_pairs(train_links_path)
    valid_links = _read_string_alignment_pairs(valid_links_path)
    test_links = _read_string_alignment_pairs(test_links_path)
    ent_links = _read_string_alignment_pairs(ent_links_path) if os.path.exists(ent_links_path) else []

    left_entities: Dict[str, int] = {}
    right_entities: Dict[str, int] = {}
    left_tokens: Dict[str, List[str]] = {}
    right_tokens: Dict[str, List[str]] = {}
    token_limit = max(text_max_len * 4, text_max_len)

    # EventEA follows the OpenEA triple/link layout but also provides explicit
    # entity names. Add them before relation and attribute text so names retain
    # priority when the semantic token budget is truncated.
    for entity_id, name in name_list_1:
        _append_tokens(left_tokens, entity_id, _surface_from_literal(name), token_limit)
        _add_entity(entity_id, left_entities, left_tokens, token_limit)

    for entity_id, name in name_list_2:
        _append_tokens(right_tokens, entity_id, _surface_from_literal(name), token_limit)
        _add_entity(entity_id, right_entities, right_tokens, token_limit)

    for head, rel, tail in rel_triples_1:
        _add_entity(head, left_entities, left_tokens, token_limit)
        _add_entity(tail, left_entities, left_tokens, token_limit)
        _append_tokens(left_tokens, head, _surface_from_identifier(rel), token_limit)

    for head, rel, tail in rel_triples_2:
        _add_entity(head, right_entities, right_tokens, token_limit)
        _add_entity(tail, right_entities, right_tokens, token_limit)
        _append_tokens(right_tokens, head, _surface_from_identifier(rel), token_limit)

    for head, attr, value in attr_triples_1:
        _add_entity(head, left_entities, left_tokens, token_limit)
        _append_tokens(left_tokens, head, _surface_from_identifier(attr), token_limit)
        _append_tokens(left_tokens, head, _surface_from_literal(value), token_limit)

    for head, attr, value in attr_triples_2:
        _add_entity(head, right_entities, right_tokens, token_limit)
        _append_tokens(right_tokens, head, _surface_from_identifier(attr), token_limit)
        _append_tokens(right_tokens, head, _surface_from_literal(value), token_limit)

    for left_id, right_id in ent_links + train_links + valid_links + test_links:
        _add_entity(left_id, left_entities, left_tokens, token_limit)
        _add_entity(right_id, right_entities, right_tokens, token_limit)

    n1 = len(left_entities)
    n2 = len(right_entities)
    total_nodes = n1 + n2
    right_offset = n1

    relation_to_global: Dict[Tuple[str, str], int] = {}

    def get_relation_id(side: str, rel_name: str) -> int:
        canonical = _canonicalize_relation_name(rel_name)
        key = ("shared", canonical) if share_cross_graph_relations else (side, canonical)
        if key not in relation_to_global:
            relation_to_global[key] = len(relation_to_global)
        return relation_to_global[key]

    edges = []
    edge_types = []
    for head, rel, tail in rel_triples_1:
        if head not in left_entities or tail not in left_entities:
            continue
        edges.append((left_entities[head], left_entities[tail]))
        edge_types.append(get_relation_id("left", rel))

    for head, rel, tail in rel_triples_2:
        if head not in right_entities or tail not in right_entities:
            continue
        edges.append((right_entities[head] + right_offset, right_entities[tail] + right_offset))
        edge_types.append(get_relation_id("right", rel))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    seq_features = _build_seq_features_from_tokens(
        left_entities=left_entities,
        right_entities=right_entities,
        left_tokens=left_tokens,
        right_tokens=right_tokens,
        text_max_len=text_max_len,
        glove_path=glove_path,
    )
    train_pairs = _pairs_from_string_links(
        pairs=train_links,
        left_entities=left_entities,
        right_entities=right_entities,
        right_global_offset=right_offset,
    )
    val_pairs = _pairs_from_string_links(
        pairs=valid_links,
        left_entities=left_entities,
        right_entities=right_entities,
        right_global_offset=right_offset,
    )
    test_pairs = _pairs_from_string_links(
        pairs=test_links,
        left_entities=left_entities,
        right_entities=right_entities,
        right_global_offset=right_offset,
    )

    payload = {
        "total_nodes": total_nodes,
        "n1": n1,
        "n2": n2,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "num_relations": len(relation_to_global),
        "seq_features": seq_features,
        "seq_feature_map": {i: seq_features[i] for i in range(total_nodes)},
        "entity_uri_map": {
            **{local_id: entity_id for entity_id, local_id in left_entities.items()},
            **{
                right_offset + local_id: entity_id
                for entity_id, local_id in right_entities.items()
            },
        },
        "entity_surface_map": {
            **{
                local_id: _surface_from_literal(left_name_map.get(
                    entity_id,
                    _surface_from_identifier(entity_id),
                ))
                for entity_id, local_id in left_entities.items()
            },
            **{
                right_offset + local_id: _surface_from_literal(right_name_map.get(
                    entity_id,
                    _surface_from_identifier(entity_id),
                ))
                for entity_id, local_id in right_entities.items()
            },
        },
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "test_pairs": test_pairs,
        "dataset_family": "openea",
        "dataset_name": pair,
        "dataset_split": split,
        "glove_path": glove_path,
        "text_max_len": text_max_len,
    }
    if not disable_cache:
        torch.save(payload, cache_path)
    return payload


def load_dbp15k_raw_split(
    root: str = "data/dbp15k",
    pair: str = "zh_en",
    split: str = "0_3",
    share_cross_graph_relations: bool = True,
) -> Dict:
    base = load_dbp15k_from_pyg(root=root, pair=pair)

    pair_dir = os.path.join(root, pair)
    split_dir = os.path.join(pair_dir, split)

    left_map, left_uri_map, left_surface_map = _read_ent_id_records(os.path.join(split_dir, "ent_ids_1"))
    right_map, right_uri_map, right_surface_map = _read_ent_id_records(os.path.join(split_dir, "ent_ids_2"))

    train_pairs = _read_pair_ids(
        path=os.path.join(split_dir, "sup_ent_ids"),
        left_raw_to_local=left_map,
        right_raw_to_local=right_map,
        right_global_offset=base["n1"],
    )
    test_pairs = _read_pair_ids(
        path=os.path.join(split_dir, "ref_ent_ids"),
        left_raw_to_local=left_map,
        right_raw_to_local=right_map,
        right_global_offset=base["n1"],
    )

    base["train_pairs"] = train_pairs
    base["test_pairs"] = test_pairs
    base["raw_split"] = split
    base["entity_uri_map"] = {
        **{local_id: entity_uri for local_id, entity_uri in left_uri_map.items()},
        **{
            base["n1"] + local_id: entity_uri
            for local_id, entity_uri in right_uri_map.items()
        },
    }
    base["entity_surface_map"] = {
        **{local_id: surface for local_id, surface in left_surface_map.items()},
        **{
            base["n1"] + local_id: surface
            for local_id, surface in right_surface_map.items()
        },
    }
    relation_edge_index, relation_edge_type, num_relations = _load_relation_aware_graph(
        left_ent_path=os.path.join(split_dir, "ent_ids_1"),
        right_ent_path=os.path.join(split_dir, "ent_ids_2"),
        triples1_path=os.path.join(split_dir, "triples_1"),
        triples2_path=os.path.join(split_dir, "triples_2"),
        left_rel_path=os.path.join(split_dir, "rel_ids_1"),
        right_rel_path=os.path.join(split_dir, "rel_ids_2"),
        sup_rel_path=os.path.join(split_dir, "sup_rel_ids"),
        right_global_offset=base["n1"],
        share_cross_graph_relations=share_cross_graph_relations,
    )
    if relation_edge_index is not None and relation_edge_type is not None:
        base["edge_index"] = relation_edge_index
        base["edge_type"] = relation_edge_type
        base["num_relations"] = num_relations
    return base
