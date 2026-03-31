import os
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

    canonical_representatives: Dict[str, Tuple[str, int]] = {}
    for side, vocab in (("left", left_vocab), ("right", right_vocab)):
        for rel_id, rel_name in vocab.items():
            node = (side, rel_id)
            canonical_name = _canonicalize_relation_name(rel_name)
            if canonical_name in canonical_representatives:
                _union(parent, node, canonical_representatives[canonical_name])
            else:
                canonical_representatives[canonical_name] = node

    if sup_rel_path is not None and os.path.exists(sup_rel_path):
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


def load_dbp15k_raw_split(
    root: str = "data/dbp15k",
    pair: str = "zh_en",
    split: str = "0_3",
) -> Dict:
    base = load_dbp15k_from_pyg(root=root, pair=pair)

    pair_dir = os.path.join(root, pair)
    split_dir = os.path.join(pair_dir, split)

    left_map = _read_ent_id_mapping(os.path.join(split_dir, "ent_ids_1"))
    right_map = _read_ent_id_mapping(os.path.join(split_dir, "ent_ids_2"))

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
    relation_edge_index, relation_edge_type, num_relations = _load_relation_aware_graph(
        left_ent_path=os.path.join(split_dir, "ent_ids_1"),
        right_ent_path=os.path.join(split_dir, "ent_ids_2"),
        triples1_path=os.path.join(split_dir, "triples_1"),
        triples2_path=os.path.join(split_dir, "triples_2"),
        left_rel_path=os.path.join(split_dir, "rel_ids_1"),
        right_rel_path=os.path.join(split_dir, "rel_ids_2"),
        sup_rel_path=os.path.join(split_dir, "sup_rel_ids"),
        right_global_offset=base["n1"],
    )
    if relation_edge_index is not None and relation_edge_type is not None:
        base["edge_index"] = relation_edge_index
        base["edge_type"] = relation_edge_type
        base["num_relations"] = num_relations
    return base


def load_dbp15k_fixed_eval_split(
    root: str = "data/dbp15k",
    pair: str = "zh_en",
) -> Dict:
    base = load_dbp15k_from_pyg(root=root, pair=pair)

    raw_pair_dir = os.path.join(root, "raw", pair)
    left_map = _read_ent_id_mapping(os.path.join(raw_pair_dir, "ent_ids_1"))
    right_map = _read_ent_id_mapping(os.path.join(raw_pair_dir, "ent_ids_2"))

    train_pairs = _read_positive_examples(
        path=os.path.join(raw_pair_dir, "train.examples.20"),
        left_raw_to_local=left_map,
        right_raw_to_local=right_map,
        right_global_offset=base["n1"],
    )
    val_pairs = _read_positive_examples(
        path=os.path.join(raw_pair_dir, "dev.examples.20"),
        left_raw_to_local=left_map,
        right_raw_to_local=right_map,
        right_global_offset=base["n1"],
    )
    test_pairs = _read_positive_examples(
        path=os.path.join(raw_pair_dir, "test.examples.1000"),
        left_raw_to_local=left_map,
        right_raw_to_local=right_map,
        right_global_offset=base["n1"],
    )

    base["train_pairs"] = train_pairs
    base["val_pairs"] = val_pairs
    base["test_pairs"] = test_pairs
    base["fixed_eval_split"] = True
    return base
