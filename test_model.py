import os
from unittest.mock import patch

import torch

from models.full_model import JointEAModel
from models.fusion import (
    CrossModalFusion,
    ParameterMatchedLateFusion,
    masked_entmax15,
)
from models.gnn_encoder import RDGCNStructuralEncoder, RREAStructuralEncoder, RelationAwareGraphConv
from models.losses import total_loss
from models.text_encoder import MultiScaleTransformerEncoder
from evaluate import select_csls_parameters
from graph_utils import build_topology_features, sample_neighbors
from train import Config, apply_runtime_overrides, build_protocol_splits


def run_mode(fusion_mode: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_nodes = 100
    seq_len = 16
    seq_dim = 300
    batch_size = 8
    num_neighbors = 4

    model = JointEAModel(
        num_nodes=num_nodes,
        text_input_dim=seq_dim,
        node_input_dim=128,
        gnn_hidden_dim=128,
        text_hidden_dim=128,
        fusion_dim=128,
        gnn_layers=2,
        gnn_share_parameters=True,
        gnn_use_depthwise_separable=True,
        text_heads=4,
        text_layers=2,
        dropout=0.1,
        fusion_mode=fusion_mode,
    ).to(device)
    model.eval()

    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 5],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 6, 1],
    ], dtype=torch.long, device=device)

    left_node_ids = torch.randint(0, num_nodes, (batch_size,), device=device)
    right_node_ids = torch.randint(0, num_nodes, (batch_size,), device=device)

    left_seq = torch.randn(batch_size, seq_len, seq_dim, device=device)
    right_seq = torch.randn(batch_size, seq_len, seq_dim, device=device)

    left_neighbor_ids = torch.randint(0, num_nodes, (batch_size, num_neighbors), device=device)
    right_neighbor_ids = torch.randint(0, num_nodes, (batch_size, num_neighbors), device=device)
    left_neighbor_mask = torch.ones(batch_size, num_neighbors, dtype=torch.long, device=device)
    right_neighbor_mask = torch.ones(batch_size, num_neighbors, dtype=torch.long, device=device)

    left_out = model(
        node_ids=left_node_ids,
        edge_index=edge_index,
        seq_features=left_seq,
        neighbor_ids=left_neighbor_ids,
        neighbor_mask=left_neighbor_mask,
    )
    right_out = model(
        node_ids=right_node_ids,
        edge_index=edge_index,
        seq_features=right_seq,
        neighbor_ids=right_neighbor_ids,
        neighbor_mask=right_neighbor_mask,
    )

    if fusion_mode in {
        "late_concat_mlp",
        "late_concat_matched",
        "mean",
        "structure_only",
        "semantic_only",
    }:
        alternate_left_out = model(
            node_ids=left_node_ids,
            edge_index=edge_index,
            seq_features=left_seq,
            neighbor_ids=torch.roll(left_neighbor_ids, shifts=1, dims=1),
            neighbor_mask=left_neighbor_mask,
        )
        assert torch.allclose(
            left_out["z_joint"],
            alternate_left_out["z_joint"],
            atol=1e-6,
        )
    if fusion_mode == "structure_only":
        assert torch.allclose(left_out["z_joint"], left_out["z_struct"], atol=1e-6)
    if fusion_mode == "semantic_only":
        assert torch.allclose(left_out["z_joint"], left_out["z_sem"], atol=1e-6)

    losses = total_loss(left_out, right_out)
    score_matrix = model.score_pairs(left_out["z_joint"], right_out["z_joint"])

    expected = (batch_size, 128)
    assert left_out["z_struct"].shape == expected
    assert left_out["z_sem_enhanced"].shape == expected
    assert left_out["z_joint"].shape == expected
    assert score_matrix.shape == (batch_size, batch_size)
    assert torch.isfinite(losses["loss"])
    assert set(losses) == {"loss", "align_loss", "structure_align_loss"}
    assert torch.allclose(losses["loss"], losses["align_loss"])
    assert losses["structure_align_loss"].item() == 0.0
    assert torch.allclose(
        left_out["z_joint"].norm(dim=-1),
        torch.ones(batch_size, device=device),
        atol=1e-5,
    )
    print(
        f"{fusion_mode}: z_joint={tuple(left_out['z_joint'].shape)}, "
        f"score={tuple(score_matrix.shape)}, loss={losses['loss'].item():.4f}"
    )


def test_internal_refinements():
    torch.manual_seed(42)
    seq_x = torch.randn(6, 12, 300)
    legacy_encoder = MultiScaleTransformerEncoder(
        in_dim=300,
        embed_dim=128,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        residual_mode="legacy_global",
    ).eval()
    gated_encoder = MultiScaleTransformerEncoder(
        in_dim=300,
        embed_dim=128,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        residual_mode="gated",
    ).eval()
    no_global_residual_encoder = MultiScaleTransformerEncoder(
        in_dim=300,
        embed_dim=128,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        residual_mode="gated_no_global",
    ).eval()
    gated_encoder.load_state_dict(legacy_encoder.state_dict())
    no_global_residual_encoder.load_state_dict(legacy_encoder.state_dict())

    with torch.no_grad():
        legacy_out = legacy_encoder(seq_x)
        gated_out = gated_encoder(seq_x)
        no_global_residual_out = no_global_residual_encoder(seq_x)
    assert not torch.allclose(legacy_out, gated_out, atol=1e-6)
    assert not torch.allclose(gated_out, no_global_residual_out, atol=1e-6)
    assert torch.allclose(gated_out.norm(dim=-1), torch.ones(6), atol=1e-5)
    assert torch.allclose(
        no_global_residual_out.norm(dim=-1),
        torch.ones(6),
        atol=1e-5,
    )

    semantic = torch.randn(6, 128)
    structural = torch.randn(6, 128)
    neighbors = torch.randn(6, 4, 128)
    neighbor_mask = torch.ones(6, 4, dtype=torch.long)
    fusion_base = CrossModalFusion(dim=128, dropout=0.0, residual_ratio=0.0).eval()
    fusion_blend = CrossModalFusion(dim=128, dropout=0.0, residual_ratio=0.05).eval()
    fusion_blend.load_state_dict(fusion_base.state_dict())

    with torch.no_grad():
        base_out = fusion_base(semantic, structural, neighbors, neighbor_mask)
        blend_out = fusion_blend(semantic, structural, neighbors, neighbor_mask)
    assert not torch.allclose(base_out, blend_out, atol=1e-6)
    assert torch.allclose(blend_out.norm(dim=-1), torch.ones(6), atol=1e-5)

    no_global_a = MultiScaleTransformerEncoder(
        in_dim=300,
        embed_dim=128,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        use_global_view=False,
    ).eval()
    no_global_b = MultiScaleTransformerEncoder(
        in_dim=300,
        embed_dim=128,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        use_global_view=False,
    ).eval()
    no_global_b.load_state_dict(no_global_a.state_dict())
    with torch.no_grad():
        for parameter in no_global_b.transformer.parameters():
            parameter.add_(torch.randn_like(parameter))
        no_global_out_a = no_global_a(seq_x)
        no_global_out_b = no_global_b(seq_x)
    assert torch.allclose(no_global_out_a, no_global_out_b, atol=1e-6)
    print("internal residual refinements: passed")


def test_variable_neighbor_entmax():
    device = torch.device("cpu")
    node_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    adjacency = {0: [1, 2], 1: [], 2: [0, 1, 3, 4]}
    neighbor_ids, neighbor_mask = sample_neighbors(
        node_ids=node_ids,
        adj_list=adjacency,
        num_neighbors=0,
        device=device,
    )
    assert neighbor_ids.shape == (3, 4)
    assert neighbor_mask.tolist() == [
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
    ]
    assert neighbor_ids[0].tolist() == [1, 2, 0, 0]
    assert neighbor_ids[1].tolist() == [1, 1, 1, 1]

    scores = torch.tensor(
        [[-2.0, 0.0, 0.5], [0.2, -0.1, 0.3], [1.0, 2.0, 3.0]],
        requires_grad=True,
    )
    mask = torch.tensor(
        [[1, 1, 1], [1, 1, 0], [0, 0, 0]],
        dtype=torch.long,
    )
    weights = masked_entmax15(scores, mask)
    assert torch.allclose(
        weights[0],
        torch.tensor([0.0, 0.3260, 0.6740]),
        atol=1e-3,
    )
    assert torch.allclose(weights[:2].sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.equal(weights[2], torch.zeros(3))
    assert weights[0, 0].item() == 0.0
    (weights * scores).sum().backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()

    fusion = CrossModalFusion(
        dim=8,
        hidden_dim=8,
        dropout=0.0,
        neighbor_attention="entmax15",
        neighbor_attention_temperature=0.25,
    ).eval()
    with torch.no_grad():
        output = fusion(
            s=torch.randn(3, 8),
            t_self=torch.randn(3, 8),
            t_nei=torch.randn(3, 4, 8),
            nei_mask=neighbor_mask,
            return_components=True,
        )
    assert output["z_joint"].shape == (3, 8)
    assert torch.equal(output["neighbor_valid_count"], torch.tensor([2, 0, 4]))
    assert torch.all(
        output["neighbor_support_size"] <= output["neighbor_valid_count"]
    )
    assert output["neighbor_support_size"][1].item() == 0
    print("variable-neighbor entmax: passed")


def test_controlled_ablation_invariants():
    early = CrossModalFusion(dim=128, hidden_dim=128, dropout=0.0).eval()
    reciprocal = CrossModalFusion(
        dim=128,
        hidden_dim=128,
        dropout=0.0,
        reciprocal_add=True,
    ).eval()
    reciprocal_residual = CrossModalFusion(
        dim=128,
        hidden_dim=128,
        dropout=0.0,
        reciprocal_residual_add=True,
    ).eval()
    late = ParameterMatchedLateFusion(dim=128, hidden_dim=512, dropout=0.0).eval()
    early_trainable = sum(p.numel() for p in early.parameters() if p.requires_grad)
    reciprocal_trainable = sum(
        p.numel() for p in reciprocal.parameters() if p.requires_grad
    )
    reciprocal_residual_trainable = sum(
        p.numel() for p in reciprocal_residual.parameters() if p.requires_grad
    )
    late_trainable = sum(p.numel() for p in late.parameters() if p.requires_grad)
    assert (
        early_trainable
        == reciprocal_trainable
        == reciprocal_residual_trainable
        == late_trainable
        == 197377
    )

    torch.manual_seed(42)
    structural = torch.randn(6, 128)
    neighbors = torch.randn(6, 4, 128)
    neighbor_mask = torch.ones(6, 4, dtype=torch.long)
    semantic_a = torch.randn(6, 128)
    semantic_b = torch.randn(6, 128)
    controls = {
        "ss": CrossModalFusion(
            dim=128,
            dropout=0.0,
            neighbor_query_mode="semantic",
            neighbor_gate_mode="semantic",
        ).eval(),
        "ts": CrossModalFusion(
            dim=128,
            dropout=0.0,
            neighbor_query_mode="structural",
            neighbor_gate_mode="semantic",
        ).eval(),
        "st": CrossModalFusion(
            dim=128,
            dropout=0.0,
            neighbor_query_mode="semantic",
            neighbor_gate_mode="structural",
        ).eval(),
        "tt": CrossModalFusion(
            dim=128,
            dropout=0.0,
            neighbor_query_mode="structural",
            neighbor_gate_mode="structural",
        ).eval(),
    }
    semantic_query = controls["ss"]
    for name, control in controls.items():
        if name != "ss":
            control.load_state_dict(semantic_query.state_dict())
    with torch.no_grad():
        outputs_a = {
            name: control(
                semantic_a,
                structural,
                neighbors,
                neighbor_mask,
                return_components=True,
            )
            for name, control in controls.items()
        }
        outputs_b = {
            name: control(
                semantic_b,
                structural,
                neighbors,
                neighbor_mask,
                return_components=True,
            )
            for name, control in controls.items()
        }

    # A structural gate is invariant to semantic changes; a semantic gate is not.
    for name in ("st", "tt"):
        assert torch.allclose(
            outputs_a[name]["structural_gate"],
            outputs_b[name]["structural_gate"],
            atol=1e-6,
        )
    for name in ("ss", "ts"):
        assert not torch.allclose(
            outputs_a[name]["structural_gate"],
            outputs_b[name]["structural_gate"],
        )

    # Only the all-structural scoring path is invariant before context fusion.
    assert torch.allclose(
        outputs_a["tt"]["semantic_attention"],
        outputs_b["tt"]["semantic_attention"],
        atol=1e-6,
    )
    for name in ("ss", "ts", "st"):
        assert not torch.allclose(
            outputs_a[name]["semantic_attention"],
            outputs_b[name]["semantic_attention"],
        )
    assert not torch.allclose(outputs_a["tt"]["z_joint"], outputs_b["tt"]["z_joint"])

    reciprocal.load_state_dict(semantic_query.state_dict())
    with torch.no_grad():
        reciprocal_a = reciprocal(
            semantic_a, structural, neighbors, neighbor_mask, return_components=True
        )
        reciprocal_b = reciprocal(
            semantic_b, structural, neighbors, neighbor_mask, return_components=True
        )
        reciprocal_struct_changed = reciprocal(
            semantic_a,
            structural + 0.5,
            neighbors,
            neighbor_mask,
            return_components=True,
        )
    expected_sum = torch.nn.functional.normalize(
        reciprocal_a["z_sem_enhanced"] + reciprocal_a["z_struct_enhanced"],
        p=2,
        dim=-1,
    )
    assert torch.allclose(reciprocal_a["z_joint"], expected_sum, atol=1e-6)
    assert not torch.allclose(
        reciprocal_a["z_struct_enhanced"],
        reciprocal_b["z_struct_enhanced"],
    )
    assert not torch.allclose(
        reciprocal_a["z_sem_enhanced"],
        reciprocal_struct_changed["z_sem_enhanced"],
    )

    reciprocal_residual.load_state_dict(semantic_query.state_dict())
    with torch.no_grad():
        reciprocal_residual_out = reciprocal_residual(
            semantic_a, structural, neighbors, neighbor_mask, return_components=True
        )
    expected_residual_sum = torch.nn.functional.normalize(
        torch.nn.functional.normalize(semantic_a, p=2, dim=-1)
        + torch.nn.functional.normalize(structural, p=2, dim=-1)
        + reciprocal_residual_out["z_sem_enhanced"]
        + reciprocal_residual_out["z_struct_enhanced"],
        p=2,
        dim=-1,
    )
    assert torch.allclose(
        reciprocal_residual_out["z_joint"], expected_residual_sum, atol=1e-6
    )

    x = torch.randn(5, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_type_a = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    edge_type_b = torch.tensor([2, 0, 1, 2], dtype=torch.long)
    typed = RelationAwareGraphConv(8, 8, 3, use_relation_types=True).eval()
    untyped = RelationAwareGraphConv(8, 8, 3, use_relation_types=False).eval()
    untyped.load_state_dict(typed.state_dict())
    with torch.no_grad():
        typed_a = typed(x, edge_index, edge_type_a)
        typed_b = typed(x, edge_index, edge_type_b)
        untyped_a = untyped(x, edge_index, edge_type_a)
        untyped_b = untyped(x, edge_index, edge_type_b)
    assert not torch.allclose(typed_a, typed_b)
    assert torch.allclose(untyped_a, untyped_b, atol=1e-6)
    print("controlled ablation invariants: passed")


def test_rdgcn_structural_encoder():
    encoder = RDGCNStructuralEncoder(
        in_dim=8,
        hidden_dim=8,
        out_dim=8,
        num_relations=3,
        num_layers=2,
        dropout=0.0,
    )
    x = torch.randn(6, 8, requires_grad=True)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 1], [1, 2, 0, 4, 5, 4]], dtype=torch.long
    )
    edge_type = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    output = encoder(x, edge_index, edge_type)
    assert output.shape == (6, 8)
    assert torch.isfinite(output).all()
    assert torch.allclose(output.norm(dim=-1), torch.ones(6), atol=1e-5)
    output.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print("RDGCN structural encoder: passed")


def test_rrea_structural_encoder():
    encoder = RREAStructuralEncoder(
        in_dim=8,
        hidden_dim=8,
        out_dim=8,
        num_relations=3,
        num_layers=2,
        dropout=0.0,
    )
    x = torch.randn(6, 8, requires_grad=True)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 1], [1, 2, 0, 4, 5, 4]], dtype=torch.long
    )
    edge_type = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    output = encoder(x, edge_index, edge_type)
    assert output.shape == (6, 8)
    assert torch.isfinite(output).all()
    assert torch.allclose(output.norm(dim=-1), torch.ones(6), atol=1e-5)
    output.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print("RREA structural encoder: passed")


def test_validation_selected_csls():
    class DummyModel:
        def encode_structure_all(self, edge_index, edge_type):
            return torch.zeros(4, 2)

    left_outputs = {
        "z_joint": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "z_struct": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "z_sem": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    }
    right_outputs = {
        "z_joint": torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.7, 0.7]]),
        "z_struct": torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.7, 0.7]]),
        "z_sem": torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.7, 0.7]]),
    }
    with patch("evaluate.encode_entity_outputs", side_effect=[left_outputs, right_outputs]):
        selection = select_csls_parameters(
            model=DummyModel(),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            edge_type=None,
            seq_features=torch.empty(4, 1, 1),
            adj_list={},
            num_neighbors=1,
            validation_pairs=[(0, 2), (1, 3)],
            candidate_right_ids=torch.tensor([2, 3, 4]),
            batch_size=2,
            device=torch.device("cpu"),
            csls_k_candidates=[1, 2],
            csls_blend_candidates=[0.5, 1.0],
        )
    assert len(selection["grid"]) == 4
    assert selection["best_k"] in {1, 2}
    assert selection["best_blend"] in {0.5, 1.0}
    assert selection["best_metrics"]["MRR"] == max(row["MRR"] for row in selection["grid"])

    cfg = Config()
    for removed_field in (
        "lambda_neg",
        "lambda_ranking",
        "num_hard_neg",
        "use_global_hard_neg",
        "warmup_epochs",
        "do_active_learning",
        "use_reranker",
        "use_relation_aware_neighbor_sampling",
    ):
        assert not hasattr(cfg, removed_field)
    cfg.select_csls_on_validation = True
    cfg.val_ratio = 0.2
    splits = build_protocol_splits(
        cfg=cfg,
        train_pairs=[(index, index + 10) for index in range(10)],
        test_pairs=[(20, 30)],
    )
    assert len(splits["train_pairs"]) == 8
    assert len(splits["val_pairs"]) == 2
    assert not set(splits["train_pairs"]) & set(splits["val_pairs"])
    print("validation-selected CSLS: passed")


def test_structural_auxiliary_infonce():
    left = {
        "z_joint": torch.randn(4, 8, requires_grad=True),
        "z_struct": torch.randn(4, 8, requires_grad=True),
    }
    right = {
        "z_joint": torch.randn(4, 8, requires_grad=True),
        "z_struct": torch.randn(4, 8, requires_grad=True),
    }
    from models.losses import total_loss

    baseline = total_loss(left, right, structure_loss_weight=0.0)
    supervised = total_loss(left, right, structure_loss_weight=0.3)
    assert baseline["structure_align_loss"].item() == 0.0
    assert supervised["structure_align_loss"].item() > 0.0
    assert supervised["loss"].item() > supervised["align_loss"].item()
    supervised["loss"].backward()
    assert left["z_struct"].grad is not None
    print("structural auxiliary InfoNCE: passed")


def test_seed_shared_structure_initialization():
    model = JointEAModel(
        num_nodes=6,
        node_input_dim=8,
        fusion_dim=8,
        gnn_hidden_dim=8,
        text_hidden_dim=8,
        use_mst=False,
        use_light_gnn=False,
        use_relation_gnn=False,
        share_seed_structure_embeddings=True,
    )
    model.set_structure_seed_pairs([(0, 3), (1, 4)])
    output = model.encode_structure_all(torch.empty(2, 0, dtype=torch.long))
    assert torch.allclose(output[0], output[3], atol=1e-6)
    assert torch.allclose(output[1], output[4], atol=1e-6)
    output.sum().backward()
    assert model.node_emb.weight.grad is not None
    print("seed-shared structural initialization: passed")


def test_topology_structure_initialization():
    edge_index = torch.tensor([[0, 0, 1, 3, 4], [1, 2, 2, 4, 5]])
    edge_type = torch.tensor([0, 1, 0, 2, 2])
    features = build_topology_features(edge_index, edge_type, 6, (3, 3))
    assert features.shape == (6, 8)
    assert torch.isfinite(features).all()
    model = JointEAModel(
        num_nodes=6,
        node_input_dim=8,
        fusion_dim=8,
        gnn_hidden_dim=8,
        text_hidden_dim=8,
        use_mst=False,
        use_light_gnn=False,
        use_relation_gnn=False,
        structure_initialization="topology",
    )
    model.set_structure_topology_features(features)
    output = model.encode_structure_all(edge_index)
    assert output.shape == (6, 8)
    assert torch.isfinite(output).all()
    print("topology structural initialization: passed")


def main():
    for fusion_mode in (
        "early_interaction",
        "reciprocal_add",
        "reciprocal_residual_add",
        "late_concat_mlp",
        "late_concat_matched",
        "mean",
        "structure_only",
        "semantic_only",
    ):
        run_mode(fusion_mode)

    test_internal_refinements()
    test_variable_neighbor_entmax()
    test_controlled_ablation_invariants()
    test_rdgcn_structural_encoder()
    test_rrea_structural_encoder()
    test_validation_selected_csls()
    test_structural_auxiliary_infonce()
    test_seed_shared_structure_initialization()
    test_topology_structure_initialization()

    with patch.dict(
        os.environ,
        {
            "KG_ALIGN_USE_CE": "0",
            "KG_ALIGN_FUSION_MODE": "late_concat_mlp",
        },
        clear=True,
    ):
        cfg = Config()
        apply_runtime_overrides(cfg)
        assert cfg.fusion_mode == "late_concat_mlp"
        assert not cfg.use_cross_modal_enhancement

    with patch.dict(os.environ, {"KG_ALIGN_USE_CE": "0"}, clear=True):
        cfg = Config()
        apply_runtime_overrides(cfg)
        assert cfg.fusion_mode == "mean"
        assert not cfg.use_cross_modal_enhancement

    with patch.dict(
        os.environ,
        {
            "KG_ALIGN_PROTOCOL": "simplified_core",
            "KG_ALIGN_SEM_RESIDUAL_MODE": "gated",
            "KG_ALIGN_CE_RESIDUAL_RATIO": "0.05",
        },
        clear=True,
    ):
        cfg = Config()
        apply_runtime_overrides(cfg)
        assert cfg.sem_residual_mode == "gated"
        assert cfg.ce_residual_ratio == 0.05

    with patch.dict(
        os.environ,
        {
            "KG_ALIGN_PROTOCOL": "simplified_core",
            "KG_ALIGN_DATASET_FAMILY": "dbp15k_raw",
        },
        clear=True,
    ):
        cfg = Config()
        apply_runtime_overrides(cfg)
        assert cfg.sem_use_global_view

    with patch.dict(
        os.environ,
        {
            "KG_ALIGN_PROTOCOL": "simplified_core",
            "KG_ALIGN_DATASET_FAMILY": "openea",
        },
        clear=True,
    ):
        cfg = Config()
        apply_runtime_overrides(cfg)
        assert cfg.sem_use_global_view

    with patch.dict(
        os.environ,
        {
            "KG_ALIGN_PROTOCOL": "simplified_core",
            "KG_ALIGN_DATASET_FAMILY": "openea",
            "KG_ALIGN_SEM_USE_GLOBAL_VIEW": "0",
        },
        clear=True,
    ):
        cfg = Config()
        apply_runtime_overrides(cfg)
        assert not cfg.sem_use_global_view

    with patch.dict(
        os.environ,
        {
            "KG_ALIGN_PROTOCOL": "simplified_core_csls",
            "KG_ALIGN_DATASET_FAMILY": "eventea",
            "KG_ALIGN_PAIR": "EN_EN_20K",
            "KG_ALIGN_OPENEA_SPLIT": ".",
        },
        clear=True,
    ):
        cfg = Config()
        apply_runtime_overrides(cfg)
        assert cfg.root == "data/eventea/EventEA"
        assert cfg.sem_use_global_view
        assert cfg.select_csls_on_validation

    with patch.dict(
        os.environ,
        {
            "KG_ALIGN_USE_RELATION_TYPES": "0",
            "KG_ALIGN_NEIGHBOR_QUERY_MODE": "hybrid",
            "KG_ALIGN_NEIGHBOR_GATE_MODE": "structural",
            "KG_ALIGN_NEIGHBOR_QUERY_SEMANTIC_WEIGHT": "0.5",
            "KG_ALIGN_FUSION_MODE": "late_concat_matched",
        },
        clear=True,
    ):
        cfg = Config()
        apply_runtime_overrides(cfg)
        assert not cfg.use_relation_types
        assert cfg.neighbor_query_mode == "hybrid"
        assert cfg.neighbor_gate_mode == "structural"
        assert cfg.neighbor_query_semantic_weight == 0.5
        assert cfg.fusion_mode == "late_concat_matched"

    with patch.dict(
        os.environ,
        {
            "KG_ALIGN_NUM_NEIGHBORS": "0",
            "KG_ALIGN_NEIGHBOR_ATTENTION": "entmax15",
            "KG_ALIGN_NEIGHBOR_ATTENTION_TEMPERATURE": "0.25",
        },
        clear=True,
    ):
        cfg = Config()
        apply_runtime_overrides(cfg)
        assert cfg.num_neighbors == 0
        assert cfg.neighbor_attention == "entmax15"
        assert cfg.neighbor_attention_temperature == 0.25
    print("fusion environment override precedence: passed")


if __name__ == "__main__":
    main()
