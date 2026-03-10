import torch

from models.full_model import JointEAModel
from models.losses import total_loss


def main():
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
        text_heads=4,
        text_layers=2,
        dropout=0.1,
    ).to(device)

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

    losses = total_loss(left_out, right_out)
    score_matrix = model.score_pairs(left_out["z_joint"], right_out["z_joint"])

    print("z_struct:", left_out["z_struct"].shape)
    print("z_sem_enhanced:", left_out["z_sem_enhanced"].shape)
    print("z_joint:", left_out["z_joint"].shape)
    print("score_matrix:", score_matrix.shape)
    print("loss:", losses["loss"].item())
    print("align_loss:", losses["align_loss"].item())


if __name__ == "__main__":
    main()
