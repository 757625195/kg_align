from typing import List, Tuple

import torch
from torch.utils.data import Dataset


class AlignmentTrainDataset(Dataset):
    def __init__(self, train_pairs: List[Tuple[int, int]]):
        self.train_pairs = train_pairs

    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, idx):
        left_id, right_id = self.train_pairs[idx]
        return {
            "left_id": torch.tensor(left_id, dtype=torch.long),
            "right_id": torch.tensor(right_id, dtype=torch.long),
        }


def collate_alignment_batch(batch):
    return {
        "left_id": torch.stack([x["left_id"] for x in batch], dim=0),
        "right_id": torch.stack([x["right_id"] for x in batch], dim=0),
    }