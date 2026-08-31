from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from training.train_ce import MixedDS, dual_head_logits


class Encoder(nn.Module):
    def forward(self, input_ids, attention_mask):
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 2)
        return SimpleNamespace(last_hidden_state=hidden)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Encoder()
        self.head = nn.Identity()
        self.drop = nn.Identity()
        self.classifier = nn.Linear(2, 1, bias=False)
        self.human_classifier = nn.Linear(2, 1, bias=False)
        self.classifier.weight.data.fill_(1)
        self.human_classifier.weight.data.fill_(2)
        self.config = SimpleNamespace(classifier_pooling="cls")


def test_mixed_dataset_marks_only_gold_rows():
    dataset = MixedDS(["a", "b"], np.array([0, 1]), np.array([1, 0]),
                      np.array([0.0, 1.0]), np.ones(2), np.array([0, 1]),
                      n_weak=1)
    assert dataset[0][-1] == 0
    assert dataset[1][-1] == 1


def test_dual_head_routes_each_row_after_one_encoder_pass():
    model = Toy()
    batch = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.ones(2, 2, dtype=torch.long),
    }
    got = dual_head_logits(model, batch, torch.tensor([False, True]))
    assert got.tolist() == [2.0, 12.0]
