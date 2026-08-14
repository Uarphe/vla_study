"""A minimal PyTorch example of single-step actions and action chunks."""

from __future__ import annotations

import math

import torch
from torch import nn


BATCH_SIZE = 4
OBS_DIM = 32
ACTION_DIM = 7
CHUNK_SIZE = 16


class SingleStepPolicy(nn.Module):
    """Predict only a_t from o_t: [B, obs_dim] -> [B, 7]."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.ReLU())
        self.action_head = nn.Linear(64, ACTION_DIM)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        feature = self.encoder(observation)
        return self.action_head(feature)


class ActionChunkPolicy(nn.Module):
    """Predict a_t ... a_{t+15} from o_t: [B, obs_dim] -> [B, 16, 7]."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.ReLU())
        self.action_head = nn.Linear(64, CHUNK_SIZE * ACTION_DIM)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        feature = self.encoder(observation)
        flat_actions = self.action_head(feature)  # [B, 16 * 7]
        return flat_actions.reshape(observation.shape[0], CHUNK_SIZE, ACTION_DIM)


def temporal_ensemble(
    chunk_history: list[tuple[int, torch.Tensor]],
    current_t: int,
    decay: float = 0.25,
) -> torch.Tensor:
    """Fuse all historical chunks that contain a prediction for current_t."""

    candidates = []
    weights = []

    for prediction_t, chunk in chunk_history:
        offset = current_t - prediction_t
        if 0 <= offset < chunk.shape[0]:
            candidates.append(chunk[offset])
            # A newer prediction has a smaller age and therefore a larger weight.
            weights.append(math.exp(-decay * offset))

    candidate_tensor = torch.stack(candidates)  # [N, 7]
    weight_tensor = torch.tensor(weights, dtype=candidate_tensor.dtype)
    weight_tensor = weight_tensor / weight_tensor.sum()
    return (candidate_tensor * weight_tensor[:, None]).sum(dim=0)  # [7]


def main() -> None:
    torch.manual_seed(0)
    observations = torch.randn(BATCH_SIZE, OBS_DIM)

    single_policy = SingleStepPolicy()
    chunk_policy = ActionChunkPolicy()

    single_action = single_policy(observations)
    action_chunk = chunk_policy(observations)

    print("single_action.shape:", single_action.shape)
    print("action_chunk.shape: ", action_chunk.shape)

    # Supervised learning target: 16 expert actions following each observation.
    expert_chunk = torch.randn(BATCH_SIZE, CHUNK_SIZE, ACTION_DIM)
    loss = nn.functional.mse_loss(action_chunk, expert_chunk)
    loss.backward()
    print("training loss:      ", round(loss.item(), 4))

    # Suppose policies queried at t=0, 1, and 2 have produced overlapping chunks.
    history = [
        (0, torch.randn(CHUNK_SIZE, ACTION_DIM)),
        (1, torch.randn(CHUNK_SIZE, ACTION_DIM)),
        (2, torch.randn(CHUNK_SIZE, ACTION_DIM)),
    ]
    executed_action = temporal_ensemble(history, current_t=2)
    print("ensemble action:    ", executed_action.shape)

    assert single_action.shape == (BATCH_SIZE, ACTION_DIM)
    assert action_chunk.shape == (BATCH_SIZE, CHUNK_SIZE, ACTION_DIM)
    assert executed_action.shape == (ACTION_DIM,)


if __name__ == "__main__":
    main()
