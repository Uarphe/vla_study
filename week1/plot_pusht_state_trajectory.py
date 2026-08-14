"""Plot the actual pusher/agent position from one PushT episode."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("HF_DATASETS_CACHE", str(Path(tempfile.gettempdir()) / "vla_study_hf_datasets_cache"))

import torch
from PIL import Image, ImageDraw
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from plot_pusht_action_trajectory import draw_time_series, draw_trajectory, font


EPISODE_ID = 0
OUTPUT_PATH = SCRIPT_DIR / "pusht_episode_0_state_trajectory.png"


def main() -> None:
    dataset = LeRobotDataset(repo_id="lerobot/pusht")
    episode = dataset.meta.episodes[EPISODE_ID]
    start = int(episode["dataset_from_index"])
    end = int(episode["dataset_to_index"])

    # Each state is the actual 2D agent/pusher position [x, y].
    # Stacking the T states in time order produces the [T, 2] state trajectory.
    states = torch.stack([dataset.hf_dataset[i]["observation.state"] for i in range(start, end)])
    actions = torch.stack([dataset.hf_dataset[i]["action"] for i in range(start, end)])
    frames = [int(dataset.hf_dataset[i]["frame_index"]) for i in range(start, end)]

    canvas = Image.new("RGB", (1500, 850), "#f5f7fa")
    draw = ImageDraw.Draw(canvas)
    draw.text((56, 24), "PushT episode 0: actual pusher trajectory", font=font(34, bold=True), fill="#172033")
    draw.text(
        (57, 70),
        f"states.shape = [{states.shape[0]}, {states.shape[1]}] = [T, D]    |    observation.state = actual [x, y] position",
        font=font(19),
        fill="#526075",
    )

    draw_time_series(
        draw,
        (50, 115, 810, 430),
        frames,
        states[:, 0].tolist(),
        (44, 123, 229),
        "Actual x position: state[:, 0]",
    )
    draw_time_series(
        draw,
        (50, 465, 810, 780),
        frames,
        states[:, 1].tolist(),
        (238, 135, 38),
        "Actual y position: state[:, 1]",
    )
    draw_trajectory(
        draw,
        (840, 115, 1450, 780),
        states.tolist(),
        title="2D actual state trajectory",
        x_label="state[:, 0] (x)",
        y_label="y",
    )

    mean_tracking_error = torch.linalg.vector_norm(actions - states, dim=1).mean().item()
    draw.text(
        (52, 810),
        f"This is the measured pusher path, not the T-block path.  Mean distance to action target: {mean_tracking_error:.2f} coordinate units.",
        font=font(17),
        fill="#5b6678",
    )
    canvas.save(OUTPUT_PATH)

    print(f"episode={EPISODE_ID}")
    print(f"dataset indices=[{start}, {end})")
    print(f"states.shape={tuple(states.shape)}")
    print(f"first state={states[0].tolist()}")
    print(f"last state={states[-1].tolist()}")
    print(f"min per dimension={states.min(dim=0).values.tolist()}")
    print(f"max per dimension={states.max(dim=0).values.tolist()}")
    print(f"mean distance from action target={mean_tracking_error:.4f}")
    print(f"saved={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
