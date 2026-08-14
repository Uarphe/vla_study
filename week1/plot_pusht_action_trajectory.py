"""Plot both action dimensions from one PushT episode.

This script intentionally uses Pillow because it is already installed in the
current LeRobot environment.  The important data operations are the three
lines that select one episode, stack its actions, and obtain a [T, D] tensor.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# Hugging Face datasets creates small lock/cache files while opening parquet.
# Use a writable temporary directory and keep generated cache files out of the repo.
os.environ.setdefault("HF_DATASETS_CACHE", str(Path(tempfile.gettempdir()) / "vla_study_hf_datasets_cache"))

import torch
from PIL import Image, ImageDraw, ImageFont
from lerobot.datasets.lerobot_dataset import LeRobotDataset


EPISODE_ID = 0
OUTPUT_PATH = SCRIPT_DIR / "pusht_episode_0_action_trajectory.png"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / name
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def map_value(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return (dst_min + dst_max) / 2
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    return [low + i * (high - low) / (count - 1) for i in range(count)]


def draw_time_series(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    frames: list[int],
    values: list[float],
    color: tuple[int, int, int],
    title: str,
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=14, fill="#ffffff", outline="#d8dee9", width=2)
    plot_left, plot_top = left + 76, top + 46
    plot_right, plot_bottom = right - 24, bottom - 48

    y_min, y_max = min(values), max(values)
    margin = max((y_max - y_min) * 0.08, 1.0)
    y_min, y_max = y_min - margin, y_max + margin

    draw.text((left + 20, top + 14), title, font=font(22, bold=True), fill="#202733")

    for tick in nice_ticks(y_min, y_max):
        y = map_value(tick, y_min, y_max, plot_bottom, plot_top)
        draw.line((plot_left, y, plot_right, y), fill="#e8ecf2", width=1)
        label = f"{tick:.0f}"
        bbox = draw.textbbox((0, 0), label, font=font(14))
        draw.text((plot_left - 10 - (bbox[2] - bbox[0]), y - 8), label, font=font(14), fill="#687386")

    x_ticks = [frames[0], 40, 80, 120, frames[-1]]
    for tick in x_ticks:
        x = map_value(tick, frames[0], frames[-1], plot_left, plot_right)
        draw.line((x, plot_top, x, plot_bottom), fill="#f0f2f6", width=1)
        label = str(tick)
        bbox = draw.textbbox((0, 0), label, font=font(14))
        draw.text((x - (bbox[2] - bbox[0]) / 2, plot_bottom + 10), label, font=font(14), fill="#687386")

    points = [
        (
            map_value(frame, frames[0], frames[-1], plot_left, plot_right),
            map_value(value, y_min, y_max, plot_bottom, plot_top),
        )
        for frame, value in zip(frames, values, strict=True)
    ]
    draw.line(points, fill=color, width=3, joint="curve")
    draw.ellipse((points[0][0] - 4, points[0][1] - 4, points[0][0] + 4, points[0][1] + 4), fill="#17a673")
    draw.ellipse((points[-1][0] - 4, points[-1][1] - 4, points[-1][0] + 4, points[-1][1] + 4), fill="#e15258")
    draw.text(((plot_left + plot_right) / 2 - 40, bottom - 28), "frame t", font=font(15), fill="#4f5b6c")


def lerp_color(start: tuple[int, int, int], end: tuple[int, int, int], ratio: float) -> tuple[int, int, int]:
    return tuple(round(a + (b - a) * ratio) for a, b in zip(start, end, strict=True))


def draw_trajectory(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    rows: list[list[float]],
    title: str = "2D action trajectory",
    x_label: str = "action[:, 0]",
    y_label: str = "dim 1",
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=14, fill="#ffffff", outline="#d8dee9", width=2)
    plot_left, plot_top = left + 76, top + 58
    plot_right, plot_bottom = right - 32, bottom - 70

    xs = [row[0] for row in rows]
    ys = [row[1] for row in rows]
    x_margin = max((max(xs) - min(xs)) * 0.08, 1.0)
    y_margin = max((max(ys) - min(ys)) * 0.08, 1.0)
    x_min, x_max = min(xs) - x_margin, max(xs) + x_margin
    y_min, y_max = min(ys) - y_margin, max(ys) + y_margin

    draw.text((left + 20, top + 14), title, font=font(22, bold=True), fill="#202733")

    for tick in nice_ticks(x_min, x_max):
        x = map_value(tick, x_min, x_max, plot_left, plot_right)
        draw.line((x, plot_top, x, plot_bottom), fill="#edf0f5", width=1)
        label = f"{tick:.0f}"
        bbox = draw.textbbox((0, 0), label, font=font(14))
        draw.text((x - (bbox[2] - bbox[0]) / 2, plot_bottom + 12), label, font=font(14), fill="#687386")

    for tick in nice_ticks(y_min, y_max):
        y = map_value(tick, y_min, y_max, plot_bottom, plot_top)
        draw.line((plot_left, y, plot_right, y), fill="#edf0f5", width=1)
        label = f"{tick:.0f}"
        bbox = draw.textbbox((0, 0), label, font=font(14))
        draw.text((plot_left - 10 - (bbox[2] - bbox[0]), y - 8), label, font=font(14), fill="#687386")

    points = [
        (
            map_value(action[0], x_min, x_max, plot_left, plot_right),
            map_value(action[1], y_min, y_max, plot_bottom, plot_top),
        )
        for action in rows
    ]

    for i in range(len(points) - 1):
        ratio = i / (len(points) - 2)
        draw.line((points[i], points[i + 1]), fill=lerp_color((44, 123, 229), (231, 76, 60), ratio), width=4)

    start, end = points[0], points[-1]
    draw.ellipse((start[0] - 7, start[1] - 7, start[0] + 7, start[1] + 7), fill="#17a673", outline="white", width=2)
    draw.ellipse((end[0] - 7, end[1] - 7, end[0] + 7, end[1] + 7), fill="#e15258", outline="white", width=2)
    draw.text((start[0] + 9, start[1] - 20), "START", font=font(14, bold=True), fill="#13845d")
    draw.text((end[0] + 9, end[1] - 20), "END", font=font(14, bold=True), fill="#c63f46")

    x_bbox = draw.textbbox((0, 0), x_label, font=font(15))
    draw.text(((plot_left + plot_right - (x_bbox[2] - x_bbox[0])) / 2, bottom - 30), x_label, font=font(15), fill="#4f5b6c")
    draw.text((left + 10, top + 76), y_label, font=font(14), fill="#4f5b6c")

    legend_y = bottom - 52
    for i in range(80):
        ratio = i / 79
        x = right - 185 + i
        draw.line((x, legend_y, x, legend_y + 8), fill=lerp_color((44, 123, 229), (231, 76, 60), ratio), width=1)
    draw.text((right - 224, legend_y - 5), "t=0", font=font(13), fill="#687386")
    draw.text((right - 95, legend_y - 5), "t=T-1", font=font(13), fill="#687386")


def main() -> None:
    dataset = LeRobotDataset(repo_id="lerobot/pusht")
    episode = dataset.meta.episodes[EPISODE_ID]
    start = int(episode["dataset_from_index"])
    end = int(episode["dataset_to_index"])

    # A single action is [D]. Stacking T consecutive actions creates [T, D].
    actions = torch.stack([dataset.hf_dataset[i]["action"] for i in range(start, end)])
    frames = [int(dataset.hf_dataset[i]["frame_index"]) for i in range(start, end)]
    action_rows = actions.tolist()

    canvas = Image.new("RGB", (1500, 850), "#f5f7fa")
    draw = ImageDraw.Draw(canvas)
    draw.text((56, 24), "PushT episode 0: action sequence", font=font(34, bold=True), fill="#172033")
    draw.text(
        (57, 70),
        f"actions.shape = [{actions.shape[0]}, {actions.shape[1]}] = [T, D]    |    10 Hz    |    {actions.shape[0]} timesteps",
        font=font(19),
        fill="#526075",
    )

    draw_time_series(draw, (50, 115, 810, 430), frames, actions[:, 0].tolist(), (44, 123, 229), "Dimension 0: action[:, 0]")
    draw_time_series(draw, (50, 465, 810, 780), frames, actions[:, 1].tolist(), (238, 135, 38), "Dimension 1: action[:, 1]")
    draw_trajectory(draw, (840, 115, 1450, 780), action_rows)

    draw.text(
        (52, 810),
        "Left: read each column through time.  Right: connect the row vectors action[t] = [a0, a1] in chronological order.",
        font=font(17),
        fill="#5b6678",
    )
    canvas.save(OUTPUT_PATH)

    print(f"episode={EPISODE_ID}")
    print(f"dataset indices=[{start}, {end})")
    print(f"actions.shape={tuple(actions.shape)}")
    print(f"first action={actions[0].tolist()}")
    print(f"last action={actions[-1].tolist()}")
    print(f"min per dimension={actions.min(dim=0).values.tolist()}")
    print(f"max per dimension={actions.max(dim=0).values.tolist()}")
    print(f"saved={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
