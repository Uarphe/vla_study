"""Visualize one [16, 7] action chunk produced by the toy policy."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from action_chunk_toy import ACTION_DIM, BATCH_SIZE, CHUNK_SIZE, OBS_DIM, ActionChunkPolicy


OUTPUT_PATH = Path(__file__).resolve().parent / "action_chunk_toy_result.png"

COLORS = [
    (38, 111, 219),
    (230, 126, 34),
    (25, 153, 112),
    (210, 70, 79),
    (126, 87, 194),
    (44, 157, 183),
    (105, 113, 128),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_name = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(str(Path("C:/Windows/Fonts") / font_name), size)
    except OSError:
        return ImageFont.load_default()


def map_value(value: float, low: float, high: float, out_low: float, out_high: float) -> float:
    if high == low:
        return (out_low + out_high) / 2
    return out_low + (value - low) / (high - low) * (out_high - out_low)


def lerp_color(a: tuple[int, int, int], b: tuple[int, int, int], ratio: float) -> tuple[int, int, int]:
    ratio = min(max(ratio, 0.0), 1.0)
    return tuple(round(x + (y - x) * ratio) for x, y in zip(a, b, strict=True))


def heat_color(value: float, limit: float) -> tuple[int, int, int]:
    white = (248, 250, 252)
    if value < 0:
        return lerp_color(white, (46, 112, 211), abs(value) / limit)
    return lerp_color(white, (222, 74, 69), value / limit)


def draw_heatmap(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], chunk: torch.Tensor) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=14, fill="white", outline="#d8dee9", width=2)
    draw.text((left + 20, top + 14), "One sample: [16, 7] action chunk", font=font(22, True), fill="#1d2738")

    grid_left, grid_top = left + 70, top + 70
    grid_right, grid_bottom = right - 24, bottom - 46
    cell_w = (grid_right - grid_left) / ACTION_DIM
    cell_h = (grid_bottom - grid_top) / CHUNK_SIZE
    limit = max(abs(float(chunk.min())), abs(float(chunk.max())))

    for d in range(ACTION_DIM):
        label = f"d{d}"
        bbox = draw.textbbox((0, 0), label, font=font(14, True))
        x = grid_left + (d + 0.5) * cell_w - (bbox[2] - bbox[0]) / 2
        draw.text((x, grid_top - 27), label, font=font(14, True), fill="#536176")

    for h in range(CHUNK_SIZE):
        label = str(h)
        bbox = draw.textbbox((0, 0), label, font=font(13))
        y = grid_top + (h + 0.5) * cell_h - (bbox[3] - bbox[1]) / 2
        draw.text((grid_left - 13 - (bbox[2] - bbox[0]), y), label, font=font(13), fill="#667287")

        for d in range(ACTION_DIM):
            x0 = grid_left + d * cell_w
            y0 = grid_top + h * cell_h
            x1 = grid_left + (d + 1) * cell_w
            y1 = grid_top + (h + 1) * cell_h
            draw.rectangle((x0, y0, x1, y1), fill=heat_color(float(chunk[h, d]), limit), outline="#ffffff", width=1)

    draw.text((left + 15, top + 62), "future\nstep h", font=font(13), fill="#536176")
    draw.text((grid_left + 160, bottom - 30), "action dimension d", font=font(14), fill="#536176")
    draw.text((right - 210, top + 20), "blue = negative", font=font(13), fill="#3868ad")
    draw.text((right - 105, top + 20), "red = positive", font=font(13), fill="#b94b48")


def draw_lines(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], chunk: torch.Tensor) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=14, fill="white", outline="#d8dee9", width=2)
    draw.text((left + 20, top + 14), "7 dimensions across the 16-step horizon", font=font(22, True), fill="#1d2738")

    plot_left, plot_top = left + 75, top + 70
    plot_right, plot_bottom = right - 30, bottom - 65
    low = float(chunk.min())
    high = float(chunk.max())
    margin = max((high - low) * 0.1, 0.01)
    low -= margin
    high += margin

    for i in range(5):
        value = low + i * (high - low) / 4
        y = map_value(value, low, high, plot_bottom, plot_top)
        draw.line((plot_left, y, plot_right, y), fill="#e7ebf1", width=1)
        label = f"{value:.2f}"
        bbox = draw.textbbox((0, 0), label, font=font(13))
        draw.text((plot_left - 10 - (bbox[2] - bbox[0]), y - 7), label, font=font(13), fill="#687386")

    for h in [0, 3, 6, 9, 12, 15]:
        x = map_value(h, 0, CHUNK_SIZE - 1, plot_left, plot_right)
        draw.line((x, plot_top, x, plot_bottom), fill="#f0f2f6", width=1)
        label = str(h)
        bbox = draw.textbbox((0, 0), label, font=font(13))
        draw.text((x - (bbox[2] - bbox[0]) / 2, plot_bottom + 11), label, font=font(13), fill="#687386")

    for d in range(ACTION_DIM):
        points = [
            (
                map_value(h, 0, CHUNK_SIZE - 1, plot_left, plot_right),
                map_value(float(chunk[h, d]), low, high, plot_bottom, plot_top),
            )
            for h in range(CHUNK_SIZE)
        ]
        draw.line(points, fill=COLORS[d], width=3, joint="curve")
        x = right - 215 + (d % 4) * 48
        y = top + 22 + (d // 4) * 24
        draw.line((x, y + 8, x + 14, y + 8), fill=COLORS[d], width=3)
        draw.text((x + 18, y), f"d{d}", font=font(13, True), fill=COLORS[d])

    draw.text(((plot_left + plot_right) / 2 - 48, bottom - 34), "future step h", font=font(14), fill="#536176")
    draw.text((left + 15, top + 80), "value", font=font(13), fill="#536176")


def main() -> None:
    # The fixed seed makes the untrained toy policy's output reproducible.
    torch.manual_seed(0)
    observations = torch.randn(BATCH_SIZE, OBS_DIM)
    policy = ActionChunkPolicy()
    with torch.no_grad():
        chunks = policy(observations)  # [B, 16, 7]

    sample_chunk = chunks[0]  # [16, 7]

    canvas = Image.new("RGB", (1500, 800), "#f5f7fa")
    draw = ImageDraw.Draw(canvas)
    draw.text((50, 24), "Toy Action Chunk output", font=font(34, True), fill="#172033")
    draw.text(
        (51, 70),
        "policy(observations).shape = [4, 16, 7]    |    visualizing chunks[0].shape = [16, 7]",
        font=font(19),
        fill="#526075",
    )

    draw_heatmap(draw, (45, 115, 700, 750), sample_chunk)
    draw_lines(draw, (730, 115, 1455, 750), sample_chunk)
    canvas.save(OUTPUT_PATH)

    print(f"chunks.shape={tuple(chunks.shape)}")
    print(f"sample_chunk.shape={tuple(sample_chunk.shape)}")
    print(f"min={sample_chunk.min().item():.4f}, max={sample_chunk.max().item():.4f}")
    print(f"saved={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
