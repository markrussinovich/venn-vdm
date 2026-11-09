"""Record the synthetic million-row sweep as an animated GIF for the documentation."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from venn.demos import million_sweep


def record(output, rows):
    frames = []
    checkpoints = {}
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    def frame(phase, completed, total):
        if completed < checkpoints.get(phase, 0) + 65536 and completed != total:
            return
        checkpoints[phase] = completed
        image = Image.new("RGB", (1100, 360), "#17202a")
        draw = ImageDraw.Draw(image)
        lines = [
            f"Venn: an actual live {total:,}-row sweep",
            "SYNTHETIC normalized Gaussian vectors, 768 dimensions",
            "Headless CPU conditional centered-cosine; no abstention",
            "",
            f"{phase.capitalize()}: {completed:,} / {total:,} rows",
            "Every output consumed and hashed; no top-k truncation",
            "",
            "Accelerated progress recording, not a latency benchmark",
            "No semantic accuracy, privacy or guardrail claim",
        ]
        for i, line in enumerate(lines):
            draw.text((20, 15 + i * 34), line, font=font, fill="#ecf0f1")
        frames.append(image)

    result = million_sweep(rows, progress=frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], loop=0,
                   duration=[150] * (len(frames) - 1) + [2000], optimize=True)
    print({"output": str(output), "consumed_rows": sum(result["counts"].values()),
           "frames": len(frames), "scope": result["scope"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1000000)
    args = parser.parse_args()
    if not 1 <= args.rows <= 1000000:
        parser.error("--rows must be in 1..1000000")
    record(args.output, args.rows)
