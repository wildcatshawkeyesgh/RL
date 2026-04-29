"""Side-by-side rollout comparison of multiple ONNX checkpoints.

Loads N ONNX actors (typically the same run at different training points),
runs each against the *same seeded* env scenario, then stitches their
per-step frames into an N-up grid video. Lets you watch the policy improve
through training -- e.g. iter 25 vs 50 vs 100 vs 200, all flying through
the same scenario simultaneously.

Layout:
  N=2 -> 1x2 (side by side)
  N=3 -> 1x3
  N=4 -> 2x2
  N=5..6 -> 2x3
  N=7..9 -> 3x3 (last cells padded with the final episode's terminal frame)

Usage:
    # Default: 2x2 grid, output alongside the first checkpoint
    uv run python -m scripts.compare_checkpoints \\
        checkpoints/v3/checkpoint_000131072.onnx  \\
        checkpoints/v3/checkpoint_001310720.onnx  \\
        checkpoints/v3/checkpoint_005242880.onnx  \\
        checkpoints/v3/checkpoint_010485760.onnx  \\
        --seed 42 --output runs/v3_progression.mp4

    # Custom cell labels
    uv run python -m scripts.compare_checkpoints \\
        ckpt_a.onnx ckpt_b.onnx ckpt_c.onnx ckpt_d.onnx \\
        --labels "iter 25" "iter 50" "iter 100" "iter 200" \\
        --seed 42 --output progression.mp4
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import gymnasium
import numpy as np
import onnxruntime as ort

import iads.gym_env  # noqa: F401  (registers env)
from ppo.render import render_frame


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "checkpoints",
        nargs="+",
        help="Two or more .onnx checkpoint paths to compare.",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional cell labels (one per checkpoint). Defaults to step number parsed from filename.",
    )
    p.add_argument("--env-config", type=str, default="iads/default.yaml")
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Env seed used for ALL checkpoints so they fly the same scenario.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output MP4 path. Default: alongside the first checkpoint as compare_<stem>.mp4.",
    )
    p.add_argument("--fps", type=int, default=10)
    return p.parse_args()


# ---------------------------------------------------------------------------


def label_from_path(p: str) -> str:
    """Extract a step number from `checkpoint_NNNNNNNNN.onnx` filenames."""
    m = re.search(r"checkpoint_(\d+)", Path(p).stem)
    if m:
        n = int(m.group(1))
        return f"step {n:,}"
    return Path(p).stem


def run_one_rollout(
    onnx_path: str, env_config: str, seed: int, label: str
) -> List[np.ndarray]:
    """Run one episode with a given ONNX actor; return all rendered frames."""
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    env = gymnasium.make("IADSPenetration-v0", config_path=env_config)
    obs_np, _info = env.reset(seed=seed)

    frames: List[np.ndarray] = []
    total_reward = 0.0
    hits = 0
    step = 0
    done = False
    truncated = False

    while not (done or truncated):
        frames.append(
            render_frame(
                env.unwrapped.sim,
                title=f"{label}  step {step}  hits {hits}  R {total_reward:+.1f}",
            )
        )
        obs_batch = obs_np.astype(np.float32).reshape(1, -1)
        mean_action = sess.run(None, {"obs": obs_batch})[0][0]
        act = np.clip(mean_action, -1.0, 1.0)
        obs_np, reward, done, truncated, _info = env.step(act)
        total_reward += float(reward)
        hits = int(_info.get("hits", hits))
        step += 1

    frames.append(
        render_frame(
            env.unwrapped.sim,
            title=(
                f"{label}  step {step}  hits {hits}  R {total_reward:+.1f}  "
                f"(terminal)"
            ),
        )
    )
    env.close()
    return frames


def grid_shape_for(n: int) -> tuple[int, int]:
    """(rows, cols) for an N-up grid. 4 -> (2, 2), 6 -> (2, 3), etc."""
    if n <= 0:
        raise ValueError("Need at least 1 checkpoint")
    if n == 1:
        return (1, 1)
    if n == 2:
        return (1, 2)
    if n == 3:
        return (1, 3)
    if n == 4:
        return (2, 2)
    if n in (5, 6):
        return (2, 3)
    if n in (7, 8, 9):
        return (3, 3)
    raise ValueError(f"Comparison of {n} checkpoints exceeds supported grid sizes (max 9)")


def stitch_grid(per_ckpt_frames: List[List[np.ndarray]]) -> List[np.ndarray]:
    """Build composite per-timestep frames in an N-up grid.

    Pads shorter episodes by repeating their final frame so all are the
    length of the longest. Pads empty grid cells with a black frame.
    """
    n = len(per_ckpt_frames)
    rows, cols = grid_shape_for(n)

    max_len = max(len(ep) for ep in per_ckpt_frames)
    # Repeat-pad shorter episodes.
    for ep in per_ckpt_frames:
        while len(ep) < max_len:
            ep.append(ep[-1])

    # Reference frame size for padding empty cells.
    h, w = per_ckpt_frames[0][0].shape[:2]
    black = np.zeros((h, w, 3), dtype=np.uint8)

    composite_frames: List[np.ndarray] = []
    for t in range(max_len):
        cells: List[np.ndarray] = []
        for i in range(rows * cols):
            if i < n:
                cells.append(per_ckpt_frames[i][t])
            else:
                cells.append(black)
        # Reshape into rows x cols, concat horizontally then vertically.
        row_imgs = []
        for r in range(rows):
            row_cells = cells[r * cols : (r + 1) * cols]
            row_imgs.append(np.concatenate(row_cells, axis=1))
        composite_frames.append(np.concatenate(row_imgs, axis=0))
    return composite_frames


def save_mp4(frames: List[np.ndarray], path: Path, fps: int) -> None:
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(path), frames, fps=fps, codec="libx264", quality=8)


def default_output(first_ckpt: str) -> Path:
    p = Path(first_ckpt)
    return p.parent / f"compare_{p.stem}.mp4"


def main():
    args = parse_args()

    if len(args.checkpoints) < 2:
        raise SystemExit("Need at least 2 checkpoints to compare.")

    if args.labels is None:
        labels = [label_from_path(c) for c in args.checkpoints]
    else:
        if len(args.labels) != len(args.checkpoints):
            raise SystemExit(
                f"Got {len(args.checkpoints)} checkpoints but {len(args.labels)} labels."
            )
        labels = args.labels

    rows, cols = grid_shape_for(len(args.checkpoints))
    print(f"Comparing {len(args.checkpoints)} checkpoints in a {rows}x{cols} grid:")

    per_ckpt_frames: List[List[np.ndarray]] = []
    for ckpt, label in zip(args.checkpoints, labels):
        ckpt_path = Path(ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        if ckpt_path.suffix.lower() != ".onnx":
            raise ValueError(f"Expected .onnx, got: {ckpt_path}")
        print(f"  - {label}  ({ckpt_path.name})")
        frames = run_one_rollout(str(ckpt_path), args.env_config, args.seed, label)
        per_ckpt_frames.append(frames)
        print(f"      ran {len(frames)} frames")

    print("Stitching grid...")
    composite = stitch_grid(per_ckpt_frames)

    out_path = Path(args.output) if args.output else default_output(args.checkpoints[0])
    save_mp4(composite, out_path, args.fps)
    print(f"Saved {len(composite)} composite frames -> {out_path}")


if __name__ == "__main__":
    main()
