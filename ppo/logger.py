"""Reusable TensorBoard logging wrapper.

Exposes a small, opinionated `TBLogger` class so other projects can drop the
PPO library in without rewriting the SummaryWriter glue. Hides the tensor
layout details add_video expects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


FrameLike = Union[np.ndarray, Sequence[np.ndarray]]


class TBLogger:
    """Thin wrapper around `SummaryWriter` for the PPO training loop.

    Usage:
        logger = TBLogger("runs", "baseline-v2", config=dataclasses.asdict(cfg))
        logger.add_scalars({"charts/episode_return": r, "losses/policy_loss": pl}, step)
        logger.add_video("rollouts/eval", list_of_HxWx3_uint8_frames, step, fps=10)
        logger.close()
    """

    def __init__(
        self,
        log_dir: str,
        run_name: str,
        *,
        config: Optional[Mapping] = None,
    ):
        self.run_dir = Path(log_dir) / run_name
        self.writer = SummaryWriter(str(self.run_dir))
        if config is not None:
            self.writer.add_text("config", format_config(config), 0)

    # -----------------------------------------------------------------
    # Scalars
    # -----------------------------------------------------------------

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, float(value), step)

    def add_scalars(self, scalars: Mapping[str, float], step: int) -> None:
        """Batch-log a dict of {tag: value}. Tags follow `<group>/<name>` so TB groups them."""
        for tag, value in scalars.items():
            self.writer.add_scalar(tag, float(value), step)

    # -----------------------------------------------------------------
    # Text
    # -----------------------------------------------------------------

    def add_text(self, tag: str, text: str, step: int = 0) -> None:
        self.writer.add_text(tag, text, step)

    # -----------------------------------------------------------------
    # Video
    # -----------------------------------------------------------------

    def add_video(
        self,
        tag: str,
        frames: FrameLike,
        step: int,
        fps: int = 10,
    ) -> None:
        """Log a video given either a list of (H, W, 3) uint8 frames or a
        (T, H, W, 3) uint8 array. Internally reshapes to TB's (1, T, C, H, W).
        """
        if isinstance(frames, np.ndarray):
            arr = frames
        else:
            arr = np.stack(list(frames), axis=0)

        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(
                f"Expected frames as (T, H, W, 3); got shape {arr.shape}"
            )
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        # (T, H, W, 3) -> (T, 3, H, W) -> (1, T, 3, H, W)
        arr = arr.transpose(0, 3, 1, 2)[np.newaxis, ...]
        self.writer.add_video(tag, torch.from_numpy(arr), step, fps=fps)

    # -----------------------------------------------------------------
    # Misc
    # -----------------------------------------------------------------

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()


def format_config(cfg: Mapping) -> str:
    """Render a config dict as a TB-friendly markdown text block."""
    lines = ["| key | value |", "|-----|-------|"]
    for k, v in cfg.items():
        lines.append(f"| `{k}` | `{v}` |")
    return "\n".join(lines)
