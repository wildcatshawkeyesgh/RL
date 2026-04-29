"""PPO config schema and YAML loader.

Defaults live in `ppo/config.yaml` (the canonical source of truth). This file
defines the typed schema (`PPOConfig` dataclass) and the loader. CLI overrides
in `scripts/ppo_impl.py` apply on top of whatever YAML file is loaded.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


@dataclass
class PPOConfig:
    """Typed schema for PPO hyperparameters. All fields required (no defaults
    inline) -- defaults come from the YAML file. Construct via `load_config`.
    """

    # Training budget.
    total_timesteps: int
    num_envs: int
    num_steps: int

    # PPO update.
    update_epochs: int
    minibatch_size: int

    # Optimization.
    learning_rate: float
    max_grad_norm: float

    # PPO clipping + GAE.
    clip_coef: float
    gamma: float
    gae_lambda: float

    # Loss coefficients.
    vf_coef: float
    ent_coef: float

    # Misc.
    normalize_advantage: bool

    # Network.
    hidden_width: int
    hidden_layers: int
    init_log_std: float

    # Env.
    env_id: str
    env_config_path: str

    # Checkpointing.
    checkpoint_dir: str
    checkpoint_every_iters: int
    onnx_export: bool

    # Logging.
    log_dir: str
    run_name: str

    # Eval video.
    video_enabled: bool
    video_every_iters: int
    video_fps: int
    video_deterministic: bool

    # Runtime.
    device: str
    seed: int


def load_config(path = None, **overrides):
    """Load a `PPOConfig` from YAML, with optional keyword overrides."""
    yaml_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    data.update(overrides)
    return PPOConfig(**data)
