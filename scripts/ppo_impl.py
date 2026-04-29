import argparse
import dataclasses
from pathlib import Path

import gymnasium
import numpy as np
import torch

import iads.gym_env
from ppo.buffer import RolloutBuffer
from ppo.config import load_config
from ppo.logger import TBLogger
from ppo.ppo import PPOAgent
from ppo.render import render_frame
from ppo.trainer import PPOTrainer
from ppo.utils import get_best_gpu


def select_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                device_id = get_best_gpu(strategy="utilization")
                device = torch.device(f"cuda:{device_id}")
                print(f"Selected GPU: {device_id}")
                return device
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def make_env(cfg, env_idx):
    def thunk():
        env = gymnasium.make(cfg.env_id, config_path=cfg.env_config_path)
        env.action_space.seed(cfg.seed + env_idx)
        return env

    return thunk


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (defaults to ppo/config.yaml)",
    )
    pre_args, remaining = pre_parser.parse_known_args()

    base = load_config(pre_args.config)

    parser = argparse.ArgumentParser(parents=[pre_parser])
    for field in dataclasses.fields(base):
        flag = "--" + field.name.replace("_", "-")
        default = getattr(base, field.name)
        if isinstance(default, bool):
            parser.add_argument(
                flag,
                type=lambda s: s.lower() in {"1", "true", "yes"},
                default=default,
            )
        elif isinstance(default, int) and not isinstance(default, bool):
            parser.add_argument(flag, type=int, default=default)
        elif isinstance(default, float):
            parser.add_argument(flag, type=float, default=default)
        else:
            parser.add_argument(flag, type=str, default=default)

    args = parser.parse_args()
    overrides = {field.name: getattr(args, field.name) for field in dataclasses.fields(base)}
    return load_config(pre_args.config, **overrides)


def save_checkpoint(
    agent,
    iteration,
    global_step,
    cfg,
    obs_dim,
    output_dir,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    pt_path = output_dir / f"checkpoint_{global_step:09d}.pt"
    torch.save(
        {
            "iteration": iteration,
            "global_step": global_step,
            "model_state_dict": agent.network.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "config": dataclasses.asdict(cfg),
            "obs_dim": obs_dim,
            "action_dim": agent.network.action_dim,
        },
        pt_path,
    )

    if cfg.onnx_export:
        onnx_path = output_dir / f"checkpoint_{global_step:09d}.onnx"
        export_actor_onnx(agent, obs_dim, onnx_path)

    return pt_path


def export_actor_onnx(agent, obs_dim, path):
    agent.network.eval()
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32, device=agent.device)

    class ActorWrapper(torch.nn.Module):
        def __init__(self, actor):
            super().__init__()
            self.actor = actor

        def forward(self, obs):
            return self.actor(obs)

    wrapper = ActorWrapper(agent.network.actor).to(agent.device).eval()
    torch.onnx.export(
        wrapper,
        dummy,
        str(path),
        input_names=["obs"],
        output_names=["action_mean"],
        dynamic_axes={"obs": {0: "batch"}, "action_mean": {0: "batch"}},
        opset_version=17,
    )
    agent.network.train()


@torch.no_grad()
def record_eval_video(
    agent,
    cfg,
    device,
    global_step,
    logger,
):
    env = gymnasium.make(cfg.env_id, config_path=cfg.env_config_path)
    obs_np, info = env.reset(seed=cfg.seed + 7919)

    frames = []
    total_reward = 0.0
    hits = 0
    step = 0
    done = False
    truncated = False

    agent.network.eval()
    while not (done or truncated):
        obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        if cfg.video_deterministic:
            action = agent.network.actor(obs_tensor)
        else:
            action, _log_prob, _entropy, _value = agent.network.get_action_and_value(obs_tensor)
        action_np = np.clip(action.squeeze(0).cpu().numpy(), -1.0, 1.0)

        frames.append(
            render_frame(
                env.unwrapped.sim,
                title=(
                    f"step={step}  hits={hits}  R={total_reward:+.1f}  "
                    f"global={global_step}"
                ),
            )
        )

        obs_np, reward, done, truncated, info = env.step(action_np)
        total_reward += float(reward)
        hits = int(info.get("hits", hits))
        step += 1

    frames.append(
        render_frame(
            env.unwrapped.sim,
            title=(
                f"step={step}  hits={hits}  R={total_reward:+.1f}  "
                f"global={global_step}  (terminal)"
            ),
        )
    )
    env.close()
    agent.network.train()

    logger.add_video("rollouts/eval", frames, step=global_step, fps=cfg.video_fps)
    return total_reward, hits, step


if __name__ == "__main__":
    cfg = parse_args()

    device = select_device(cfg.device)
    print(f"Device: {device}")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    envs = gymnasium.vector.SyncVectorEnv([make_env(cfg, i) for i in range(cfg.num_envs)])
    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    print(f"obs_dim={obs_dim}  action_dim={action_dim}  num_envs={cfg.num_envs}")

    agent = PPOAgent(obs_dim, action_dim, cfg, device)
    buffer = RolloutBuffer(cfg.num_steps, cfg.num_envs, obs_dim, action_dim, device)

    output_dir = Path(cfg.checkpoint_dir) / cfg.run_name
    logger = TBLogger(cfg.log_dir, cfg.run_name, config=dataclasses.asdict(cfg))

    trainer = PPOTrainer(
        cfg=cfg,
        envs=envs,
        agent=agent,
        buffer=buffer,
        device=device,
        logger=logger,
        output_dir=output_dir,
        save_checkpoint_fn=save_checkpoint,
        record_eval_video_fn=record_eval_video,
    )
    trainer.train()
