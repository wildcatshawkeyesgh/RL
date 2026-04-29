import argparse
from pathlib import Path

import gymnasium
import numpy as np
import onnxruntime as ort

import iads.gym_env
from ppo.render import render_frame


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .onnx file (the actor exported by scripts/ppo_impl.py).",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default="iads/default.yaml",
        help="Scenario YAML/JSON path. Defaults to iads/default.yaml.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to record back-to-back (concatenated into one video).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Env reset seed (per first-episode reset). Default: random.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Where to save the MP4. Default: alongside the checkpoint as <stem>.mp4.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="MP4 / live-animation playback rate. Default 10.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Show frames in a matplotlib window instead of writing MP4 (needs a display).",
    )
    return parser.parse_args()


def run_one_episode(
    session,
    env,
    seed,
    episode_idx,
):
    obs_np, info = env.reset(seed=seed)

    frames = []
    total_reward = 0.0
    hits = 0
    step = 0
    done = False
    truncated = False

    while not (done or truncated):
        frames.append(
            render_frame(
                env.unwrapped.sim,
                title=(
                    f"ep {episode_idx + 1}  step {step}  hits {hits}  "
                    f"R {total_reward:+.1f}"
                ),
            )
        )

        obs_batch = obs_np.astype(np.float32).reshape(1, -1)
        mean_action = session.run(None, {"obs": obs_batch})[0][0]
        action = np.clip(mean_action, -1.0, 1.0)

        obs_np, reward, done, truncated, info = env.step(action)
        total_reward += float(reward)
        hits = int(info.get("hits", hits))
        step += 1

    frames.append(
        render_frame(
            env.unwrapped.sim,
            title=(
                f"ep {episode_idx + 1}  step {step}  hits {hits}  "
                f"R {total_reward:+.1f}  (terminal)"
            ),
        )
    )

    return frames, total_reward, hits, step


def show_live(frames, fps):
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 7.5))
    ax.set_axis_off()
    img = ax.imshow(frames[0])

    def update(frame_idx):
        img.set_data(frames[frame_idx])
        return [img]

    interval_ms = max(1, int(1000 / fps))
    _animation = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval_ms,
        blit=True,
        repeat=False,
    )
    plt.show()


def save_mp4(frames, output_path, fps):
    import imageio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        str(output_path),
        frames,
        fps=fps,
        codec="libx264",
        quality=8,
    )


def default_output_path(checkpoint_path):
    path = Path(checkpoint_path)
    return path.parent / f"{path.stem}.mp4"


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    session = ort.InferenceSession(str(checkpoint_path), providers=["CPUExecutionProvider"])
    input_shape = session.get_inputs()[0].shape
    output_shape = session.get_outputs()[0].shape
    print(f"Loaded {checkpoint_path.name}")
    print(f"  inputs  : obs {input_shape}")
    print(f"  outputs : action_mean {output_shape}")

    env = gymnasium.make(
        "IADSPenetration-v0",
        config_path=args.env_config,
    )

    all_frames = []
    for episode in range(args.episodes):
        seed = (args.seed + episode) if args.seed is not None else None
        frames, total_reward, hits, steps = run_one_episode(session, env, seed, episode)
        print(
            f"  episode {episode + 1}: return={total_reward:+8.2f}  hits={hits}  steps={steps}"
        )
        all_frames.extend(frames)

    env.close()

    if args.live:
        show_live(all_frames, args.fps)
        return

    output_path = Path(args.output) if args.output else default_output_path(args.checkpoint)
    save_mp4(all_frames, output_path, args.fps)
    print(f"Saved {len(all_frames)} frames -> {output_path}")


if __name__ == "__main__":
    main()
