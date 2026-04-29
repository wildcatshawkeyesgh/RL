import time

import numpy as np
import torch


def get_info_value(info, key, env_idx, default = 0):
    val = info.get(key, default)
    if isinstance(val, (np.ndarray, list, tuple)):
        if env_idx < len(val):
            return val[env_idx]
        return default
    return val


class PPOTrainer:
    def __init__(
        self,
        cfg,
        envs,
        agent,
        buffer,
        device,
        logger,
        output_dir,
        save_checkpoint_fn = None,
        record_eval_video_fn = None,
    ):
        self.cfg = cfg
        self.envs = envs
        self.agent = agent
        self.buffer = buffer
        self.device = device
        self.logger = logger
        self.output_dir = output_dir
        self.save_checkpoint_fn = save_checkpoint_fn
        self.record_eval_video_fn = record_eval_video_fn

    def train(self):
        cfg = self.cfg
        device = self.device
        envs = self.envs
        agent = self.agent
        buffer = self.buffer
        logger = self.logger

        obs_dim = envs.single_observation_space.shape[0]

        obs_np, _info = envs.reset(seed=cfg.seed)
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        done = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)

        episode_return = np.zeros(cfg.num_envs, dtype=np.float64)
        episode_length = np.zeros(cfg.num_envs, dtype=np.int64)

        completed_returns = []
        completed_lengths = []
        completed_hits = []
        completed_real_destroyed = []
        completed_decoys_destroyed = []

        total_iterations = max(1, cfg.total_timesteps // (cfg.num_envs * cfg.num_steps))
        global_step = 0
        start_time = time.time()

        for iteration in range(1, total_iterations + 1):
            for step in range(cfg.num_steps):
                global_step += cfg.num_envs

                with torch.no_grad():
                    action, log_prob, value = agent.select_action(obs)

                action_np = action.cpu().numpy()
                action_np = np.clip(action_np, -1.0, 1.0)

                next_obs_np, reward_np, terminated_np, truncated_np, info = envs.step(
                    action_np
                )
                done_np = np.logical_or(terminated_np, truncated_np).astype(np.float32)

                buffer.add(
                    step=step,
                    obs=obs,
                    action=action,
                    log_prob=log_prob,
                    reward=torch.as_tensor(reward_np, dtype=torch.float32, device=device),
                    done=done,
                    value=value,
                )

                episode_return += reward_np
                episode_length += 1
                for env_idx in range(cfg.num_envs):
                    if done_np[env_idx]:
                        completed_returns.append(float(episode_return[env_idx]))
                        completed_lengths.append(int(episode_length[env_idx]))
                        completed_hits.append(int(get_info_value(info, "hits", env_idx, 0)))
                        completed_real_destroyed.append(
                            int(get_info_value(info, "real_destroyed", env_idx, 0))
                        )
                        completed_decoys_destroyed.append(
                            int(get_info_value(info, "decoys_destroyed", env_idx, 0))
                        )
                        episode_return[env_idx] = 0.0
                        episode_length[env_idx] = 0

                obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
                done = torch.as_tensor(done_np, dtype=torch.float32, device=device)

            with torch.no_grad():
                _action, _log_prob, last_value = agent.select_action(obs)
            buffer.compute_gae(last_value, done, cfg.gamma, cfg.gae_lambda)

            stats = agent.update(buffer)

            steps_per_second = int(global_step / max(1e-6, time.time() - start_time))
            if completed_returns:
                mean_return = float(np.mean(completed_returns))
                mean_length = float(np.mean(completed_lengths))
                mean_hits = float(np.mean(completed_hits))
                mean_real_destroyed = float(np.mean(completed_real_destroyed))
                mean_decoys_destroyed = float(np.mean(completed_decoys_destroyed))
            else:
                mean_return = mean_length = mean_hits = mean_real_destroyed = mean_decoys_destroyed = 0.0

            logger.add_scalars(
                {
                    "charts/episode_return": mean_return,
                    "charts/episode_length": mean_length,
                    "charts/hits_per_episode": mean_hits,
                    "charts/real_destroyed_per_episode": mean_real_destroyed,
                    "charts/decoys_destroyed_per_episode": mean_decoys_destroyed,
                    "charts/episodes_per_iter": len(completed_returns),
                    "charts/sps": steps_per_second,
                    "losses/policy_loss": stats.policy_loss,
                    "losses/value_loss": stats.value_loss,
                    "losses/entropy": stats.entropy,
                    "losses/approx_kl": stats.approx_kl,
                    "losses/clip_frac": stats.clip_frac,
                    "losses/explained_variance": stats.explained_variance,
                    "losses/ratio_max": stats.ratio_max,
                    "losses/ratio_min": stats.ratio_min,
                    "policy/log_std_mean": float(agent.network.log_std.mean().item()),
                },
                global_step,
            )

            print(
                f"iter={iteration:5d}  step={global_step:9d}  sps={steps_per_second:5d}  "
                f"ret={mean_return:7.2f}  len={mean_length:5.1f}  "
                f"hits={mean_hits:.2f}  pl={stats.policy_loss:+.3f}  "
                f"vl={stats.value_loss:.3f}  H={stats.entropy:.3f}  "
                f"kl={stats.approx_kl:.4f}  cf={stats.clip_frac:.3f}"
            )

            completed_returns.clear()
            completed_lengths.clear()
            completed_hits.clear()
            completed_real_destroyed.clear()
            completed_decoys_destroyed.clear()

            if iteration % cfg.checkpoint_every_iters == 0 or iteration == total_iterations:
                if self.save_checkpoint_fn is not None:
                    checkpoint_path = self.save_checkpoint_fn(
                        agent, iteration, global_step, cfg, obs_dim, self.output_dir
                    )
                    print(f"  [checkpoint] {checkpoint_path}")

            if cfg.video_enabled and (
                iteration % cfg.video_every_iters == 0 or iteration == total_iterations
            ):
                if self.record_eval_video_fn is not None:
                    episode_reward, hits, episode_steps = self.record_eval_video_fn(
                        agent, cfg, device, global_step, logger
                    )
                    print(
                        f"  [video] eval episode: ret={episode_reward:+.2f} hits={hits} "
                        f"steps={episode_steps}"
                    )

        envs.close()
        logger.close()
