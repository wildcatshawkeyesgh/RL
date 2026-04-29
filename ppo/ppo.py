from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from ppo.model import ActorCritic


@dataclass
class UpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_frac: float
    explained_variance: float
    ratio_max: float
    ratio_min: float


class PPOAgent:
    def __init__(
        self,
        obs_dim,
        action_dim,
        config,
        device,
    ):
        self.config = config
        self.device = device
        self.network = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_width=config.hidden_width,
            hidden_layers=config.hidden_layers,
            init_log_std=config.init_log_std,
        ).to(device)
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=config.learning_rate
        )

    @torch.no_grad()
    def select_action(self, obs):
        action, log_prob, _entropy, value = self.network.get_action_and_value(obs)
        return action, log_prob, value

    def update(self, buffer):
        cfg = self.config

        with torch.no_grad():
            y_pred = buffer.values.flatten()
            y_true = buffer.returns.flatten()
            var_y = y_true.var()
            explained_variance = float(
                1.0 - ((y_true - y_pred).var() / (var_y + 1e-8)).item()
            )

        advantages = buffer.advantages
        if cfg.normalize_advantage:
            buffer.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        last_policy_loss = last_value_loss = last_entropy = last_kl_divergence = last_clip_frac = 0.0
        ratio_max_seen = float("-inf")
        ratio_min_seen = float("inf")

        for _epoch in range(cfg.update_epochs):
            for minibatch_obs, minibatch_actions, minibatch_old_log_probs, minibatch_advantages, minibatch_returns in buffer.get_minibatches(
                cfg.minibatch_size
            ):
                new_log_prob, entropy, new_value = self.network.get_action_and_value(
                    minibatch_obs, minibatch_actions
                )

                log_ratio = new_log_prob - minibatch_old_log_probs
                ratio = log_ratio.exp()

                surrogate_1 = -minibatch_advantages * ratio
                surrogate_2 = -minibatch_advantages * torch.clamp(
                    ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef
                )
                policy_loss = torch.max(surrogate_1, surrogate_2).mean()

                value_loss = 0.5 * ((new_value - minibatch_returns) ** 2).mean()

                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + cfg.vf_coef * value_loss
                    + cfg.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), cfg.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean()
                    ratio_max = float(ratio.max().item())
                    ratio_min = float(ratio.min().item())

                last_policy_loss = float(policy_loss.item())
                last_value_loss = float(value_loss.item())
                last_entropy = float(-entropy_loss.item())
                last_kl_divergence = float(approx_kl.item())
                last_clip_frac = float(clip_frac.item())
                if ratio_max > ratio_max_seen:
                    ratio_max_seen = ratio_max
                if ratio_min < ratio_min_seen:
                    ratio_min_seen = ratio_min

        return UpdateStats(
            policy_loss = last_policy_loss,
            value_loss = last_value_loss,
            entropy = last_entropy,
            approx_kl = last_kl_divergence,
            clip_frac = last_clip_frac,
            explained_variance = explained_variance,
            ratio_max = ratio_max_seen,
            ratio_min = ratio_min_seen,
        )
