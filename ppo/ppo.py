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
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        self.optimizer, T_max=300, eta_min=0.000003)

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

        policy_losses = []
        value_losses = []
        entropies = []
        kl_divergences = []
        clip_fracs = []
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

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(-entropy_loss.item()))
                kl_divergences.append(float(approx_kl.item()))
                clip_fracs.append(float(clip_frac.item()))
                if ratio_max > ratio_max_seen:
                    ratio_max_seen = ratio_max
                if ratio_min < ratio_min_seen:
                    ratio_min_seen = ratio_min
        self.scheduler.step()
        return UpdateStats(
            policy_loss = sum(policy_losses) / len(policy_losses),
            value_loss = sum(value_losses) / len(value_losses),
            entropy = sum(entropies) / len(entropies),
            approx_kl = sum(kl_divergences) / len(kl_divergences),
            clip_frac = sum(clip_fracs) / len(clip_fracs),
            explained_variance = explained_variance,
            ratio_max = ratio_max_seen,
            ratio_min = ratio_min_seen,
        )
