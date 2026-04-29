import torch


class RolloutBuffer:
    def __init__(
        self,
        num_steps,
        num_envs,
        obs_dim,
        action_dim,
        device,
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        S, E = num_steps, num_envs
        self.obs = torch.zeros(S, E, obs_dim, dtype=torch.float32, device=device)
        self.actions = torch.zeros(S, E, action_dim, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(S, E, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(S, E, dtype=torch.float32, device=device)
        self.dones = torch.zeros(S, E, dtype=torch.float32, device=device)
        self.values = torch.zeros(S, E, dtype=torch.float32, device=device)

        self.advantages = torch.zeros(S, E, dtype=torch.float32, device=device)
        self.returns = torch.zeros(S, E, dtype=torch.float32, device=device)

    def add(
        self,
        step,
        obs,
        action,
        log_prob,
        reward,
        done,
        value,
    ):
        self.obs[step] = obs
        self.actions[step] = action
        self.log_probs[step] = log_prob.detach()
        self.rewards[step] = reward
        self.dones[step] = done
        self.values[step] = value.detach()

    @torch.no_grad()
    def compute_gae(
        self,
        last_value,
        last_done,
        gamma,
        gae_lambda,
    ):
        S = self.num_steps
        gae = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for t in reversed(range(S)):
            if t == S - 1:
                next_value = last_value
                next_nonterminal = 1.0 - last_done
            else:
                next_value = self.values[t + 1]
                next_nonterminal = 1.0 - self.dones[t + 1]
            delta = (
                self.rewards[t]
                + gamma * next_value * next_nonterminal
                - self.values[t]
            )
            gae = delta + gamma * gae_lambda * next_nonterminal * gae
            self.advantages[t] = gae
        self.returns[:] = self.advantages + self.values

    def get_minibatches(self, batch_size, generator = None):
        total = self.num_steps * self.num_envs
        flat_obs = self.obs.reshape(total, self.obs_dim)
        flat_actions = self.actions.reshape(total, self.action_dim)
        flat_log_probs = self.log_probs.reshape(total)
        flat_advantages = self.advantages.reshape(total)
        flat_returns = self.returns.reshape(total)

        idx = torch.randperm(total, generator=generator, device=self.device)
        for start in range(0, total, batch_size):
            end = start + batch_size
            batch_indices = idx[start:end]
            yield (
                flat_obs[batch_indices],
                flat_actions[batch_indices],
                flat_log_probs[batch_indices],
                flat_advantages[batch_indices],
                flat_returns[batch_indices],
            )
