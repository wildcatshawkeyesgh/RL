import math

import torch
import torch.nn as nn
from torch.distributions import Normal


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_width, hidden_layers, output_dim, output_gain):
        super().__init__()
        layers = []
        previous = input_dim
        for _ in range(hidden_layers):
            layers.append(self.init_linear(nn.Linear(previous, hidden_width), math.sqrt(2)))
            layers.append(nn.Tanh())
            previous = hidden_width
        layers.append(self.init_linear(nn.Linear(previous, output_dim), output_gain))
        self.net = nn.Sequential(*layers)

    def init_linear(self, layer, gain):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)
        return layer

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_width = 128,
        hidden_layers = 2,
        init_log_std = 0.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.actor = MLP(obs_dim, hidden_width, hidden_layers, action_dim, output_gain = 0.01)
        self.critic = MLP(obs_dim, hidden_width, hidden_layers, 1, output_gain = 1.0)
        self.log_std = nn.Parameter(
            torch.full((action_dim,), float(init_log_std), dtype=torch.float32)
        )

    def get_action_and_value(self, obs, action = None):
        mean = self.actor(obs)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        value = self.critic(obs).squeeze(-1)

        if action is None:
            sampled = dist.sample()
            log_prob = dist.log_prob(sampled).sum(-1)
            entropy = dist.entropy().sum(-1)
            return sampled, log_prob, entropy, value

        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value

    def actor_mean(self, obs):
        return self.actor(obs)
