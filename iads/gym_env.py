"""
gym_env.py - Gymnasium wrapper for IADS Penetration simulation.

Designed for a basic PPO agent with 4 agents (2 missiles + 2 decoys). The
agent has direct continuous control over every missile from t=0. On reset(),
a random launch zone is selected and all missiles spawn at random positions
inside that zone, heading +y (north).

Observation layout (flat float32, all values clipped to [0, 1]):
  Per agent (NUM_AGENTS x AGENT_FEATURES):
    pos (x, y), vel (vx, vy), heading (sin, cos), altitude,
    alive flag, is_decoy flag, evading flag, agent_id one-hot (NUM_AGENTS dims)
  Per SAM (NUM_SAMS x SAM_FEATURES):
    pos (x, y), range, active flag, tracking_agent multi-hot (NUM_AGENTS dims)
  Per radar (NUM_RADARS x RADAR_FEATURES):
    pos (x, y), range, active flag, tracking_agent multi-hot (NUM_AGENTS dims)
  Per target (NUM_TARGETS x TARGET_FEATURES):
    pos (x, y), alive flag
  Globals:
    time_remaining (normalized)

Action space: Box(-1, 1, shape=(NUM_AGENTS * 3,), float32)
  Per agent: (heading_delta, altitude_delta, evasion).
    heading_delta  scaled to +/- 15 deg per step.
    altitude_delta scaled to +/- 2 km per step.
    evasion        engaged when value >= 0.

Reward (per the PPO brief):
  +100  per real missile reaching a target
  -20   per real missile destroyed (intercepted or fuel-exhausted)
  -5    per decoy destroyed
  -20   once at episode end if it timed out with zero hits
  +0.05 * (prev_min_dist - curr_min_dist)  potential-based progress shaping,
        using minimum distance from any living REAL missile to any alive target
  -0.01 per step
"""

import json
from pathlib import Path

import gymnasium
import numpy as np
import yaml
from gymnasium import spaces

from iads.sim_engine import EntityType, InterceptorState, IADSSimulation, MissileState

# ---------------------------------------------------------------------------
# Per-entity feature counts (functions of NUM_AGENTS, fixed at 4 for the brief)
# ---------------------------------------------------------------------------
NUM_AGENTS = 4  # 2 real missiles + 2 decoys
PER_AGENT_ACTION_DIM = 3  # heading_delta, altitude_delta, evasion

# Per-agent obs:
#   pos(2) + vel(2) + heading_sincos(2) + alt(1)
# + alive(1) + is_decoy(1) + evading(1)
# + agent_id one-hot (NUM_AGENTS)
_AGENT_FEATURES_BASE = 10
_AGENT_FEATURES = _AGENT_FEATURES_BASE + NUM_AGENTS

# Per-SAM/radar:  pos(2) + range(1) + active(1) + tracking_multi_hot(NUM_AGENTS)
_THREAT_FEATURES_BASE = 4
_THREAT_FEATURES = _THREAT_FEATURES_BASE + NUM_AGENTS

# Per-target: pos(2) + alive(1)
_TARGET_FEATURES = 3


def load_scenario_config(path) -> dict:
    """Load an IADS scenario config from YAML or JSON, picked by extension."""
    p = Path(path)
    with open(p) as f:
        if p.suffix.lower() in (".yaml", ".yml"):
            return yaml.safe_load(f) or {}
        return json.load(f)


class IADSEnv(gymnasium.Env):
    """Gymnasium wrapper for IADS Penetration with the PPO-brief layout."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config_path: str = None,
        config_overrides: dict = None,
        render_mode: str = None,
    ):
        super().__init__()

        self._base_config = {}
        if config_path is not None:
            self._base_config = load_scenario_config(config_path)
        if config_overrides is not None:
            self._base_config.update(config_overrides)

        self.render_mode = render_mode

        # Resolve effective config so we can size observations.
        _tmp = IADSSimulation(dict(self._base_config) if self._base_config else None)
        cfg = _tmp.config
        del _tmp

        self.num_radars = cfg["num_radars"]
        self.num_sams = cfg["num_sams"]
        self.num_targets = cfg["num_targets"]
        self.num_real = cfg["num_real_missiles"]
        self.num_decoys = cfg["num_decoy_missiles"]
        self.num_missiles = self.num_real + self.num_decoys
        if self.num_missiles != NUM_AGENTS:
            raise ValueError(
                f"This env is fixed to NUM_AGENTS={NUM_AGENTS} "
                f"(2 real + 2 decoys). Got {self.num_real} real + "
                f"{self.num_decoys} decoys = {self.num_missiles}."
            )

        self.area_width = cfg["area_width"]
        self.area_height = cfg["area_height"]
        self.max_time = cfg["max_time"]
        self.max_speed = float(cfg.get("missile_speed", 3.0))
        self.alt_max = 25.0  # matches sim_engine clip in steer_missile

        self._reward_target_hit = float(cfg.get("reward_target_hit", 100.0))
        self._reward_missile_destroyed = float(
            cfg.get("reward_missile_destroyed", -20.0)
        )
        self._reward_decoy_destroyed = float(cfg.get("reward_decoy_destroyed", -5.0))
        self._reward_timeout_zero_hits = float(
            cfg.get("reward_timeout_zero_hits", -20.0)
        )
        self._reward_progress_alpha = float(cfg.get("reward_progress_alpha", 0.05))
        self._reward_time_penalty = float(cfg.get("reward_time_penalty", -0.01))
        self._reward_decoy_bait = float(cfg.get("reward_decoy_bait", 0.0))
        self._reward_decoy_bait_radius = float(
            cfg.get("reward_decoy_bait_radius_km", 15.0)
        )

        self._layout = self.compute_obs_layout()
        obs_dim = self._layout["total"]

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(NUM_AGENTS * PER_AGENT_ACTION_DIM,),
            dtype=np.float32,
        )

        # Runtime state (populated by reset)
        self.sim: IADSSimulation = None
        self._launch_zone_idx: int = 0
        self._np_random = np.random.default_rng()
        self._prev_min_dist: float = 0.0
        self._target_hits_this_ep: int = 0

    # -----------------------------------------------------------------------
    # Observation layout
    # -----------------------------------------------------------------------

    def compute_obs_layout(self) -> dict:
        agents_start = 0
        agents_end = agents_start + NUM_AGENTS * _AGENT_FEATURES
        sams_start = agents_end
        sams_end = sams_start + self.num_sams * _THREAT_FEATURES
        radars_start = sams_end
        radars_end = radars_start + self.num_radars * _THREAT_FEATURES
        targets_start = radars_end
        targets_end = targets_start + self.num_targets * _TARGET_FEATURES
        globals_start = targets_end
        globals_end = globals_start + 1  # time_remaining

        return dict(
            agents_start=agents_start,
            agents_end=agents_end,
            sams_start=sams_start,
            sams_end=sams_end,
            radars_start=radars_start,
            radars_end=radars_end,
            targets_start=targets_start,
            targets_end=targets_end,
            globals_start=globals_start,
            globals_end=globals_end,
            total=globals_end,
        )

    # -----------------------------------------------------------------------
    # Gymnasium API
    # -----------------------------------------------------------------------

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        config = dict(self._base_config) if self._base_config else {}
        if seed is not None:
            config["seed"] = seed
            config["layout_seed"] = seed

        self.sim = IADSSimulation(config)
        self.launch_all_missiles()
        self._target_hits_this_ep = 0
        self._prev_min_dist = self.compute_min_real_to_target_dist()

        obs = self.build_observation()
        info = self.build_info()
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        expected = NUM_AGENTS * PER_AGENT_ACTION_DIM
        if action.shape[0] != expected:
            raise ValueError(
                f"Expected action of shape ({expected},), got {action.shape}"
            )

        self.apply_action(action)
        events, _done = self.sim.step()

        # Brief reward terms.
        reward = self._reward_time_penalty

        for ev in events:
            t = ev["type"]
            if t == "target_hit":
                reward += self._reward_target_hit
                self._target_hits_this_ep += 1
            elif t == "real_intercepted":
                reward += self._reward_missile_destroyed
            elif t == "decoy_intercepted":
                reward += self._reward_decoy_destroyed
                if self._reward_decoy_bait != 0.0 and self.is_decoy_baiting(ev):
                    reward += self._reward_decoy_bait
            elif t == "fuel_exhausted":
                if ev.get("is_real"):
                    reward += self._reward_missile_destroyed
                else:
                    reward += self._reward_decoy_destroyed

        # Potential-based progress shaping on minimum distance from any living
        # REAL missile to any alive target. Decoys do not count.
        curr_min_dist = self.compute_min_real_to_target_dist()
        reward += self._reward_progress_alpha * (self._prev_min_dist - curr_min_dist)
        self._prev_min_dist = curr_min_dist

        # Episode termination
        all_targets_dead = all(not t.alive for t in self.sim.targets)
        real_in_flight = any(
            m.missile_type == EntityType.REAL_MISSILE
            and m.state == MissileState.IN_FLIGHT
            for m in self.sim.missiles
        )
        terminated = all_targets_dead or not real_in_flight
        truncated = (self.sim.time >= self.sim.max_time) and not terminated

        if truncated and self._target_hits_this_ep == 0:
            reward += self._reward_timeout_zero_hits

        obs = self.build_observation()
        info = self.build_info()
        return obs, float(reward), terminated, truncated, info

    def render(self):
        return None

    def close(self):
        self.sim = None

    # -----------------------------------------------------------------------
    # Launch
    # -----------------------------------------------------------------------

    def launch_all_missiles(self):
        """Pick a random launch zone, spawn all missiles inside it heading +y."""
        zones = self.sim.config["launch_zones"]
        self._launch_zone_idx = int(self._np_random.integers(0, len(zones)))
        zone = zones[self._launch_zone_idx]

        for m in self.sim.missiles:
            x = float(self._np_random.uniform(zone["x_min"], zone["x_max"]))
            y = float(self._np_random.uniform(zone["y_min"], zone["y_max"]))
            m.pos = np.array([x, y, m.altitude], dtype=float)
            m.velocity = np.array([0.0, m.speed, 0.0], dtype=float)
            m.state = MissileState.IN_FLIGHT
            m.waypoints = []
            m.current_waypoint_idx = 0
            m.target_id = None
            m.trail = [m.pos.copy()]
            m.distance_traveled = 0.0
            m.evading = False

    # -----------------------------------------------------------------------
    # Action handling
    # -----------------------------------------------------------------------

    def apply_action(self, action: np.ndarray):
        """Apply 12-dim action: per-agent (heading, altitude, evasion)."""
        for i, missile in enumerate(self.sim.missiles):
            if missile.state != MissileState.IN_FLIGHT:
                continue
            b = i * PER_AGENT_ACTION_DIM
            heading_delta = float(action[b + 0]) * 15.0
            altitude_delta = float(action[b + 1]) * 2.0
            evasion = bool(action[b + 2] >= 0.0)
            self.sim.steer_missile(missile.id, heading_delta, altitude_delta)
            self.sim.set_missile_evading(missile.id, evasion)

    # -----------------------------------------------------------------------
    # Reward helpers
    # -----------------------------------------------------------------------

    def is_decoy_baiting(self, decoy_intercept_event: dict) -> bool:
        """True if a still-alive real missile is within the bait radius of the
        decoy at the moment it was intercepted. Uses the event's logged
        `position` so it stays correct even if sim state changes during the step.
        """
        pos = decoy_intercept_event.get("position")
        if pos is None:
            return False
        decoy_xy = np.asarray(pos, dtype=float)[:2]
        radius = self._reward_decoy_bait_radius
        for m in self.sim.missiles:
            if (
                m.missile_type == EntityType.REAL_MISSILE
                and m.state == MissileState.IN_FLIGHT
            ):
                if float(np.linalg.norm(m.pos[:2] - decoy_xy)) <= radius:
                    return True
        return False

    def compute_min_real_to_target_dist(self) -> float:
        """Minimum 2D distance from any living REAL missile to any alive target."""
        alive_tgts = [t for t in self.sim.targets if t.alive]
        live_reals = [
            m
            for m in self.sim.missiles
            if m.missile_type == EntityType.REAL_MISSILE
            and m.state == MissileState.IN_FLIGHT
        ]
        if not alive_tgts or not live_reals:
            return 0.0
        best = float("inf")
        for m in live_reals:
            for t in alive_tgts:
                d = float(np.linalg.norm(t.pos[:2] - m.pos[:2]))
                if d < best:
                    best = d
        return best

    # -----------------------------------------------------------------------
    # Observation builder
    # -----------------------------------------------------------------------

    def build_observation(self) -> np.ndarray:
        L = self._layout
        obs = np.zeros(L["total"], dtype=np.float32)
        self.fill_agents(obs)
        self.fill_sams(obs)
        self.fill_radars(obs)
        self.fill_targets(obs)
        self.fill_globals(obs)
        return obs

    def fill_agents(self, obs: np.ndarray):
        L = self._layout
        W = self.area_width
        H = self.area_height
        v_norm = max(self.max_speed, 1e-3) * 2.0  # so |v|/v_norm in ~[0, 0.5]

        for i, m in enumerate(self.sim.missiles):
            b = L["agents_start"] + i * _AGENT_FEATURES
            alive = m.state == MissileState.IN_FLIGHT

            obs[b + 0] = np.clip(m.pos[0] / W, 0.0, 1.0)
            obs[b + 1] = np.clip(m.pos[1] / H, 0.0, 1.0)

            obs[b + 2] = np.clip((m.velocity[0] / v_norm) + 0.5, 0.0, 1.0)
            obs[b + 3] = np.clip((m.velocity[1] / v_norm) + 0.5, 0.0, 1.0)

            speed_2d = float(np.hypot(m.velocity[0], m.velocity[1]))
            if speed_2d > 1e-2:
                heading = float(np.arctan2(m.velocity[0], m.velocity[1]))
                obs[b + 4] = (np.sin(heading) + 1.0) / 2.0
                obs[b + 5] = (np.cos(heading) + 1.0) / 2.0
            else:
                obs[b + 4] = 0.5
                obs[b + 5] = 0.5

            obs[b + 6] = np.clip(m.altitude / self.alt_max, 0.0, 1.0)
            obs[b + 7] = float(alive)
            obs[b + 8] = float(m.missile_type == EntityType.DECOY_MISSILE)
            obs[b + 9] = float(m.evading)

            # Agent ID one-hot.
            obs[b + _AGENT_FEATURES_BASE + i] = 1.0

    def fill_sams(self, obs: np.ndarray):
        L = self._layout
        W = self.area_width
        H = self.area_height
        diag = np.sqrt(W * W + H * H)

        # Build SAM-tracking-which-agents lookup from in-flight interceptors.
        sam_tracking = {sam.id: set() for sam in self.sim.sams}
        for ic in self.sim.interceptors:
            if ic.state != InterceptorState.IN_FLIGHT:
                continue
            agent_idx = ic.target_missile_id
            if 0 <= agent_idx < NUM_AGENTS:
                sam_tracking.setdefault(ic.sam_id, set()).add(agent_idx)

        for si, sam in enumerate(self.sim.sams):
            b = L["sams_start"] + si * _THREAT_FEATURES
            obs[b + 0] = np.clip(sam.pos[0] / W, 0.0, 1.0)
            obs[b + 1] = np.clip(sam.pos[1] / H, 0.0, 1.0)
            obs[b + 2] = np.clip(sam.engagement_range / diag, 0.0, 1.0)
            obs[b + 3] = float(sam.alive and sam.interceptors_remaining > 0)
            for agent_idx in sam_tracking.get(sam.id, ()):
                obs[b + _THREAT_FEATURES_BASE + agent_idx] = 1.0

    def fill_radars(self, obs: np.ndarray):
        L = self._layout
        W = self.area_width
        H = self.area_height
        diag = np.sqrt(W * W + H * H)

        for ri, radar in enumerate(self.sim.radars):
            b = L["radars_start"] + ri * _THREAT_FEATURES
            obs[b + 0] = np.clip(radar.pos[0] / W, 0.0, 1.0)
            obs[b + 1] = np.clip(radar.pos[1] / H, 0.0, 1.0)
            obs[b + 2] = np.clip(radar.detection_range / diag, 0.0, 1.0)
            obs[b + 3] = float(radar.alive)
            # Tracking multi-hot: which agents this radar currently has a track on.
            for agent_idx in radar.tracks.keys():
                if 0 <= agent_idx < NUM_AGENTS:
                    obs[b + _THREAT_FEATURES_BASE + agent_idx] = 1.0

    def fill_targets(self, obs: np.ndarray):
        L = self._layout
        W = self.area_width
        H = self.area_height
        for ti, tgt in enumerate(self.sim.targets):
            b = L["targets_start"] + ti * _TARGET_FEATURES
            obs[b + 0] = np.clip(tgt.pos[0] / W, 0.0, 1.0)
            obs[b + 1] = np.clip(tgt.pos[1] / H, 0.0, 1.0)
            obs[b + 2] = float(tgt.alive)

    def fill_globals(self, obs: np.ndarray):
        L = self._layout
        # Time remaining (1.0 at start, 0.0 at timeout).
        obs[L["globals_start"]] = float(
            np.clip(1.0 - self.sim.time / self.max_time, 0.0, 1.0)
        )

    # -----------------------------------------------------------------------
    # Info dict
    # -----------------------------------------------------------------------

    def build_info(self) -> dict:
        targets_destroyed = sum(1 for t in self.sim.targets if not t.alive)
        real_alive = sum(
            1
            for m in self.sim.missiles
            if m.missile_type == EntityType.REAL_MISSILE
            and m.state == MissileState.IN_FLIGHT
        )
        decoys_alive = sum(
            1
            for m in self.sim.missiles
            if m.missile_type == EntityType.DECOY_MISSILE
            and m.state == MissileState.IN_FLIGHT
        )
        real_destroyed = sum(
            1
            for m in self.sim.missiles
            if m.missile_type == EntityType.REAL_MISSILE
            and m.state in (MissileState.INTERCEPTED, MissileState.MISSED)
        )
        decoys_destroyed = sum(
            1
            for m in self.sim.missiles
            if m.missile_type == EntityType.DECOY_MISSILE
            and m.state in (MissileState.INTERCEPTED, MissileState.MISSED)
        )
        interceptors_fired = sum(
            sam.num_interceptors - sam.interceptors_remaining for sam in self.sim.sams
        )
        return {
            "launch_zone": int(self._launch_zone_idx),
            "targets_destroyed": targets_destroyed,
            "hits": self._target_hits_this_ep,
            "real_alive": real_alive,
            "decoys_alive": decoys_alive,
            "real_destroyed": real_destroyed,
            "decoys_destroyed": decoys_destroyed,
            "interceptors_fired": interceptors_fired,
            "time_step": self.sim.time,
        }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

gymnasium.register(
    id="IADSPenetration-v0",
    entry_point="iads.gym_env:IADSEnv",
)
