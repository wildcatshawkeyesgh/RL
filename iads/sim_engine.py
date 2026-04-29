"""
IADS Penetration - Core Simulation Engine
==========================================
Handles all entity logic, movement, detection, engagement, and state updates.
No dependencies on Gymnasium or Pygame. Can be used standalone.
"""

import numpy as np
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum, auto
from copy import deepcopy


# ============================================================
# ENUMS
# ============================================================

class EntityType(Enum):
    RADAR = auto()
    SAM = auto()
    TARGET = auto()
    REAL_MISSILE = auto()
    DECOY_MISSILE = auto()
    INTERCEPTOR = auto()


class MissileState(Enum):
    READY = auto()
    IN_FLIGHT = auto()
    HIT_TARGET = auto()
    INTERCEPTED = auto()
    MISSED = auto()  # out of fuel


class InterceptorState(Enum):
    LOADED = auto()
    IN_FLIGHT = auto()
    HIT = auto()
    MISSED = auto()


# ============================================================
# ENTITY DATACLASSES
# ============================================================

@dataclass
class Radar:
    id: int
    pos: np.ndarray
    detection_range: float = 150.0
    classification_rate: float = 0.05
    tracks: Dict[int, float] = field(default_factory=dict)
    alive: bool = True


@dataclass
class SAM:
    id: int
    pos: np.ndarray
    engagement_range: float = 60.0
    num_interceptors: int = 8
    interceptors_remaining: int = 8
    reload_time: float = 5.0
    cooldown: float = 0.0
    pk: float = 0.70
    alive: bool = True
    proximity_delay: float = 3.0  # extra delay for closely-spaced targets
    proximity_threshold: float = 5.0  # km - what counts as "closely spaced"
    last_engage_positions: List[Tuple[float, np.ndarray]] = field(default_factory=list)


@dataclass
class Target:
    id: int
    pos: np.ndarray
    value: float = 1.0
    alive: bool = True


@dataclass
class Missile:
    id: int
    missile_type: EntityType
    pos: np.ndarray
    velocity: np.ndarray
    speed: float = 3.0
    max_speed: float = 3.0
    state: MissileState = MissileState.READY
    target_id: Optional[int] = None
    waypoints: List[np.ndarray] = field(default_factory=list)
    current_waypoint_idx: int = 0
    altitude: float = 10.0
    fuel: float = 300.0
    distance_traveled: float = 0.0
    rcs: float = 1.0
    evading: bool = False
    evasion_fuel_mult: float = 1.5
    evasion_speed_mult: float = 0.7
    trail: List[np.ndarray] = field(default_factory=list)
    is_tracked: bool = False
    num_tracking_radars: int = 0


@dataclass
class Interceptor:
    id: int
    sam_id: int
    target_missile_id: int
    pos: np.ndarray
    velocity: np.ndarray
    speed: float = 8.0
    state: InterceptorState = InterceptorState.IN_FLIGHT
    pk: float = 0.70
    predicted_intercept_point: np.ndarray = field(default_factory=lambda: np.zeros(3))
    trail: List[np.ndarray] = field(default_factory=list)


# ============================================================
# SIMULATION ENGINE
# ============================================================

class IADSSimulation:
    """Core simulation engine for IADS penetration scenario."""

    def __init__(self, config: dict = None):
        self.config = self.default_config()
        if config:
            self.config.update(config)
        if not self.config.get('missile_fuel'):
            self.config['missile_fuel'] = 1.5 * self.config['area_height']
        self.dt = 1.0
        self.time = 0
        self.max_time = self.config.get('max_time', 200)

        self.radars: List[Radar] = []
        self.sams: List[SAM] = []
        self.targets: List[Target] = []
        self.missiles: List[Missile] = []
        self.interceptors: List[Interceptor] = []
        self.next_interceptor_id = 0

        # Event log for reward computation and replay
        self.events: List[dict] = []
        self.state_log: List[dict] = []

        # Track merge state
        self.merged_tracks: List[List[int]] = []  # groups of missile IDs seen as one track
        self.track_merge_distance = self.config.get('track_merge_distance', 3.0)

        # RNG
        self._rng = np.random.default_rng(self.config.get('seed'))

        self.setup_scenario()

    @staticmethod
    def default_config():
        return {
            'max_time': 200,
            'area_width': 300.0,
            'area_height': 300.0,
            # Defender
            'num_radars': 4,
            'num_sams': 8,
            'num_targets': 3,
            'radar_detection_range': 150.0,
            'radar_classification_rate': 0.05,
            'sam_engagement_range': 80.0,
            'sam_interceptors': 8,
            'sam_pk': 0.70,
            'sam_reload_time': 5.0,
            'sam_proximity_delay': 3.0,
            'sam_proximity_threshold': 5.0,
            # Attacker
            'num_real_missiles': 4,
            'num_decoy_missiles': 4,
            'missile_speed': 3.0,
            'missile_altitude': 10.0,
            'missile_fuel': None,  # resolved to 1.5 * area_height in __init__ if not set
            'decoy_rcs': 1.2,
            'real_rcs': 1.0,
            'evasion_pk_reduction': 0.30,
            'evasion_fuel_multiplier': 1.5,
            'evasion_speed_multiplier': 0.7,
            # Interceptor
            'interceptor_speed': 8.0,
            # Detection
            'track_merge_distance': 3.0,
            'altitude_detection_factor': 0.3,
            # Target
            'hit_radius': 5.0,
            # Layout
            'defense_template': None,  # None = random choice
            'perturbation_radius': 15.0,
            'layout_seed': None,
            # Launch zones: list of dicts {x_min, x_max, y_min, y_max}
            'launch_zones': [
                {'x_min': 20.0, 'x_max': 80.0, 'y_min': 5.0, 'y_max': 25.0},
                {'x_min': 130.0, 'x_max': 190.0, 'y_min': 5.0, 'y_max': 25.0},
                {'x_min': 220.0, 'x_max': 280.0, 'y_min': 5.0, 'y_max': 25.0},
            ],
            # Reward terms are computed in iads/gym_env.py per the PPO brief;
            # the sim only emits events.
            # General
            'seed': None,
        }

    # ============================================================
    # SCENARIO SETUP
    # ============================================================

    def setup_scenario(self):
        """Generate defense layout from templates with perturbation."""
        cfg = self.config
        w = cfg['area_width']
        h = cfg['area_height']
        rng = np.random.default_rng(cfg.get('layout_seed'))

        templates = self.get_defense_templates(w, h)
        template_name = cfg.get('defense_template')
        if template_name and template_name in templates:
            template = templates[template_name]
        else:
            template = templates[rng.choice(list(templates.keys()))]

        perturb = cfg['perturbation_radius']

        # Create targets
        for i, base_pos in enumerate(template['targets']):
            pos = self.perturb_pos(base_pos, perturb * 0.5, w, h, rng)
            pos[2] = 0.0
            self.targets.append(Target(id=i, pos=pos, value=1.0))

        # Create SAMs
        for i, base_pos in enumerate(template['sams']):
            pos = self.perturb_pos(base_pos, perturb, w, h, rng)
            pos[2] = 0.0
            self.sams.append(SAM(
                id=i, pos=pos,
                engagement_range=cfg['sam_engagement_range'],
                num_interceptors=cfg['sam_interceptors'],
                interceptors_remaining=cfg['sam_interceptors'],
                pk=cfg['sam_pk'],
                reload_time=cfg['sam_reload_time'],
                proximity_delay=cfg['sam_proximity_delay'],
                proximity_threshold=cfg['sam_proximity_threshold'],
            ))

        # Create radars
        for i, base_pos in enumerate(template['radars']):
            pos = self.perturb_pos(base_pos, perturb, w, h, rng)
            pos[2] = 0.0
            self.radars.append(Radar(
                id=i, pos=pos,
                detection_range=cfg['radar_detection_range'],
                classification_rate=cfg['radar_classification_rate'],
            ))

        # Create missiles (not yet positioned - launch zone selected by player/agent)
        total_real = cfg['num_real_missiles']
        total_decoy = cfg['num_decoy_missiles']

        for i in range(total_real):
            self.missiles.append(Missile(
                id=i,
                missile_type=EntityType.REAL_MISSILE,
                pos=np.zeros(3),
                velocity=np.zeros(3),
                speed=cfg['missile_speed'],
                max_speed=cfg['missile_speed'],
                altitude=cfg['missile_altitude'],
                fuel=cfg['missile_fuel'],
                rcs=cfg['real_rcs'],
                evasion_fuel_mult=cfg['evasion_fuel_multiplier'],
                evasion_speed_mult=cfg['evasion_speed_multiplier'],
            ))

        for i in range(total_decoy):
            mid = total_real + i
            self.missiles.append(Missile(
                id=mid,
                missile_type=EntityType.DECOY_MISSILE,
                pos=np.zeros(3),
                velocity=np.zeros(3),
                speed=cfg['missile_speed'],
                max_speed=cfg['missile_speed'],
                altitude=cfg['missile_altitude'],
                fuel=cfg['missile_fuel'],
                rcs=cfg['decoy_rcs'],
                evasion_fuel_mult=cfg['evasion_fuel_multiplier'],
                evasion_speed_mult=cfg['evasion_speed_multiplier'],
            ))

    def get_defense_templates(self, w: float, h: float) -> dict:
        """Define defense layout templates."""
        templates = {}

        # Template 1: Spread defense - even coverage
        templates['spread'] = {
            'targets': [
                np.array([w * 0.25, h * 0.85, 0.0]),
                np.array([w * 0.50, h * 0.90, 0.0]),
                np.array([w * 0.75, h * 0.85, 0.0]),
            ],
            'sams': [
                np.array([w * 0.15, h * 0.55, 0.0]),
                np.array([w * 0.35, h * 0.50, 0.0]),
                np.array([w * 0.55, h * 0.55, 0.0]),
                np.array([w * 0.75, h * 0.50, 0.0]),
                np.array([w * 0.25, h * 0.70, 0.0]),
                np.array([w * 0.50, h * 0.65, 0.0]),
                np.array([w * 0.75, h * 0.70, 0.0]),
                np.array([w * 0.50, h * 0.78, 0.0]),
            ],
            'radars': [
                np.array([w * 0.20, h * 0.40, 0.0]),
                np.array([w * 0.45, h * 0.38, 0.0]),
                np.array([w * 0.70, h * 0.40, 0.0]),
                np.array([w * 0.50, h * 0.55, 0.0]),
            ],
        }

        # Template 2: Layered defense - depth
        templates['layered'] = {
            'targets': [
                np.array([w * 0.30, h * 0.88, 0.0]),
                np.array([w * 0.50, h * 0.92, 0.0]),
                np.array([w * 0.70, h * 0.88, 0.0]),
            ],
            'sams': [
                np.array([w * 0.20, h * 0.42, 0.0]),
                np.array([w * 0.50, h * 0.40, 0.0]),
                np.array([w * 0.80, h * 0.42, 0.0]),
                np.array([w * 0.30, h * 0.58, 0.0]),
                np.array([w * 0.65, h * 0.56, 0.0]),
                np.array([w * 0.25, h * 0.72, 0.0]),
                np.array([w * 0.50, h * 0.70, 0.0]),
                np.array([w * 0.75, h * 0.72, 0.0]),
            ],
            'radars': [
                np.array([w * 0.30, h * 0.35, 0.0]),
                np.array([w * 0.65, h * 0.35, 0.0]),
                np.array([w * 0.45, h * 0.52, 0.0]),
                np.array([w * 0.50, h * 0.68, 0.0]),
            ],
        }

        # Template 3: Left-heavy
        templates['left_heavy'] = {
            'targets': [
                np.array([w * 0.20, h * 0.87, 0.0]),
                np.array([w * 0.45, h * 0.90, 0.0]),
                np.array([w * 0.75, h * 0.85, 0.0]),
            ],
            'sams': [
                np.array([w * 0.12, h * 0.48, 0.0]),
                np.array([w * 0.28, h * 0.45, 0.0]),
                np.array([w * 0.18, h * 0.62, 0.0]),
                np.array([w * 0.35, h * 0.60, 0.0]),
                np.array([w * 0.25, h * 0.75, 0.0]),
                np.array([w * 0.45, h * 0.72, 0.0]),
                np.array([w * 0.65, h * 0.55, 0.0]),
                np.array([w * 0.80, h * 0.68, 0.0]),
            ],
            'radars': [
                np.array([w * 0.20, h * 0.38, 0.0]),
                np.array([w * 0.40, h * 0.42, 0.0]),
                np.array([w * 0.30, h * 0.58, 0.0]),
                np.array([w * 0.70, h * 0.48, 0.0]),
            ],
        }

        # Template 4: Right-heavy
        templates['right_heavy'] = {
            'targets': [
                np.array([w * 0.25, h * 0.85, 0.0]),
                np.array([w * 0.55, h * 0.90, 0.0]),
                np.array([w * 0.80, h * 0.87, 0.0]),
            ],
            'sams': [
                np.array([w * 0.20, h * 0.55, 0.0]),
                np.array([w * 0.35, h * 0.68, 0.0]),
                np.array([w * 0.55, h * 0.48, 0.0]),
                np.array([w * 0.65, h * 0.60, 0.0]),
                np.array([w * 0.72, h * 0.45, 0.0]),
                np.array([w * 0.82, h * 0.62, 0.0]),
                np.array([w * 0.75, h * 0.75, 0.0]),
                np.array([w * 0.60, h * 0.73, 0.0]),
            ],
            'radars': [
                np.array([w * 0.30, h * 0.48, 0.0]),
                np.array([w * 0.60, h * 0.38, 0.0]),
                np.array([w * 0.75, h * 0.42, 0.0]),
                np.array([w * 0.68, h * 0.58, 0.0]),
            ],
        }

        # Trim to match config counts
        n_sams = self.config['num_sams']
        n_radars = self.config['num_radars']
        n_targets = self.config['num_targets']
        for name, tmpl in templates.items():
            tmpl['sams'] = tmpl['sams'][:n_sams]
            tmpl['radars'] = tmpl['radars'][:n_radars]
            tmpl['targets'] = tmpl['targets'][:n_targets]

        return templates

    def perturb_pos(self, base: np.ndarray, radius: float,
                     w: float, h: float, rng) -> np.ndarray:
        """Perturb a position within bounds."""
        pos = base.copy()
        pos[0] += rng.uniform(-radius, radius)
        pos[1] += rng.uniform(-radius, radius)
        pos[0] = np.clip(pos[0], 5.0, w - 5.0)
        pos[1] = np.clip(pos[1], 5.0, h - 5.0)
        return pos

    # ============================================================
    # LAUNCH
    # ============================================================

    def launch_missile(self, missile_id: int, launch_pos: np.ndarray,
                       waypoints: List[np.ndarray],
                       target_id: Optional[int] = None) -> bool:
        """Launch a missile from a position with waypoints."""
        m = self.missiles[missile_id]
        if m.state != MissileState.READY:
            return False

        m.pos = launch_pos.copy()
        m.pos[2] = m.altitude
        m.state = MissileState.IN_FLIGHT
        m.waypoints = [np.array(wp, dtype=float) for wp in waypoints]
        m.current_waypoint_idx = 0
        m.target_id = target_id
        m.trail = [m.pos.copy()]

        if len(m.waypoints) > 0:
            direction = m.waypoints[0] - m.pos
            dist = np.linalg.norm(direction)
            if dist > 0:
                m.velocity = (direction / dist) * m.speed

        return True

    def launch_salvo(self, launch_pos: np.ndarray,
                     missile_plans: List[dict]) -> bool:
        """
        Launch all missiles simultaneously.
        missile_plans: list of dicts with keys:
            'missile_id': int
            'waypoints': list of [x, y, z] arrays
            'target_id': optional int (for real missiles)
        """
        for plan in missile_plans:
            self.launch_missile(
                plan['missile_id'],
                launch_pos,
                plan['waypoints'],
                plan.get('target_id'),
            )
        return True

    # ============================================================
    # COMMANDS (for in-flight overrides)
    # ============================================================

    def set_missile_evading(self, missile_id: int, evading: bool):
        """Toggle evasive maneuvers on a missile."""
        m = self.missiles[missile_id]
        if m.state == MissileState.IN_FLIGHT:
            m.evading = evading
            if evading:
                m.speed = m.max_speed * m.evasion_speed_mult
            else:
                m.speed = m.max_speed

    def steer_missile(self, missile_id: int, heading_delta: float = 0.0,
                      altitude_delta: float = 0.0):
        """
        Override a missile's heading and altitude.
        heading_delta: degrees to turn (positive = right)
        altitude_delta: km to change altitude
        """
        m = self.missiles[missile_id]
        if m.state != MissileState.IN_FLIGHT:
            return

        # Rotate velocity vector by heading_delta
        if abs(heading_delta) > 0.01 and np.linalg.norm(m.velocity[:2]) > 0.01:
            angle = np.radians(heading_delta)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            vx, vy = m.velocity[0], m.velocity[1]
            m.velocity[0] = vx * cos_a - vy * sin_a
            m.velocity[1] = vx * sin_a + vy * cos_a
            # Renormalize to current speed
            horiz_speed = np.linalg.norm(m.velocity[:2])
            if horiz_speed > 0:
                m.velocity[:2] = (m.velocity[:2] / horiz_speed) * m.speed

        # Adjust altitude
        if abs(altitude_delta) > 0.01:
            m.altitude = max(0.5, min(25.0, m.altitude + altitude_delta))

        # Clear remaining waypoints since we're manually steering
        m.waypoints = []
        m.current_waypoint_idx = 0

    # ============================================================
    # STEP
    # ============================================================

    def step(self) -> Tuple[List[dict], bool]:
        """
        Advance simulation by one timestep.
        Returns: (events_this_step, done)
        """
        self.time += 1
        step_events = []

        self.update_missiles(step_events)
        self.update_interceptors()
        self.update_track_merging()
        self.update_radars()
        self.defender_ai(step_events)
        self.resolve_intercepts(step_events)
        self.check_target_hits(step_events)
        self.log_state()

        self.events.extend(step_events)
        done = self.check_done()

        return step_events, done

    # ============================================================
    # UPDATE FUNCTIONS
    # ============================================================

    def update_missiles(self, step_events: list):
        """Move all in-flight missiles."""
        for m in self.missiles:
            if m.state != MissileState.IN_FLIGHT:
                continue

            # Fuel cost
            fuel_cost = m.speed * self.dt
            if m.evading:
                fuel_cost *= m.evasion_fuel_mult
            m.distance_traveled += fuel_cost

            if m.distance_traveled >= m.fuel:
                m.state = MissileState.MISSED
                step_events.append({
                    'type': 'fuel_exhausted',
                    'time': self.time,
                    'missile_id': m.id,
                    'is_real': m.missile_type == EntityType.REAL_MISSILE,
                    'position': m.pos.copy().tolist(),
                })
                continue

            # Navigate toward current waypoint if we have one
            if m.current_waypoint_idx < len(m.waypoints):
                target_wp = m.waypoints[m.current_waypoint_idx].copy()
                if target_wp[2] < 0.1:
                    target_wp[2] = m.altitude

                direction = target_wp - m.pos
                dist = np.linalg.norm(direction)

                if dist < m.speed * self.dt * 1.5:
                    m.pos = target_wp.copy()
                    m.pos[2] = m.altitude
                    m.current_waypoint_idx += 1
                    # Set velocity toward next waypoint
                    if m.current_waypoint_idx < len(m.waypoints):
                        next_dir = m.waypoints[m.current_waypoint_idx] - m.pos
                        nd = np.linalg.norm(next_dir)
                        if nd > 0:
                            m.velocity = (next_dir / nd) * m.speed
                else:
                    m.velocity = (direction / dist) * m.speed
                    m.pos += m.velocity * self.dt
            else:
                # No waypoints left - continue on current heading
                if np.linalg.norm(m.velocity) < 0.01:
                    m.velocity = np.array([0.0, m.speed, 0.0])
                else:
                    m.velocity = (m.velocity / np.linalg.norm(m.velocity)) * m.speed
                m.pos += m.velocity * self.dt

            # Enforce altitude
            m.pos[2] = m.altitude

            # Enforce x/y walls: clip position to map bounds and zero out the
            # outward velocity component so the missile slides along the wall
            # instead of pushing into it. Next-step velocity renormalization
            # handles the rest. Self-corrects out of corners (a corner-stuck
            # missile drops to ~0 velocity, then renormalizes to default +y).
            W = self.config['area_width']
            H = self.config['area_height']
            if m.pos[0] < 0.0:
                m.pos[0] = 0.0
                if m.velocity[0] < 0.0:
                    m.velocity[0] = 0.0
            elif m.pos[0] > W:
                m.pos[0] = W
                if m.velocity[0] > 0.0:
                    m.velocity[0] = 0.0
            if m.pos[1] < 0.0:
                m.pos[1] = 0.0
                if m.velocity[1] < 0.0:
                    m.velocity[1] = 0.0
            elif m.pos[1] > H:
                m.pos[1] = H
                if m.velocity[1] > 0.0:
                    m.velocity[1] = 0.0

            # Record trail
            m.trail.append(m.pos.copy())

    def update_interceptors(self):
        """Move all in-flight interceptors toward intercept points."""
        for intc in self.interceptors:
            if intc.state != InterceptorState.IN_FLIGHT:
                continue

            # Update predicted intercept point based on current target position
            target_m = self.get_missile(intc.target_missile_id)
            if target_m and target_m.state == MissileState.IN_FLIGHT:
                intc.predicted_intercept_point = self.predict_intercept(
                    intc.pos, intc.speed, target_m.pos, target_m.velocity
                )

            direction = intc.predicted_intercept_point - intc.pos
            dist = np.linalg.norm(direction)

            if dist < intc.speed * self.dt * 1.5:
                intc.pos = intc.predicted_intercept_point.copy()
            else:
                intc.velocity = (direction / dist) * intc.speed
                intc.pos += intc.velocity * self.dt

            intc.trail.append(intc.pos.copy())

    def predict_intercept(self, launcher_pos: np.ndarray, intc_speed: float,
                           target_pos: np.ndarray, target_vel: np.ndarray) -> np.ndarray:
        """Predict intercept point with leading."""
        intercept_pos = target_pos.copy()
        for _ in range(3):
            dist = np.linalg.norm(intercept_pos - launcher_pos)
            t = dist / max(intc_speed, 0.01)
            intercept_pos = target_pos + target_vel * t
        return intercept_pos

    def update_track_merging(self):
        """Compute which missiles appear as merged tracks to radar."""
        in_flight = [m for m in self.missiles if m.state == MissileState.IN_FLIGHT]
        self.merged_tracks = []
        assigned = set()

        for m in in_flight:
            if m.id in assigned:
                continue
            group = [m.id]
            assigned.add(m.id)
            for other in in_flight:
                if other.id in assigned:
                    continue
                dist = np.linalg.norm(m.pos - other.pos)
                if dist <= self.track_merge_distance:
                    group.append(other.id)
                    assigned.add(other.id)
            self.merged_tracks.append(group)

    def get_track_group(self, missile_id: int) -> Optional[List[int]]:
        """Get the merged track group for a missile."""
        for group in self.merged_tracks:
            if missile_id in group:
                return group
        return None

    def update_radars(self):
        """Update radar detections and classification."""
        for radar in self.radars:
            if not radar.alive:
                continue

            tracked_ids = set()

            for group in self.merged_tracks:
                # Use the centroid of the group for detection
                group_missiles = [m for m in self.missiles
                                  if m.id in group and m.state == MissileState.IN_FLIGHT]
                if not group_missiles:
                    continue

                centroid = np.mean([m.pos for m in group_missiles], axis=0)
                avg_alt = np.mean([m.altitude for m in group_missiles])
                dist = np.linalg.norm(centroid - radar.pos)

                alt_factor = 1.0 + self.config['altitude_detection_factor'] * (avg_alt / 15.0)
                effective_range = radar.detection_range * alt_factor

                if dist <= effective_range:
                    # Track this group as a single entity (use lowest ID as representative)
                    rep_id = min(group)
                    tracked_ids.add(rep_id)

                    if rep_id not in radar.tracks:
                        radar.tracks[rep_id] = 0.0

                    # Classification: check if any in group are decoys
                    has_decoy = any(m.missile_type == EntityType.DECOY_MISSILE
                                    for m in group_missiles)
                    has_real = any(m.missile_type == EntityType.REAL_MISSILE
                                  for m in group_missiles)

                    if len(group) == 1:
                        # Single track - classify normally
                        m = group_missiles[0]
                        if m.missile_type == EntityType.DECOY_MISSILE:
                            radar.tracks[rep_id] = min(1.0,
                                radar.tracks[rep_id] + radar.classification_rate)
                        else:
                            radar.tracks[rep_id] = min(0.1,
                                radar.tracks[rep_id] + radar.classification_rate * 0.1)
                    else:
                        # Merged track - harder to classify, looks ambiguous
                        radar.tracks[rep_id] = min(0.2,
                            radar.tracks[rep_id] + radar.classification_rate * 0.05)

            # Remove tracks no longer detected
            lost = [tid for tid in radar.tracks if tid not in tracked_ids]
            for tid in lost:
                del radar.tracks[tid]

        # Update missile tracking status
        for m in self.missiles:
            if m.state != MissileState.IN_FLIGHT:
                m.is_tracked = False
                m.num_tracking_radars = 0
                continue
            group = self.get_track_group(m.id)
            rep_id = min(group) if group else m.id
            count = sum(1 for r in self.radars if r.alive and rep_id in r.tracks)
            m.is_tracked = count > 0
            m.num_tracking_radars = count

    def get_best_classification(self, missile_id: int) -> float:
        """Best classification confidence across all radars for a track."""
        group = self.get_track_group(missile_id)
        rep_id = min(group) if group else missile_id
        best = 0.0
        for radar in self.radars:
            if radar.alive and rep_id in radar.tracks:
                best = max(best, radar.tracks[rep_id])
        return best

    def defender_ai(self, step_events: list):
        """Scripted defender SAM engagement logic."""
        for sam in self.sams:
            if not sam.alive or sam.interceptors_remaining <= 0:
                continue

            if sam.cooldown > 0:
                sam.cooldown -= self.dt
                continue

            # Find engageable tracks
            candidates = []
            for group in self.merged_tracks:
                group_missiles = [m for m in self.missiles
                                  if m.id in group and m.state == MissileState.IN_FLIGHT]
                if not group_missiles:
                    continue

                centroid = np.mean([m.pos for m in group_missiles], axis=0)
                dist = np.linalg.norm(centroid - sam.pos)

                if dist > sam.engagement_range:
                    continue

                rep_id = min(group)
                if not self.is_tracked_by_any(rep_id):
                    continue

                classification = self.get_best_classification(rep_id)
                if classification > 0.8:
                    continue

                already_engaged = any(
                    intc.target_missile_id in group
                    and intc.state == InterceptorState.IN_FLIGHT
                    for intc in self.interceptors
                )

                # Check proximity interference constraint
                proximity_blocked = False
                for t, pos in sam.last_engage_positions:
                    if self.time - t < sam.proximity_delay:
                        if np.linalg.norm(centroid - pos) < sam.proximity_threshold:
                            proximity_blocked = True
                            break

                if proximity_blocked:
                    continue

                priority = -dist + (50.0 if not already_engaged else 0.0)
                # Pick a specific missile from the group to target
                target_m = group_missiles[0]
                candidates.append((target_m, centroid, priority))

            if not candidates:
                continue

            candidates.sort(key=lambda x: x[2], reverse=True)
            target_m, centroid, _ = candidates[0]

            intercept_point = self.predict_intercept(
                sam.pos, self.config['interceptor_speed'],
                target_m.pos, target_m.velocity
            )
            direction = intercept_point - sam.pos
            dist = np.linalg.norm(direction)

            intc = Interceptor(
                id=self.next_interceptor_id,
                sam_id=sam.id,
                target_missile_id=target_m.id,
                pos=sam.pos.copy(),
                velocity=(direction / max(dist, 0.01)) * self.config['interceptor_speed'],
                speed=self.config['interceptor_speed'],
                pk=sam.pk,
                predicted_intercept_point=intercept_point,
                trail=[sam.pos.copy()],
            )
            self.interceptors.append(intc)
            self.next_interceptor_id += 1

            sam.interceptors_remaining -= 1
            sam.cooldown = sam.reload_time
            sam.last_engage_positions.append((self.time, centroid.copy()))

            step_events.append({
                'type': 'interceptor_launched',
                'time': self.time,
                'sam_id': sam.id,
                'target_missile_id': target_m.id,
                'interceptor_id': intc.id,
            })

            # Check if SAM is now depleted
            if sam.interceptors_remaining <= 0:
                step_events.append({
                    'type': 'sam_depleted',
                    'time': self.time,
                    'sam_id': sam.id,
                })

    def is_tracked_by_any(self, rep_id: int) -> bool:
        return any(r.alive and rep_id in r.tracks for r in self.radars)

    def resolve_intercepts(self, step_events: list):
        """Resolve interceptors that have reached their targets."""
        for intc in self.interceptors:
            if intc.state != InterceptorState.IN_FLIGHT:
                continue

            target_m = self.get_missile(intc.target_missile_id)
            if target_m is None or target_m.state != MissileState.IN_FLIGHT:
                intc.state = InterceptorState.MISSED
                continue

            dist_to_target = np.linalg.norm(intc.pos - target_m.pos)
            engagement_radius = intc.speed * self.dt * 3.0

            if dist_to_target <= engagement_radius:
                pk = intc.pk
                if target_m.evading:
                    pk *= (1.0 - self.config['evasion_pk_reduction'])

                if self._rng.random() < pk:
                    intc.state = InterceptorState.HIT
                    target_m.state = MissileState.INTERCEPTED
                    event_type = ('real_intercepted'
                                  if target_m.missile_type == EntityType.REAL_MISSILE
                                  else 'decoy_intercepted')
                    step_events.append({
                        'type': event_type,
                        'time': self.time,
                        'missile_id': target_m.id,
                        'interceptor_id': intc.id,
                        'position': target_m.pos.copy().tolist(),
                    })
                else:
                    intc.state = InterceptorState.MISSED
                    step_events.append({
                        'type': 'interceptor_miss',
                        'time': self.time,
                        'missile_id': target_m.id,
                        'interceptor_id': intc.id,
                    })

            # Check if interceptor has overshot significantly
            dist_to_pip = np.linalg.norm(intc.pos - intc.predicted_intercept_point)
            if dist_to_pip < intc.speed * self.dt and dist_to_target > engagement_radius * 2:
                intc.state = InterceptorState.MISSED

    def check_target_hits(self, step_events: list):
        """
        Proximity detonation: any real missile within hit_radius of any alive
        target destroys that target. Missile picks the closest alive target.
        """
        hit_radius = self.config['hit_radius']

        for m in self.missiles:
            if m.state != MissileState.IN_FLIGHT:
                continue
            if m.missile_type != EntityType.REAL_MISSILE:
                continue

            closest_target = None
            closest_dist = float('inf')
            for t in self.targets:
                if not t.alive:
                    continue
                d = float(np.linalg.norm(m.pos[:2] - t.pos[:2]))
                if d <= hit_radius and d < closest_dist:
                    closest_dist = d
                    closest_target = t

            if closest_target is None:
                continue

            m.state = MissileState.HIT_TARGET
            closest_target.alive = False
            step_events.append({
                'type': 'target_hit',
                'time': self.time,
                'missile_id': m.id,
                'target_id': closest_target.id,
                'position': closest_target.pos.copy().tolist(),
            })

    def check_done(self) -> bool:
        """Check if simulation is over."""
        if self.time >= self.max_time:
            return True

        all_resolved = all(
            m.state not in (MissileState.READY, MissileState.IN_FLIGHT)
            for m in self.missiles
        )
        return all_resolved

    def get_missile(self, missile_id: int) -> Optional[Missile]:
        for m in self.missiles:
            if m.id == missile_id:
                return m
        return None

    # ============================================================
    # STATE LOGGING
    # ============================================================

    def log_state(self):
        """Log current state for replay."""
        state = {
            'time': self.time,
            'missiles': [],
            'interceptors': [],
            'events': [e for e in self.events if e.get('time') == self.time],
        }
        for m in self.missiles:
            state['missiles'].append({
                'id': m.id,
                'type': m.missile_type.name,
                'state': m.state.name,
                'pos': m.pos.tolist(),
                'velocity': m.velocity.tolist(),
                'altitude': m.altitude,
                'fuel_remaining': max(0, m.fuel - m.distance_traveled),
                'evading': m.evading,
                'is_tracked': m.is_tracked,
                'num_tracking_radars': m.num_tracking_radars,
                'target_id': m.target_id,
            })
        for intc in self.interceptors:
            if intc.state == InterceptorState.IN_FLIGHT:
                state['interceptors'].append({
                    'id': intc.id,
                    'sam_id': intc.sam_id,
                    'target_missile_id': intc.target_missile_id,
                    'pos': intc.pos.tolist(),
                    'state': intc.state.name,
                })
        self.state_log.append(state)

    def get_static_state(self) -> dict:
        """Get the static defense layout for display/observation."""
        return {
            'radars': [{'id': r.id, 'pos': r.pos.tolist(),
                        'detection_range': r.detection_range,
                        'alive': r.alive} for r in self.radars],
            'sams': [{'id': s.id, 'pos': s.pos.tolist(),
                      'engagement_range': s.engagement_range,
                      'interceptors_remaining': s.interceptors_remaining,
                      'num_interceptors': s.num_interceptors,
                      'alive': s.alive} for s in self.sams],
            'targets': [{'id': t.id, 'pos': t.pos.tolist(),
                         'value': t.value, 'alive': t.alive} for t in self.targets],
            'launch_zones': self.config['launch_zones'],
            'area_width': self.config['area_width'],
            'area_height': self.config['area_height'],
        }

    def save_replay(self, filepath: str):
        """Save complete replay data to JSON."""
        replay = {
            'config': {k: v for k, v in self.config.items()
                       if not callable(v)},
            'static': self.get_static_state(),
            'frames': self.state_log,
            'events': self.events,
        }
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, Enum):
                return obj.name
            return obj

        with open(filepath, 'w') as f:
            json.dump(replay, f, default=convert, indent=2)
