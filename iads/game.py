"""
IADS Penetration - Pygame Game Interface
==========================================
Human-playable game with planning and execution phases.
Supports top-down and 3D perspective views.

Controls:
  PLANNING PHASE:
    Left Click    - Place waypoint for selected missile
    Right Click   - Remove last waypoint for selected missile
    Tab           - Cycle selected missile
    T             - Assign selected target (for real missiles)
    Space         - Confirm plan and launch
    1/2/3         - Select launch zone

  EXECUTION PHASE:
    Left Click    - Select a missile
    Right Click   - Deselect missile
    A/D           - Steer selected missile left/right
    W/S           - Climb/dive selected missile
    E             - Toggle evasive maneuver
    Tab           - Cycle between in-flight missiles
    Space         - Pause/resume
    +/-           - Speed up/slow down

  BOTH PHASES:
    V             - Toggle top-down / 3D view
    ESC           - Quit
"""

import sys
import os
import math
import json
import numpy as np

try:
    import pygame
    import pygame.gfxdraw
except ImportError:
    print("Pygame is required. Install with: pip install pygame")
    sys.exit(1)

from iads.sim_engine import (
    IADSSimulation, EntityType, MissileState, InterceptorState,
    Missile, Interceptor, Radar, SAM, Target
)


# ============================================================
# COLORS
# ============================================================
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_BG = (15, 20, 30)
GRID_COLOR = (30, 40, 55)
PANEL_BG = (20, 28, 40)
PANEL_BORDER = (50, 65, 90)

# Entity colors
BLUE_MISSILE = (50, 140, 255)
BLUE_MISSILE_LIGHT = (100, 175, 255)
DECOY_COLOR = (50, 200, 200)
DECOY_LIGHT = (100, 225, 225)
RED_SAM = (220, 50, 50)
RED_SAM_LIGHT = (255, 100, 100)
RED_RADAR = (255, 140, 50)
RED_RADAR_LIGHT = (255, 180, 100)
TARGET_COLOR = (255, 220, 50)
TARGET_HIT_COLOR = (100, 100, 100)
INTERCEPTOR_COLOR = (255, 60, 60)
GREEN = (50, 200, 80)
YELLOW = (255, 220, 50)
ORANGE = (255, 160, 50)

# Range ring colors
RADAR_RANGE_COLOR = (255, 140, 50, 40)
SAM_RANGE_COLOR = (220, 50, 50, 40)
LAUNCH_ZONE_COLOR = (50, 200, 80, 60)
WAYPOINT_COLOR = (150, 150, 255)
SELECTED_COLOR = (255, 255, 100)
TRAIL_ALPHA = 100


# ============================================================
# GAME PHASES
# ============================================================
class GamePhase:
    PLANNING = 'planning'
    EXECUTING = 'executing'
    DONE = 'done'


# ============================================================
# MAIN GAME CLASS
# ============================================================
class IADSGame:
    def __init__(self, config: dict = None, screen_width: int = 1400,
                 screen_height: int = 900):
        pygame.init()
        pygame.display.set_caption("IADS Penetration - Strike Planner")

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("consolas", 13)
        self.font_med = pygame.font.SysFont("consolas", 16)
        self.font_large = pygame.font.SysFont("consolas", 22)
        self.font_title = pygame.font.SysFont("consolas", 28, bold=True)

        # Layout
        self.panel_width = 320
        self.map_width = screen_width - self.panel_width
        self.map_height = screen_height

        # Sim
        self.config = config or IADSSimulation.default_config()
        self.sim = IADSSimulation(self.config)
        self.static = self.sim.get_static_state()

        # Game state
        self.phase = GamePhase.PLANNING
        self.view_3d = False
        self.paused = False
        self.sim_speed = 1
        self.base_sim_rate = 5.0  # sim steps per second at 1x speed
        self.tick_accumulator = 0.0

        # Planning state
        self.selected_missile_idx = 0
        self.selected_launch_zone = 0
        self.missile_waypoints: dict = {m.id: [] for m in self.sim.missiles}
        self.missile_targets: dict = {m.id: None for m in self.sim.missiles}
        self.selecting_target = False

        # Execution state
        self.selected_flight_missile = None

        # 3D camera
        self.cam_rot_x = 30.0
        self.cam_rot_z = 0.0
        self.cam_zoom = 1.0
        self.dragging_3d = False
        self.drag_start = None

        # Score tracking
        self.score = 0.0
        self.targets_hit = 0
        self.missiles_lost = 0
        self.decoys_used = 0

        # Event flash effects
        self.flash_effects: list = []

    # ============================================================
    # COORDINATE TRANSFORMS
    # ============================================================

    def world_to_screen_2d(self, world_pos) -> Tuple:
        """Convert world (km) to screen pixels for top-down view."""
        from typing import Tuple
        w = self.static['area_width']
        h = self.static['area_height']
        margin = 20
        sx = margin + (world_pos[0] / w) * (self.map_width - 2 * margin)
        sy = margin + ((h - world_pos[1]) / h) * (self.map_height - 2 * margin)
        return (int(sx), int(sy))

    def screen_to_world_2d(self, screen_pos) -> np.ndarray:
        """Convert screen pixels to world (km) for top-down view."""
        w = self.static['area_width']
        h = self.static['area_height']
        margin = 20
        wx = ((screen_pos[0] - margin) / (self.map_width - 2 * margin)) * w
        wy = h - ((screen_pos[1] - margin) / (self.map_height - 2 * margin)) * h
        return np.array([wx, wy, 0.0])

    def world_to_screen_3d(self, world_pos) -> Tuple:
        """Simple 3D projection for perspective view."""
        from typing import Tuple
        w = self.static['area_width']
        h = self.static['area_height']

        # Center and normalize
        x = (world_pos[0] / w - 0.5) * 2.0
        y = (world_pos[1] / h - 0.5) * 2.0
        z = world_pos[2] / 50.0 if len(world_pos) > 2 else 0.0

        # Apply rotation
        rx = math.radians(self.cam_rot_x)
        rz = math.radians(self.cam_rot_z)

        # Rotate around Z axis
        x1 = x * math.cos(rz) - y * math.sin(rz)
        y1 = x * math.sin(rz) + y * math.cos(rz)
        z1 = z

        # Rotate around X axis (tilt)
        y2 = y1 * math.cos(rx) - z1 * math.sin(rx)
        z2 = y1 * math.sin(rx) + z1 * math.cos(rx)
        x2 = x1

        # Simple perspective projection
        zoom = self.cam_zoom
        fov = 3.0
        scale = fov / (fov + y2 + 2.0) * zoom
        sx = self.map_width / 2 + x2 * scale * self.map_width * 0.35
        sy = self.map_height / 2 - z2 * scale * self.map_height * 0.35

        return (int(sx), int(sy))

    # ============================================================
    # DRAWING - TOP DOWN VIEW
    # ============================================================

    def draw_topdown(self):
        """Draw the top-down tactical map view."""
        # Background
        map_rect = pygame.Rect(0, 0, self.map_width, self.map_height)
        pygame.draw.rect(self.screen, DARK_BG, map_rect)

        # Grid
        w = self.static['area_width']
        h = self.static['area_height']
        grid_step = 50  # km
        for gx in range(0, int(w) + 1, grid_step):
            sx, _ = self.world_to_screen_2d([gx, 0])
            pygame.draw.line(self.screen, GRID_COLOR, (sx, 0), (sx, self.map_height), 1)
        for gy in range(0, int(h) + 1, grid_step):
            _, sy = self.world_to_screen_2d([0, gy])
            pygame.draw.line(self.screen, GRID_COLOR, (0, sy), (self.map_width, sy), 1)

        # Launch zones
        if self.phase == GamePhase.PLANNING:
            for i, zone in enumerate(self.static['launch_zones']):
                tl = self.world_to_screen_2d([zone['x_min'], zone['y_max']])
                br = self.world_to_screen_2d([zone['x_max'], zone['y_min']])
                rect = pygame.Rect(tl[0], tl[1], br[0] - tl[0], br[1] - tl[1])
                color = GREEN if i == self.selected_launch_zone else (40, 80, 50)
                surf = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
                surf.fill((*color, 50))
                self.screen.blit(surf, rect.topleft)
                pygame.draw.rect(self.screen, color, rect, 2)
                label = self.font_small.render(f"Zone {i+1}", True, color)
                self.screen.blit(label, (rect.x + 5, rect.y + 3))

        # Radar range rings
        for r in self.sim.radars:
            if not r.alive:
                continue
            center = self.world_to_screen_2d(r.pos)
            # Calculate pixel radius
            edge = self.world_to_screen_2d([r.pos[0] + r.detection_range, r.pos[1]])
            radius = abs(edge[0] - center[0])
            surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (255, 140, 50, 25), (radius, radius), radius)
            pygame.draw.circle(surf, (255, 140, 50, 60), (radius, radius), radius, 1)
            self.screen.blit(surf, (center[0] - radius, center[1] - radius))

        # SAM range rings
        for s in self.sim.sams:
            if not s.alive:
                continue
            center = self.world_to_screen_2d(s.pos)
            edge = self.world_to_screen_2d([s.pos[0] + s.engagement_range, s.pos[1]])
            radius = abs(edge[0] - center[0])
            surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (220, 50, 50, 20), (radius, radius), radius)
            pygame.draw.circle(surf, (220, 50, 50, 50), (radius, radius), radius, 1)
            self.screen.blit(surf, (center[0] - radius, center[1] - radius))

        # Targets
        for t in self.sim.targets:
            pos = self.world_to_screen_2d(t.pos)
            color = TARGET_COLOR if t.alive else TARGET_HIT_COLOR
            self.draw_star(pos, 10, color)
            label = self.font_small.render(f"T{t.id}", True, color)
            self.screen.blit(label, (pos[0] + 12, pos[1] - 8))

        # Radar sites
        for r in self.sim.radars:
            pos = self.world_to_screen_2d(r.pos)
            color = RED_RADAR if r.alive else TARGET_HIT_COLOR
            pygame.draw.circle(self.screen, color, pos, 7)
            pygame.draw.circle(self.screen, RED_RADAR_LIGHT, pos, 7, 2)
            # Radar "dish" indicator
            pygame.draw.arc(self.screen, RED_RADAR_LIGHT,
                           (pos[0] - 12, pos[1] - 12, 24, 24),
                           math.radians(30), math.radians(150), 2)

        # SAM sites
        for s in self.sim.sams:
            pos = self.world_to_screen_2d(s.pos)
            color = RED_SAM if s.alive and s.interceptors_remaining > 0 else TARGET_HIT_COLOR
            pygame.draw.polygon(self.screen, color, [
                (pos[0], pos[1] - 9),
                (pos[0] - 7, pos[1] + 6),
                (pos[0] + 7, pos[1] + 6),
            ])
            # Ammo indicator
            ammo_text = self.font_small.render(f"{s.interceptors_remaining}", True, WHITE)
            self.screen.blit(ammo_text, (pos[0] + 10, pos[1] - 5))

        # Missile trails
        for m in self.sim.missiles:
            if len(m.trail) < 2:
                continue
            color = BLUE_MISSILE if m.missile_type == EntityType.REAL_MISSILE else DECOY_COLOR
            points = [self.world_to_screen_2d(p) for p in m.trail[-60:]]
            if len(points) >= 2:
                pygame.draw.lines(self.screen, (*color[:3],), False, points, 1)

        # Waypoints (planning phase)
        if self.phase == GamePhase.PLANNING:
            for m in self.sim.missiles:
                wps = self.missile_waypoints.get(m.id, [])
                if not wps:
                    continue
                is_selected = (m.id == self.sim.missiles[self.selected_missile_idx].id)
                color = SELECTED_COLOR if is_selected else WAYPOINT_COLOR
                # Draw waypoint path
                zone = self.static['launch_zones'][self.selected_launch_zone]
                start = self.world_to_screen_2d([
                    (zone['x_min'] + zone['x_max']) / 2,
                    (zone['y_min'] + zone['y_max']) / 2,
                ])
                points = [start] + [self.world_to_screen_2d(wp) for wp in wps]
                if len(points) >= 2:
                    pygame.draw.lines(self.screen, color, False, points, 2 if is_selected else 1)
                for i, wp in enumerate(wps):
                    sp = self.world_to_screen_2d(wp)
                    pygame.draw.circle(self.screen, color, sp, 4 if is_selected else 3)
                    if is_selected:
                        num = self.font_small.render(str(i + 1), True, color)
                        self.screen.blit(num, (sp[0] + 6, sp[1] - 10))

        # Missiles in flight
        for m in self.sim.missiles:
            if m.state != MissileState.IN_FLIGHT:
                continue
            pos = self.world_to_screen_2d(m.pos)
            is_real = m.missile_type == EntityType.REAL_MISSILE
            base_color = BLUE_MISSILE if is_real else DECOY_COLOR

            # Size varies with altitude
            size = max(4, min(10, int(4 + m.altitude / 5)))

            # Draw missile
            if is_real:
                pygame.draw.polygon(self.screen, base_color, [
                    (pos[0], pos[1] - size),
                    (pos[0] - size // 2, pos[1] + size // 2),
                    (pos[0] + size // 2, pos[1] + size // 2),
                ])
            else:
                pygame.draw.circle(self.screen, base_color, pos, size // 2 + 1)

            # Tracking indicator
            if m.is_tracked:
                pygame.draw.circle(self.screen, ORANGE, pos, size + 4, 2)

            # Evasion indicator
            if m.evading:
                pygame.draw.circle(self.screen, YELLOW, pos, size + 7, 1)

            # Selection highlight
            if (self.phase == GamePhase.EXECUTING and
                self.selected_flight_missile is not None and
                m.id == self.selected_flight_missile):
                pygame.draw.circle(self.screen, SELECTED_COLOR, pos, size + 10, 2)

            # Label
            label_color = BLUE_MISSILE_LIGHT if is_real else DECOY_LIGHT
            mtype = "R" if is_real else "D"
            label = self.font_small.render(f"{mtype}{m.id}", True, label_color)
            self.screen.blit(label, (pos[0] + size + 3, pos[1] - 8))

        # Interceptors
        for intc in self.sim.interceptors:
            if intc.state != InterceptorState.IN_FLIGHT:
                continue
            pos = self.world_to_screen_2d(intc.pos)
            pygame.draw.circle(self.screen, INTERCEPTOR_COLOR, pos, 3)
            # Line to target
            target_m = self.sim._get_missile(intc.target_missile_id)
            if target_m and target_m.state == MissileState.IN_FLIGHT:
                tpos = self.world_to_screen_2d(target_m.pos)
                pygame.draw.line(self.screen, (255, 60, 60, 100), pos, tpos, 1)

        # Flash effects
        self.draw_flash_effects()

    def draw_star(self, pos, size, color):
        """Draw a star shape."""
        points = []
        for i in range(10):
            angle = math.radians(i * 36 - 90)
            r = size if i % 2 == 0 else size * 0.4
            points.append((
                pos[0] + r * math.cos(angle),
                pos[1] + r * math.sin(angle),
            ))
        pygame.draw.polygon(self.screen, color, points)

    # ============================================================
    # DRAWING - 3D VIEW
    # ============================================================

    def draw_3d(self):
        """Draw the 3D perspective view."""
        map_rect = pygame.Rect(0, 0, self.map_width, self.map_height)
        pygame.draw.rect(self.screen, DARK_BG, map_rect)

        w = self.static['area_width']
        h = self.static['area_height']

        # Ground grid
        grid_step = 50
        for gx in range(0, int(w) + 1, grid_step):
            p1 = self.world_to_screen_3d([gx, 0, 0])
            p2 = self.world_to_screen_3d([gx, h, 0])
            if self.in_map(p1) and self.in_map(p2):
                pygame.draw.line(self.screen, GRID_COLOR, p1, p2, 1)
        for gy in range(0, int(h) + 1, grid_step):
            p1 = self.world_to_screen_3d([0, gy, 0])
            p2 = self.world_to_screen_3d([w, gy, 0])
            if self.in_map(p1) and self.in_map(p2):
                pygame.draw.line(self.screen, GRID_COLOR, p1, p2, 1)

        # Targets
        for t in self.sim.targets:
            pos = self.world_to_screen_3d(t.pos)
            if self.in_map(pos):
                color = TARGET_COLOR if t.alive else TARGET_HIT_COLOR
                self.draw_star(pos, 8, color)

        # SAMs
        for s in self.sim.sams:
            pos = self.world_to_screen_3d(s.pos)
            if self.in_map(pos):
                color = RED_SAM if s.alive and s.interceptors_remaining > 0 else TARGET_HIT_COLOR
                pygame.draw.polygon(self.screen, color, [
                    (pos[0], pos[1] - 7),
                    (pos[0] - 5, pos[1] + 4),
                    (pos[0] + 5, pos[1] + 4),
                ])

        # Radars
        for r in self.sim.radars:
            pos = self.world_to_screen_3d(r.pos)
            if self.in_map(pos):
                color = RED_RADAR if r.alive else TARGET_HIT_COLOR
                pygame.draw.circle(self.screen, color, pos, 5)

        # Missiles
        for m in self.sim.missiles:
            if m.state != MissileState.IN_FLIGHT:
                continue
            pos = self.world_to_screen_3d(m.pos)
            ground_pos = self.world_to_screen_3d([m.pos[0], m.pos[1], 0])
            if self.in_map(pos):
                # Altitude line
                pygame.draw.line(self.screen, (50, 50, 70), ground_pos, pos, 1)
                # Missile
                is_real = m.missile_type == EntityType.REAL_MISSILE
                color = BLUE_MISSILE if is_real else DECOY_COLOR
                size = 5
                if is_real:
                    pygame.draw.polygon(self.screen, color, [
                        (pos[0], pos[1] - size),
                        (pos[0] - size // 2, pos[1] + size // 2),
                        (pos[0] + size // 2, pos[1] + size // 2),
                    ])
                else:
                    pygame.draw.circle(self.screen, color, pos, 3)

                if m.is_tracked:
                    pygame.draw.circle(self.screen, ORANGE, pos, size + 3, 1)

            # Trail
            if len(m.trail) >= 2:
                trail_pts = [self.world_to_screen_3d(p) for p in m.trail[-40:]]
                valid = [p for p in trail_pts if self.in_map(p)]
                if len(valid) >= 2:
                    color = BLUE_MISSILE if m.missile_type == EntityType.REAL_MISSILE else DECOY_COLOR
                    pygame.draw.lines(self.screen, color, False, valid, 1)

        # Interceptors
        for intc in self.sim.interceptors:
            if intc.state != InterceptorState.IN_FLIGHT:
                continue
            pos = self.world_to_screen_3d(intc.pos)
            if self.in_map(pos):
                pygame.draw.circle(self.screen, INTERCEPTOR_COLOR, pos, 3)

        self.draw_flash_effects()

    def in_map(self, pos) -> bool:
        return 0 <= pos[0] < self.map_width and 0 <= pos[1] < self.map_height

    # ============================================================
    # DRAWING - SIDE PANEL
    # ============================================================

    def draw_panel(self):
        """Draw the right-side info panel."""
        x = self.map_width
        panel_rect = pygame.Rect(x, 0, self.panel_width, self.screen_height)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect)
        pygame.draw.line(self.screen, PANEL_BORDER, (x, 0), (x, self.screen_height), 2)

        y = 15
        pad = x + 15

        # Title
        title = self.font_title.render("IADS STRIKE", True, BLUE_MISSILE_LIGHT)
        self.screen.blit(title, (pad, y))
        y += 35

        # Phase indicator
        phase_color = GREEN if self.phase == GamePhase.PLANNING else ORANGE
        phase_text = self.phase.upper()
        if self.phase == GamePhase.EXECUTING and self.paused:
            phase_text += " (PAUSED)"
        label = self.font_med.render(phase_text, True, phase_color)
        self.screen.blit(label, (pad, y))
        y += 25

        # Time and speed
        if self.phase != GamePhase.PLANNING:
            effective = self.base_sim_rate * self.sim_speed
            time_text = self.font_small.render(
                f"Time: {self.sim.time}/{self.sim.max_time}  Speed: {self.sim_speed}x ({effective:.0f} steps/s)",
                True, WHITE)
            self.screen.blit(time_text, (pad, y))
        y += 20

        # View mode
        view_text = self.font_small.render(
            f"View: {'3D' if self.view_3d else 'Top-Down'} [V to switch]",
            True, (150, 150, 150))
        self.screen.blit(view_text, (pad, y))
        y += 25

        # Divider
        pygame.draw.line(self.screen, PANEL_BORDER, (pad, y), (x + self.panel_width - 15, y), 1)
        y += 10

        # MISSILES
        header = self.font_med.render("MISSILES", True, BLUE_MISSILE_LIGHT)
        self.screen.blit(header, (pad, y))
        y += 22

        for i, m in enumerate(self.sim.missiles):
            is_real = m.missile_type == EntityType.REAL_MISSILE
            type_str = "REAL" if is_real else "DCOY"
            color = BLUE_MISSILE if is_real else DECOY_COLOR

            # Status
            if m.state == MissileState.READY:
                status = "READY"
                status_color = (150, 150, 150)
            elif m.state == MissileState.IN_FLIGHT:
                status = "FLIGHT"
                status_color = GREEN
            elif m.state == MissileState.HIT_TARGET:
                status = "HIT!"
                status_color = TARGET_COLOR
            elif m.state == MissileState.INTERCEPTED:
                status = "LOST"
                status_color = RED_SAM
            else:
                status = "MISS"
                status_color = (150, 150, 150)

            # Highlight if selected
            if self.phase == GamePhase.PLANNING and i == self.selected_missile_idx:
                pygame.draw.rect(self.screen, (40, 50, 70),
                               (pad - 5, y - 2, self.panel_width - 25, 18))

            if (self.phase == GamePhase.EXECUTING and
                self.selected_flight_missile is not None and
                m.id == self.selected_flight_missile):
                pygame.draw.rect(self.screen, (40, 50, 70),
                               (pad - 5, y - 2, self.panel_width - 25, 18))

            text = self.font_small.render(f"{type_str} {m.id}", True, color)
            self.screen.blit(text, (pad, y))
            st = self.font_small.render(status, True, status_color)
            self.screen.blit(st, (pad + 70, y))

            if m.state == MissileState.IN_FLIGHT:
                fuel_pct = max(0, (m.fuel - m.distance_traveled) / m.fuel * 100)
                fuel_text = self.font_small.render(f"F:{fuel_pct:.0f}%", True, (150, 150, 150))
                self.screen.blit(fuel_text, (pad + 130, y))
                if m.is_tracked:
                    trk = self.font_small.render("TRK", True, ORANGE)
                    self.screen.blit(trk, (pad + 190, y))
                if m.evading:
                    evd = self.font_small.render("EVD", True, YELLOW)
                    self.screen.blit(evd, (pad + 230, y))

            # Waypoint count in planning
            if self.phase == GamePhase.PLANNING:
                wps = self.missile_waypoints.get(m.id, [])
                wp_text = self.font_small.render(f"WP:{len(wps)}", True, (150, 150, 150))
                self.screen.blit(wp_text, (pad + 130, y))
                if is_real and self.missile_targets.get(m.id) is not None:
                    tgt = self.font_small.render(f"->T{self.missile_targets[m.id]}", True, TARGET_COLOR)
                    self.screen.blit(tgt, (pad + 190, y))

            y += 18

        y += 10
        pygame.draw.line(self.screen, PANEL_BORDER, (pad, y), (x + self.panel_width - 15, y), 1)
        y += 10

        # SAM STATUS
        header = self.font_med.render("DEFENSE", True, RED_SAM_LIGHT)
        self.screen.blit(header, (pad, y))
        y += 22

        for s in self.sim.sams:
            ammo_pct = s.interceptors_remaining / s.num_interceptors
            color = GREEN if ammo_pct > 0.5 else ORANGE if ammo_pct > 0 else RED_SAM
            bar_w = 80
            bar_h = 8
            pygame.draw.rect(self.screen, (40, 40, 40), (pad + 60, y + 2, bar_w, bar_h))
            pygame.draw.rect(self.screen, color,
                           (pad + 60, y + 2, int(bar_w * ammo_pct), bar_h))
            text = self.font_small.render(f"SAM {s.id}", True, (180, 180, 180))
            self.screen.blit(text, (pad, y))
            count = self.font_small.render(f"{s.interceptors_remaining}/{s.num_interceptors}",
                                           True, (150, 150, 150))
            self.screen.blit(count, (pad + 150, y))
            y += 16

        y += 10
        pygame.draw.line(self.screen, PANEL_BORDER, (pad, y), (x + self.panel_width - 15, y), 1)
        y += 10

        # SCORE
        header = self.font_med.render("SCORE", True, TARGET_COLOR)
        self.screen.blit(header, (pad, y))
        y += 22

        targets_alive = sum(1 for t in self.sim.targets if t.alive)
        targets_total = len(self.sim.targets)
        targets_hit = targets_total - targets_alive
        text = self.font_small.render(f"Targets Hit: {targets_hit}/{targets_total}", True, WHITE)
        self.screen.blit(text, (pad, y))
        y += 16

        real_lost = sum(1 for m in self.sim.missiles
                       if m.missile_type == EntityType.REAL_MISSILE
                       and m.state == MissileState.INTERCEPTED)
        text = self.font_small.render(f"Real Missiles Lost: {real_lost}", True, WHITE)
        self.screen.blit(text, (pad, y))
        y += 16

        decoy_used = sum(1 for m in self.sim.missiles
                        if m.missile_type == EntityType.DECOY_MISSILE
                        and m.state == MissileState.INTERCEPTED)
        text = self.font_small.render(f"Decoys Absorbed: {decoy_used}", True, WHITE)
        self.screen.blit(text, (pad, y))
        y += 16

        intc_total = len(self.sim.interceptors)
        text = self.font_small.render(f"Interceptors Fired: {intc_total}", True, WHITE)
        self.screen.blit(text, (pad, y))
        y += 25

        # CONTROLS HELP
        pygame.draw.line(self.screen, PANEL_BORDER, (pad, y), (x + self.panel_width - 15, y), 1)
        y += 10
        header = self.font_med.render("CONTROLS", True, (150, 150, 150))
        self.screen.blit(header, (pad, y))
        y += 20

        if self.phase == GamePhase.PLANNING:
            controls = [
                "LClick: Place waypoint",
                "RClick: Remove waypoint",
                "Tab: Cycle missile",
                "T: Assign target",
                "1/2/3: Select launch zone",
                "Space: LAUNCH",
                "V: Toggle view",
            ]
        elif self.phase == GamePhase.EXECUTING:
            controls = [
                "LClick: Select missile",
                "A/D: Steer left/right",
                "W/S: Climb/dive",
                "E: Toggle evasion",
                "Tab: Cycle missile",
                "Space: Pause",
                "+/-: Speed up/down",
                "V: Toggle view",
            ]
        else:
            controls = [
                "R: Restart",
                "ESC: Quit",
            ]

        for ctrl in controls:
            text = self.font_small.render(ctrl, True, (120, 120, 140))
            self.screen.blit(text, (pad, y))
            y += 15

    # ============================================================
    # FLASH EFFECTS
    # ============================================================

    def add_flash(self, world_pos, color, size=20, duration=15):
        self.flash_effects.append({
            'pos': world_pos.copy() if isinstance(world_pos, np.ndarray) else np.array(world_pos),
            'color': color,
            'size': size,
            'duration': duration,
            'frame': 0,
        })

    def draw_flash_effects(self):
        to_remove = []
        for i, fx in enumerate(self.flash_effects):
            progress = fx['frame'] / fx['duration']
            if progress >= 1.0:
                to_remove.append(i)
                continue
            alpha = int(255 * (1.0 - progress))
            size = int(fx['size'] * (1.0 + progress * 2))
            if self.view_3d:
                pos = self.world_to_screen_3d(fx['pos'])
            else:
                pos = self.world_to_screen_2d(fx['pos'])
            if self.in_map(pos):
                surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                pygame.draw.circle(surf, (*fx['color'][:3], alpha), (size, size), size)
                self.screen.blit(surf, (pos[0] - size, pos[1] - size))
            fx['frame'] += 1
        for i in reversed(to_remove):
            self.flash_effects.pop(i)

    # ============================================================
    # INPUT HANDLING
    # ============================================================

    def handle_planning_input(self, event):
        """Handle input during planning phase."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            if mx >= self.map_width:
                return  # Clicked on panel

            world = self.screen_to_world_2d((mx, my))
            current_missile = self.sim.missiles[self.selected_missile_idx]

            if event.button == 1:  # Left click - place waypoint
                max_wp = self.config.get('max_waypoints', 10)
                wps = self.missile_waypoints[current_missile.id]
                if len(wps) < max_wp:
                    wp = np.array([world[0], world[1], current_missile.altitude])
                    wps.append(wp)

            elif event.button == 3:  # Right click - remove last waypoint
                wps = self.missile_waypoints[current_missile.id]
                if wps:
                    wps.pop()

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_TAB:
                self.selected_missile_idx = (self.selected_missile_idx + 1) % len(self.sim.missiles)

            elif event.key == pygame.K_t:
                # Cycle target assignment for real missiles
                current_missile = self.sim.missiles[self.selected_missile_idx]
                if current_missile.missile_type == EntityType.REAL_MISSILE:
                    current_target = self.missile_targets.get(current_missile.id)
                    num_targets = len(self.sim.targets)
                    if current_target is None:
                        self.missile_targets[current_missile.id] = 0
                    else:
                        next_t = (current_target + 1) % (num_targets + 1)
                        self.missile_targets[current_missile.id] = next_t if next_t < num_targets else None

            elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                zone_idx = event.key - pygame.K_1
                if zone_idx < len(self.static['launch_zones']):
                    self.selected_launch_zone = zone_idx

            elif event.key == pygame.K_SPACE:
                self.execute_launch()

    def handle_execution_input(self, event):
        """Handle input during execution phase."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            if mx >= self.map_width:
                return

            if event.button == 1:  # Select nearest in-flight missile
                world = self.screen_to_world_2d((mx, my))
                best_dist = float('inf')
                best_id = None
                for m in self.sim.missiles:
                    if m.state != MissileState.IN_FLIGHT:
                        continue
                    d = np.linalg.norm(m.pos[:2] - world[:2])
                    if d < best_dist and d < 20:
                        best_dist = d
                        best_id = m.id
                self.selected_flight_missile = best_id

            elif event.button == 3:
                self.selected_flight_missile = None

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.paused = not self.paused

            elif event.key == pygame.K_TAB:
                # Cycle through in-flight missiles
                in_flight = [m for m in self.sim.missiles if m.state == MissileState.IN_FLIGHT]
                if in_flight:
                    if self.selected_flight_missile is None:
                        self.selected_flight_missile = in_flight[0].id
                    else:
                        ids = [m.id for m in in_flight]
                        try:
                            idx = ids.index(self.selected_flight_missile)
                            self.selected_flight_missile = ids[(idx + 1) % len(ids)]
                        except ValueError:
                            self.selected_flight_missile = ids[0]

            elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                self.sim_speed = min(8, self.sim_speed * 2)
            elif event.key == pygame.K_MINUS:
                self.sim_speed = max(1, self.sim_speed // 2)

            # Steering controls
            elif self.selected_flight_missile is not None:
                if event.key == pygame.K_a:
                    self.sim.steer_missile(self.selected_flight_missile, heading_delta=-15.0)
                elif event.key == pygame.K_d:
                    self.sim.steer_missile(self.selected_flight_missile, heading_delta=15.0)
                elif event.key == pygame.K_w:
                    self.sim.steer_missile(self.selected_flight_missile, altitude_delta=2.0)
                elif event.key == pygame.K_s:
                    self.sim.steer_missile(self.selected_flight_missile, altitude_delta=-2.0)
                elif event.key == pygame.K_e:
                    m = self.sim._get_missile(self.selected_flight_missile)
                    if m:
                        self.sim.set_missile_evading(m.id, not m.evading)

    def handle_done_input(self, event):
        """Handle input when simulation is done."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self.restart()

    # ============================================================
    # LAUNCH EXECUTION
    # ============================================================

    def execute_launch(self):
        """Validate plan and launch the salvo."""
        zone = self.static['launch_zones'][self.selected_launch_zone]
        launch_pos = np.array([
            (zone['x_min'] + zone['x_max']) / 2,
            (zone['y_min'] + zone['y_max']) / 2,
            self.config.get('missile_altitude', 10.0),
        ])

        plans = []
        for m in self.sim.missiles:
            wps = self.missile_waypoints.get(m.id, [])
            if not wps:
                # Auto-generate a simple forward path if no waypoints set
                wps = [np.array([launch_pos[0], self.static['area_height'] * 0.5, m.altitude]),
                       np.array([launch_pos[0], self.static['area_height'] * 0.8, m.altitude])]

            plan = {
                'missile_id': m.id,
                'waypoints': wps,
            }
            if m.missile_type == EntityType.REAL_MISSILE:
                tid = self.missile_targets.get(m.id)
                if tid is not None:
                    plan['target_id'] = tid
                    # Add target position as final waypoint if not already close
                    target_pos = self.sim.targets[tid].pos
                    if len(wps) == 0 or np.linalg.norm(wps[-1][:2] - target_pos[:2]) > 10:
                        wps.append(np.array([target_pos[0], target_pos[1], m.altitude]))

            plans.append(plan)

        self.sim.launch_salvo(launch_pos, plans)
        self.phase = GamePhase.EXECUTING
        self.tick_accumulator = 0.0

    def restart(self):
        """Restart the game with a new scenario."""
        self.sim = IADSSimulation(self.config)
        self.static = self.sim.get_static_state()
        self.phase = GamePhase.PLANNING
        self.paused = False
        self.sim_speed = 1
        self.selected_missile_idx = 0
        self.selected_launch_zone = 0
        self.missile_waypoints = {m.id: [] for m in self.sim.missiles}
        self.missile_targets = {m.id: None for m in self.sim.missiles}
        self.selected_flight_missile = None
        self.flash_effects = []

    # ============================================================
    # MAIN LOOP
    # ============================================================

    def run(self):
        """Main game loop."""
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.WINDOWRESIZED:
                    self.screen_width = event.x
                    self.screen_height = event.y
                    self.map_width = self.screen_width - self.panel_width
                    self.map_height = self.screen_height
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_v:
                        self.view_3d = not self.view_3d

                    # Phase-specific input
                    if self.phase == GamePhase.PLANNING:
                        self.handle_planning_input(event)
                    elif self.phase == GamePhase.EXECUTING:
                        self.handle_execution_input(event)
                    elif self.phase == GamePhase.DONE:
                        self.handle_done_input(event)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.phase == GamePhase.PLANNING:
                        self.handle_planning_input(event)
                    elif self.phase == GamePhase.EXECUTING:
                        self.handle_execution_input(event)

                # 3D camera rotation
                if self.view_3d:
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
                        self.dragging_3d = True
                        self.drag_start = event.pos
                    elif event.type == pygame.MOUSEBUTTONUP and event.button == 2:
                        self.dragging_3d = False
                    elif event.type == pygame.MOUSEMOTION and self.dragging_3d:
                        dx = event.pos[0] - self.drag_start[0]
                        dy = event.pos[1] - self.drag_start[1]
                        self.cam_rot_z += dx * 0.3
                        self.cam_rot_x = max(5, min(80, self.cam_rot_x + dy * 0.3))
                        self.drag_start = event.pos
                    elif event.type == pygame.MOUSEWHEEL:
                        self.cam_zoom = max(0.3, min(3.0, self.cam_zoom + event.y * 0.1))

            # Simulation stepping — accumulator-based for human-playable speed
            if self.phase == GamePhase.EXECUTING and not self.paused:
                dt_ms = self.clock.get_time()
                steps_per_sec = self.base_sim_rate * self.sim_speed
                self.tick_accumulator += dt_ms / 1000.0
                step_interval = 1.0 / steps_per_sec
                while self.tick_accumulator >= step_interval:
                    self.tick_accumulator -= step_interval
                    events, done = self.sim.step()

                    # Process events for visual effects
                    for e in events:
                        if 'position' in e:
                            pos = np.array(e['position'])
                        else:
                            pos = None

                        if e['type'] == 'real_intercepted' and pos is not None:
                            self.add_flash(pos, RED_SAM, size=25, duration=20)
                        elif e['type'] == 'decoy_intercepted' and pos is not None:
                            self.add_flash(pos, ORANGE, size=15, duration=15)
                        elif e['type'] == 'target_hit' and pos is not None:
                            self.add_flash(pos, TARGET_COLOR, size=35, duration=25)
                        elif e['type'] == 'interceptor_launched':
                            sam = self.sim.sams[e['sam_id']]
                            self.add_flash(sam.pos, RED_SAM_LIGHT, size=10, duration=8)

                    if done:
                        self.phase = GamePhase.DONE
                        break

            # Drawing
            self.screen.fill(BLACK)

            if self.view_3d:
                self.draw_3d()
            else:
                self.draw_topdown()

            self.draw_panel()

            # Phase-specific overlays
            if self.phase == GamePhase.DONE:
                self.draw_end_screen()

            pygame.display.flip()
            self.clock.tick(30)

        # Save replay on exit
        try:
            self.sim.save_replay("replays/last_game.json")
        except Exception:
            pass

        pygame.quit()

    def draw_end_screen(self):
        """Draw end-of-mission summary overlay."""
        overlay = pygame.Surface((self.map_width, self.map_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        cx = self.map_width // 2
        cy = self.map_height // 2

        targets_hit = sum(1 for t in self.sim.targets if not t.alive)
        total_targets = len(self.sim.targets)

        title = self.font_title.render("MISSION COMPLETE", True, WHITE)
        self.screen.blit(title, (cx - title.get_width() // 2, cy - 80))

        if targets_hit == total_targets:
            result = self.font_large.render("ALL TARGETS DESTROYED", True, GREEN)
        elif targets_hit > 0:
            result = self.font_large.render(f"{targets_hit}/{total_targets} TARGETS HIT", True, ORANGE)
        else:
            result = self.font_large.render("MISSION FAILED", True, RED_SAM)
        self.screen.blit(result, (cx - result.get_width() // 2, cy - 40))

        restart = self.font_med.render("Press R to restart | ESC to quit", True, (150, 150, 150))
        self.screen.blit(restart, (cx - restart.get_width() // 2, cy + 20))


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    """Run the IADS Penetration game."""
    import argparse
    parser = argparse.ArgumentParser(description="IADS Penetration Strike Planner")
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--width', type=int, default=1400, help='Screen width')
    parser.add_argument('--height', type=int, default=900, help='Screen height')
    parser.add_argument('--template', type=str, choices=['spread', 'layered', 'left_heavy', 'right_heavy'],
                       help='Defense template')
    parser.add_argument('--seed', type=int, help='Random seed')
    args = parser.parse_args()

    from iads.gym_env import load_scenario_config

    config = IADSSimulation.default_config()

    # Auto-load default.yaml (or .json fallback) from script directory if present
    here = os.path.dirname(os.path.abspath(__file__))
    for fname in ("default.yaml", "default.json"):
        path = os.path.join(here, fname)
        if os.path.exists(path):
            config.update(load_scenario_config(path))
            break

    # --config overrides on top of the default
    if args.config:
        config.update(load_scenario_config(args.config))

    if args.template:
        config['defense_template'] = args.template
    if args.seed is not None:
        config['seed'] = args.seed
        config['layout_seed'] = args.seed

    config['max_waypoints'] = config.get('max_waypoints', 10)

    game = IADSGame(config=config, screen_width=args.width, screen_height=args.height)
    game.run()


if __name__ == "__main__":
    main()
