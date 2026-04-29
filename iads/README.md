# iads/

IADS-specific simulation, Gymnasium environment, and human-playable game. These files are the domain layer — everything here knows about missiles, radars, SAMs, and targets.

---

## Files

### `sim_engine.py` — Headless Simulation Engine
Pure Python + numpy. No GUI, no Gym, no Torch. Used by both the game and the RL pipeline.

**Key classes:**

| Class | Role |
|-------|------|
| `IADSSimulation` | Main sim. `step()` advances one timestep and returns `(events, done)`. |
| `Radar` | Detects missiles in range; builds classification confidence over time |
| `SAM` | Scripted defender; fires interceptors at tracked unclassified missiles |
| `Missile` | Attacker unit; follows waypoints, burns fuel, can evade |
| `Interceptor` | Defender projectile; pursues missile to predicted intercept point |
| `Target` | Fixed asset; destroyed when a real missile hits within `hit_radius` (3 km) |

**Each `step()` does (in order):**
1. Move missiles (waypoint following, fuel burn)
2. Move interceptors (pursuit steering)
3. Update track merging (missiles within 3 km → one radar contact)
4. Update radar detection and classification confidence
5. Run SAM AI (engage trackable, unclassified targets)
6. Resolve intercepts (probabilistic kill roll against Pk)
7. Check target hits (distance threshold)
8. Return events list + done flag

**Key methods:**
- `launch_salvo(launch_pos, missile_plans)` — position missiles and start execution
- `steer_missile(missile_id, heading_delta, altitude_delta)` — per-step steering
- `set_missile_evading(missile_id, evading)` — toggle evasion (-30% Pk, +50% fuel burn)
- `compute_reward(events)` — event-driven reward from the step's events list

### `gym_env.py` — Gymnasium Wrapper
Registers `IADSPenetration-v0`. Wraps `IADSSimulation` as a standard Gymnasium env.

- **Obs:** `Box(0, 1, shape=(N,), float32)` — flat normalized vector, N computed from config (~369 with defaults). Four blocks: world state, decision context (15 dims: type one-hot + subject one-hot + progress), plan state (accumulates as planning proceeds), dynamic state (zeros during planning).
- **Action:** `Box(-1, 1, shape=(29,), float32)` — dims 0-4 are decision type one-hot (informational), dims 5-28 are the 24-dim value interpreted based on current decision type.
- **Sequential planning phase:** Each planning decision is its own step (reward=0, sim clock frozen). Fixed order: launch zone (1 step) → target assign per real missile (×4) → waypoint count per missile (×8) → waypoint placement per missile (×num_waypoints). Total planning: 21–93 steps.
- **Execution phase:** Steps after planning — agent steers missiles using dims 5-28 as 24-dim continuous control (per-missile heading, altitude, evasion for up to 8 missiles).
- **Info dict:** `decision_type` (0–4), `decision_subject` (int), `is_planning` (bool), `planning_step` (int), `episode_planning_length` (int).
- **Reward modes:** `semi_dense` (default), `sparse`, `dense`

**Decision types:**

| Value | Name | Actor output | Subject |
|-------|------|-------------|---------|
| 0 | launch_zone | Categorical(3) | salvo |
| 1 | target_assign | Categorical(3) | real missile index |
| 2 | num_waypoints | Categorical(10) | missile index |
| 3 | waypoint_place | Continuous(2) | missile index |
| 4 | execution | Continuous(24) | all missiles |

```python
import iads.gym_env  # must import to trigger registration
env = gymnasium.make("IADSPenetration-v0")
env = gymnasium.make("IADSPenetration-v0", config_path="iads/default.json", reward_mode="sparse")
envs = gymnasium.make_vec("IADSPenetration-v0", num_envs=8)
```

### `game.py` — Pygame Game
Human-playable interface. Creates an `IADSSimulation`, renders it, and maps keyboard/mouse input to engine calls. Supports top-down and 3D perspective views. Frame-rate independent via time accumulator.

```bash
uv run python -m iads.game
uv run python -m iads.game --template layered --seed 42
```

### `default.json` — Default Scenario Config
Auto-loaded by `game.py` at startup. Pass `--config` to overlay additional overrides.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `area_width` / `area_height` | 400 / 600 km | Map size |
| `max_time` | 200 steps | Episode time limit |
| `num_radars` | 3 | Radar range: 100 km |
| `num_sams` | 6 | SAM range: 70 km, Pk: 0.70, 6 interceptors, 100-step reload |
| `num_real_missiles` | 4 | Strike missiles (must hit targets) |
| `num_decoy_missiles` | 4 | Decoys (draw fire) |
| `missile_speed` | 4.0 km/step | — |
| `missile_fuel` | 300 km | Max range |
| `evasion_pk_reduction` | 0.30 | Evasion reduces Pk by 30% |

Defense layout templates: `spread`, `layered`, `left_heavy`, `right_heavy`

---

## Portability Note
`sim_engine.py` has no dependencies beyond numpy. Both `game.py` and `gym_env.py` import from `iads.sim_engine`. The rest of the project (`mindwipe/`, `orchestra/`) never imports from `iads/` directly except through the Gymnasium registration.
