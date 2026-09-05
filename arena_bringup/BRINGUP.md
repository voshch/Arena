# Arena bringup usage

Most Arena sessions start with `arena launch`. It is a bash composite that
either:

- **brings up a fresh runtime** if none exists (sim + `arena_node` via
  `arena_runtime.launch.py`, then waits for `/arena/register_env`), or
- **attaches additively** to an already-running runtime, with a sim-mismatch
  check on `sim:=`.

Either way it then spawns `env.n` task-generator envs and, unless
`headless:=true` (or explicit `viz:=false`), runs `arena viz --all` so each
env gets a rviz window.

For the decoupled flow (runtime stays up, envs and viz come and go), the
three underlying verbs can be used independently. See
[CLI verbs](#cli-verbs) below.

The full argument surface for `arena_runtime.launch.py` and
`task_generator.launch.py` is in [launch/README.md](launch/README.md).

---

## Three-verb model

| Verb | Launch file | What it starts |
|---|---|---|
| `arena runtime [args]` | `arena_runtime.launch.py` | Sim + `arena_node`, no envs |
| `arena env [args]` | `task_generator.launch.py` | One task-generator env; waits for `/arena/register_env` (10s warning cadence if runtime is absent) |
| `arena viz [target]` | (ros2 run) | Attaches rviz to a running env; see [arena viz](#arena-viz) |

`arena launch` orchestrates all three (skipping the runtime step if one is
already up) and is the canonical entry point for most sessions.

---

## Minimum-viable invocations

### 1. Smoke check - dummy sim, empty map

No physics engine. Useful for verifying that the ROS graph comes up without
hardware or GPU. The `sim` argument defaults to `gazebo`; pass `sim:=dummy`
explicitly for this plumbing-only mode.

```bash
arena launch \
    sim:=dummy \
    world:=map_empty \
    robot:=jackal \
    task.robots:=explore \
    task.obstacles:=random
```

| Arg | Implication |
|---|---|
| `sim:=dummy` | No physics engine; a `map->dummy` static TF is published instead. Must be explicit - the default is `gazebo`. |
| `world:=map_empty` | Loads the empty map from `arena_simulation_setup` |
| `robot:=jackal` | Single jackal; `mobile` adapter defaults to `none` (dummy sim has no nav2) |
| `task.robots:=explore` | Robot gets fresh random goals continuously |
| `task.obstacles:=random` | Random static/dynamic obstacles placed each episode |

The `human` arg defaults to `dummy` when `sim=dummy`, so pedestrians stay
idle until driven from a `human_steering` panel. `arena launch` auto-attaches
one per env (suppressed by `headless:=true`), and `arena human` re-attaches.
`human.steering:=true` forces the panel on any backend (possess engine-driven
peds), `human.steering:=false` suppresses it.

---

### 2. Gazebo + jackal + random obstacles

```bash
arena launch \
    sim:=gazebo \
    world:=map_empty \
    robot:=jackal \
    robot.mobile.local_planner:=teb \
    task.robots:=explore \
    task.obstacles:=random
```

| Arg | Implication |
|---|---|
| `sim:=gazebo` | Starts gz-sim 8 (dart physics, ogre renderer). `human` defaults to `arena` (arena_humansim) |
| `world:=map_empty` | Resolved to `arena_simulation_setup/worlds/map_empty/worlds/map_empty.world`; falls back to `configs/gazebo/empty.sdf` if absent |
| `robot.mobile.local_planner:=teb` | TEB local planner; `mobile` adapter defaults to `nav2` for gazebo, the override lands as ROS param `robot.mobile.local_planner` and is forwarded to nav2's bringup |
| `headless` | Omitted → `false` (sim GUI visible, rviz shown). Pass `headless:=true` to hide the sim GUI (viz also suppressed unless `viz:=true` is set explicitly) |
| `lockstep:=true` | Start the lockstep scheduler at bringup: the sim advances tick by tick, gated on every registered hard channel, instead of free-running. Comes up paused (frozen at tick zero) unless `lockstep.paused:=false`. Continue with `arena lockstep resume`. Toggle later with `arena lockstep on\|off` |
| `lockstep.paused:=false` | Autostarted lockstep begins stepping immediately instead of waiting for `arena lockstep resume` |
| `lockstep.channels:="a;b"` | Semicolon-separated `name\|topic\|type\|period_s\|hard-or-soft` entries registered under caller `launch`, extra channels, producers self-register their own. `{env}` expands per env |
| `lockstep.rtf:=N` | Target real-time factor for the lockstep scheduler, 0 or empty = unpaced |
| `env.bootstrap_timeout:=N` | Seconds an env may spend never-ACTIVE before eviction, measured from reservation, empty = node default (never) |

To suppress the arena_humansim node when no human obstacles are needed,
add `human:=dummy` to the command above.

---

### 3. Gazebo + jackal + HumanSim

```bash
arena launch \
    sim:=gazebo \
    world:=map_empty \
    robot:=jackal \
    human:=arena \
    tm_robots:=explore \
    tm_obstacles:=random
```

`human:=arena` starts the `arena_humansim` node (subsystem mode) in the
task-generator namespace. Human pedestrian models are managed by the
arena_humansim adapter; the `tm_obstacles` mode controls non-human obstacles
separately.

---

### 4. Isaac + multi-robot via task.config

Isaac must be installed and `arena feature isaac` must be set up before launch.

```bash
arena launch \
    sim:=isaac \
    world:=map_empty \
    task.config:=$(ros2 pkg prefix arena_bringup)/share/arena_bringup/configs/tasks/default.yaml \
    task.obstacles:=random \
    headless:=true
```

| Arg | Implication |
|---|---|
| `sim:=isaac` | Runs `arena feature isaac launch` via bash. `mobile` adapter defaults to `nav2` |
| `task.config:=<path>` | Structured `TaskModeSpec` YAML; overrides `task.robots`. Use to split a fleet across multiple task modes |
| `headless:=true` | Sim GUI hidden; rviz suppressed (no GUI at all) |

Multi-robot fleet with two modes:

```bash
cat > /tmp/fleet.yaml << 'EOF'
task_modes:
  - kind: scenario
    produces: GOTO_POSE
    assignments: [jackal_0]
    config: {}
  - kind: explore
    produces: GOTO_POSE
    assignments: []
    config: {}
EOF

arena launch \
    sim:=isaac \
    world:=map_empty \
    robot:=jackal \
    task.config:=/tmp/fleet.yaml \
    task.obstacles:=random \
    headless:=true
```

`jackal_0` follows a scenario; every other jackal explores. See
[configs/tasks/README.md](configs/tasks/README.md) for the full schema.

---

### 5. Multiple parallel environments

```bash
arena launch \
    sim:=gazebo \
    world:=map_empty \
    robot:=jackal \
    env.n:=3
```

| Arg | Implication |
|---|---|
| `env.n:=3` | Three task-generator instances under `arena/env_0/task_generator_node`, `arena/env_1/...`, `arena/env_2/...`. `arena_node` self-orchestrates the fleet via `/arena/spawn_env`. |

Slot positions are placed by the shelf packer in `arena_node` based on each env's `WorldExtent`; spacing is governed by the `slot_buffer` ROS parameter on `arena_node` (default 5 m).

---

### 6. Runtime / client mode (dynamic envs)

The runtime (`arena_node`) and the simulator can be launched without any
task-generator envs, then envs and viz can be added or removed at will.

```bash
# Terminal 1: runtime only.
arena runtime sim:=gazebo world:=map_empty
```

Then attach pieces from other terminals:

```bash
# Add an env. Multiple invocations stack (different robot/task each).
arena env robot:=jackal task.robots:=explore task.obstacles:=random

# Or use arena launch, which detects the existing runtime and attaches
# additively rather than bringing up a fresh one. Errors on sim:= mismatch.
arena launch sim:=gazebo env.n:=1 robot:=burger task.robots:=random

# Attach rviz to an existing env (auto-pick, by id, or all).
arena viz
arena viz 0
arena viz --all
```

`arena env` and `arena viz` both wait forever (10s warning cadence) if the
runtime or env isn't up yet, so terminal ordering doesn't matter.

To tear an env down by id, call the cleanup service (or use
`arena cleanup <env_id>`, see [CLI verbs](#cli-verbs)).

---

### 7. RL training mode

```bash
arena train sim:=gazebo world:=map_empty robot:=jackal \
    train_config:=/path/to/train_config.yaml
```

Training includes `arena_runtime.launch.py` directly (runtime-only, no
auto-spawn). `train_agent.py` reads `n_envs` from the YAML and spawns envs
via `/arena/spawn_env`.

---

## headless and viz

| Arg | Default | Meaning |
|---|---|---|
| `headless` | `false` | `true` = hide the sim GUI (server-only mode for Gazebo). Implicitly sets `viz:=false` unless `viz:=true` is explicit. |
| `viz` | `true` | Controls whether `arena viz --all` is called after envs come up. Ignored when `headless:=true` unless overridden. |
| `human.steering` | `auto` | Per-env `human_steering` panel. `auto` = attach when the resolved `human` backend is `dummy` (and not headless). `true` = always attach, wins over `headless`. `false` = never. |
| `humansim.markers` | `2` | Debug marker level of the `arena` human backend: `0` = off (also hides its rviz panels), `1` = agent bodies, headings and infrastructure, `2` = adds goals, paths, waypoints, vision cones and force vectors. Lower it to cut per-tick marker cost in crowded scenarios. |
| `viz.view` | `map` | Camera view in rviz: `map` (TopDownOrtho), `robot` (Orbit on robot base), `robot3p` (ThirdPersonFollower on robot base). |
| `viz.robot` | `0` | Robot index in the fleet for `viz.view:=robot*`. `all` spawns one rviz window per robot. Ignored when `view=map`. |

Examples:

```bash
# Sim GUI visible, rviz shown (default)
arena launch sim:=gazebo

# Sim GUI hidden, no rviz
arena launch sim:=gazebo headless:=true

# Sim GUI hidden, rviz shown (explicit override)
arena launch sim:=gazebo headless:=true viz:=true

# Sim GUI visible, no rviz
arena launch sim:=gazebo viz:=false
```

---

## arena viz

Attaches rviz to one or more running envs after launch (out-of-band).

```bash
arena viz                              # auto-pick if exactly one env is running
arena viz <env_id>                     # match by env id (last path component)
arena viz --ns <ns>                    # explicit namespace
arena viz --all                        # one rviz window per running env
arena viz view:=robot3p                # third-person follower on robot 0
arena viz 1 view:=robot robot:=-1      # orbit the last robot of env 1
arena viz --all robot:=all             # all envs, every robot, one window each
```

Any `key:=value` is forwarded straight to
[rviz_config.launch.py](../utils/rviz_utils/launch/rviz_config.launch.py). Under
`arena launch` the same tokens take a `viz.` prefix to disambiguate from
runtime args; `arena viz` accepts the prefixed form too. The launch file
declares `view` and `robot` as ROS params on the rviz_config node.

Waits forever for a matching env to appear (10s warning cadence), mirroring
`arena env`'s wait for the runtime. Once at least one env is up: a single
match with no arg auto-picks; multiple matches with no arg print the list

### Backend selection

`rviz_utils` and [`rerun_utils`](../utils/rerun_utils) both consume the same
[`arena_viz`](../utils/arena_viz) manifest. Pick one (or both) via the
`backend:=` token; the default comes from `$ARENA_VIZ_BACKEND` (falling
back to `rviz`).

```bash
arena viz                                 # rviz (default)
arena viz 0 backend:=rerun                # rerun web viewer against env_0
arena viz --all backend:=rviz,rerun       # both side by side, every env
ARENA_VIZ_BACKEND=rerun arena viz --all   # rerun as the session default
```

Backend-specific knobs pass straight through:

| Backend | Args | Notes |
|---|---|---|
| `rviz` | `view:=map\|robot\|robot3p`, `robot:=<int>\|all` | `robot:=all` fans out one window per robot. |
| `rerun` | `web_port:=<int>`, `grpc_port:=<int>` | Defaults 9090/9876; auto-bumped per env under `--all` to avoid collisions. Requires `pip install 'rerun-sdk>=0.21'` once. |

Rerun opens a browser-based viewer (no X server, container-friendly); for
the default `web_port:=9090` go to `http://localhost:9090/`.
and exit non-zero with a hint to use `--all` or `<env_id>`.

---

## arena robot

Add, remove, and list robots in a running fleet at runtime, on top of the
launch-time `robot:=` seed.

```bash
arena robot jackal                       # spawn one jackal (unique auto name)
arena robot jackal mobile:=manual        # per-instance adapter selection
arena robot jackal count:=2              # two jackals
arena robot rm jackal_0                  # despawn by name
arena robot ls                           # list the current fleet
```

`<key>:=<value>` tokens describe the robot and are forwarded to the spawn
service: `name` sets the instance name, `count` spawns that many, and every
other key reaches `Robot.parse` (e.g. `mobile:=`, `arm:=`). `--flags` control
the command. Target an env with `--env <id>` or `--ns <ns>` (required when more
than one env is running); `arena robot ls --all` lists every env.

Spawns and despawns apply on the next episode reset. By default the command
waits until the robot appears in / disappears from `state/robots` (10s warning
cadence); `--nowait` returns as soon as the request is accepted (printing the
assigned name).

---

## Common options

```bash
log_level:=debug     # verbose output from all nodes
use_sim_time:=false  # real-time clock (unusual, only for real robots)
complexity:=2        # AMCL (position unknown); 3 = SLAM
record.dir:=/tmp/arena_run  # enable data recording (record.auto:=false keeps the recorder off)
task.fail_on_collision:=true  # abort the episode as FAILED when the robot footprint contacts a wall, static obstacle, or pedestrian (default false)
```

### sim:=

Default is `gazebo`. Valid values:

| Value | Meaning |
|---|---|
| `gazebo` (default) | gz-sim 8, dart physics, ogre renderer. `human` defaults to `arena` (arena_humansim). |
| `isaac` | Isaac Sim via `arena feature isaac launch`. `mobile` defaults to `nav2`. |
| `dummy` | No physics engine; a static `map->dummy` TF is published. For plumbing-only checks (no GPU, no controllers). Must be passed explicitly. |

### debug:= and optim:=

Two open, dev-oriented flag namespaces. `debug:=a,b` is shorthand for
`debug.a:=1 debug.b:=1` (same for `optim`); the dotted form is canonical and
wins on conflict, so `optim:=obstacles optim.obstacles:=full` leaves obstacles
at full fidelity. Unknown flags are silently ignored - these are throwaway
testing knobs, not a supported experiment surface.

| Flag | Effect |
|---|---|
| `debug:=aiomonitor` | Open an aiomonitor console on the env's asyncio loop (port `20101 + env_id*10`). |
| `debug:=map_server` | Force-launch the map server even when no adapter requested it. |
| `debug.map_source:=disk` | Treat the world's on-disk `<level>/map.png` as the single occupancy truth (nav2, spawn sampling, pedestrian collision walls, and robot collision events) instead of rasterizing `world.yaml`. Single-level worlds with an on-disk map only; errors otherwise. Default `compute` rasterizes from `world.yaml`. |
| `optim.obstacles:=bbox` | Spawn static obstacles as bounding-box primitives (read from each asset's `annotation.yaml`) instead of full meshes; assets without a `bounding_box` annotation fall back to the mesh. |
| `optim.obstacles:=none` | Silently skip all static obstacle spawns (world + episode). Pedestrians, walls, and floors are kept. |

`optim.obstacles` is a graded knob: `full` (default, `0`), `bbox` (`1`), `none`
(`2`); pass the alias or the number. Bare `optim:=obstacles` is shorthand for
`bbox`. The legacy `optim:=no_obstacles` still maps to `none`.

Combine freely: `debug:=aiomonitor,map_server`, `optim.obstacles:=bbox`, or the
dotted equivalents. `optim:=no_camera` / `optim:=no_lidar` also exist (they strip
those sensors from the robot URDF on the simulator side) but are best-effort and
unsupported.

## Cap-scoped overrides

Per-robot caps (`mobile`, `arm`, `lift`) are configured at launch via three
shapes of argument:

| Shape | Example | Lands as | Purpose |
|---|---|---|---|
| `robot.<cap>:=<kind>` | `robot.mobile:=rosnav_rl`, `robot.mobile:=drl` | `robot.<cap>_adapter` | Pick which `Bringup` runs for the cap. |
| `robot.<cap>.<key>:=<val>` | `robot.mobile.local_planner:=teb`, `robot.mobile.planner:=drlvo` | `robot.<cap>.<key>` | Override a value from `caps/<cap>.yaml`. |
| `<adapter-kwarg>:=<val>` | `global_planner:=smac`, `global_planner:=nav2/navfn` | `robot.<cap>.<key>` (via the adapter's launch file) | Adapter-internal launch kwargs (nav2 planner names, or the `<family>/<kind>` form consumed by the `drl` adapter). |

One robot-level timeout also takes the `robot.` prefix: `robot.ready_timeout`
(adapter readiness, default unbounded, `-1` means unbounded).

The cap-scoped form is the recommended style because it's self-documenting and
maps unambiguously to a single cap. To discover what `<key>`s a cap accepts,
read the robot's `arena_robots/.../robots/<name>/caps/<cap>.yaml`: every
top-level key there is overridable by the same name with the cap prefix.

Contestant args in benchmark YAMLs use the same shapes and the same forwarding
rules (see
[benchmark README](../arena_evaluation/arena_evaluation/configs/benchmark/README.md#contestant-args)).

### robot:=auto

The `robot` argument defaults to `auto`. Instead of selecting a robot
explicitly, `auto` resolves to a robot at launch time based on the active
planner's declared `action_type` and `sensor_needs`:

| Condition | Resolved robot |
|---|---|
| `action_type: omnidirectional` | `ridgeback_plus` |
| `action_type: differential_drive` | `jackal` |
| `sensor_needs` includes `image` or `depth` | `turtlebot` (overrides kinematics match) |
| `mobile.kind` is not `drl`, or no planner set | `jackal` (fallback) |

The resolution result is printed to the boot log:

```
arena: auto -> planner=rlrvo robot=ridgeback_plus [...]
```

`auto` is composable as a per-token value in multi-robot lists. Any `auto`
token is substituted independently; explicit tokens are left as-is:

```
robot:=auto              # single auto-resolved robot
robot:=auto[2]           # two auto-resolved robots
robot:=auto,jackal       # one auto-resolved + one jackal
robot:=jackal,auto,turtlebot[2]   # mixed fleet
```

When an explicit robot is named and its kinematic class or sensor set
disagrees with the chosen planner, a mismatch warning is emitted; the bridge
applies the canonical projection for that planner in that case.

## Deprecated launch args

Public launch args moved into dotted namespaces. The old names still work
(the launch layer prints a yellow warning and the new key wins if both are
given) and will be removed in a future release.

| Old | New |
|---|---|
| `isaac.physics` | `sim.isaac.physics` |
| `record_data_dir` | `record.dir` |
| `disable_auto_recorder` | `record.auto` (inverted: `disable_auto_recorder:=true` is `record.auto:=false`, default `true`) |
| `env_n` | `env.n` |
| `env_id` | `env.id` |
| `ns` | `env.ns` |
| `managed` | `env.managed` |
| `tm_robots` | `task.robots` |
| `tm_obstacles` | `task.obstacles` |
| `tm_modules` | `task.modules` |
| `task_config` | `task.config` |
| `scenario_file` | `task.scenario` |
| `parameter_file` | `task.params` |
| `episodes` | `task.episodes` |
| `auto_reset` | `task.auto_reset` |
| `fail_on_collision` | `task.fail_on_collision` |
| `mobile`, `mobile.<key>` | `robot.mobile`, `robot.mobile.<key>` |
| `arm`, `arm.<key>` | `robot.arm`, `robot.arm.<key>` |
| `planner` | `robot.planner` |
| `train_mode` | `robot.train` |

## CLI verbs

`source arena` (from `~/arena_ws`) loads a bash function that wraps the
common entry points. Verbs relevant to bringup:

| Verb | Wraps | Purpose |
|---|---|---|
| `arena launch [args]` | bash composite | All-in-one: `arena runtime` + N × `arena env` + optional `arena viz --all`. |
| `arena runtime [args]` | `arena_runtime.launch.py` | Runtime-only launch (sim + `arena_node`, no envs). |
| `arena env [args]` | `task_generator.launch.py` | Attach one task-generator env to a running runtime. |
| `arena viz [target]` | `ros2 run rviz_utils rviz_config` | Attach rviz to a running env; see [arena viz](#arena-viz). |
| `arena human [target]` | `rqt --standalone human_steering` | Attach the pedestrian-steering panel to a running env. Target/`--ns` resolution matches [arena viz](#arena-viz). Interactive control works on any backend serving `human/*`: driving a ped possesses it, the engine reclaims it about a second after you stop, `human:=dummy` is the engine-less puppet stage. `arena launch` auto-attaches one panel per env whenever the resolved human backend is `dummy`, unless `headless:=true`. This verb re-attaches after closing it. |
| `arena robot <model>\|rm\|ls` | `runtime/spawn_robot`, `runtime/despawn_robot`, `state/robots` | Spawn, despawn, or list robots in a running fleet; see [arena robot](#arena-robot). |
| `arena cleanup <env_id>` | `/arena/cleanup_namespace` service | Force-clean an env's namespace by id (calls the service for both the `env_<id>_` and `env_<id>/` prefixes, covering gazebo and isaac layouts). |
| `arena train [args]` | `arena_training` feature launcher | RL training entry, see section 7 above. |

None of these verbs killall anything. `arena launch` checks for an existing
runtime via `/arena/register_env`: if present, it attaches additively
(spawning `env.n` more envs against the existing runtime) and errors out
only if `sim:=` on the command line mismatches the running runtime's `sim`
parameter. `arena runtime` will fail if another `/arena` node is already
registered (ROS doesn't allow duplicate node names); kill the prior one
manually or call `arena cleanup` on its envs first.

### Shell completion

`source arena` registers TAB completion for `arena` in bash and zsh (zsh
registration waits for `compinit` if it has not run yet). Verb names, subverbs,
flags, supervisor knobs, and package names complete without ROS. Launch-arg
names and values (`sim:=`, `world:=`, `robot:=`, `task.*:=`, `human:=`) come from
a manifest cached under `${XDG_CACHE_HOME:-~/.cache}/arena/`, regenerated in the
background on the first TAB after a rebuild or on demand with `arena complete --refresh`.
`robot:=` and `robot.planner:=` list only what `arena feature robots|planners ls`
marks installed. Dotted keys fold to their group at each level (`arena launch
<TAB>` shows `task`, `task<TAB>` shows `task.robots` and siblings), a leaf key
completes straight to `key:=`, and `:` after a group name picks the group's own key.

## Benchmark mode

Benchmark runs are driven by the `arena evaluation benchmark` CLI verb. Requires
`arena feature evaluation install` first.

```bash
arena evaluation benchmark sim:=gazebo headless:=true suite:=basic contest:=basic
```

Suite and contest config, runner semantics, and output layout are documented in
[arena_evaluation/configs/benchmark/](../arena_evaluation/arena_evaluation/configs/benchmark/README.md).
