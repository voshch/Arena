# Cam: viewport-camera control

`arena_cam` is a standalone ROS 2 client that drives the Arena viewport
cameras over the `/arena/viewport/*` contract: services `set_view`,
`set_reference_frame`, `set_projection`, `capture`, stream topic `cmd_view`, and
feedback topic `camera_pose`. It lets you script camera moves (flybys,
orbits, tracking, dolly-zoom) against a LIVE sim from Python or a YAML
shot file, and fly the same cameras from the keyboard (see
[Driving interactively](#driving-interactively)). The wire namespace is
`/arena/viewport/*`; the Python/CLI surface is named `cam`.

## Mental model

A `Camera` is a chain of verbs queued into a timeline. `.play()` connects
to the live sim, runs every action in order, then disconnects.

Poses are expressed in the CURRENT reference frame. `add("track", ...)` attaches
the reference to a sim entity so any verb authored AFTER it runs in that
entity's moving frame: an `orbit` authored after `add("track", entity="env_0/jackal")`
orbits the robot even while it drives.

`look` (alias `cut`) is one-shot: it snaps the camera via the `set_view`
service and returns, releasing the camera to manual control. Streamed verbs
(`orbit`, `move_to`, `flyby`, `dolly`, `dolly_zoom`, `hold`, `pan`, `tilt`,
`zoom`) publish on `cmd_view` at 60 Hz and hold the camera for their full
duration.

The reference modes control how much of a tracked entity's rotation the
camera offset inherits:

| mode | what is inherited |
| --- | --- |
| `full` | full 6-DOF pose |
| `yaw` | position + yaw only (no pitch/roll) |
| `position` | position only |

`add("latch")` freezes the reference at its current world pose, stopping
tracking without a visible jump. `add("world")` resets the reference to
the origin.

The `camera_pose` feedback topic seeds the starting position of the cursor,
so `move_to`, `dolly`, and similar verbs glide smoothly from wherever the
camera currently sits.

Streamed verbs stamp each keyframe `now + 0.3s` (`LEAD`) and publish at
best-effort depth 64. The plugin buffers keyframes in a deque and each render
frame interpolates the local pose at clock-now between the two bracketing
entries (lerp position, slerp orientation, clamped at the ends). Publish rate
and transport jitter do not drive smoothness: gaps up to ~300 ms are
invisible because keyframes lead the clock. Trade-off: the live view trails
real time by ~LEAD (0.3 s), acceptable for scripted playback but not for
zero-latency interactive flying.

---

## Python API

There is one generic entry point on `Camera`:

```python
cam.add(name, **params)   # chainable, returns self
```

`name` may be a verb (atomic) or a shot (composite). Verbs are resolved
first; if the name is not a verb, it is looked up in the shots registry.
An unknown name raises `ValueError` listing both vocabularies. The call
shape is identical regardless of whether the name is a verb or a shot.

```python
from arena_cam import Camera

cam = Camera()
cam.add("projection", "perspective")
cam.add("look", eye=(8, 8, 6), target=(0, 0, 0.5))
cam.add("track", entity="env_0/jackal", mode="yaw")
cam.add("orbit", radius=4, elevation_deg=30, sweep_deg=360, duration=8, ease="inout")
cam.add("hold", duration=2)
cam.play()
```

```python
# shots expand transparently at the same call site
cam = Camera()
cam.add("establishing", radius=8)
cam.play()
```

All angle arguments are **radians**. fov is radians (0 = keep current).
`sweep_deg` / `start_deg` are accepted as degree conveniences on verbs that
support them.

### Verb reference

**Discrete verbs** snap the camera immediately via a service call and
return. **Streamed verbs** publish on `cmd_view` for their full duration.

#### Discrete

**`look` / `cut`** -- `eye`, `target`, `fov=0.0`
One-shot framing in the current reference frame. Calls `set_view` and
releases the camera to manual control. `fov` in radians, 0 leaves it
unchanged.

**`track`** -- `entity`, `mode="full"`
Attach the reference frame to a sim entity identified by its `sim_path`
(e.g. `"env_0/jackal"`). `mode` is `"full"`, `"yaw"`, or `"position"`.
All subsequent verbs run in this entity's frame.

**`world`** -- no params
Reset the reference frame to the world origin.

**`latch`** -- no params
Freeze the reference at its current world pose. Stops tracking without
a visible camera jump.

**`reference`** -- `pose`, `mode="full"`
Pin the reference frame to a constant world pose. `pose` is
`((x, y, z), (w, x, y, z))`.

**`projection`** -- string `"perspective"` or `"orthographic"`, OR dict `{projection: ...}`
Set the Gazebo camera projection mode via `set_projection`.

#### Streamed

**`hold`** -- `duration`
Keep the current framing for `duration` seconds.

**`move_to`** -- `eye`, `target`, `duration`, `ease="inout"`, `fov=None`, `world_orientation=False`
Glide from the current pose to look from `eye` toward `target` over
`duration` seconds. `fov` in radians, `None` keeps the current value.

**`orbit`** -- `radius`, `elevation_deg` or `elevation=0.0`, `center=(0,0,0)`, `sweep_deg` or `sweep` (default 2π), `start_deg` or `start_angle=None`, `duration=8.0`, `ease="inout"`, `look_height=0.0`, `fov=None`, `world_orientation=False`
Sweep azimuth on a sphere of `radius` around `center`, always looking at it, so
the subject keeps a constant distance. `elevation` is the angle above the
horizontal (0 = level ring, +90° straight down); the aim point is
`(center_x, center_y, center_z + look_height)`. When `start_angle` is `None`
the orbit starts at the current camera angle around `center`. Negative sweep
goes clockwise.

**`dolly`** -- `distance`, `duration=2.0`, `ease="inout"`
Move along the view axis. Positive `distance` pushes toward the target,
negative pulls back.

**`dolly_zoom`** -- `target`, `from_fov`, `to_fov`, `duration=3.0`, `ease="inout"`
Vertigo (Hitchcock) effect: change fov while dollying so `target` stays
the same apparent size in frame. Both fov values are radians.

**`flyby`** -- `eyes`, `target`, `fov=None`, `duration=8.0`, `ease="inout"`, `world_orientation=False`
Catmull-Rom spline through `eyes` (a list of `(x,y,z)` waypoints, prepended
with the current cursor position), looking at `target` throughout.

**`pan`** -- `sweep_deg` or `sweep` (default π/2), `duration=4.0`, `ease="inout"`
Rotate the view in place horizontally: eye fixed, yaw sweeps, pitch held.

**`tilt`** -- `sweep_deg` or `sweep` (default π/6), `duration=4.0`, `ease="inout"`
Rotate the view in place vertically: eye fixed, pitch sweeps, yaw held.

**`zoom`** -- `from_fov`, `to_fov`, `duration=2.0`, `ease="inout"`
Change fov only, eye and aim fixed. `from_fov` is explicit because fov
cannot be read back from the sim.

`ease` choices: `linear`, `in`, `out`, `inout` (default), `sine_inout`.

**`play()`**
Connect to the live sim, run the timeline, disconnect. Blocks until done.

---

## Shots

A shot is a named, parametrized, reusable composite that expands to a flat
list of atomic actions. From the call site it looks identical to a verb:
`cam.add("establishing", radius=8)`. Internally, resolution checks verbs
first, then shots; the caller never needs to know which it is.

### Shot schema

```yaml
params:                          # defaults; override at the call/include site
  radius: 6.0
  duration: 10.0

projection: perspective          # optional; desugars to a leading `projection` step
reference:                       # optional; desugars to a leading track/latch/reference step
  entity: env_0/jackal
  mode: yaw

timeline:                        # list of single-key steps {name: params}
  - look:  { eye: ["${radius}", "${radius}", 4], target: [0, 0, 0.5] }
  - orbit: { radius: "${radius}", elevation_deg: 30, sweep_deg: 360, duration: "${duration}" }
  - hold:  { duration: 2 }
```

`projection:` and `reference:` are header shortcuts: they desugar into
leading timeline steps, so the effective schema is just `params:` plus a
flat `timeline:`. A step's `name` key may be a verb OR another shot, so a
"meta-shot" can include other shots in its timeline.

### `${param}` substitution

Inside a shot's timeline, a value that is exactly the string `"${key}"` is
replaced by the parameter's value, preserving type (a list stays a list, a
number stays a number). A string with `${key}` embedded in surrounding text
is interpolated as text. Params come from the shot's `params:` defaults,
merged with (overridden by) params passed at the call or include site. A
missing key is a `ValueError`.

### Recursion and cycles

Resolution is eager: a shot is flattened to atomic actions at author time
(package import for YAML shots, `Camera.add` for inline). A shot may
include another shot in its timeline; that nested shot is itself expanded
recursively. Cycles are detected at load time and raise `ValueError`
("shot cycle: a -> b -> a").

### Bundled shots and auto-registration

Data shots live in [`configs/cam/shots/<name>.yaml`](../configs/cam/shots/)
and are auto-registered by file stem at package import. The stem may not
shadow an existing verb name (import-time `ValueError`). Two shots ship:

- `establishing` -- a single-take orbit with a look, orbit, and hold.
- `tour` -- a meta-shot: includes `establishing`, sets a reference, then
  runs `dolly` and `pan` verbs.

---

## Extending: the verb registry and shots

A new verb is one `@primitive("name")`-decorated class in `camera.py` that
owns its name, its `from_params` parsing, and its behaviour. Adding a verb
means writing one decorated class; it becomes usable from `Camera.add`, shot
YAML, and the CLI with no other edits. The registry lives in `registry.py`
(`PRIMITIVES`).

A new shot is a YAML file dropped into `configs/cam/shots/<name>.yaml`. It
is auto-registered at import with no other edits required. The stem must not
shadow a verb name.

Skeleton for a new streamed verb:

```python
from .registry import primitive
from .camera import _Segment, _Cursor

@primitive("my_verb")
class _MyVerb(_Segment):
    def __init__(self, duration: float, ease: str) -> None:
        super().__init__(duration, ease, False)

    @classmethod
    def from_params(cls, params: dict) -> "_MyVerb":
        return cls(params.get("duration", 2.0), params.get("ease", "inout"))

    def plan(self, cursor: _Cursor):
        # return (sampler, end_cursor)
        ...
```

For a discrete verb, subclass `_Action` and implement `run(node, cursor)` instead
of `plan`.

---

## Shot YAML

A shot file describes a `Camera` timeline declaratively. Load and run it
with `load_shot`:

```python
from arena_cam import load_shot
load_shot("my_shot.yaml").play()
```

`load_shot` treats the argument as a file path. The registry name form is
for `Camera.add` and the CLI.

### Schema

```yaml
params:                          # optional; shot parameter defaults
  radius: 5.0

projection: perspective          # optional, perspective|orthographic

reference:                       # optional; one of:
  entity: env_0/jackal           #   track a sim entity
  mode: yaw                      #   full|yaw|position (default full)
# --- OR ---
reference:
  latch: true                    #   freeze at current world pose
# --- OR ---
reference:
  pose:                          #   pin to a constant world pose
    - [x, y, z]                  #   position
    - [w, x, y, z]               #   quaternion
  mode: full

timeline:                        # list of single-key maps {name: params}
  - look:   { eye: [8, 8, 6], target: [0, 0, 0.5], fov: 1.0 }
  - orbit:  { radius: 4, elevation_deg: 30, sweep_deg: 360, duration: 8, ease: inout }
  - hold:   { duration: 2 }
```

Each timeline step is a single-key map `{name: params}` where `name` is a
verb or a shot name. Steps are routed through `Camera.add`, so every
registered verb and shot works in YAML automatically. All angles are
**radians** except the degree conveniences (`sweep_deg`, `start_deg`,
`elevation_deg`) on `orbit`, `pan`, and `tilt`. fov is always radians.

### Timeline verb parameters

| verb | required | optional |
| --- | --- | --- |
| `look` / `cut` | `eye`, `target` | `fov` (rad, default 0) |
| `track` | `entity` | `mode` |
| `world` | | |
| `latch` | | |
| `reference` | `pose` | `mode` |
| `projection` | string value | |
| `hold` | `duration` | |
| `move_to` | `eye`, `target`, `duration` | `ease`, `fov`, `world_orientation` |
| `orbit` | `radius` | `elevation` (rad) or `elevation_deg`, `center`, `sweep` (rad) or `sweep_deg`, `start_angle` (rad) or `start_deg`, `duration`, `ease`, `look_height`, `fov`, `world_orientation` |
| `dolly` | `distance` | `duration`, `ease` |
| `dolly_zoom` | `target`, `from_fov`, `to_fov` | `duration`, `ease` |
| `flyby` | `eyes` (list of [x,y,z]), `target` | `fov`, `duration`, `ease`, `world_orientation` |
| `pan` | | `sweep` (rad) or `sweep_deg`, `duration`, `ease` |
| `tilt` | | `sweep` (rad) or `sweep_deg`, `duration`, `ease` |
| `zoom` | `from_fov`, `to_fov` | `duration`, `ease` |

### Full example

```yaml
projection: perspective

reference:
  entity: env_0/jackal
  mode: yaw

timeline:
  - look:   { eye: [6, 0, 3], target: [0, 0, 0.5] }
  - orbit:  { radius: 5, elevation_deg: 35, sweep_deg: 360, duration: 10, ease: sine_inout }
  - pan:    { sweep_deg: 45, duration: 2 }
  - latch:  {}
  - move_to: { eye: [12, 12, 8], target: [0, 0, 0], duration: 4 }
  - hold:   { duration: 2 }
```

---

## CLI

`arena cam` wraps the console script `cam = arena_cam.cli:main`.
The interface is fully generic: one positional name dispatches verbs, shots,
and files uniformly.

### `arena cam <name> [key=value ...] [--sim] [--viz [ENV_ID]]`

Run a verb, a shot, or a YAML file against the live sim.

- `<name>` is a verb name, a shot name, or a path to a `.yaml`/`.yml`
  file. A `.yaml`/`.yml` suffix, a `/` in the name, or an existing file
  on disk routes the argument through `load_shot`; otherwise it is a
  registry lookup (verbs first, then shots).
- `key=value` tokens are params passed to the verb or shot.
- Target flags select which viewport cameras the shot drives:
  - no flag drives everything (the sim GUI camera and every env's rviz camera);
  - `--sim` drives only the sim GUI camera (`/arena/viewport/*`);
  - `--viz` drives every rviz camera, `--viz <env_id>` just one;
  - the flags compose, e.g. `--sim --viz 0`.

  The sim camera is in world coordinates. Each rviz camera is per-env, so a shot
  in absolute coordinates is localized into each env's frame (the registry
  `reference` offset) before being sent. Once a shot sets a reference frame,
  poses are relative to it and go out as authored, and only the reference pose
  itself is localized. Record mode needs the selection to resolve to a single
  camera.

```
arena cam look eye=8,8,6 target=0,0,0.5
arena cam orbit radius=4 elevation_deg=30 sweep_deg=360 duration=8
arena cam establishing radius=8
arena cam tour duration=12 --sim
arena cam shots/my_shot.yaml radius=5 --viz 0
```

**Param coercion** for `key=value` tokens:

- `true` / `false` become `bool`.
- A token containing commas becomes a tuple of coerced parts
  (e.g. `eye=8,8,6` becomes `(8.0, 8.0, 6.0)`).
- Otherwise: try `float`, else keep as `str`.

Limitation: `key=value` covers scalars and x,y,z tuples. List-of-points
params (e.g. `flyby` `eyes`) and deep nesting are file-only. Use
`arena cam show <name>` to check whether a name needs a file.

### `arena cam list`

Print the full catalog: verbs and shots.

```
arena cam list
```

### `arena cam show <name>`

Print a verb's constructor params, or a shot's `params:` and `timeline:`.

```
arena cam show orbit
arena cam show establishing
```

---

## Driving interactively

`arena cam drive` opens an rqt panel that flies the same cameras from the
keyboard, over the same selection a shot uses (no flag drives everything,
`--sim` and `--viz [ENV_ID]` narrow it).

```
arena cam drive               # every viewport camera
arena cam drive --sim         # the sim GUI camera only
arena cam drive --viz 0       # env 0's rviz camera
```

The panel drives the camera while its window is active and releases it as soon as
the window is not, so there is nothing to take or hand back. Keys set a target
velocity each axis damps toward, so a held key accelerates and a released one
coasts down. `Space` stops the motion without releasing. While the panel is idle
it mirrors the live camera pose, so the readout stays true and `P` captures where
the camera actually is.

| key | fly | orbit (`Tab` toggles) |
| --- | --- | --- |
| `W` / `S` | dolly along the view axis | elevation over the focus |
| `A` / `D` | truck left / right | azimuth about the focus |
| `Q` / `E` | down / up along world Z | radius (true subject distance) |
| arrows | look (yaw / pitch) | pan the focus |
| `Shift` / `Ctrl` | boost / crawl | boost / crawl |
| `[` / `]` | fov | fov |

Anywhere: `F` frames the target entity, `Shift+F` cycles its reference mode
(`full` / `yaw` / `position`), `H` returns to the world frame, `1` / `3` / `7`
snap to front / right / top, `Space` brakes, `P` captures a still to
`$ARENA_DATA_DIR/recordings/stills/`. Keys are ignored while a text field has
focus, so the target box stays typeable.

`[` / `]` are a held axis like the movement keys, not a step: fov ramps at
0.5 rad/s while held, so a tap moves it by a degree or two.

Framing sets the reference frame to the entity and orbits its local origin, so the
camera follows it as it drives. The target picker lists live robot frames
(`env_0/jackal`); any other `sim_path` can be typed in.

Two behaviours differ from scripted playback:

- **Lead.** Keyframes lead by 30 ms instead of 0.3 s, trading jitter tolerance for
  responsiveness. Raise the panel's lead slider if a loaded GUI stutters. What is
  left on top of that is the GUI's own frame time: rviz only moves the camera on a
  render frame, so a heavy scene adds its frame interval to the lead.
- **Release.** Streaming is continuous while engaged, including at rest, because the
  plugin takes the camera back after ~400 ms of silence. Releasing is just stopping
  the stream; the camera stays where it was left.

The horizon never tilts and no roll is produced. fov stays untouched until a fov key
is pressed, so the panel never overrides the sim's own value by default.

---

## Recording

`Camera.record(out_dir, fps=30, lockstep=False)` is the offline sibling of
`play()`: it renders the timeline to a numbered PPM sequence (`frame_00000.ppm`,
...) by capturing each frame at an exact pose through the `capture` service, so
the result is smooth and deterministic regardless of render speed. A bare
`out_dir` name lands under `$ARENA_DATA_DIR/recordings/` and an absolute or
slash-bearing path is used as given. Recording into a non-empty directory errors
(a take is never silently clobbered); pass `-f` / `--force` to overwrite it.

```python
cam = Camera()
cam.add("orbit", radius=4, elevation_deg=30, sweep_deg=360, duration=8)
cam.record("orbit", fps=30)        # -> $ARENA_DATA_DIR/recordings/orbit/
cam.record("orbit", fps=30, lockstep=True)  # physics-lockstep take
```

On the CLI, `record` (output dir), `fps`, and `lockstep` are reserved params,
popped before the rest reach the verb, shot, or file:

```
arena cam tour record=tour fps=30
arena cam orbit radius=4 duration=8 record=orbit
arena cam orbit radius=4 duration=8 record=orbit lockstep=true
```

A streamed segment emits `round(duration * fps)` frames, a discrete `look` /
`cut` one frame, and the state-only verbs (`track`, `world`, `latch`,
`reference`, `projection`) none. Frames are raw `P6` PPM; assemble with e.g.
`ffmpeg -framerate <fps> -i frame_%05d.ppm -pix_fmt yuv420p out.mp4`.

Recording is camera-locked by default: the camera path is deterministic but the
scene advances at the sim's own rate, so pause the sim first for a fully
reproducible take. Pass `lockstep=True` (CLI: `lockstep=true`) for physics-lockstep
recording instead: with a lockstep run active the cam rides it as a hard channel
gated at `1/fps`, and without one it takes its own sim hold and steps physics by
`1/fps` between frames. Either way a frame is captured only once the scene has
reached that sim time. Gazebo only.

---

## Requirements

Two backends host the `/arena/viewport/*` contract (`set_view`,
`set_reference_frame`, `set_projection`, `capture`, plus the `cmd_view` /
`camera_pose` topics):

- **Gazebo**, under `/arena`, via the `ViewportCamera` GUI system plugin. It is a
  GUI plugin, so nothing exists in headless mode (`gz sim -s`).
- **rviz**, under each env's namespace, via the `rviz_viewport_control` view
  controller that [rviz_config.py](../../rviz_utils/rviz_utils/scripts/rviz_config.py)
  writes into the generated config.

Isaac hosts no part of the contract, so nothing here works under `sim:=isaac`.

`capture` is Gazebo-only; the rviz side answers `capture not yet supported`, so
recording and `P` need the sim camera. `Cam` waits up to 10 seconds for the
services and logs an error if they are absent.
