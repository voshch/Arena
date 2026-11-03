# human_steering

rqt panel that drives any backend serving `human/*` (the dummy backend is
the reference implementation). Launch with `arena human` (or `rqt
--standalone human_steering --args --ns <ns>`), see
[BRINGUP.md](../../arena_bringup/BRINGUP.md).

## Architecture

The panel is the motion engine, not a viewer. It composes each pedestrian's
intent (waypoints, teleop, clips, pose sliders, gaze) into full `Pedestrian`
state and streams it out. The backend only validates, caches, fills
gait for peds the panel is not driving, and relays the result to the
physics sim.

```
canvas / sliders / teleop / clips -> Driver (compose slider > gaze > clip > gait)
                                   -> human/stream (all held peds)
backend: validate, merge into cache
                                   -> publish full roster (fills gait for idle peds)
                                   -> relay to the physics/engine sim
```

The driver streams every ped the panel holds (`Intent.held`). The first
authoring action on a ped (teleop, a route, a teleport, a posed joint, a
clip, gaze) claims it, and the claim survives the motion ending: a ped
whose teleop deadman expires or whose route completes keeps streaming its
standing pose, so an engine backend cannot snatch it back mid-session.
Every batch on `human/stream` is the panel's complete claim set. Only an
explicit Release (or an episode change, or the ped despawning) drops a
ped from the batch, and the backend releases it the moment it is omitted,
since a batch's absence is as meaningful as its presence. Closing the
panel stops the stream and the backend's 1 s timeout releases everything.

## Panel layout

Toolbar on top, then a horizontal splitter. A roster rail on the left
(~170 px). The map canvas in the center, all the stretch, never
collapsible. A controls rail on the right (~460 px): Drive, Clips, Pose,
and FK preview in a scroll area.

Roster rows show a colored state pill (IDLE, WALKING, RUNNING from the
roster mirror, or TELEOP while the panel drives that ped), current speed,
and route progress (`Driver.roster_status`). A held ped (in
`Driver.held_names()`, the panel's local claim set, no backend
round-trip) gets a filled-circle marker in the driver's held color, matching
the canvas's held ring.

The canvas auto-fits to content the first time it arrives. A toolbar Fit
action and the F key do the same on demand. The wheel zooms, anchored under
the cursor. The Select tool pans by drag. An origin crosshair and a corner
hint ("map: ready · static: N walls · peds: N") keep the canvas legible
before any data arrives. Ped and waypoint markers stay a constant screen
size at any zoom. A held ped draws a distinct ring color, and the selected
ring wins when a ped is both selected and held.

Pose engagement is implicit. Move a joint's slider or edit its value box
and it engages: the row turns bold and accented, and a small "x" appears.
Click "x" to release the joint back to the bus. "Clear pose" releases every
engaged joint for the selected ped. Group headers (Torso / Head, arms,
legs) only collapse and expand, they hold no state of their own. An empty
clip library disables Play and Stop. Sliders span `gait.LIMITS` by default,
`--unlimited` widens them to `0..2pi` and skips the driver's compose-time
clamp.

STOP and Release are both toolbar buttons (and canvas context-menu entries)
acting on the selected ped, and they differ in scope. STOP kills motion
only: it clears waypoints and teleop and nothing else, posed joints, an
active clip, and gaze all stay engaged, and the ped stays claimed,
standing where it stopped.
Release hands the ped back entirely: it clears the whole intent, mode,
route, teleop, posed joints, clip, gaze, so the ped drops out of
`human/stream` on the next tick and the backend takes it back over.
Releasing the selected ped also clears the selection, and the canvas only
keeps waypoint previews and gaze markers for held peds, so a released or
despawned ped takes its route overlay with it.

Selection is shared between the roster list and the canvas: picking a
roster row selects on the canvas, and a Select-tool click, including one on
empty space that clears it, is reported back to the roster. Every
manipulation tool (Walk to, + Waypoint, Teleport, Gaze) acts on that one
selection, a click on empty space still targets the selected ped. Select
clicks to pick a ped and drags to pan the view. Walk to sends the selected
ped to one point. + Waypoint appends a stop to that ped's route, or starts
a new looping route if it has none. The route and its progress survive
switching tools. Teleport drags the ped directly. Teleop is a toolbar tool:
an application-wide event filter catches every arrow key press and release,
including auto-repeats, before any slider, list, or the canvas itself does,
so held keys chord together (e.g. forward + turn) and never pan the map,
and drives the selected ped at the Drive speed no matter which widget has
focus. Holding Shift runs: double the Drive speed, floored at the run
threshold so the animation state always flips to RUNNING. Gaze parks a look
target the head tracks continuously.

## Modules

| Module | ROS/Qt | Role |
| --- | --- | --- |
| [`plugin.py`](human_steering/plugin.py) | rqt/Qt | `rqt_gui_py.plugin.Plugin` shell: `--ns` / `--unlimited` args, QTimer(50 ms) tick, wires panel + driver |
| [`panel.py`](human_steering/panel.py) | Qt | toolbar, roster, Drive/Clips/Pose groups, FK preview, status bar |
| [`canvas.py`](human_steering/canvas.py) | Qt+ROS | `QGraphicsView`: map/marker underlay, ped rendering, direct-manipulation tools |
| [`driver.py`](human_steering/driver.py) | ROS (Qt-free) | the motion engine: roster mirror, per-ped intent, integration, composition, publishing |
| [`compose.py`](human_steering/compose.py) | none | per-joint precedence (slider > gaze > clip > gait) and gaze angle solving |
| [`integrate.py`](human_steering/integrate.py) | none | waypoint follower and teleop dead-reckoning, dt-aware (freezes at dt=0) |
| [`clips.py`](human_steering/clips.py) | none | wire-clip loading, cosine sampling, blending |
| [`fk.py`](human_steering/fk.py) | none* | forward kinematics of the RViz body URDF and front/side projection for the pose preview |

\* `fk.py` builds the URDF with `rviz_utils.hri.urdf` at call time. It has no
top-level ROS/Qt import, so it stays importable without a sourced install,
only the preview call fails.

`driver.py` is Qt-free and importable on its own. Every ROS message type it
touches is guarded the same way
[`gait.py`](../../arena_runtime/arena_runtime/arena_runtime/humans/gait.py)
guards `sensor_msgs`. Only constructing a `Driver` against a real node
needs a sourced install.

### Headless scripting example

```python
import rclpy
from rclpy.parameter import Parameter
from human_steering.driver import Driver, resolve_namespace

rclpy.init()
node = rclpy.create_node(
    "human_steering_script",
    # stream stamps must come from the sim clock
    parameter_overrides=[Parameter("use_sim_time", value=True)],
)
rclpy.spin_once(node, timeout_sec=1.0)  # let the graph populate

namespaces = resolve_namespace(node, target="env_0")
driver = Driver(node, namespaces)

driver.set_waypoints("agent_3", [(2.0, 1.0), (2.0, 4.0)], loop=True, speed=1.0)

rate = node.create_rate(20.0)  # matches the GUI's own 20 Hz stream cadence
while rclpy.ok():
    rclpy.spin_once(node, timeout_sec=0.0)
    driver.tick()
    rate.sleep()
```

## Surface

| Name | Kind | Namespace | Type | QoS | Notes |
| --- | --- | --- | --- | --- | --- |
| `human/stream` | topic (driver publishes) | env (`<env_ns>/human/stream`) | `arena_people_msgs/Pedestrians` | reliable, volatile, depth 1 | all held peds, bare joint names, 20 Hz. Each batch is the publisher's complete claim set: a ped omitted from a valid batch is released instantly on the backend. Releasing the last held ped publishes exactly one empty batch (the "I claim nothing" manifest) so that release is instant too, instead of continuously, and 1 s of silence is the backend's own crash-fallback timeout. Headers must carry a sim-clock stamp, the backend drops batches stamped before its stream gate opened (the plugin flips its node to `use_sim_time` for exactly this) |
| `human/move` | service (driver calls) | env (`<env_ns>/human/move`) | `arena_people_msgs/srv/MovePedestrians` | default | teleport intent, only `name`+`pose` honored by the backend |
| `human/<token>/cmd_vel` | topic (driver subscribes) | env (`<env_ns>/human/<token>/cmd_vel`) | `geometry_msgs/Twist` | depth 10 | per-ped teleop input, `token` = ped name with runs of non-alphanumerics collapsed to `_` (see `clips.slug`). Same input path and 0.5 s deadman as the panel's own Teleop tool, and the first command claims the ped until `Driver.release` |
| `arena_peds` | topic (driver subscribes) | env (`<env_ns>/arena_peds`) | `arena_people_msgs/Pedestrians` | depth 10 | roster mirror: names, ids, poses, `model_uri`, `twist`/`animation_state` (roster-row speed and state pill) |
| `state/episode` | topic (driver subscribes) | node (`<node_ns>/state/episode`) | `arena_runtime_msgs/EpisodeRecord` | depth 20, transient-local | `episode_id` changes clear every per-ped intent |
| `runtime/require_map` | service (canvas calls) | node (`<node_ns>/runtime/require_map`) | `std_srvs/Trigger` | default | retried every ~3 s until it returns success (the env node's initialized flag can lag viz_manifest), gates the occupancy underlay, static markers render regardless |
| `pedestrian_markers/static*` | topic (canvas subscribes) | env (`<env_ns>/pedestrian_markers/static*`) | `visualization_msgs/MarkerArray` | depth 1, transient-local | wall/object baseline, always rendered |
| `map` | topic (canvas subscribes, after the Trigger) | node (`<node_ns>/map`) | `nav_msgs/OccupancyGrid` | depth 1, transient-local | occupancy underlay |
| `state/viz_manifest` | topic (discovery) | node (`<node_ns>/state/viz_manifest`) | any | n/a | manifest-suffix convention: one discovery pass finds `<node_ns>`, `env_ns = dirname(node_ns)`, `map topic = <node_ns>/map` (see `_meta/tools/viz`) |

Namespace resolution is one pass over `node.get_topic_names_and_types()`
for the `/state/viz_manifest` suffix, the same discovery convention
`arena viz` uses. `--ns` (or the CLI's positional target) matches the way
`arena viz <env_id>` matches env ids.

Backend detection is `human/move` service discovery. Absent (or not yet
resolved), the panel shows a banner ("this env exposes no human control
endpoints") and disables Drive/Clips/Pose/tool controls. The roster and
canvas stay live either way.

## Wire-clip schema

Clip sources: `dirname(model_uri)/clips/wire.json` (the model bundle's own
clips) and `$ARENA_DATA_DIR/peds/poses/*.json` (user-authored poses, merged
on top). A pose is a one-sample clip in the same schema.

```jsonc
{
  "version": 1,
  "rate_hz": 30,
  "clips": {
    "<name>": {
      "duration": 1.2,
      "cyclic": true,
      "tracks": {
        "<bare base joint>": [0.0, 0.05, 0.1, "..."]
      },
      "root_z": [0.0, 0.01, "..."]
    }
  }
}
```

- Track keys are the bare semantic joint names from `GaitGenerator.JOINT_NAMES`
  (`gait.py`), root-relative, same anatomical shoulder convention as the rest
  of the wire contract (see the backend's `JOINTS.md`). Values are
  pre-clamped to `gait.LIMITS` at authoring time, the driver does not
  re-clamp clip samples.
- `rate_hz` is the sample rate for every track in the file. `duration` is in
  seconds.
- `cyclic=true` loops the sampler (`t modulo duration`), `cyclic=false`
  clamps and holds the last frame past `duration`.
- `root_z` is optional: a per-frame hip-height offset, sampled the same way
  as any joint track.
- Sampling is cosine-interpolated between keyframes. Starting or stopping a
  clip blends against the previous pose over 0.3 s (`clips.BLEND_S`)
  instead of cutting instantly.

## Constants

| Constant | Value | Where |
| --- | --- | --- |
| stream rate / QoS | 20 Hz, depth 1, reliable/volatile | `driver.STREAM_HZ`, `driver._stream_qos` |
| keepalive | 2 Hz, wall-clock, publish-only | backend-side (the dummy backend), not this package |
| teleop deadman | 0.5 s | `integrate.DEADMAN_S` |
| run threshold | 1.8 m/s | `driver.RUN_THRESHOLD_MPS` |
| clip blend | 0.3 s | `clips.BLEND_S` |
| waypoint arrival | 0.2 m | `integrate.ARRIVAL_M` |
| wire clip rate | 30 Hz | `clips.WIRE_RATE_HZ` |
| gaze head height | 1.6 m | `compose.GAZE_HEAD_HEIGHT_M` |
| Qt tick | 50 ms | `plugin.TICK_MS` |
| require_map retry | ~3 s | `canvas.REQUIRE_MAP_RETRY_TICKS` |

## Tests

`tests/` is ROS-free (sampler, composer, integrator, FK
smoke on a synthetic joint tree). `tests/ros/` needs a sourced ROS install
(namespace discovery against a real node). `tests/conftest.py` mirrors
[`task_generator/tests/conftest.py`](../../task_generator/tests/conftest.py):
it skips anything under `tests/ros/` when `rclpy` is not importable.

```
python3 -m pytest utils/human_steering/tests -x -q
```
