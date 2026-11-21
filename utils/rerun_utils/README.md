# rerun_utils

Web-native Rerun viewer for the [arena_viz](../arena_viz) contract. Mirrors
[rviz_utils](../rviz_utils) but renders by streaming `rr.log(...)` calls
against [`rr.serve_web()`](https://rerun.io/docs/howto/visualization/sharing-recordings#serve_web)
instead of generating an rviz YAML config.

## Install

`rerun-sdk` is a pip dependency, not a ROS package:

```bash
pip install 'rerun-sdk>=0.38'  # named transform frames; tested with 0.38.1
# optional, for ROBOT_MODEL rendering:
pip install rerun-loader-urdf-python
```

## Launch

```bash
ros2 launch rerun_utils rerun_bridge.launch.py ns:=/env_0 \
    web_port:=9090 grpc_port:=9876
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `ns` | required | Env namespace, e.g. `/env_0`. |
| `web_port` | `9090` | Port of the rerun web viewer. |
| `grpc_port` | `9876` | Port of the rerun gRPC server. |
| `memory_limit` | `512MiB` | What the gRPC server buffers for viewers that connect later, in rerun size syntax. |
| `max_rate_hz` | `30.0` | Most updates a second logged per topic and per tf frame, `0` = uncapped. |

Then open `http://localhost:9090/` in any browser. No X server, no desktop
environment, works inside a container.

The bridge follows the same lifecycle as `rviz_utils`:

1. Wait for `arena_env_node`'s `initialized` param.
2. Read `prefix`, `env_id` and `tf_namespace`.
3. Wait for one latched `state/robots` (fleet) and one latched
   `state/viz_manifest` (the env + per-robot display list).
4. Start `rr.serve_web()`.
5. Start a `<tf_namespace>/tf` + `<tf_namespace>/tf_static` mirror that logs every transform as a named-frame
   `rr.Transform3D(child_frame=..., parent_frame=...)` under `env_<id>/tf/<child_frame>`, so rerun chains the
   frames as tf does. A frame that is no transform's child (the env's `map`) is tied to the env's origin.
6. Walk the manifest, dispatch each `AdapterDisplay` to its renderer.
7. Each renderer creates a ROS subscription on the bridge node; callbacks
   log through the bridge's `Stream`.
8. Send the layout: the 3D scene on `/`, and beside it a 2D view per IMAGE display and a plot of the last 10 s per IMU
   display. Panels start collapsed.

## Logging

`rerun_utils/stream.py` decides how everything reaches the viewer:

- The recording flushes its batches once per `max_rate_hz` period (33 ms by default and when uncapped).
- Streams are logged on the `sim_time` timeline from their header stamps (the node's sim clock for unstamped ones).
- One-time data is logged static: `tf_static`, the ties of tf roots, the map (costmaps stay temporal), latched marker
  arrays and the IMU series names. When the buffer is full the server drops the oldest temporal data first, so a
  viewer that connects late still gets these. Streams logged static would be dropped together with them.
- Each topic and each tf frame logs at most `max_rate_hz` times a second. A message that arrives too early waits,
  replaced by any newer one, and is logged once its slot comes, so the view always ends on the latest message. A
  message that waits is converted only when it is logged.
- A dynamic transform that moved less than 1e-5 from the last one logged into its frame is skipped.
- 8-bit color and mono images are sent as JPEG (quality 80), `16UC1` and `32FC1` depth as 16-bit PNG in millimeters,
  other images raw.
- Point clouds keep their finite points, every k-th of them when there are more than 20000.

## MCP tools

The package registers the `viz` MCP domain (`rerun_utils/mcp.py`, entry point `arena_mcp.domains`), which runs the same launch file for an env without a terminal:

- `viz_rerun_start(env_id=None, web_port=0, grpc_port=0, memory_limit="512MiB", max_rate_hz=30.0)` starts the bridge of the env's task generator detached in its own process group, logging to `$ROS_LOG_DIR/rerun_view_env_<id>_<time>.log`. A port of 0 picks a free one. It returns once the web port accepts connections: `{env_id, ns, pid, web_port, grpc_port, url, started}`. A live view is returned unchanged with `started: false`.
- `viz_rerun_stop(env_id=None)` sends SIGINT to the group, then SIGTERM and SIGKILL after the reaper grace windows: `{env_id, stopped}`.
- `viz_rerun_list()` returns `{views: [{env_id, ns, pid, web_port, grpc_port, url, alive}]}`.

`url` is `http://127.0.0.1:<web>/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A<grpc>%2Fproxy`, the form the bridge logs. Views are one JSON file per env under `/tmp/arena-viz/<ROS_DOMAIN_ID>/`, so the MCP server, the CLI and the dashboard share them, and a file whose process is gone is dropped when read. Both this domain and `arena viz backend:=rerun` build the launch command with `arena_bringup.viz_backends.rerun_launch_argv`.

## Renderer registry

One module per `DisplayKind`, registered via decorator:

```python
# rerun_utils/renderers/path.py
@register(DisplayKind.PATH)
def render_path(d: AdapterDisplay, robot: RobotDescriptor | None, ctx: RendererCtx) -> None:
    style = StyleSpec.from_json(d.style_json)
    entity = display_path(ctx.env_id, robot, d.name)
    color = style.color or (0, 200, 0)
    def cb(msg: Path) -> None:
        pts = [(p.pose.position.x, p.pose.position.y, p.pose.position.z) for p in msg.poses]
        rr.log(entity, rr.LineStrips3D([pts], colors=[color]))
    ctx.node.create_subscription(Path, d.topic, cb, 10)
```

The registry asserts at import time that every `DisplayKind` member has a
renderer.

### Current coverage

| Kind | Status |
|---|---|
| `MAP` | `rr.Mesh3D` quad at the grid's metric size, origin and frame, textured free / cost / occupied / unknown; a grid styled as a costmap is `rr.Points3D` of its cells with a cost, slightly above the map |
| `TF` | no-op (handled by `TFMirror` unconditionally) |
| `PATH` | `rr.LineStrips3D` in the path's frame |
| `POSE` | `rr.Arrows3D` (one arrow at pose+yaw) in the pose's frame |
| `ODOM` | `rr.Arrows3D` in the odometry's frame |
| `LASER_SCAN` | `rr.Points3D` in the scan's frame |
| `POINTS_3D` | `rr.Points3D` of the finite points, at most 20000, in the cloud's frame |
| `IMAGE` | `rr.EncodedImage` (JPEG) for 8-bit color and mono, `rr.EncodedDepthImage` (PNG) for depth, `rr.Image` otherwise |
| `IMU` | one `rr.Scalars` series each for `accel` and `gyro`, named `x`, `y`, `z` |
| `POLYGON` | `rr.LineStrips3D` (closed loop) in the polygon's frame |
| `PEDESTRIANS` | a reduced capsule view (one capsule and a heading arrow per person) of `<env ns>/arena_peds`, standing on the pose z; the display's ROS4HRI topic has no rerun skeleton plugin |
| `ROBOT_MODEL` | URDF via `rerun-loader-urdf-python` (optional, warn-and-skip if absent) |
| `FOOT_CONTACT` | warn-and-skip |
| `TRAJECTORY` | warn-and-skip (MoveIt-only) |
| `PLANNING_SCENE` | warn-and-skip (MoveIt-only) |
| `INTERACTIVE_MARKERS` | nothing logged (rviz-only drag handles) |
| `MARKER_ARRAY` | cubes, spheres, cylinders, arrows, lines, points, cube/sphere lists, triangles and text per namespace and frame; added markers kept until deleted |

Marker fidelity against rviz. Arrow shafts follow `scale` (two-point arrows: `scale.x`; pose arrows: `scale.y`); the
arrow head size is fixed by rerun. Not drawn: mesh resources (`MESH_RESOURCE`, `mesh_use_embedded_materials`), per-point
colors on line strips and lists, the `scale.z` height of text (rerun labels have one size), the second axis of a
cylinder's ellipse (the larger of `scale.x` and `scale.y` sets the radius), and `frame_locked` (data always rides its
frame's transform at the time it is logged).

### Entity-path convention

`rerun_utils/entity_paths.py`:

- env-level displays: `env_<id>/<display_name>`
- per-robot displays: `env_<id>/robots/<robot.name>/<display_name>`
- TF tree: one entity per frame, `env_<id>/tf/<child_frame>`, holding the transform into that frame
- frame names are global to a recording: the bridge names the recording `arena_env_<id>`, so run one bridge per env (the
  `env_<id>` entity root is only the tie-in point of the env's tf roots, not a frame namespace)
- data in a ROS frame logs `rr.CoordinateFrame(<frame_id>)` next to it (`entity_paths.in_frame`), so it rides the TF
  chain wherever its entity sits; marker arrays log under `<display>/<namespace>/<frame>/<kind>`.

## Style hints

The renderer reads styling from `StyleSpec.from_json(d.style_json)`:

- `color`, `alpha`, `line_width`, `enabled`, `decay`: honored when meaningful for the archetype.
- `latched`: MARKER_ARRAY subscribes reliable + transient-local.
- `extra={"rerun": {...}}` is the escape hatch (not used by any current renderer; reserved for future per-viz nudges).
