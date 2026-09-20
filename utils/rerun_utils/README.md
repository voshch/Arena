# rerun_utils

Web-native Rerun viewer for the [arena_viz](../arena_viz) contract. Mirrors
[rviz_utils](../rviz_utils) but renders by streaming `rr.log(...)` calls
against [`rr.serve_web()`](https://rerun.io/docs/howto/visualization/sharing-recordings#serve_web)
instead of generating an rviz YAML config.

## Install

`rerun-sdk` is a pip dependency, not a ROS package:

```bash
pip install 'rerun-sdk>=0.21'
# optional, for ROBOT_MODEL rendering:
pip install rerun-loader-urdf-python
```

## Launch

```bash
ros2 launch rerun_utils rerun_bridge.launch.py ns:=/env_0 \
    web_port:=9090 grpc_port:=9876
```

Then open `http://localhost:9090/` in any browser. No X server, no desktop
environment, works inside a container.

The bridge follows the same lifecycle as `rviz_utils`:

1. Wait for `task_generator_node`'s `initialized` param.
2. Read `prefix` + `env_id`.
3. Wait for one latched `state/robots` (fleet) and one latched
   `state/viz_manifest` (the env + per-robot display list).
4. Start `rr.serve_web()`.
5. Start a `/tf` + `/tf_static` mirror that logs every transform as
   `rr.Transform3D` under `env_<id>/tf/<child_frame>`.
6. Walk the manifest, dispatch each `AdapterDisplay` to its renderer.
7. Each renderer creates a ROS subscription on the bridge node; callbacks
   call `rr.log(...)`.

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
| `MAP` | `rr.SegmentationImage` (free/occupied/unknown trinary) |
| `TF` | no-op (handled by `TFMirror` unconditionally) |
| `PATH` | `rr.LineStrips3D` |
| `POSE` | `rr.Arrows3D` (one arrow at pose+yaw) |
| `ODOM` | `rr.Arrows3D` |
| `LASER_SCAN` | `rr.Points3D` in sensor frame |
| `POINTS_3D` | `rr.Points3D` in sensor frame |
| `IMAGE` | `rr.Image` |
| `IMU` | `rr.Scalars` per axis |
| `POLYGON` | `rr.LineStrips3D` (closed loop) |
| `PEDESTRIANS` | `rr.Boxes3D` from MarkerArray |
| `ROBOT_MODEL` | URDF via `rerun-loader-urdf-python` (optional, warn-and-skip if absent) |
| `FOOT_CONTACT` | warn-and-skip |
| `TRAJECTORY` | warn-and-skip (MoveIt-only) |
| `PLANNING_SCENE` | warn-and-skip (MoveIt-only) |

### Entity-path convention

`rerun_utils/entity_paths.py`:

- env-level displays: `env_<id>/<display_name>`
- per-robot displays: `env_<id>/robots/<robot.name>/<display_name>`
- TF tree: `env_<id>/tf/<child_frame>`
- frame-anchored displays (LaserScan, PointCloud2): logged under `env_<id>/tf/<frame_id>/<display_name>` so they ride the TF transform automatically.

## Style hints

The renderer reads styling from `StyleSpec.from_json(d.style_json)`:

- `color`, `alpha`, `line_width`, `enabled`, `decay`: honored when meaningful for the archetype.
- `latched`: MARKER_ARRAY subscribes reliable + transient-local.
- `extra={"rerun": {...}}` is the escape hatch (not used by any current renderer; reserved for future per-viz nudges).
