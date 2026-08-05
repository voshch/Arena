# Navigation adapters (task_generator Python half)

task_generator's `Adapter` ABC composes an `arena_robots.bringup.Bringup` and
an `arena_robots.clients.Client`. It maps scenario Phases to arena IDL goals
and dispatches them through the Client, the same path any external consumer
uses. See [arena_robots/DRIVING.md](../../../../../arena_robots/DRIVING.md) for
the public API.

## The `Adapter` ABC

Defined in [`__init__.py`](__init__.py):

```python
@register_adapter
class MyAdapter(Adapter):
    kind       = "my_stack"              # matches Bringup.kind
    accepts    = frozenset({TaskKind.GOTO_POSE})
    bringup_cls = MyBringup
    client_cls  = GotoPoseClient

    async def dispatch_phase(self, phase, robot) -> None:
        goal = GotoPose.Goal()
        goal.target = self._phase_to_pose_stamped(phase)
        await self.client.send_goal(goal)
```

The base class constructor builds `self.bringup` and `self.client` from
`bringup_cls` / `client_cls`; subclasses only implement `dispatch_phase`.

`is_phase_done` polls `self.client.is_done()`. Mobile adapters inherit
[`MobileAdapter`](mobile/__init__.py), whose `on_reset` teleports the robot
to `ctx.start_pose`. Subclass overrides chain `super().on_reset(...)` before
any post-teleport work.

## Existing kinds

| File | Bringup | Client |
|---|---|---|
| `nav2.py` | `Nav2Bringup` | `GotoPoseClient` |
| `none.py` | `NoneBringup` | `GotoPoseClient` |
| `external.py` | `ExternalBringup` | `GotoPoseClient` |

`ExternalBringup` reads `goal_topic`, `cmd_vel_topic`, `launch_file`,
`requires`, and `extra` from `caps/mobile.yaml > external:`, configure them
there, not as constructor arguments.

## Adding a new adapter kind

1. Create `arena_robots/bringup/<kind>.py`: `Bringup` subclass with `kind`,
   `requires`, and `launch_description()`.
2. Add handler(s) in `arena_robots/task_server_handlers/<task_kind>/<kind>.py`,
   then register a lazy loader in the per-kind `__init__.py`:
   `@HANDLERS.register((TaskKind.GOTO_POSE, "<kind>"))`
   The loader imports the module and returns the handler class, so its msgs
   deps are only pulled when the bringup is selected at runtime.
3. Create `task_generator/tasks/robots/adapters/<kind>.py`: `Adapter`
   subclass with `kind`, `accepts`, `bringup_cls`, `client_cls`, and
   `dispatch_phase`.
4. Eager-import the new module from `RobotManager.__init__` so registration
   fires before `get_adapter` runs.
5. Set `mobile: <kind>` (or `adapters: {mobile: <kind>}`) in the scenario robot entry or `robot_setup.yaml`.

## Declaring visualization displays

Adapters surface displays to any visualizer (rviz, rerun, ...) through
`AdapterDisplayHint` entries. The base `Adapter` already declares
`ROBOT_MODEL` + `ODOM`, and `MobileAdapter` adds `POSE` (goal) + `PATH`
(plan), so subclasses only declare adapter-owned displays.

```python
from arena_viz import DisplayKind, StyleSpec

@AdapterMeta.attach(
    accepts={TaskKind.GOTO_POSE},
    bringup=MyBringup,
    client=GotoPoseClient,
    cap="mobile",
    displays=[
        AdapterDisplayHint(
            name="Local Plan",
            topic="{ns}/local_plan",
            topic_type="nav_msgs/Path",
            kind=DisplayKind.PATH,
            style_json=StyleSpec(color=(255, 0, 0), line_width=0.05).to_json(),
        ),
    ],
)
class MyAdapter(MobileAdapter):
    ...
```

`{ns}` and `{robot}` in `topic` / `style_json` are substituted at manifest
publish time. `kind` is a canonical [`DisplayKind`](../../../../../utils/arena_viz/arena_viz/kinds.py)
value; each visualizer ships a renderer per kind. `style_json` is a
serialized [`StyleSpec`](../../../../../utils/arena_viz/arena_viz/style.py),
viz-neutral fields (color, alpha, line_width, enabled) plus an `extra`
escape hatch keyed by visualizer (`extra={"rviz": {"Color Scheme": "costmap"}}`)
for the rare per-viz nudge.

## Opting into the collision tracker

`CollisionTrackerNode` (in
[`manager/robot_manager/collision_tracker.py`](../../../manager/robot_manager/collision_tracker.py))
is a shim for adapters that don't run nav2's own `collision_monitor` (RL, bare
cmd_vel, research stacks). It takes the mobile cap's `polygons_dict` and
publishes two topics under the robot namespace each tick (sim-time gated):

| Topic | Type | Content |
|---|---|---|
| `<robot_ns>/collision_monitor_state` | `nav2_msgs/CollisionMonitorState` | Scalar `polygon_name` + `action_type` for the most severe active polygon. Drop-in for the same observation nav2's `collision_monitor` emits, training consumers don't care which produced it. |
| `<robot_ns>/collision_events` | `arena_robots_msgs/CollisionEvents` | Per-obstacle attribution: one `CollisionEvent` per (cap polygon × hit entity). `obstacle_id` is the entity name (`<wall>` for world walls/doors), `polygon_name` is the cap polygon. `distance` is reserved for future signed-clearance reporting; v1 is intersect-only and always emits `0.0`. |

Geometry comes from `EnvironmentManager.walls_geometry` (MultiLineString) and
`EnvironmentManager.static_polygons` (name-keyed footprint dict); both are kept
in sync by the spawn/respawn pipeline, so the tracker needs no per-episode
refresh hook. Pedestrians are read from the `arena_peds` topic.

Opt in by mounting the node in `__init__` and adding it to the task_generator's
executor:

```python
class MyMobileAdapter(Adapter):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._collision: CollisionTrackerNode | None = None
        polys = self.rm.robot.model.resolve_sync().caps.mobile.polygons_dict
        if not polys:
            return
        self._collision = CollisionTrackerNode(self.rm, polys)
        self.rm.node.executor.add_node(self._collision)
```

See [`mobile/test_collision.py`](mobile/test_collision.py) for a working example, a debug
adapter that mounts the tracker and drives a constant forward cmd_vel. The
`nav2` adapter deliberately does not mount it: nav2's own `collision_monitor`
already publishes `collision_monitor_state` on the same topic.
