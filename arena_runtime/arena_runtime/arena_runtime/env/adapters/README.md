# Navigation adapters (env Python half)

The env layer's `Adapter` ABC composes an `arena_robots.bringup.Bringup` and
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

Goto completion is judged by the task runner from the robot pose. Gesture and
reach phases complete on `self.client.is_done()`, and `on_phase_tick(phase, robot)` runs every
judge tick while a phase is active, with the phase as `dispatch_phase` received it (Nav2
uses it to redispatch a finished action while the judge has not seen the robot arrive,
never while `robot.waiting`). Mobile adapters inherit
[`MobileAdapter`](mobile/__init__.py), whose `on_reset` teleports the robot
to `ctx.start_pose`. Subclass overrides chain `super().on_reset(...)` before
any post-teleport work. The episode's first phase is dispatched before `on_reset`
runs, so an override that cancels the in-flight action sends `robot.active_dispatch_phase()`
again after the teleport (rosnav_rl, external) or redispatches it on a later tick (Nav2).
`cancel_goals()` cancels every client's in-flight goal, `RobotManager.stop_task` calls it on
each bound adapter.

## MoveIt arm scene

Before every `ReachPhase` dispatch the `moveit` arm adapter rebuilds the arena-owned collision
objects of the arm's MoveIt scene through `<arm ns>/apply_planning_scene`: one diff that removes
the ids of the previous dispatch (tracked per mount) and adds the current ones, so reset needs no
sync and a base that moved between reaches is followed. Ids are `arena/<entity>` for a static
object and `arena/<entity>/<link>` for each link of an articulated one, `<entity>` the obstacle
name as the scenario authors it. Only the env's static and articulated obstacles enter, no walls,
pedestrians or robots, and objects whose bounds lie farther than the arm cap's `scene_radius` from
the base are left out.

Objects are expressed in the robot's MoveIt root frame, the root link of its
`robot_description`, posed from the base TF in the map frame because the SRDFs carry no virtual
joint and the planning frame moves with the base. Geometry lives in
[`arm/scene.py`](arm/scene.py):

- An articulated object takes the collision meshes and origins of its built bundle URDF
  (`<model>.urdf`, meshes under `links/<link>/`), each link posed by URDF forward kinematics at
  the joint values the simulator holds for the entity now (the state behind its `q.<joint>`
  fields). The obstacle scale multiplies the link translations and the vertices.
- A rigid object takes the annotated `bounding_box` as one box.
- OBJ files are read by a standard-library reader (`v` and `f` lines, `i`, `i/j` and `i/j/k`
  corners, negative indices, fan triangulation).

### Object frames

A `ReachPhase.frame` naming an episode entity (`chest`) or one of its links (`chest/<link>`)
is resolved at dispatch into a `PoseStamped` in the robot base frame: robot pose and entity pose
in the map frame, the URDF link frame at the current joint values, then the authored target in
that frame. With `relative: true` the offset is rotated into the base frame instead of composed.
Any other frame keeps `reach_frame`. A linear reach in an entity's frame sets the goal's
`contact` to the object ids of all its links, so the arm may touch that object and nothing else.
The frame is a snapshot at dispatch: a link frame moves with its joint afterwards, so a target
meant to stay put is given in the entity frame.
A reach dispatched before the entity its frame names is registered (an episode's first phase
dispatches before the episode objects spawn) is sent from a background task once the entity
appears, at most 10 s of wall time later. A cancel or the next dispatch drops that task.

## Existing kinds

| File | Bringup | Client |
|---|---|---|
| `nav2.py` | `Nav2Bringup` | `GotoPoseClient` |
| `meshnav.py` | `MeshNavBringup` | `GotoPoseClient` |
| `none.py` | `NoneBringup` | `GotoPoseClient` |
| `external.py` | `ExternalBringup` | `GotoPoseClient` |
| `drl.py` | `DrlBringup` | `GotoPoseClient` |
| `vla.py` | `DrlBringup` | `GotoPoseClient` |

`VlaAdapter` is a `DrlAdapter` for planners that declare `goal_inputs`
(see [arena_planners/planners/README.md](../../../../../arena_planners/planners/README.md)).
It reveals the goal inputs `robot.mobile.goal` selects, falls back to
`robot.mobile.instruction` for goals without one, and refuses robots without
an `image` sensor. A goto without authored `text` is worded by
[`shared/route.py`](../../../../../arena_simulation_setup/src/arena_simulation_setup/shared/route.py):
`robot.mobile.wording:=route` gives walking directions from the robot's pose
through the world's doors and openings ("Walk through the main hallway, then
take the door on your left into the pharmacy."), `robot.mobile.wording:=goal`
names the target ("Go to the pharmacy."). Unset, a bare pose gets directions
and a named target is named. A pose the zones do not cover falls back to its
coordinates. `EpisodeRecord.phases` keeps the instruction and its source per
phase.

While a phase holds at its goal (`hold_time`, `until`), `DrlAdapter` cancels the
planner through its edge node, which zeroes `cmd_vel`, since a learned policy has
no stop of its own and the park judge wants the robot still. It redispatches the
phase if the robot leaves the tolerance before the hold ends.

`DrlAdapter` (and so `VlaAdapter`) can send the signals its planner lists
under `signals:` in `planner.yaml`. For a goto phase with a `signal`, it
passes the expected signal to the planner in the reset payload and ends the
phase when that signal arrives: within the phase's tolerance it succeeds,
otherwise the episode fails as `signaled <signal> <d> m from goal`.
`RobotManager.submit_task` rejects a phase whose signal the robot's adapter
cannot send.

`ExternalBringup` reads `goal_topic`, `cmd_vel_topic`, `launch_file`,
`requires`, and `extra` from `caps/mobile.yaml > external:`, configure them
there, not as constructor arguments.

`MeshNavAdapter` (`robot.mobile_adapter:=meshnav`, needs `arena feature meshnav
install`) plans with MeshNav (Move Base Flex mesh navigation) on a triangle mesh
of every walkable floor, elevator cabin, stair and escalator. Before bringup it
builds the mesh of the spawned world in the simulator frame with
[`shared/navmesh.py`](../../../../../arena_simulation_setup/src/arena_simulation_setup/shared/navmesh.py),
bridging height steps up to the robot's `meshnav.max_step` and closing the doors
that never open for the robot (`mechanism_map.static_shut_doors`: no activation
distance, or a locked gate it is not authorized for, unless a pressure plate
drives the door or an `unlock_on` regime unlocks it), and writes it once as
`<tmp>/arena_meshnav/<hash>.ply`, one file per distinct mesh. Goals are placed on
their level and sent in the robot's `odom` frame, which is the simulator frame.
Its `routes_levels` is set, so `RobotManager.submit_task` routes its goals with
the stair kinds the robot climbs as `walks` of `elevator_routing.route_phases`:
elevator rides get their legs and ride phase as for nav2, while stairs and
escalators get no phases and the mesh plans over them. Before sending a goal the
adapter looks up the mesh vertices nearest to the robot and the goal, and when
they lie in separate components (no stair, escalator or door joins them, and no
ride was inserted) it logs a warning and fails the phase through
`RobotManager.fail_phase` instead of sending a goal MBF cannot reach. Ride phases
are exempt, and so is any point more than 1 m from the mesh.

## Adding a new adapter kind

1. Create `arena_robots/bringup/<kind>.py`: `Bringup` subclass with `kind`,
   `requires`, and `launch_description()`.
2. Add handler(s) in `arena_robots/task_server_handlers/<task_kind>/<kind>.py`,
   then register a lazy loader in the per-kind `__init__.py`:
   `@HANDLERS.register((TaskKind.GOTO_POSE, "<kind>"))`
   The loader imports the module and returns the handler class, so its msgs
   deps are only pulled when the bringup is selected at runtime.
3. Create `arena_runtime/env/adapters/<kind>.py`: `Adapter`
   subclass with `kind`, `accepts`, `bringup_cls`, `client_cls`, and
   `dispatch_phase`.
4. Eager-import the new module from `RobotManager.__init__` so registration
   fires before `get_adapter` runs.
5. Set `mobile: <kind>` (or `adapters: {mobile: <kind>}`) in the scenario robot entry or `robot_setup.yaml`.

## Lift and gripper adapters

`lift` and `gripper` adapters accept no task phase (`accepts=()`, no client): they own a ROS command
surface instead. `robot.lift` / `robot.gripper` (default `direct`) select the kind for every robot
advertising the cap, `adapters: {lift: none}` in a robot setup file overrides one robot.

| Cap | Kind | Command topic under `<robot_ns>` | Type | Effect |
|---|---|---|---|---|
| `lift` | `direct` | `lift/<instance>/command` | `std_msgs/Float64` | height clamped to the instance's `limits: [lower, upper]` (unclamped without it), published on `<controller>/commands` (`Float64MultiArray`) |
| `gripper` | `direct` | `gripper/<instance>/command` | `control_msgs/GripperCommand` | position clamped between `positions.open` and `positions.close`, `max_effort` capped at the instance's (0 = the instance's), sent as a `GripperCommand` goal to `<controller>/gripper_cmd` |
| both | `none` | - | - | binds nothing |

`direct` echoes the clamped command on `lift/<instance>/applied` and `gripper/<instance>/applied`
(same types, transient-local). The gripper goal is fire and forget: the adapter does not await the
result. `<instance>` is the instance key of `caps/lift.yaml` / `caps/gripper.yaml`, or the mount name
for component caps. The `robots_lift` and `robots_gripper` MCP tools publish these commands and
return the echo. Topic names and clamping live in
[`arena_robots/actuators.py`](../../../../../arena_robots/arena_robots/arena_robots/actuators.py).

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
for the rare per-viz nudge. `robot_colored=True` replaces the style's color with
the robot's own color from
[`env/colors.py`](../colors.py), which its handles
share, so one robot's paths and handles read as one color.

## Opting into the collision tracker

`CollisionTrackerNode` (in
[`env/robot_manager/collision_tracker.py`](../robot_manager/collision_tracker.py))
is mounted by `MobileAdapter` whenever the mobile cap declares a `footprint`
or a `polygons_dict`. It publishes two topics under the robot namespace each
tick (sim-time gated):

| Topic | Type | Content |
|---|---|---|
| `<robot_ns>/collision_monitor_state` | `nav2_msgs/CollisionMonitorState` | Scalar `polygon_name` + `action_type` for the most severe active polygon. Drop-in for the same observation nav2's `collision_monitor` emits, training consumers don't care which produced it. |
| `<robot_ns>/collision_events` | `arena_robots_msgs/CollisionEvents` | Footprint contacts, the single definition of a collision: at most one `CollisionEvent` for the top-ranked grid cell under the footprint plus one per pedestrian disc it intersects. `kind` is one of the `KIND_*` constants; `obstacle_id` is the static name, `<wall>` for an authored wall, `<map>` for map occupancy with nothing authored behind it, or the pedestrian name. `distance` is reserved for future signed-clearance reporting; v1 is intersect-only and always emits `0.0`. Under `fail_on_collision` a non-empty message ends the episode with outcome `collision`, so that outcome implies a non-empty message in the bag. Without a `footprint` the topic is not published. |

Static geometry is one window read on `EnvironmentManager.collision_grid`
(see [env/README.md](../README.md)), kept in sync by the
spawn/respawn pipeline, so the tracker needs no per-episode refresh hook.
`EnvironmentManager.static_polygons` only supplies the reported centroid.
Pedestrians are read from the `arena_peds` topic.

An adapter that does not subclass `MobileAdapter` mounts the node in
`__init__` and adds it to the env node's executor:

```python
class MyMobileAdapter(Adapter):
    def __init__(self, robot_manager: RobotManager, **bringup_kwargs: object) -> None:
        super().__init__(robot_manager, **bringup_kwargs)
        mobile = self.rm.robot.model.resolve_sync().caps.mobile
        self._collision = CollisionTrackerNode(self.rm, mobile.polygons_dict, footprint=mobile.footprint)
        self.rm.node.executor.add_node(self._collision)
```

[`mobile/test_collision.py`](mobile/test_collision.py) is a `MobileAdapter`
subclass that drives a constant forward cmd_vel, for exercising the tracker.
