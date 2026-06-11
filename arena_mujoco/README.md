# arena_mujoco

MuJoCo simulation backend for Arena-Rosnav. Selects with `sim:=mujoco`.

Two packages live here:

| Package | Role |
| --- | --- |
| `arena_mujoco/` | Out-of-process MuJoCo server node (`run_mujoco`) and all service handlers |
| `arena_mujoco_msgs/` | Custom message and service types for the server interface |

---

## Architecture

```
arena_runtime (per-env)                 arena_mujoco process
  MujocoHost (SimLifecycle)            ┌─────────────────────────────────┐
    pause / unpause / cleanup ────────►│  MujocoController (rclpy node)  │
                                       │  step loop: 50 Hz wall-clock    │
  MujocoSimulator (per env_id)         │  or explicit Step service       │
    SpawnUrdf / SpawnPrims / ─────────►│  /clock publisher               │
    SpawnWalls / SpawnCeilings /       │                                 │
    SpawnFloors / SpawnPedestrians /   │  SceneStore                     │
    EditPrims / DeletePrims /          │    _EnvScene (per env_id)       │
    MovePedestrians / UpdatePeds /     │      MjSpec / MjModel / MjData  │
    DeletePedestrians / ResetWorld /   │      body name registry         │
    Step                               │      mocap pool (64 slots/env)  │
                                       └───────────┬─────────────────────┘
                                                   │ hooks (per robot spawn)
                                        ┌──────────┼──────────────────┐
                                        │          │                  │
                                   control.py  odom.py          sensors/
                                   pre_compile  post_spawn       pre_compile
                                   post_spawn   pump (~50 Hz)    post_spawn
                                   pump                          pump
                                   JointState subs/pub      LaserScan / Image /
                                   data.ctrl write          CameraInfo / IMU /
                                                            Contact publishers
```

**SceneStore** owns one `_EnvScene` per numeric `env_id`. Each scene holds a live `MjSpec`, compiled `MjModel`, and `MjData`. Adding or removing a body marks the scene dirty; `recompile()` rebuilds model and data, carrying joint state forward by name. Every service call that adds geometry ends with a `recompile` for that env. Sibling envs are never touched.

**Hooks** (`arena_mujoco.hooks`) decouple the control, odom, and sensor subsystems from `SpawnUrdf` and the step loop. Four hook types:

- `pre_compile(env_id, spec, robot_root_body, robot_entry, joint_names)` - mutates the spec before recompile (add actuators, cameras, sites).
- `post_spawn(env_id, name, robot_entry, joint_names)` - runs after recompile when model and data are live; creates ROS publishers and subscribers.
- `despawn(env_id, robot_entry)` - runs when a robot's body is deleted (by name, by env prefix or by `ResetWorld`); destroys the publishers and subscriptions `post_spawn` created.
- `pump()` - called once per tick of sim time; drains staged commands and publishes sensor data.

Modules register themselves at import time (`control.py`, `odom.py`, `sensors/__init__.py`).

---

## Physics step loop

`MujocoController._advance(n)` is the one stepping path. It works through `n` physics steps (`dt = 0.002 s`, pinned in `scene.PHYSICS_DT`) in ticks of 10 steps, and after each tick drains the pumps (control commands, joint states, odom and TF, sensors) and then publishes `/clock` from the store clock. Odom and TF go out on every tick, the transforms of all robots in one `/tf` message (every nav2 node of every env subscribes to `/tf`, so a message per robot made publishing the server's main cost at 8 envs), including the short last tick of a lockstep window, so the clock never runs ahead of the latest transform. A consumer that looks a transform up at `now()` would otherwise wait in sim time while the held sim waits for that consumer. Sensors and odom rate-limit on sim time, so their cadence is the same at any real-time factor.

- Free-running: a 50 Hz wall timer calls `_advance(10)`, one tick per 20 ms, real-time factor 1.
- Paused: the timer only drains the pumps, so joint states keep flowing to the controller_manager while the clock stands still.
- Lockstep: `mujoco/Step n` calls `_advance(n)` regardless of the pause flag and answers the clock it reached. `MujocoHost.step_seconds` drives it, so `arena lockstep` and the benchmark runner step MuJoCo like the other simulators.

---

## Service surface

All services are on the global `/mujoco/` namespace. `env_id` selects the target env; the server calls `ensure_env(env_id)` on first contact. Every service keeps 1000 requests (`services/utils.SERVICE_QOS`): the server answers on one thread, and with the default depth of ten the requests of several envs starting at once were dropped while it was busy.

### Lifecycle (registered directly on the node)

| Service | Type | Description |
| --- | --- | --- |
| `mujoco/PauseSimulation` | `std_srvs/Trigger` | Halt step loop |
| `mujoco/UnpauseSimulation` | `std_srvs/Trigger` | Resume step loop |
| `mujoco/Step` | `arena_mujoco_msgs/Step` | Advance all envs by `n` steps (RL determinism) |
| `mujoco/ResetWorld` | `arena_mujoco_msgs/ResetWorld` | Prefix-delete all bodies for `env_id`, recompile |
| `mujoco/DeletePrims` | `arena_mujoco_msgs/DeletePrims` | Delete named bodies; trailing `/` triggers prefix delete |

### World content (via `services` registry)

| Service | Type | Description |
| --- | --- | --- |
| `mujoco/SpawnUrdf` | `arena_mujoco_msgs/SpawnUrdf` | Load URDF, attach into env spec, fire hooks |
| `mujoco/SpawnPrims` | `arena_mujoco_msgs/SpawnPrims` | Spawn box or mesh obstacles |
| `mujoco/EditPrims` | `arena_mujoco_msgs/EditPrims` | Update pose and scale of existing prims |
| `mujoco/SpawnWalls` | `arena_mujoco_msgs/SpawnWalls` | Spawn oriented box walls from start/end points |
| `mujoco/SpawnFloors` | `arena_mujoco_msgs/SpawnFloors` | Spawn flat floor slabs |
| `mujoco/SpawnCeilings` | `arena_mujoco_msgs/SpawnCeilings` | Spawn flat ceiling slabs (2 cm thick) |

### Pedestrians (reuse `arena_people_msgs`)

| Service | Type | Description |
| --- | --- | --- |
| `mujoco/SpawnPedestrians` | `arena_people_msgs/SpawnPedestrians` | Assign mocap pool slots, set initial pose |
| `mujoco/MovePedestrians` | `arena_people_msgs/MovePedestrians` | Teleport to explicit pose, restarts the clip |
| `mujoco/UpdatePedestrians` | `arena_people_msgs/UpdatePedestrians` | Continuous pose update (twist ignored) |
| `mujoco/DeletePedestrians` | `arena_people_msgs/DeletePedestrians` | Park mocap body, recycle handle |

**env_id routing for pedestrians.** `SpawnPedestrians` carries no `env_id` field (inherited from `arena_people_msgs`). The server parses the `env_N/` prefix from each pedestrian name to derive the target env.

---

## How entities are realized

### Robots (URDF)

`SpawnUrdf`:

1. Rewrites the URDF for MuJoCo compatibility: reads the `<gazebo reference><sensor>` declarations (`UrdfSensor`: kind, link, pose, rate, camera fov, resolution and clip range), then drops `ros2_control`, `transmission`, and `gazebo` subtrees; folds dash-hostile link/joint names to underscores; points every `<mesh>` at the first part of `mesh_parts(file)`; injects `<mujoco><compiler discardvisual="false" fusestatic="false" strippath="false"/></mujoco>`.
2. Drops any `<visual>` or `<collision>` whose mesh cannot be read, with a warning, as Gazebo does. Parses the cleaned URDF with `MjSpec.from_file`, then gives each mesh geom the material of its file and adds a sibling geom per further material of that file.
3. Attaches the robot spec under the env worldbody via `attach_body(prefix)`, which rewrites all ids to `<prefix><id>`. If `localization=True`, a freejoint is added to the base link so `set_body_pose` can drive it.
4. Fires `pre_compile` hooks (actuator wiring, sensor element injection), then `recompile`, then `post_spawn` hooks (ROS publishers and subscribers). A robot whose sensor elements or model do not compile is rolled back, including the actuators and sensors that referenced its bodies, and the response carries an empty `path`.

The `MujocoSimulator` side additionally generates a bridge URDF with `TopicBasedSystem` hw-plugin entries before calling `SpawnUrdf`, so the external `controller_manager` reads the same topic names as the server.

### Obstacles

`SpawnPrims` dispatches on `prim.mesh_path`:

- Non-empty path: `add_mesh_body` adds one `mjGEOM_MESH` geom per material of the file, with that material's color or diffuse texture. Those geoms are visual only (`contype = conaffinity = 0`): cameras and ray casts see them, physics does not. Physics collides with the file's convex decomposition instead (see below), so a robot fits under a table but not through its legs.
- Empty path: `add_box_body` with `scale` as full extents (halved to MuJoCo half-extents internally, 1 mm at least per side). Boxes collide.

`MujocoSimulator.obstacle_spawn` and `obstacle_delete` send every obstacle of a call in one request: a service client keeps ten requests, so a burst of per-obstacle calls loses some and each lost one waits out its timeout. Per obstacle it takes the model's OBJ, else the mesh its SDF visual points at (the asset database ships SDF plus DAE), else the annotated bounding box as a box, and skips with a warning when the asset has none of them. `optim.obstacles:=bbox` prefers the bounding box.

`EditPrims` with `scale` resizes a prim in place: a box takes the scale as its side lengths, mesh geoms switch to their file at that scale. `EditPrims` with `pose` writes the pose into the spec as well as the model, so a moved obstacle stays moved across later recompiles.

### Collision hulls (`mesh_mat.collision_hulls`)

MuJoCo collides a mesh by its convex hull, which fills the space under a table. Each obstacle mesh is therefore decomposed once with CoACD (concavity 0.1, 16 pieces at most) into `<stem>_hull<n>.obj` files in the mesh cache, and every piece becomes a colliding geom in geom group 3 with at most 32 hull vertices. Group 3 is hidden from cameras, the viewer (key `3` shows it) and ray casts. `SpawnPrims` decomposes the uncached files of a request on up to 8 threads. A cold cache makes the first load of a furnished world slow (hospital_1: 45 files, 595 pieces, about two minutes on smu-gpu01), later loads read the cache. A file CoACD cannot decompose keeps its visual and collides with nothing, with a warning.

### Meshes (`mesh_mat.mesh_parts`)

MuJoCo reads OBJ, MSH and STL up to 200000 faces itself. Every other file goes through trimesh once: the scene is split per material into `<stem>_<digest>_<n>.obj` parts (plus a PNG per textured part, 1024 px at most) under `$TMPDIR/arena_mjmesh/<digest>/`, keyed by path, mtime and size, with a `parts.json` manifest that later processes reuse. A file above 200000 faces becomes one convex hull per sub-mesh. Parts keep the diffuse color the file declares. A file that declares none (plain STL) leaves the geom's own color alone, so a URDF `<material>` still applies.

### When the scene recompiles

Every structural change needs a fresh `MjModel`, and each compile also rebuilds the camera renderers and reloads the viewer. `DeletePrims` and box-only `SpawnPrims` therefore only edit the spec. The next tick, step or pose query compiles once for the whole burst (`SceneStore.compile_dirty`), so removing a world's 143 obstacles is one compile, not 143. `EditPrims` scale changes wait for that compile too. Requests that can fail on their assets (mesh prims, walls, floors, ceilings, robots, pedestrian skins) compile before they answer. A compile that follows deletions first frees the meshes, materials and textures nothing uses any more.

### Rollback

`SceneStore.recompile` drops the bodies and assets added since the last successful compile when the spec no longer compiles, recompiles, and raises `RejectedBodies`. `SpawnPrims` then retries its prims one at a time so one bad asset costs one obstacle. The wall, floor, ceiling and robot services report the batch as failed. The server keeps running either way.

### Walls, floors, ceilings

All three use `add_box_body`. Walls are built from start/end points: length from XY distance, height from Z span, center and yaw-quaternion derived inline. Floors and ceilings are thin slabs (2 cm) centered on the supplied XY position.

### Materials (`arena_mujoco_msgs/Material`)

`ensure_material(spec, cache, mat_msg)` in `mesh_mat.py` turns the `.mdl` file at `path` into one tiled material, cached per env:

1. the image in its OmniPBR `diffuse_texture` slot,
2. else a referenced image whose name ends in `diff`, `diffuse`, `albedo` or `basecolor`,
3. else the diffuse color the file declares (`diffuse_tint = color(...)`),
4. else, with a warning, a muted color derived from the material name.

Textures are bound by file, so MuJoCo applies its own image and UV conventions. Only the albedo is used, normal and roughness maps are not.

### Lighting

Each env spec carries a shadowless directional light from above plus a raised headlight ambient term. Ceilings would otherwise put every indoor camera frame in shadow.

### Pedestrians (kinematic mocap pool)

On first pedestrian spawn for an env, `alloc_mocap(env_id, 64)` pre-creates 64 capsule mocap bodies parked at `(0, 0, -1000)` and compiles once. Spawn and delete reuse handles from the free list. `set_mocap_pose` writes `data.mocap_pos/quat` directly, heading included; no physics integration, no recompile.

Each `/mujoco/arena_peds` message names its pedestrian's actor SDF in `model_uri`, the same field Gazebo's skeleton plugin reads. The first time a name carries one, `humans.actor_rig` loads the actor's skinned COLLADA mesh and its clips with pycollada and caches the result as `rig.npz` in the mesh cache: one skin part per material (512 px textures, parts whose texture is mostly transparent are dropped), the vertices and weights each bone moves, and per clip the pose of every bone at every key. The pooled body then gets one mocap body per bone (31 for the shipped actors), each sitting at its joint, and one MuJoCo skin per part bound to them, and its capsule moves to group 4, which costs one recompile. A released body keeps its skins, so a roster that repeats recompiles nothing. Deleting a pooled body deletes its bones and skins.

| Element | Cameras | Lidar rays | Contacts |
| --- | --- | --- | --- |
| capsule without skins (group 0) | yes | yes | yes |
| capsule under skins (group 4) | no | yes | yes |
| skins | yes | no | no |

**Animation from the wire's joints.** Every `/mujoco/arena_peds` message carries the pedestrian's ROS4HRI joint angles in `joint_state` (the task generator fills them from its gait and gesture layer, the stream Isaac renders). `humans.joint_poses` does what `arena_isaac`'s `ExternalPoseProvider` does: each angle turns its bone about an axis in that bone's own frame (`bone_map.BONE_MAP`, a copy of Isaac's `peds/providers/bone_map.py`), composed onto the actor's neutral stance, which is the first key of its `idle` clip with the root joint at rest. Forward kinematics through the actor's skeleton then gives every bone's world pose. Bones the map does not name hold the neutral stance, the root does not bob. Pedestrians of one skeleton layout are posed in one batch, about 0.3 ms per message.

The shipped actors are the bundles Isaac converts (same 31 CMU bones, same order), so the result is Isaac's pose. Isaac's own converter and provider code, run on all 19 actors in four poses that move vertices by more than a meter, put every skinned vertex within 0.14 mm of this path. A unit test repeats that comparison on the test actor and another keeps the two bone maps identical, both skip without an `arena_isaac` checkout. Seen side by side through the turtlebot camera, Isaac and MuJoCo show the same stance, stride and wave.

**Animation from clips.** A pedestrian whose message has no `joint_state` (a `MovePedestrians` teleport, a publisher that sends poses only) plays its actor's clips instead, like Gazebo's skeleton plugin: `walk` when it covers ground at 0.05 m/s or more, with the cursor advanced by the distance covered over the clip's own stride (1.475 m in 6.17 s for the shipped actors) so the feet stay planted, else `idle` in sim time. The root joint keeps its clip rotation and bob while its steady advance over the clip is removed, since the pose on the wire already carries it. `MovePedestrians` restarts the clip.

---

## ros2_control bridge wire format

The `control.py` module wires per-joint position and velocity actuators into the MjSpec before each robot recompile:

- Position actuator: `gaintype=FIXED kp=1000`, `biastype=AFFINE [0, -kp, -kv]` (stiff position servo).
- Velocity actuator: `gaintype=FIXED kv=100`, `biastype=AFFINE [0, 0, -kv]` (damped velocity servo).

ROS topics (per-robot, under the robot's namespace):

| Topic | Direction | Type | Description |
| --- | --- | --- | --- |
| `<ns>/mujoco/joint_commands_velocity` | in | `sensor_msgs/JointState` | Velocity setpoints; `.velocity` fields written to velocity actuator `data.ctrl` |
| `<ns>/mujoco/joint_commands_position` | in | `sensor_msgs/JointState` | Position setpoints; `.position` fields written to position actuator `data.ctrl` |
| `<ns>/mujoco/joint_states` | out | `sensor_msgs/JointState` | `qpos`, `qvel`, `qfrc_actuator` from the live model, published at ~100 Hz |

The external `controller_manager` (launched by `MujocoSimulator._launch_robot_stack`) provides the actual control law. The server side only reads setpoints from the command topics and exposes actuation state back to the bridge.

---

## Sensors

`sensors/core.py` provides `SensorPublisher` (abstract base) and the `@register_sensor(type_str)` decorator. Concrete publishers self-register at import:

| Module | Type key | MuJoCo element added | ROS output |
| --- | --- | --- | --- |
| `sensors/laser.py` | `laserscan`, `pointcloud` | named site on the sensor frame's body | `sensor_msgs/LaserScan`, or the same fan as `sensor_msgs/PointCloud2`, via `mj_multiRay` |
| `sensors/camera.py` | `image`, `depth`, `camera_info`, camera-backed `pointcloud` | one `<camera>` per backing URDF sensor | `sensor_msgs/Image` (`rgb8`, `32FC1` planar meters), `sensor_msgs/CameraInfo`, `sensor_msgs/PointCloud2` |
| `sensors/imu.py` | `imu` | site plus framequat, gyro and accelerometer sensors | `sensor_msgs/Imu` |
| `sensors/contact.py` | `contact` | site marking the frame's body | `ros_gz_interfaces/Contacts` |

`SpawnUrdf` carries the robot's effective sensor list (`RobotView.effective_sensors` of its assembly request, so `jackal[top=camera/oakd_rgbd]` brings the camera's sensors) with topics already resolved. The framework adds each sensor's MuJoCo elements to the spec at pre-compile time, then instantiates publishers in post-spawn. A name that repeats in the list (an assembly part declaring `imu` next to the base's own) gets a `_1`, `_2` suffix for its elements. Elements are named `<robot root body>_sensor_<name>_...` and looked up by that exact name, so two robots in one env keep their own sensors. A recompile renumbers every model element, so the pump calls `rebind()` on a publisher before the first publish against a new model.

**Mounting.** Every sensor sits at the `<pose>` of its backing URDF `<sensor>` on that element's `<gazebo reference>` link (the child link when the reference names a joint), which is where Gazebo measures from. Without a backing element it sits at the origin of its `SensorSpec.frame` link, and on the robot root when the model has no such body. Messages are stamped with `SensorSpec.frame` and sim time.

**Rates.** The pump runs once per 20 ms tick and publishes a sensor when its next due time has passed, so the average rate is exact (a 30 Hz camera alternates 20 and 40 ms gaps) and nothing publishes above 50 Hz.

**Lidar.** Each lidar takes its own beam count, angle limits (both ends included), ring elevations, range limits and rate from its URDF sensor element, and falls back to the robot's `caps/mobile.yaml` `laser:` block and then to a 360 beam ring. `LaserScan` carries the ring `ros_gz_bridge` picks from a multi-ring scan (index `rings // 2`). `PointCloud2` carries every ring, cast only while subscribed. Ranges follow REP 117: `+inf` without a return inside the maximum, `-inf` below the minimum. The lidar does not see the robot that carries it: for the cast it moves that robot's geoms into geom group 5, which the ray mask skips together with group 2 (ceilings) and group 3 (collision hulls). Other robots, pedestrian capsules and obstacle meshes stay visible. A fan of 1024 rays or more is split over up to 8 threads, each casting on its own copy of the `MjData` (`mj_multiRay` keeps scratch memory there): the jackal's 10240 rays against 30 hospital meshes take 8 ms instead of 32 ms.

**Cameras.** Specs that share a `sensor:` share one MuJoCo camera and one render per sim time. A `pointcloud` spec is camera-backed when its `sensor:` also carries an image, depth or camera-info spec, or names a URDF camera sensor. The camera looks along +x of its mounting pose with z up, the Gazebo convention, and takes resolution, horizontal fov, clip range and rate from the URDF element (640x480, 60 degrees, 0.1 to 100 m, 30 Hz without one). The renderer's near plane is fixed at 1 cm (`stat.extent` is pinned, parked pedestrian bodies would otherwise push it out to a meter). Depth is `+inf` past the far clip and `-inf` before the near clip. The cloud holds the finite depth pixels in body axes (x forward, y left, z up), matching Gazebo's rgbd `points`. Image, depth and cloud render only while subscribed. The camera sees its own robot. The renderer pool (`_RendererPool`) holds one offscreen `mujoco.Renderer` per env and resolution, rebuilt when a recompile swaps the model. A GL failure degrades to no images rather than crashing the server. `MUJOCO_GL` defaults to `egl`.

**Contact.** `ros_gz_interfaces/Contacts`, the type `ros_gz_bridge` gives a Gazebo contact sensor: one `Contact` per pair of a geom of the frame's link and a penetrating geom outside the robot's own kinematic tree, floor included, with world positions, normals pointing into the link's geom and penetration depths. The list is empty while the link touches nothing.

---

## Launch

```bash
arena launch sim:=mujoco
```

The `arena_runtime` launch wires `run_mujoco.launch.py` when `sim:=mujoco`. To start the server alone:

```bash
ros2 launch arena_mujoco run_mujoco.launch.py [headless:=True] [log_level:=info]
```

`--headless False` (the default) attempts to attach `mujoco.viewer` in passive mode for env 0 after it is first compiled. The viewer draws on its own thread from copies of the sim state (`MujocoController._draw_frames`): a renderer that cannot keep up skips frames instead of holding the sim (hospital_1 over xrdp: rtf 0.41 with the sync on the sim thread, 0.98 off it). Its camera frames everything standing in the env after each recompile until the user moves it.

The sim thread hands the viewer a copy of env 0's data at the end of a tick, at most 30 times per wall second. The viewer syncs only the state (`sync(state_only=True)`) and copies the whole model only after `set_body_pose` moved a static body in place (`SceneStore.model_edits`). A full sync copies every model array the viewer reads, and in a furnished world that is mostly the mesh BVHs: hospital_1 with six pedestrians compiles to a 1 GB model with 10 M visible triangles, and one full sync copies 589 MB, against 0.7 ms for the state.

When the display renders GL in software (llvmpipe under xrdp, VNC or `ssh -X`) and VirtualGL reaches a GPU, the launch runs the server under `vglrun -d egl`: GL renders on the GPU and VirtualGL copies the finished frames into the window. A local display with GPU GLX, a machine without a GPU and `headless:=True` start the server as is. `arena feature mujoco install` installs VirtualGL 3.1.5 from its release `.deb`. hospital_1 over xrdp on smu-gpu01: 2 fps in software, 34 to 51 fps on the RTX PRO 6000 with 25 to 30 new sim states per second, rtf unchanged.

---

## Open items

- **Untextured assets.** 105 of the 148 database objects carry no texture in their DAE (43 do). They render in the DAE's plain diffuse color, black for some.
- **No robot declares a `contact` sensor.** The publisher has no user yet. Gazebo would bridge the same `ros_gz_interfaces/Contacts`, but `arena_robots` leaves `contact` out of its bridge table on purpose. Isaac publishes `isaacsim_msgs/ContactSensor`.
- **One executor thread.** Service and subscription callbacks, the pumps and the publishing run on the server's single executor thread. Physics steps (`SceneStore.step`, up to 16 workers), large ray fans and convex decomposition run on thread pools. With 8 jackal envs at real time the thread is 56 % busy.
- **More than about 4 envs in one runtime.** 4 jackal envs with nav2 come up in 21 s and hold real time (server 34 % of a core, lockstep rtf 2.0). At 8 the per-env ROS stacks fail first: nav2 lifecycle calls time out and episodes abort while the server is still half idle.

## Debugging

`kill -USR1 <run_mujoco pid>` prints every thread's Python stack to the server's stderr without stopping it.

## Tests

`tests/unit/` runs on real MuJoCo models:

- `test_scene.py`: name resolution, prefix deletes, state carry across a recompile, moved static bodies staying moved, the shared clock, the pedestrian pool and its lidar-height capsule, rollback of uncompilable additions.
- `test_sensors.py`: URDF sensor parsing (joint references included), lidar layout from its URDF element, ray casts that skip the carrier and follow REP 117, a fan cast across threads, camera orientation, depth range and clipping, sensor pose pitch, cloud axes, one shared camera per backing sensor, the fixed near plane, contact pairs and the contact message. The render tests skip without an offscreen GL context.
- `test_meshes.py`: per-material mesh parts, texture size limit, mesh bodies colliding as convex pieces (a ball rests on a table top and stays under it), rays passing through colliders, prim rescaling, the STL face limit, robot rollback, the material albedo order, one compile per delete burst.
- `test_humans.py`: actor SDF parsing, clip keys, bone poses per clip with the root advance removed, wire joint angles turning bones from the idle stance, skinned vertices matching Isaac's converter and provider, the bone map matching Isaac's, the skin a spawned pedestrian wears as the renderer draws it, rays hitting the capsule under it, wire joints taking precedence over clips, the walk clip advancing by distance and idle when standing, bones and skins leaving with their body.

Run with `arena test arena_mujoco`.
