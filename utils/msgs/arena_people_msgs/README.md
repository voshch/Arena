# arena_people_msgs

rosidl interfaces for pedestrian state and control.

`BaseHumanSimulator` subclasses reach the physics engine through
`PedestrianITF` (`pedestrian_spawn`/`move`/`delete`/`update`). With an
out-of-process engine (Isaac Sim) that crosses the wire as the services
below, engine side hosting the server. `Move` is a teleport with phase reset,
`Update` is smooth continuous motion. The base human simulator also reuses
`MovePedestrians` as the env-level `human/move` teleport request, reading
only `name` and `pose`.

## Services (`srv/`)

| File | Purpose |
|---|---|
| `SpawnPedestrians.srv` | Create pedestrian actors from `SpawnPedestrian` descriptors. |
| `DeletePedestrians.srv` | Remove pedestrian actors by name. |
| `MovePedestrians.srv` | Explicit teleport: receivers reset animation phase and trajectory state so actors do not blink mid-stride. |
| `UpdatePedestrians.srv` | Continuous per-tick state: receivers treat it as smooth motion, no animation-phase reset, no trajectory re-seed. Isaac's handler reads `joint_state` verbatim from the request. |

## Messages (`msg/`)

| File | Purpose |
|---|---|
| `Pedestrian.msg` | Standard pedestrian representation: `name`/`id`, `pose`/`twist`, `animation_state` (`IDLE`/`WALKING`/`RUNNING`/`PANIC`/`SURPRISED`/`CURIOUS`/`THREATENING`), `gesture`, `joint_state` (empty = synthesize fallback gait, non-empty = upstream override), `gait_phase`, `model_uri`. |
| `SpawnPedestrian.msg` | One pedestrian to spawn: a `Pedestrian` plus `model_ref`, a backend-interpreted asset reference (Gazebo: SDF path or inline XML, Isaac: character name). |
| `Gesture.msg` | Upper-body gesture intent: `kind` (registered gesture name, empty = none), `at` (world-frame target point) and `opts` (JSON object string with per-kind options, empty = defaults). Composes with `animation_state`, resolved to joints by the animation layer. |
| `Pedestrians.msg` | `Pedestrian[]` with a `Header`, the wire type for the `arena_peds` bus and the `human_steering` GUI's `human/stream`. |
