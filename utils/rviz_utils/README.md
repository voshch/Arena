# rviz_utils

ROS4HRI visualization bridge and rviz config utilities for Arena.

## hri_producer

[`rviz_utils/scripts/hri_producer.py`](rviz_utils/scripts/hri_producer.py)

Subscribes `<env_ns>/arena_peds` and projects each pedestrian into the
REP-155 `<env_ns>/humans/` namespace: tracked-id lists, per-person engagement,
per-body `joint_states`, URDF latched on `bodies/<id>/urdf`, and TF
`body_<id>`.  Drives a pool of `robot_state_publisher` subprocesses
([`hri/body_pool.py`](rviz_utils/hri/body_pool.py)) so `hri_rviz/Skeletons3D`
can render animated skeletons.

**Relay mode (primary path).** When `arena_peds.joint_state.name` is non-empty,
`hri_producer` re-suffixes each bare semantic joint name with the body ID
(`<joint>_<body_id>`) and publishes directly.  The joint data originates from
`GaitGenerator` inside `BaseHumanSimulator.publish_arena_peds`.

**Fallback gait.** When `joint_state` arrives empty (e.g. from the Isaac
adapter, which does not fill joint_state on the bus), `hri_producer` runs its
own `GaitGenerator` instance to synthesize a gait from the pedestrian's speed
and `animation_state`.  The fallback applies per-body in the same message, so
mixed messages (some peds with joint_state, some without) are handled correctly.

Joint name convention: bare names on the bus (`l_r_hip`, `l_knee`, ...) are
suffixed per body before publishing to `joint_states` so they match the
`human_description` URDF rig.  See
[`task_generator/simulators/human/gait.py`](../../task_generator/task_generator/simulators/human/gait.py)
for the full 20-joint semantic name set.

## rviz config generation

[`rviz_utils/scripts/rviz_config.py`](rviz_utils/scripts/rviz_config.py) renders
[`config/rviz_default.rviz`](config/rviz_default.rviz) for one env and launches rviz2 on it
(`arena viz`). The `view:=map` camera is framed on the env's static map: the generator waits
up to 5 s for the latched `<ns>/map` grid and orbits its centre at 1.45 m of distance per
metre of map span, so a 20 m office and a 35 m hospital both fill the viewport. Without a map
it falls back to the fixed frame (focal (15, 10), distance 50).
