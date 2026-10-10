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

Body and joint frames are published on `<env_ns>/humans/tf`. Nodes that
listen only to the env's `/tf` never receive them. `hri_producer` subscribes
to `arena_peds` only while `humans/tf` or `humans/faces/tracked` has a subscriber
(checked once per second), so an unwatched env costs it no roster work and no
`joint_states` traffic. A consumer that needs pedestrians in the world tree subscribes
both: `rviz_config` relays the env `/tf` and `<env_ns>/humans/tf` into
`<env_ns>/viewer/tf` for rviz, and the evaluation recorder merges both into
`/tf` in the bag.

**Faces.** Each body is also a REP-155 face with the same id, listed on
`humans/faces/tracked`. While a face's `humans/faces/<id>/view` has a subscriber, `hri_producer` computes `face_<id>` (origin between the two URDF eye
visuals of `head_<id>`, x forward, z up) by forward kinematics of the neck chain
from the tick's rig joint positions ([`hri/face.py`](rviz_utils/hri/face.py)), adds
`gaze_<id>` (same origin, optical convention) and publishes both on the global
`/tf`, plus `humans/tf` when the env remaps its `/tf`. The view (`arena_people_msgs/FaceView`) carries a
square pinhole `CameraInfo` of `view.fov_deg` (default 60) in `gaze_<id>` and
`clip_near`, `view.clip_near` (default 0.15 m), the near clip that hides the
pedestrian's own head but not its hands. `arena cam pov` and
the `cam drive` panel look through it.

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
