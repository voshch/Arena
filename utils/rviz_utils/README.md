# rviz_utils

ROS4HRI visualization bridge and rviz config utilities for Arena.

## hri_producer

[`rviz_utils/scripts/hri_producer.py`](rviz_utils/scripts/hri_producer.py)

Subscribes `<env_ns>/arena_peds` and projects each pedestrian into the
REP-155 `<env_ns>/humans/` namespace: tracked-id lists, per-person engagement,
per-body `joint_states`, URDF latched on `bodies/<id>/urdf`, and TF
`body_<id>`. [`hri/body_frames.py`](rviz_utils/hri/body_frames.py) computes the
joint frames in-process by the rules of `robot_state_publisher` (moving joints per
body at most every 50 ms of the tick's stamp, fixed joints latched on `/tf_static`), so
`hri_rviz/Skeletons3D` can render animated skeletons without a process or DDS
participant per body. Only the first `max_bodies` (32) of an env get joint frames, the
others keep their URDF, root frame and face.

Body and joint frames are published on `<env_ns>/humans/tf`. Nodes that
listen only to the env's `/tf` never receive them. `hri_producer` subscribes
to `arena_peds` only while `humans/tf` or `humans/faces/tracked` has a subscriber
(checked once per second), so an unwatched env costs it no roster work and no
joint frames. `bodies/<id>/joint_states` is sent only while it has a subscriber. A consumer that needs pedestrians in the world tree subscribes
both: `rviz_config` relays the env `/tf` and `<env_ns>/humans/tf` into
`<env_ns>/viewer/tf` for rviz, and the evaluation recorder merges both into
`/tf` in the bag.

**Faces.** Each body is also a REP-155 face with the same id, listed on
`humans/faces/tracked`. While a face's `humans/faces/<id>/view` has a subscriber, `hri_producer` computes `face_<id>` (origin between the two URDF eye
visuals of `head_<id>`, x forward, z up) by forward kinematics of the neck chain
from the tick's rig joint positions ([`hri/face.py`](rviz_utils/hri/face.py)), adds
`gaze_<id>` (same origin, optical convention) and publishes both on the global
`/tf`, plus `humans/tf` when the env remaps its `/tf`. The view (`arena_people_msgs/FaceView`) carries a
square pinhole `CameraInfo` of `view.fov_deg` (launch argument, default 60) in `gaze_<id>` and
`clip_near`, `view.clip_near` (default 0.15 m), the near clip that hides the
pedestrian's own head but not its hands. `arena cam pov` and
the `cam drive` panel look through it.

**Body model.** Each body's URDF comes from `rviz_utils.hri.urdf`, built from
the body rig template (`arena_simulation_setup` `configs/body/template.yaml`):
root `body_<id>` at the pelvis, a three-axis ball per rig joint (links
`<joint>_<id>`, joints `<joint>_<rx|ry|rz>_<id>`, fingers included), the root
offset on `root_<x|y|z>_<id>`, and the REP-155 body frames (`waist`, `torso`,
`head`, `l|r_shoulder`, `elbow`, `wrist`, `hip`, `knee`, `ankle`) as fixed
links. `hri_producer` turns `Pedestrian.body` into those joint positions.

**Fallback gait.** A pedestrian without a body pose gets a gait from
`hri_producer`'s own `GaitGenerator`, from its speed and `animation_state`,
posed over its model's stance. Mixed messages are handled per body.
