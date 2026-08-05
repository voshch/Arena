# Human skeleton joint contract

Frozen interface shared by the gait generator (`GaitGenerator`), the HRI producer node,
and every pedestrian renderer (RViz, Gazebo, Isaac). `Pedestrian.joint_state` is the
animation single source of truth (SSOT): it carries the 24 base joint names from
`GaitGenerator.JOINT_NAMES`, and every renderer resolves its own convention from that
one field.

## 1. Wire contract (normative)

### Naming convention

Every joint/link carries a `_<ID>` suffix where **`<ID> = str(pedestrian.id)`**.
`sensor_msgs/JointState.name[i]` MUST be the full suffixed name (e.g. `l_r_hip_7`) so it
matches the body's generated URDF. The gait generator takes the agent id and suffixes it.

### Publish all 24

Publish a position for all 24 base joints each tick (unanimated -> 0.0) so
`robot_state_publisher` does not warn. Fixed joints (`torso`, `head`, `l/r_wrist`) are
not part of the contract and never appear in `JointState`. Per-joint advisory limits
live in `GaitGenerator.LIMITS`: generators may clamp their output to them, the stream
and presentation layers never enforce them. Outside the shoulder triples they match the
Section 2 table.

### Value semantics

For every joint except the two shoulder triples, wire values are identical to the
ros4hri `human_description` URDF interpretation of that joint's axis (Section 2 states
each joint's raw meaning). This includes `r_waist`/`y_waist` and `l/r_ankle`: their
body-aligned URDF axes already coincide with the wire semantic.

The shoulder triples (`l/r_y_shoulder`, `l/r_p_shoulder`, `l/r_r_shoulder`) are the one
exception: their wire values are anatomical, not raw URDF axis values.

- `p_shoulder`: sagittal flexion of the whole arm. Positive = forward, same sign
  convention on both sides. Antiphase is baked into the emitted values (`GaitGenerator`
  emits `l = A*sin(phi+pi)`, `r = A*sin(phi)`), the sign convention itself is not
  mirrored.
- `y_shoulder` / `r_shoulder`: arm azimuth and axial twist, Section 1a.

### Section 1a: shoulder triple composition (both sides identical sign convention)

The wire triple (y, p, r) defines the rotation of the upper arm relative to the torso
frame (REP-155 body-aligned: X fwd, Y left, Z up), composed intrinsically:

```
R_arm = Rz(y) * R_axis((0,-1,0), p) * R_axis(d, r)
where d = Rz(y) * R_axis((0,-1,0), p) applied to (0,0,-1)   (current limb long axis)
```

- `p`: sagittal flexion, positive forward.
- `y`: azimuth about body-up, positive CCW from above, SAME sign both sides (renderers
  own any mirroring, the wire never mirrors).
- `r`: twist about the limb long axis, positive right-handed about `d` (distal
  direction), same sign both sides. Anatomical internal/external meaning therefore
  mirrors between sides, this is deliberate, the wire is geometric.
- Rest pose `(0,0,0)` = arm hanging. Singularity of the decomposition at `p=0` affects
  only producers that EXTRACT angles (converters), never renderers that compose.
  Extractors must regularize (warm start + damping).

Consumers must not interpret the shoulder-triple values through the raw URDF axes in
Section 2's table, that mapping is the RViz adapter's job, not the contract's.

### Torso triple

`r_waist`/`y_waist`/`waist` mirror the head triple's order (roll, yaw, pitch). URDF
chain: `body -> r_waist -> y_waist -> waist -> torso`, all joint origins `rpy="0 0 0"`
(frames body-aligned at rest). Axes: `r_waist` `(1,0,0)`, `y_waist` `(0,0,1)`, `waist`
`(0,1,0)`. With body-aligned frames the raw URDF axis meaning coincides with the wire
semantic, so the RViz adapter passes all three through untouched (same as the head
triple). Legs stay parented to `body`: `y_waist` is exactly the body-torque seam (torso
turns, pelvis and legs do not).

### Ankles

`l_ankle`/`r_ankle` are revolute about `(0,-1,0)`, same sagittal family as knees and
elbows. Positive lifts the toes (dorsiflexion).

### Gait synthesis

Phase `phi` per agent integrates from speed: `phi += 2*pi*cadence*dt`, where
`cadence ~= clamp(0.4 + 0.55*speed, 0.4, 2.2)` Hz (frozen: the Gazebo plugin's phase
lock mirrors this formula). Legs antiphase, arms contralateral (arm swings with the
opposite leg).

- **walk** (`WALKING`, speed-scaled gain `g = clamp(speed/1.2, 0.2, 1.0)`): per-joint
  profiles baked from the polished CMU 12_01 clip (`arena_humans` posture pipeline),
  mean + 3 sine harmonics per signal, all scaled by `g` (`_WALK_PROFILE` in `gait.py`).
  Limb pairs share one canonical profile with the right side evaluated at `phi + pi`,
  so L/R antiphase is exact by construction. Antiphase means `l(phi) = r(phi + pi)`,
  NOT `l = -r`: the profiles carry nonzero means (hips average forward-flexed, elbows
  ~18 deg bent). Torso roll/yaw, ankle, and shoulder azimuth/twist DOFs stay 0.0.
- **run** (`RUNNING`): the same profiles at 1.6x amplitude, `cadence` higher.
- **idle** (`IDLE` and the behavior states `PANIC/SURPRISED/CURIOUS/THREATENING` for now):
  near-zero limbs, tiny breathing sway `waist = 0.03*sin(phi)` and a slow gaze wander
  `y_head = 0.06*sin(0.3*phi)`, `p_head = 0.02*sin(0.5*phi + 1.0)`. (Behavior
  states get richer posture later, baseline treats them as idle for the rig.)

State selection comes from `Pedestrian.animation_state`. Speed is `hypot(twist.linear.x, y)`.

### Determinism

Per-agent phase keyed by id, cleared on despawn. Seed initial `phi` from `id`
(e.g. `(id % 360) * pi/180`) so agents are not in lockstep. Read `dt`, never wall-clock.

## 2. ros4hri URDF rendering (RViz adapter)

The ros4hri `human_description` package is a vendored fork at
`src/deps/human_description` (branch `arena-v2`) carrying the torso triple and the
revolute ankles. It preserves the stock ros4hri link/frame set.

### URDF generation (per body)

The producer generates one URDF per body via xacro:

```
xacro <share>/human_description/urdf/human-tpl.xacro id:=<ID> height:=<H>
```

- Only `id` (string) and `height` (float, default `1.65`) are real xacro args, the other
  proportion knobs in `create_human_urdf.py` do not map to xacro args and are inert.
- The generated URDF is set as ROS param `human_description_<ID>` and consumed by a
  `robot_state_publisher` for that body. Root link is `body_<ID>` (REP-155 hip-origin frame).

### Mirror convention

Every joint origin in `human-tpl.xacro` is `rpy="0 0 0"`, so all link frames are
body-aligned at rest (REP-155: X forward, Y left, Z up).

The arm macro reflects axes per side: `l_p_shoulder` axis `(1,0,0)`, `r_p_shoulder`
axis `(-1,0,0)`, `l_y_shoulder` `(0,0,-1)`, `r_y_shoulder` `(0,0,1)`, `l_r_shoulder`
`(0,0,1)`, `r_r_shoulder` `(0,0,-1)`. So the raw URDF meaning of `p_shoulder` is
**lateral abduction** (rotation about body-forward X), positive = outward on both
sides, not a sagittal swing.

Leg sagittal joints are not reflected: `l_r_hip`/`r_r_hip` are both `(0,-1,0)`,
positive = forward. Knees, elbows, and ankles are `(0,-1,0)` on both sides. `l_p_hip`/
`r_p_hip` are mirrored (`(1,0,0)`/`(-1,0,0)`), positive = abduct outward on both sides,
which coincides with the semantic meaning (gait idles both at `0.0`), so only the
shoulder triples need the Section 1 exception. The torso triple (`r_waist`/`y_waist`/
`waist`) is not reflected either: single links on the midline, same axes as the head
triple.

### Articulated joints (24 revolute, all others fixed)

| # | base name | axis | limits [lo, hi] (rad) | role |
|---|---|---|---|---|
| 1 | `r_waist` | (1,0,0) | [-0.6, 0.6] | torso roll relative to pelvis |
| 2 | `y_waist` | (0,0,1) | [-0.8, 0.8] | torso yaw relative to pelvis |
| 3 | `waist` | (0,1,0) | [-0.2, 1.0] | torso forward lean |
| 4 | `r_head` | (1,0,0) | [-1.0, 1.0] | head roll |
| 5 | `y_head` | (0,0,1) | [-1.4, 1.4] | head yaw |
| 6 | `p_head` | (0,-1,0) | [-1.5, 1.5] | head pitch |
| 7 | `l_y_shoulder` | (0,0,-1) | [-1.1, 1.9] | L shoulder yaw |
| 8 | `l_p_shoulder` | (1,0,0) | [-0.4, 3.3] | **L arm abduction (raw axis)** * |
| 9 | `l_r_shoulder` | (0,0,1) | [-1.7, 1.5] | L shoulder roll |
| 10 | `l_elbow` | (0,-1,0) | [0.0, 2.5] | **L elbow** |
| 11 | `r_y_shoulder` | (0,0,1) | [-1.1, 1.9] | R shoulder yaw |
| 12 | `r_p_shoulder` | (-1,0,0) | [-0.4, 3.3] | **R arm abduction (raw axis)** * |
| 13 | `r_r_shoulder` | (0,0,-1) | [-1.7, 1.5] | R shoulder roll |
| 14 | `r_elbow` | (0,-1,0) | [0.0, 2.5] | **R elbow** |
| 15 | `l_y_hip` | (0,0,-1) | [-0.1, 0.6] | L hip yaw |
| 16 | `l_p_hip` | (1,0,0) | [-0.4, 3.3] | L hip abduction |
| 17 | `l_r_hip` | (0,-1,0) | [-0.4, 0.7] | **L leg sagittal swing** |
| 18 | `l_knee` | (0,-1,0) | [-2.5, 0.0] | **L knee** |
| 19 | `r_y_hip` | (0,0,-1) | [-0.1, 0.6] | R hip yaw |
| 20 | `r_p_hip` | (-1,0,0) | [-0.4, 3.3] | R hip abduction |
| 21 | `r_r_hip` | (0,-1,0) | [-0.4, 0.7] | **R leg sagittal swing** |
| 22 | `r_knee` | (0,-1,0) | [-2.5, 0.0] | **R knee** |
| 23 | `l_ankle` | (0,-1,0) | [-0.9, 0.6] | **L ankle sagittal (dorsiflexion)** |
| 24 | `r_ankle` | (0,-1,0) | [-0.9, 0.6] | **R ankle sagittal (dorsiflexion)** |

Bold = the gait-driving DOFs. * Raw URDF axis meaning, the wire-contract value carried
in this DOF is anatomical flexion (Section 1a), the `rig.py` adapter below performs the
conversion. Declared shoulder limits are independent of the wire advisory limits in
`GaitGenerator.LIMITS`: `y_u`/`r_u` on `l/r_y_shoulder`/`l/r_r_shoulder` are
ZXZ-extraction outputs, not passthrough of the wire `y`/`r`.

### `rig.py` adapter obligation

`rviz_utils`' `hri_producer` translates Section 1's semantic wire values into this raw
URDF frame before `robot_state_publisher` sees them, via `rviz_utils/hri/rig.py`.
`rig.py` stays stateless. Its obligations per DOF group:

- **Torso triple and ankles**: passthrough, the raw URDF axis meaning already coincides
  with the wire semantic (see Torso triple / Ankles above and Mirror convention).
- **Shoulder triples**: the only math. `rig.py` composes `R_arm` from the wire triple
  (Section 1a) and extracts the URDF chain values via closed-form ZXZ Euler extraction:
  left `Rz(-y_u) Rx(p_u) Rz(r_u)`, right `Rz(y_u) Rx(-p_u) Rz(-r_u)`. Each side solves
  its own `(y_u, p_u, r_u)`, the two triples are not shared. Every rotation has two ZXZ
  decompositions, `(a, b, c)` and `(a+pi, -b, c+pi)`. The canonical middle-angle-in-
  `[0,pi]` pick lands on `a=-pi/2` for backward flexion, so `rig.py` flips to the second
  branch whenever the canonical first angle is negative, which keeps `a=c=pi/2` across
  the whole `y=r=0` family regardless of the sign of `p`.

  Degenerate branch (`|sin(middle angle)| ~ 0`): fold the full Z-rotation into the THIRD
  angle and hold the first at `pi/2`. For both sides this is the ZXZ conjugation
  `Rz(-pi/2) * Rx(f) * Rz(pi/2) = R((0,-1,0), f)`: the constant `pi/2` pre/post twists
  turn the abduction DOF (`Rx`) into a rotation about `(0,-1,0)`, the same sagittal axis
  already used by the hip/knee/elbow/ankle joints.

  `robot_state_publisher` does not enforce URDF joint limits, so the exact `pi/2` on
  `r_shoulder` (declared limit `1.5`) is intended, not a bug.

## 3. Other renderers

- **Isaac**: `peds` `bone_map.py` speaks the semantic convention (same-sign flexion,
  antiphase baked into the values). `BONE_MAP` is regenerated from the contract rig by
  `tests/peds/test_bone_map_parity.py`: shoulder azimuth/twist on the Arm bones,
  `waist`/`r_waist`/`y_waist` each spread over the LowerBack/Spine/Spine1 chain with
  0.5/0.3/0.2 weights, head DOFs over neck plus head, ankles on the Foot bones. Axes
  come from the measured-probe procedure, not eyeballing. `ExternalPoseProvider`
  replaces mapped bones with the wire pose instead of composing it over the walking
  clip.
- **Gazebo**: clip fidelity only. gz-sim 8 actors expose no per-bone skeleton control
  (see `arena_gz_plugins` `PedSkeletonPlugin.cc` header), the plugin follows
  `animation_state`/pose and ignores `joint_state` by design. Full per-bone motion
  parity for gz is a separate clip-export track.
