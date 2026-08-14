# Human skeleton joint contract

Frozen interface shared by the gait generator (`GaitGenerator`), the HRI producer node,
and every pedestrian renderer (RViz, Gazebo, Isaac). `Pedestrian.joint_state` is the
animation single source of truth (SSOT): it carries the 30 base joint names from
`GaitGenerator.JOINT_NAMES`, and every renderer resolves its own convention from that
one field. (Contract v2: the single torso triple became the three-triple spine stack
of the Spine stack section. Every pre-stack recording carries 24 names and needs the
zero-fill migration described there.)

## 1. Wire contract (normative)

### Naming convention

Every joint/link carries a `_<ID>` suffix where **`<ID> = str(pedestrian.id)`**.
`sensor_msgs/JointState.name[i]` MUST be the full suffixed name (e.g. `l_r_hip_7`) so it
matches the body's generated URDF. The gait generator takes the agent id and suffixes it.

### Publish all 36

Publish a position for all 36 base joints each tick (unanimated -> 0.0) so
`robot_state_publisher` does not warn. Fixed joints (`torso`, `head`, `l/r_wrist`) are
not part of the contract and never appear in `JointState`. Per-joint advisory limits
live in `GaitGenerator.LIMITS`: generators may clamp their output to them, the stream
and presentation layers never enforce them. Outside the shoulder triples they match the
Section 2 table.

### Value semantics

For every joint except the two shoulder triples, wire values are identical to the
ros4hri `human_description` URDF interpretation of that joint's axis (Section 2 states
each joint's raw meaning). This includes the whole spine stack (`r/y_waist`,
`r/y_spine`, `spine`, `r/y_chest`, `chest`) and `l/r_ankle`: their body-aligned URDF
axes already coincide with the wire semantic.

The shoulder triples (`l/r_y_shoulder`, `l/r_p_shoulder`, `l/r_r_shoulder`) are the one
exception: their wire values are anatomical, not raw URDF axis values.

- `p_shoulder`: sagittal flexion of the whole arm. Positive = forward, same sign
  convention on both sides. Antiphase is baked into the emitted values (`GaitGenerator`
  emits `l = A*sin(phi+pi)`, `r = A*sin(phi)`), the sign convention itself is not
  mirrored.
- `y_shoulder` / `r_shoulder`: arm azimuth and axial twist, Section 1a.

### Section 1a: shoulder triple composition (both sides identical sign convention)

The wire triple (y, p, r) defines the rotation of the upper arm relative to its
parent COLLAR frame (v3: body-aligned at collar rest, REP-155 X fwd, Y left, Z up),
composed intrinsically:

```
R_arm = Rz(y) * R_axis((0,-1,0), p) * R_axis((0,0,-1), r)
```

`(0,0,-1)` is the rest limb axis. As the last INTRINSIC factor the `r` rotation is a
pure twist, equivalent to the world-frame rotation `R_axis(d, r)` applied after the
first two, `d = R_arm applied to (0,0,-1)` (current limb long axis). Composing
`R_axis(d, r)` with world-frame `d` as the last factor instead is NOT a twist: it
swings the bone tip once y or p is nonzero.

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

### Spine stack (waist / spine / chest triples)

The torso is three stacked segments of `torso_height/3` each, every segment carrying a
roll/yaw/pitch triple in the head triple's order: `r_waist`/`y_waist`/`waist` at the
pelvis seam, `r_spine`/`y_spine`/`spine` one segment up, `r_chest`/`y_chest`/`chest`
two segments up. URDF chain:
`body -> r_waist -> y_waist -> waist -> r_spine -> y_spine -> spine -> r_chest ->
y_chest -> chest -> torso`, ALL joint origins `rpy="0 0 0"` (every frame body-aligned
at rest). Axes per cluster: roll `(1,0,0)`, yaw `(0,0,1)`, pitch `(0,1,0)`. With
body-aligned frames the raw URDF axis meaning coincides with the wire semantic, so the
RViz adapter passes all nine through untouched (same as the head triple) -- new spine
DOFs MUST keep this property: a wire DOF that needs a composition adapter is a design
smell (see the Section 1a shoulder history). Arms and head keep parenting to `torso`
(now the top of the chest segment), legs stay parented to `body`: `y_waist` remains
the body-torque seam (torso turns, pelvis and legs do not).

Segment values are LOCAL (each relative to the segment below), so the chest world
rotation is the composition of all three triples. A producer with only a lumped torso
estimate puts it on the waist triple and zeros spine/chest (exact wire-v1 rendering).
Distributing a lumped bend as waist 0.5 / spine 0.3 / chest 0.2 reproduces what the
Isaac renderer's weighted spread used to fake. Migration of v1 (24-name) recordings:
insert the six spine/chest DOFs as 0.0.

### Collars (v3)

`l/r_y_collar`, `l/r_p_collar`: two DOFs per clavicle, joints collocated at the
sternum (torso origin), the shoulder socket hangs off `p_collar` at the clavicle
length. `y_collar` is protraction (positive swings the socket FORWARD, both sides),
`p_collar` is elevation/shrug (positive up, both sides). Axes are mirrored in the
URDF (`y`: `(0,0,-1)` left / `(0,0,1)` right, `p`: `(1,0,0)` left / `(-1,0,0)`
right) exactly like `p_hip`, so the raw URDF meaning coincides with the wire
semantic and the RViz adapter passes both through untouched. The shoulder triple
(Section 1a) composes on top of the collar frame.

### Ankles

`l/r_y_ankle` is foot yaw (toe direction) about `(0,0,-1)`, the same unmirrored
axis family as `y_hip`, applied BEFORE the sagittal ankle in the chain
(`knee -> y_ankle -> ankle`). `l_ankle`/`r_ankle` are revolute about
`(0,-1,0)`, same sagittal family as knees and elbows. Positive lifts the toes
(dorsiflexion).

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
  ~18 deg bent). Torso roll/yaw, the spine/chest triples, collars, ankles (both
  DOFs), and shoulder azimuth/twist stay 0.0 (the baked profile predates the spine
  stack and stays lumped on the waist triple).
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
`src/deps/human_description` (branch `arena-v2`) carrying the spine stack, the
collars, and the two-DOF ankles. It preserves the stock ros4hri link/frame set.

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
`(0,0,1)`, `r_r_shoulder` `(0,0,-1)`, and the collar pair per the Collars section.
So the raw URDF meaning of `p_shoulder` is **lateral abduction** (rotation about
body-forward X), positive = outward on both sides, not a sagittal swing.

Leg sagittal joints are not reflected: `l_r_hip`/`r_r_hip` are both `(0,-1,0)`,
positive = forward. Knees, elbows, and ankles are `(0,-1,0)` on both sides. `l_p_hip`/
`r_p_hip` are mirrored (`(1,0,0)`/`(-1,0,0)`), positive = abduct outward on both sides,
which coincides with the semantic meaning (gait idles both at `0.0`), so only the
shoulder triples need the Section 1 exception. The spine stack is not reflected
either: single links on the midline, same axes as the head triple.

### Articulated joints (36 revolute, all others fixed)

| # | base name | axis | limits [lo, hi] (rad) | role |
|---|---|---|---|---|
| 1 | `r_waist` | (1,0,0) | [-0.6, 0.6] | lumbar roll relative to pelvis |
| 2 | `y_waist` | (0,0,1) | [-0.8, 0.8] | lumbar yaw relative to pelvis |
| 3 | `waist` | (0,1,0) | [-0.2, 1.0] | lumbar forward lean |
| 4 | `r_spine` | (1,0,0) | [-0.3, 0.3] | mid-spine roll (local) |
| 5 | `y_spine` | (0,0,1) | [-0.4, 0.4] | mid-spine yaw (local) |
| 6 | `spine` | (0,1,0) | [-0.1, 0.5] | mid-spine forward lean (local) |
| 7 | `r_chest` | (1,0,0) | [-0.3, 0.3] | chest roll (local) |
| 8 | `y_chest` | (0,0,1) | [-0.4, 0.4] | chest yaw (local) |
| 9 | `chest` | (0,1,0) | [-0.1, 0.5] | chest forward lean (local) |
| 10 | `r_head` | (1,0,0) | [-1.0, 1.0] | head roll |
| 11 | `y_head` | (0,0,1) | [-1.4, 1.4] | head yaw |
| 12 | `p_head` | (0,-1,0) | [-1.5, 1.5] | head pitch |
| 13 | `l_y_collar` | (0,0,-1) | [-0.5, 0.5] | L clavicle protraction |
| 14 | `l_p_collar` | (1,0,0) | [-0.2, 0.6] | L clavicle elevation (shrug) |
| 15 | `l_y_shoulder` | (0,0,-1) | [-1.1, 1.9] | L shoulder yaw |
| 16 | `l_p_shoulder` | (1,0,0) | [-0.4, 3.3] | **L arm abduction (raw axis)** * |
| 17 | `l_r_shoulder` | (0,0,1) | [-1.7, 1.5] | L shoulder roll |
| 18 | `l_elbow` | (0,-1,0) | [0.0, 2.5] | **L elbow** |
| 19 | `r_y_collar` | (0,0,1) | [-0.5, 0.5] | R clavicle protraction |
| 20 | `r_p_collar` | (-1,0,0) | [-0.2, 0.6] | R clavicle elevation (shrug) |
| 21 | `r_y_shoulder` | (0,0,1) | [-1.1, 1.9] | R shoulder yaw |
| 22 | `r_p_shoulder` | (-1,0,0) | [-0.4, 3.3] | **R arm abduction (raw axis)** * |
| 23 | `r_r_shoulder` | (0,0,-1) | [-1.7, 1.5] | R shoulder roll |
| 24 | `r_elbow` | (0,-1,0) | [0.0, 2.5] | **R elbow** |
| 25 | `l_y_hip` | (0,0,-1) | [-0.1, 0.6] | L hip yaw |
| 26 | `l_p_hip` | (1,0,0) | [-0.4, 3.3] | L hip abduction |
| 27 | `l_r_hip` | (0,-1,0) | [-0.4, 0.7] | **L leg sagittal swing** |
| 28 | `l_knee` | (0,-1,0) | [-2.5, 0.0] | **L knee** |
| 29 | `r_y_hip` | (0,0,-1) | [-0.1, 0.6] | R hip yaw |
| 30 | `r_p_hip` | (-1,0,0) | [-0.4, 3.3] | R hip abduction |
| 31 | `r_r_hip` | (0,-1,0) | [-0.4, 0.7] | **R leg sagittal swing** |
| 32 | `r_knee` | (0,-1,0) | [-2.5, 0.0] | **R knee** |
| 33 | `l_y_ankle` | (0,0,-1) | [-0.6, 0.6] | L foot yaw (toe direction) |
| 34 | `l_ankle` | (0,-1,0) | [-0.9, 0.6] | **L ankle sagittal (dorsiflexion)** |
| 35 | `r_y_ankle` | (0,0,-1) | [-0.6, 0.6] | R foot yaw (toe direction) |
| 36 | `r_ankle` | (0,-1,0) | [-0.9, 0.6] | **R ankle sagittal (dorsiflexion)** |

The spine/chest limits are per-segment halves of the lumped waist range: the stack's
advisory sum intentionally exceeds the old lumped range (real spines bend further
distributed than a single lumbar hinge).

Bold = the gait-driving DOFs. * Raw URDF axis meaning, the wire-contract value carried
in this DOF is anatomical flexion (Section 1a), the `rig.py` adapter below performs the
conversion. Declared shoulder limits are independent of the wire advisory limits in
`GaitGenerator.LIMITS`: `y_u`/`r_u` on `l/r_y_shoulder`/`l/r_r_shoulder` are
ZXZ-extraction outputs, not passthrough of the wire `y`/`r`.

### `rig.py` adapter obligation

`rviz_utils`' `hri_producer` translates Section 1's semantic wire values into this raw
URDF frame before `robot_state_publisher` sees them, via `rviz_utils/hri/rig.py`.
`rig.py` stays stateless. Its obligations per DOF group:

- **Spine stack (all nine DOFs), collars, and ankles (both DOFs)**: passthrough,
  the raw URDF axis meaning already coincides with the wire semantic (see Spine
  stack / Collars / Ankles above and Mirror convention).
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
  `tests/peds/test_bone_map_parity.py`: shoulder azimuth/twist on the Arm bones, the
  spine stack 1:1 onto the LowerBack/Spine/Spine1 chain (waist triple -> LowerBack,
  spine triple -> Spine, chest triple -> Spine1 -- the pre-stack 0.5/0.3/0.2 weighted
  spread of the lumped waist is retired), head DOFs over neck plus head, ankles on the
  Foot bones. Axes come from the measured-probe procedure, not eyeballing.
  `ExternalPoseProvider` replaces mapped bones with the wire pose instead of composing
  it over the walking clip.
- **Gazebo**: clip fidelity only. gz-sim 8 actors expose no per-bone skeleton control
  (see `arena_gz_plugins` `PedSkeletonPlugin.cc` header), the plugin follows
  `animation_state`/pose and ignores `joint_state` by design. Full per-bone motion
  parity for gz is a separate clip-export track.
