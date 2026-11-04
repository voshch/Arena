# Human skeleton joint contract

The 52 semantic joints every pose producer speaks: the gait generator (`GaitGenerator`), gestures,
clips and possession. Each tick the publisher compiles them into the wire pose `Pedestrian.body`
over the ped's bundle ([BODY.md](BODY.md)), which every renderer reads. (Contract v2: the single
torso triple became the three-triple spine stack of the Spine stack section. Every pre-stack
recording carries 24 names and needs the zero-fill migration described there. Contract v4 added
the two-DOF wrists, every pre-v4 recording carries 36 names and loads with the wrists at 0.0.
Contract v5 added the twelve finger DOFs, every pre-v5 recording carries 40 names and loads with
the fingers at 0.0.)

## 1. Semantic contract (normative)

### Naming convention

Every joint/link carries a `_<ID>` suffix where **`<ID> = str(pedestrian.id)`**.
`sensor_msgs/JointState.name[i]` MUST be the full suffixed name (e.g. `l_r_hip_7`) so it
matches the body's generated URDF. The gait generator takes the agent id and suffixes it.

### Publish all 52

Publish a position for all 52 base joints each tick (unanimated -> 0.0) so
every joint frame stays on `humans/tf`. Fixed joints (`torso`, `head`) are
not part of the contract and never appear in `JointState`. Per-joint advisory limits
live in `GaitGenerator.LIMITS`: generators may clamp their output to them, the stream
and presentation layers never enforce them. Outside the shoulder triples they match the
Section 2 table.

### Value semantics

For every joint except the two shoulder triples, a wire value is a rotation about that
joint's raw axis (Section 2 states each joint's raw meaning). This includes the whole spine stack (`r/y_waist`,
`r/y_spine`, `spine`, `r/y_chest`, `chest`), `l/r_ankle` and the wrists: their body-aligned URDF
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

### Wrists (v4)

Rest pose has the palms facing the thighs. `l/r_r_wrist` is forearm twist about the
limb axis, positive pronation (the palm turns backward), applied BEFORE the flexion in
the chain (`elbow -> r_wrist -> wrist`). `l/r_wrist` is flexion/extension, positive
extension (the hand swings outward). Both are mirrored in the URDF like `p_hip`
(twist `(0,0,-1)` left / `(0,0,1)` right, flexion `(1,0,0)` left / `(-1,0,0)` right),
so the raw URDF meaning coincides with the wire semantic on both sides. The `wrist`
link keeps its ros4hri name and carries the hand.

### Fingers (v5)

`l/r_curl_thumb`, `l/r_curl_index`, `l/r_curl_middle`, `l/r_curl_ring`, `l/r_curl_pinky` and
`l/r_spread`: twelve DOFs, six per hand, on the digit joints of [BODY.md](BODY.md). The rest
hand has straight fingers beside each other, thumb forward, palm toward the thigh.
`curl_<digit>` flexes the whole digit toward the palm (positive = closing, both sides) and
sets the base and middle knuckles to its angle and the tip knuckle to 0.7 of it. `spread` fans the four
fingers apart in the palm plane (positive = index toward the thumb, pinky away, both sides) with
the weights 1, 1/3, -1/3 and -1 on the knuckles of index, middle, ring and pinky. The thumb has
no spread. Axes are mirrored like the collars (`curl`: `(-1,0,0)` left / `(1,0,0)` right, thumb
`(0,0,-1)` left / `(0,0,1)` right, `spread`: `(0,-1,0)` both sides, at the hanging rest), so the wire is geometric. Recordings that predate them
carry no finger values and are read as 0.0 (the loaders zero-fill). The fingers hold the rest hand
unless a playing body clip declares a shape in `hands.py` (handshake grip, hug, counting) or a point
gesture curls every digit of its arm's hand but the index, each fading with its slot weight.

### Driven joints

A ped whose model ships a `.rig` with drivers carries those joints
after the 52, named by the driver keys (wheelchair bundles: `l_wheel`, `r_wheel`). A
driven joint is turned by the ped's motion, not by the gait phase. A
`from: distance` driver is a wheel of `radius` meters, `lateral` meters left of the
root, that rolls without slipping over the path of its own contact point: a turn
spins the two wheels at different rates, a pivot in place counter-rotates them and
a sideways displacement does not roll them. Values are radians, positive rolling
forward, unbounded and never wrapped, so renderers can interpolate linearly between
frames.

The publisher integrates them from successive ped poses ([`drivers.py`](drivers.py))
and sets them on every ped of such a model, possessed ones included. The possession
stream itself carries only the 52. Renderers turn each driver's `bone` about its `axis` from
the `.rig` and skip it on a skeleton without the bone. The RViz body has no
such joints and ignores them.

A `from: velocity` driver takes its angle from the ped's forward speed and yaw
rate instead of integrating a wheel over distance. Its driver carries `bone`,
`from` and the bone-local `axis`, without `radius` or `lateral`. Electric
wheelchair bundles use `joystick_pitch` (Y axis, positive forward) and
`joystick_roll` (X axis, positive right) on the same `Joystick` bone. The publisher
computes these angles in radians, the wheel integrator leaves them untouched and
renderers read their `.rig` drivers as usual.

`JoystickDriver` in `drivers.py` supplies the electric chair's two joystick
angles after gait generation, alongside the wheel angles. It samples speed and yaw
rate together over a 0.2 s pose-timestamp window and holds the resulting stick
between samples, including repeated frames. Its per-agent cache is discarded on
despawn and reset.
The publisher enables it when the model's `.rig` declares both `joystick_pitch`
and `joystick_roll` with `from: velocity`, regardless of agent type. Any other
`from: velocity` entry rejects the rig.

The same driver selects one of seven calibrated right-arm poses from the joystick's
`(pitch, roll)` pair: neutral, forward, reverse, forward left/right and pivot
left/right. Its small table holds absolute radians for the three shoulder joints,
elbow and two wrist joints. `with_driven` replaces those gait angles together with
the joystick angles, so both change in the same published frame. The arm pose is
left out while a gesture, possession or a producer's own `joint_state` drives one of
its joints, the joystick still moves. These poses were calibrated for
`arenian_wheelchair_electric` with `seated_wheelchair_electric`:
forward/reverse pitch and pivot roll are 0.6 rad in magnitude, while forward turns
use 0.4 rad roll. The palm follows the joystick's roll, with the shoulder, elbow
and wrist angles calibrated together. All poses were checked from four views,
with 5 to 7 mm between the hand mesh and the joystick and no
intersections with its shaft, grip or housing. A changed model or
seated profile requires recalibration. Fingers keep their existing pose, there is
no runtime IK, grasping or interpolation.

### Wrists (v4)

`l/r_r_wrist`, `l/r_wrist`: two DOFs per hand at the distal end of the forearm, chain
`elbow -> r_wrist -> wrist`. The rest hand hangs along the forearm, fingers down, thumb
forward, palm toward the thigh. `r_wrist` is forearm pronation (positive turns the palm
backward, both sides), `wrist` is the hand hinge (positive = extension, back of the hand
toward the forearm, both sides). Axes are mirrored in the URDF exactly like the collars
(`r_wrist`: `(0,0,-1)` left / `(0,0,1)` right, `wrist`: `(1,0,0)` left / `(-1,0,0)`
right), so the RViz adapter passes both through untouched. The `halt` gesture and
the electric-chair operating poses write them, other default animations leave them
at 0.0. Recordings that predate them carry no wrist values and are read as 0.0 (the
loaders zero-fill, same migration as the spine stack).

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
  ~18 deg bent). The sagittal ankle is `-(hip + knee)`, which keeps the stance foot
  flat (the hip, knee and ankle hinges are parallel at wire zero), and the ankle limit
  lets the swing foot trail. Torso roll/yaw, the spine/chest triples, collars, ankle
  yaw, wrists, and shoulder azimuth/twist stay 0.0 (the baked profile predates the spine
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

## 2. RViz rendering

`rviz_utils.hri.urdf` builds each body's URDF from the body rig template of [BODY.md](BODY.md):
root `body_<ID>` (REP-155 hip-origin frame), a three-axis ball per rig joint, the REP-155
body frames as fixed links. `hri_producer` poses it from `Pedestrian.body`, so RViz shows the
same pose the sims render, fingers included.

### Mirror convention

Every semantic DOF turns about a body-aligned axis at rest (REP-155: X forward, Y left, Z up).

The arm axes reflect per side: `l_p_shoulder` axis `(1,0,0)`, `r_p_shoulder`
axis `(-1,0,0)`, `l_y_shoulder` `(0,0,-1)`, `r_y_shoulder` `(0,0,1)`, `l_r_shoulder`
`(0,0,1)`, `r_r_shoulder` `(0,0,-1)`, and the collar pair per the Collars section.
So the raw URDF meaning of `p_shoulder` is **lateral abduction** (rotation about
body-forward X), positive = outward on both sides, not a sagittal swing.

Leg sagittal joints are not reflected: `l_r_hip`/`r_r_hip` are both `(0,-1,0)`,
positive = forward. Knees, elbows, and ankles are `(0,-1,0)` on both sides. The wrist pair is mirrored,
see the Wrists section. `l_p_hip`/
`r_p_hip` are mirrored (`(1,0,0)`/`(-1,0,0)`), positive = abduct outward on both sides,
which coincides with the semantic meaning (gait idles both at `0.0`), so only the
shoulder triples need the Section 1 exception. The spine stack is not reflected
either: single links on the midline, same axes as the head triple.

### The 40 body DOFs

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
| 37 | `l_r_wrist` | (0,0,-1) | [-1.4, 1.4] | L forearm twist, positive pronation |
| 38 | `l_wrist` | (1,0,0) | [-1.4, 1.2] | L wrist, positive extension |
| 39 | `r_r_wrist` | (0,0,1) | [-1.4, 1.4] | R forearm twist, positive pronation |
| 40 | `r_wrist` | (-1,0,0) | [-1.4, 1.2] | R wrist, positive extension |

The spine/chest limits are per-segment halves of the lumped waist range: the stack's
advisory sum intentionally exceeds the old lumped range (real spines bend further
distributed than a single lumbar hinge).

Bold = the gait-driving DOFs. * Raw axis, the wire value carried in this DOF is anatomical
flexion (Section 1a).

