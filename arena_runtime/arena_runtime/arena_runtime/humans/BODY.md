# Body rig contract

Frozen interface between pedestrian pose producers (gait, gestures, possession, clips,
importers) and pedestrian renderers (Isaac, MuJoCo, Gazebo, RViz). The rig follows the
SMPL-X joint convention: SMPL-X joint names and hierarchy, one local rotation per joint,
every rest frame aligned with the body. Arena adopts the convention only, no SMPL-X model
file is part of Arena. The 54 semantic DOFs of [JOINTS.md](JOINTS.md) stay the authoring
vocabulary and compile to this rig (Section 6).

## 1. Joints

The 22 body joints are always defined. A pose may carry any subset, a joint it leaves out
keeps its stance.

| joint | parent | turns |
|---|---|---|
| `pelvis` | (body frame) | the whole skeleton |
| `left_hip` | `pelvis` | left thigh |
| `right_hip` | `pelvis` | right thigh |
| `spine1` | `pelvis` | lumbar segment |
| `left_knee` | `left_hip` | left shank |
| `right_knee` | `right_hip` | right shank |
| `spine2` | `spine1` | mid-spine segment |
| `left_ankle` | `left_knee` | left foot |
| `right_ankle` | `right_knee` | right foot |
| `spine3` | `spine2` | chest segment |
| `left_foot` | `left_ankle` | left toes |
| `right_foot` | `right_ankle` | right toes |
| `neck` | `spine3` | neck |
| `left_collar` | `spine3` | left clavicle |
| `right_collar` | `spine3` | right clavicle |
| `head` | `neck` | head |
| `left_shoulder` | `left_collar` | left upper arm |
| `right_shoulder` | `right_collar` | right upper arm |
| `left_elbow` | `left_shoulder` | left forearm |
| `right_elbow` | `right_shoulder` | right forearm |
| `left_wrist` | `left_elbow` | left hand |
| `right_wrist` | `right_elbow` | right hand |

Optional joints, same naming as SMPL-X: `jaw`, `left_eye_smplhf` and `right_eye_smplhf`
under `head`, and per hand five chains of three under the wrist, `<side>_<digit><k>` with
digit `index`, `middle`, `pinky`, `ring` or `thumb` and `k` 1 to 3, each `k` the parent
of `k + 1`. Digit joints turn the finger segments: `<side>_<digit>1` the knuckle (the
thumb's carpometacarpal joint), `2` and `3` the next two joints out.

A joint is named after the point it sits at, not the bone it turns: `left_hip` turns the
thigh, `left_ankle` the foot, `left_foot` the toes.

## 2. Frames and zero pose

Every joint frame is aligned with the body frame at the zero pose: X forward, Y left, Z up
(REP-103). A joint's rotation is relative to its parent's frame, so a joint's world
rotation is the product of the rotations from `pelvis` down to it.

The zero pose (all rotations identity) is the template T-pose: legs straight down, feet
pointing forward, arms straight out to the sides, palms down, spine and head upright.
Joint positions at the zero pose are the template's (Section 4) or a bundle's own.

SMPL-X uses X left, Y up, Z forward with identity rest frames, so one constant matrix
converts every SMPL-X joint rotation and translation:

```
C = [[0, 0, 1],
     [1, 0, 0],
     [0, 1, 0]]      v_arena = C v_smplx      R_arena = C R_smplx C^T
```

The Arena template is not the SMPL-X rest skeleton, so a rotation converted by `C` still
needs a rest correction per joint. `rest_offsets` with the SMPL-X rest skeleton as the
bind pose (identity frames) gives each joint's `O`, and

```
R_local_arena = O_parent R_local_converted O_joint^T      O_parent = identity at the pelvis
```

puts every bone along the direction it has on the source body. Rest skeletons of sources
come from their files, or `REST_OFFSETS` in `adapters/arena_animation_retarget/smplx.py`.

## 3. Wire

`Pedestrian.body` is an `arena_people_msgs/BodyPose`:

- `joints`: Arena joint names, unique, any order.
- `rotations`: one unit quaternion per entry of `joints`, the joint relative to its parent.
- `root_offset`: the displacement of the rig's root joint (parent -1, `pelvis`) from its rest
  position, in the body frame, m.

`Pedestrian.pose` is the body frame (origin at the soles under the pelvis, Z up the body),
and the `pelvis` rotation is relative to it. A source with a world-frame root (SMPL-X
`global_orient` and `transl`) splits it: the heading of the pelvis X axis projected on the
ground plane and the ground position go into `pose`, the rest of the rotation into
`pelvis`, and the height above rest into `root_offset.z`. Renderers that plant the soles
keep doing so and override `root_offset.z` for a standing ped.

`Pedestrian.joint_state` carries only the driven joints of the bundle's `.rig`
(wheels, joystick), as scalar radians.

Wire to rig is by name only. A renderer poses the joints named in both `joints` and the
rig's `joints/names`, ignores wire names the rig lacks, and holds no joint list of its own.

## 4. Template

`arena_simulation_setup/configs/body/template.yaml` is the reference skeleton at the zero
pose:

```yaml
format: arena-body-template/1
joints:
  pelvis: {parent: null, offset: [x, y, z]}
  left_hip: {parent: pelvis, offset: [x, y, z]}
  ...
```

Every joint of Section 1 that the template defines has a `parent` and an `offset` from the
parent joint in Arena axes, `pelvis` its position in the body frame. The template defines
the 22 body joints and the 30 digit joints. They sit at the joint cube centers of the MakeHuman
base mesh (MPFB2 data, CC0), the cubes the `cmu_mb` rig places its bone heads at, with the arms
and legs laid straight at their segment lengths, the toes pointing forward, the midline joints on
the sagittal plane and the right side the mirror of the left. The digits lie flat with the palm
down: the four fingers straight along the hand axis (`+Y` left, `-Y` right), side by side from
the index on the thumb side (`+X`) to the pinky, and the thumb straight along the direction
MakeHuman places it, forward and toward the palm. Producers that derive
rotations from joint positions and the forward kinematics of `shared/body.py` use the
template's rest directions.

## 5. Bundle rig (`.rig`)

A human bundle carries its posing contract as `rig/<name>.rig` beside `rig/<name>.sdf`. A human
`model_uri` resolves to `rig/<name>.sdf` when it exists, so `<name>.rig` sits next to the SDF
and everything that finds files beside the SDF works unchanged. `rig/<name>.sdf` points at
`../meshes/<name>.dae` and `../clips`, or at its own `rig/meshes` skin. The old
`<name>.sdf`, `meshes/`, `clips/` and `rig.yaml` are never written, old code reads them as before.

The file is a numpy zip of `group/member` arrays like `.anim`, stored and not deflated,
`allow_pickle=False`. `arena_assets/rig.py` has `write_rig`, `load_rig` and the loaded `Rig`.

| member | shape, type | meaning |
|---|---|---|
| `rig/version` | scalar int | oldest reader that can use the file |
| `rig/settings` | JSON string | `skeleton`, `source_rig`, `neutral`, `rest`, `root_offset` |
| `joints/names` | (J) str | joint names, parents first |
| `joints/parents` | (J) int32 | index into `joints/names`, -1 for a root |
| `joints/stance` | (J, 4) float64 | world rotation in the neutral stance, xyzw |
| `bones/names` | (B) str | skin bones that follow a joint |
| `bones/joints` | (B) int32 | index into `joints/names` |
| `bones/weights` | (B) float64 | share of the joint's local rotation, 1 for a single bone |
| `bones/offsets` | (B, 4) float64 | bone world rotation in the zero pose, xyzw |
| `soles/joints` | (S) int32 | optional, joints whose bones carry the sole points, left heel pair then right |
| `drivers/names` | (D) str | optional, driven wire joints |
| `drivers/bones` | (D) str | optional, bone each one turns |
| `drivers/axes` | (D, 3) float64 | optional, bone-local axis |
| `drivers/settings` | (D) str | optional, JSON per driver: `from` (`distance` or `velocity`), `radius`, `lateral` |

- `bones/offsets` is the bone's world rotation with the bind skeleton posed into the zero pose.
  A renderer sets the bone's world rotation to `G_joint * offset`, `G_joint` composed from the
  body pose along `joints/parents`, so the file alone poses the skin. No reader holds a joint list.
- Several bone rows for one joint spread it along a chain, in chain order: each bone takes its
  `weights` share of the joint's local rotation angle about the same axis, the weights sum to 1,
  and the last bone's world rotation is `G_joint * offset`.
- Bones no row names keep their rotation in the neutral stance relative to their parent bone.
- `joints/stance` is each joint's world rotation in the bundle's neutral stance, the first frame
  of its `neutral` (else `idle`) clip with the root and leg chain from the skin, thighs and
  shins hanging straight down and the feet keeping their world rotation.
- A driver turns its bone about the bone-local axis by the driven joint's angle.
- `rig/settings` keys `neutral`, `rest` and `root_offset` carry the values of the bundle's
  `rig.yaml`. `root_offset` is the vertical drop of the body root for seated bundles.
- `skeleton` is descriptive (`smplx`), `source_rig` names the bone map below.

Versioning: `RIG_READER = 1` is the highest `rig/version` the code reads. A file above it logs a
warning naming the fix and is treated as absent. Unknown groups, members and settings keys are
ignored. A new group or member keeps `rig/version`, changing the meaning of a member raises it.

A bundle without a `.rig` has no body posing: the ped plays its clips and one warning names
`rig_human`. Nothing upgrades a bundle at load time.

`rig_human <bundle>` (arena_simulation_setup) writes `rig/<name>.rig` from the bundle's skin
(`rig/meshes/<name>.dae` when present, else `meshes/<name>.dae`) and, unless the bundle has
its own `rig/meshes` skin, `rig/<name>.sdf`. It reads `neutral`, `rest`, `root_offset` and
`drivers` from an existing `rig.yaml` and never writes it.
The bone map follows the skin's bones:

- `cmu_mb`: the 22 body joints onto the CMU bones (`Hips`, `LowerBack`, `LeftUpLeg`, ...),
  the neck spread over `Neck` and `Neck1`. Its skins have no per-finger bones.
- `game_engine`: the MakeHuman game engine skeleton, 52 joints: the 22 body joints onto
  `pelvis`, `spine_01` to `spine_03`, `neck_01` (one bone, weight 1.0), `head`,
  `clavicle_<l|r>`, `upperarm_<l|r>`, `lowerarm_<l|r>`, `hand_<l|r>`, `thigh_<l|r>`,
  `calf_<l|r>`, `foot_<l|r>` and `ball_<l|r>`, and each digit joint `<side>_<digit><k>` onto
  `<digit>_0<k>_<l|r>`. The `hand` bone's direction is set by `middle_01`.

A bundle without a neutral clip takes its stance from the skin's bind pose.

## 6. Semantic DOFs

`semantic_to_body` in `arena_simulation_setup/shared/body.py` compiles the 54 wire DOFs of
JOINTS.md, and `body_to_semantic` extracts them back.

| JOINTS.md DOFs | joint |
|---|---|
| `r_waist`, `y_waist`, `waist` | `spine1` |
| `r_spine`, `y_spine`, `spine` | `spine2` |
| `r_chest`, `y_chest`, `chest` | `spine3` |
| `r_head`, `y_head`, `p_head` | `neck`, `head` |
| `<s>_y_collar`, `<s>_p_collar` | `<side>_collar` |
| `<s>_y_shoulder`, `<s>_p_shoulder`, `<s>_r_shoulder` | `<side>_shoulder` |
| `<s>_elbow` | `<side>_elbow` |
| `<s>_r_wrist`, `<s>_wrist`, `<s>_d_wrist` | `<side>_wrist` |
| `<s>_y_hip`, `<s>_p_hip`, `<s>_r_hip` | `<side>_hip` |
| `<s>_knee` | `<side>_knee` |
| `<s>_y_ankle`, `<s>_ankle` | `<side>_ankle` |
| `<s>_curl_<digit>` | `<side>_<digit>1`, `<side>_<digit>2`, `<side>_<digit>3` |
| `<s>_spread` | `<side>_index1`, `<side>_middle1`, `<side>_ring1`, `<side>_pinky1` |

Each group composes in the ros4hri URDF chain order of JOINTS.md Section 2 about the axes
listed there, the shoulder triple per JOINTS.md Section 1a. That gives each joint a rotation
`D` in a frame aligned with the body at the semantic rest, and with `T` the bundle's stance:

```
R_local_joint = T_parent^T D_joint T_joint
```

All 54 at zero therefore pose the bundle's neutral stance. Without a `.rig` the default stance holds
identity except `Rx(-pi/2)` (left) and `Rx(pi/2)` (right) on the shoulder, elbow, wrist and the
15 digit joints of each hand, the arms hung from the T-pose. A stance that lacks a digit joint
hangs it like its wrist.

`<s>_curl_<digit>` flexes the digit about `-X` (left) or `X` (right), the thumb about `-Z` (left)
or `Z` (right), positive curling the fingertip toward the palm, the thumb across it, the angle of
the base and middle knuckles, the tip knuckle at 0.7 of it. `<s>_spread`
turns `index1`, `middle1`, `ring1` and `pinky1` about `-Y` with the weights 1, 1/3, -1/3 and -1,
positive moving the index toward the thumb and the pinky away from it, in the palm plane at
the hanging stance. The thumb has no spread. A digit DOF left out is 0, a straight digit. `y_head` goes 0.6 to `neck` and 0.4 to `head`, `p_head` 0.35 to `neck`
and 0.65 to `head`, and `r_head` to `head`. `pelvis`, `left_foot` and `right_foot` have no
semantic DOF and hold the stance.

## 7. Clips

A clip on the body rig is an `.anim` file (`tree/assets/Animation.py`) with a `body` group
beside its `clip` group. Older code ignores the group and plays `clip/angles`, so `anim/requires`
stays 1:

| member | shape | content |
|---|---|---|
| `body/joints` | (J,) | Arena joint names |
| `body/rotations` | (T, J, 4) | parent-relative quaternions, x y z w |
| `body/root_offset` | (T, 3) | m, body frame |

The `clip` group keeps its meaning and carries the clip's 42 DOFs. The player blends those as
any clip and, after `semantic_to_body`, the body group over the result with the slot's weight, a
joint when the slot owns its DOFs and `pelvis`, `left_foot`, `right_foot` and `pelvis_offset` when
it owns the whole body. `pelvis_tilt` clips keep their `root/quat`.
