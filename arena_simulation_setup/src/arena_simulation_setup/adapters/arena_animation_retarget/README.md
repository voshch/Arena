# arena_animation_retarget

Retargets SMPL-X body motion onto Arena's 42-DOF human wire, producing clips for
`arena_simulation_setup/assets/Common/Animation`. Sources include text-to-motion output from Kimodo and AMASS-style SMPL-X parameter files. Conversion has two stages: a rotation retarget, then whole-body IK solved as one batch on the GPU. An optional third (`--hold`) re-solves the arms so a hand holds a prop.

```bash
# one clip, or a whole tree of them (directories are searched recursively)
ros2 run arena_simulation_setup animation_retarget wave.npz -o arena_simulation_setup/assets/Common/Animation --loop
ros2 run arena_simulation_setup animation_retarget ~/amass/CMU -o $ANIM --name-from-path --prefix amass_ \
    --report retarget.csv --skip-existing
python -m arena_simulation_setup.adapters.arena_animation_retarget ...   # same, without ros2
```

```python
from arena_simulation_setup.adapters.arena_animation_retarget import load, retarget, retarget_batch, write_clip
clip = retarget(load("wave.npz"))                       # Motion -> Clip (20 fps)
clips = retarget_batch([load(f) for f in files])        # one batched IK for all of them
write_clip(clip, ANIM, "wave_smplx", loop=True)
```

Each source becomes `<out>/<name>/<name>.anim` plus `annotation.yaml`. The `clip` group holds what Arena's loader reads (`joint_names`, `angles`, `t`, `root_xy_yaw`, `animation_state`), the `root` group `z` and `quat`: the pelvis height and orientation.

The wire has no pelvis-tilt joint. By default the tilt is folded into the waist and hips, so the body keeps its shape on an upright root. That works for bending and kneeling, but a body lying on the ground needs its hips turned past their range and the skin twists. With `--no-fold` the body stays pelvis-relative and the annotation gets `pelvis_tilt: true`: Arena's player then tilts the whole ped by `root/quat`, its pelvis over the agent's position and its lowest point on the floor. Use it for falls and lying clips.

## Batch processing and GPU

- Worker processes (`--jobs`, default 8) load the files and run the rotation retarget, at about 17k frames/s per process.
- Frames are gathered into chunks of up to `--chunk-frames` (default 400k, about 6 GB of GPU memory).
- Each chunk is solved in one L-BFGS run on `--device` (default: CUDA if available). Clips share the solve but never couple: smoothing stays inside each clip.
- Measured on an RTX PRO 6000: 205k frames (1.9 h at 30 fps) in 16 s at 150 iterations, about 13k frames/s. Torch on CPU manages about 760 frames/s; the earlier per-frame scipy fit, about 145. Small batches are dominated by CUDA start-up and per-iteration overhead (around 10 s).
- A file that fails to load, such as an AMASS `shape.npz`, is reported and skipped. The exit code is 1 if anything failed.
- `--report` writes every clip's fidelity metrics to a CSV; `--skip-existing` resumes an interrupted batch.

## Inputs (`.npz`, detected by keys)

| source | keys | world up |
|---|---|---|
| rotation output (Kimodo) | `global_rot_mats` + `posed_joints`, or `local_rot_mats` (+ `root_positions`) | Y |
| SMPL-X parameters (AMASS) | `poses`, or `root_orient`/`global_orient` + `pose_body`/`body_pose`; `trans`/`transl`, `betas` | Z |

Only the 22 body joints are used, source finger data is not retargeted. The frame rate comes from `fps` or `mocap_frame_rate` (default 30) and can be overridden with `--fps-in`. `--up` overrides the world up axis. Parameter files need a rest skeleton: the default is the body Kimodo generates with, or pass `--model SMPLX_NEUTRAL.npz` to build the exact skeleton for each file's `betas` (numpy only, no `smplx` package needed).

## Method

### 1. Rotation retarget (closed form, vectorized)

Each Arena segment takes the SMPL-X segment's rotation away from that segment's own rest pose. This is what makes rest-shape differences disappear:
- SMPL-X's clavicle rises 25° at rest. A position fit turns that into a constant shrug.
- Its rest limbs are kinked sideways (left arm 19°, legs 9–14°). Arena's one-plane hinges can only copy a sideways kink by twisting the whole limb, turning the knees out about 60°.

Instead, each limb's rest map keeps the in-plane part of the kink as the rest hinge angle and drops the sideways part. The limb then lies along SMPL-X's rest line (shoulder→wrist, hip→ankle). Spine, head, collars, wrists and ankles come straight out of rotation decompositions. Shoulder angles are kept on the branch nearest the previous frame.

### 2. Whole-body IK (`ik.py`)

On its own, the rotation retarget puts a bent arm's hand 6–20% of arm length off, because Arena's forearm is 0.6× its upper arm while SMPL-X's are equal. The IK starts from the rotation retarget and minimizes one objective over all frames at once:

| term | what it matches |
|---|---|
| reach | hand − shoulder and ankle − hip: the source vector scaled to Arena's limb length (per side, against the source's rest reach) |
| hands, hand_neck | left − right hand and hands − neck, absolute. Weighted in only while the source hands are within about 25 cm of each other or the head, so claps close and hands reach the face. |
| feet | stance: left − right ankle relative to the hips (Arena's hips are 2.3× wider than SMPL-X's) |
| dirs, twist | rest-corrected bone directions and the upper bones' forward axes. These keep the elbow and knee swivel and the limb twist, i.e. the style, where reach leaves them free. |
| hand_rot, foot_fwd | hand orientation; toe direction |
| prior | stay near the rotation retarget, strongest on the spine, head and collars |
| limits | JOINTS.md hard limits where anatomy fixes them (elbows and knees bend one way; wrists, ankles, collars, head), enforced by a stiff penalty and then a clamp |
| smooth | second differences within each clip |

`--method rotation` keeps only the bone-direction terms, which reproduces the per-frame fit of earlier versions.
`--method none` returns the closed-form rotation retarget unchanged. Weights live in `ik.IKConfig`.

On 10 Kimodo clips:
- **Hands:** within 0.3–3.5% of arm length (rotation retarget: 6–20%).
- **Feet:** within 2% (was 5–17%).
- **Arm lines:** within 4°.
- **Jitter:** peak acceleration 4–17°/frame², against 12–85 for the rotation retarget.

The price is individual bone directions, 6–30° from SMPL-X's. That is unavoidable: with different proportions, a hand in the same place needs a differently angled upper arm.

## Two-person interactions (`--pair`, `pair.py`)

```bash
ros2 run arena_simulation_setup animation_retarget hug.npz shake/ -o $ANIM --pair --prefix intermask_
```

```python
from arena_simulation_setup.adapters.arena_animation_retarget.pair import load_pair, retarget_pairs
res = retarget_pairs([load_pair("hug.npz")])[0]   # res.clips = (a, b), res.placements, res.contact
```

Each source holds two people and becomes two clips, `<name>_a` and `<name>_b`. `--no-fold`, `--check` and `--no-score` do not apply: contacts are solved on upright bodies and pairs are scored by their contact error. Sources:
- an `.npz` whose arrays carry `p0_` / `p1_` key prefixes, e.g. the InterMask (Inter-X) export: `p0_root_orient`,
  `p0_pose_body`, `p0_trans`, …;
- a directory holding `P1.npz` and `P2.npz`, the Inter-X dataset layout.

Arena plays every clip in place: the simulator poses each human at its agent's pose and ignores the clip's root motion. So a pair needs a fixed stand point per person, and the hands must meet the partner where the partner's Arena body actually is. Arena's arm proportions differ, so independent retargets miss each other by 8–45 cm.

The pair mode works in four steps:

1. **Window** (`--window auto`): keep the frames where the two touch, padded by 0.75 s. Arena's interaction manager walks agents into formation before it plays a clip. `all` keeps every frame, for interactions without contact such as talking or waving.
2. **Placement:** each person's pelvis position and heading, averaged with weights from contact strength. This is how the source stands while touching.
3. **Contacts:** every wrist and elbow is matched to the nearest of the partner's body points. Within 35 cm it counts as a contact, weighted by distance. The offset is kept in the partner's heading frame.
4. **IK with anchors:** both people go through the batched IK. Each contact then becomes an `anchor` target (an IK term that is off for single clips): the partner's Arena point plus the source offset. Hand-to-hand contacts meet halfway. The IK runs again (`--passes`, default 2).

Each clip's `annotation.yaml` gets a `pair` block:

```yaml
pair: {role: a, partner: intermask_hug_b, partner_offset: [0.28, -0.01, 2.92], separation: 0.28,
       source_window_s: [0.83, 4.8]}
```

`partner_offset` is where the partner stands seen from this person: x forward, y left (m), then relative heading
(rad).

The report's `contact_mean` / `contact_p95` columns compare each contact vector on the two Arena bodies with the source's, in cm. `contact_mean_noanchor` gives the same measure without anchors. On 11 InterMask samples the error went from 9–45 cm to 1.4–6 cm.

## Holding a prop (`--hold`, `hold.py`)

```bash
ros2 run arena_simulation_setup animation_retarget iv_walk.npz -o $ANIM --name walk_ivpole \
    --hold iv --hold-body Hospital/patient_male_caucasian_old
ros2 run arena_simulation_setup animation_retarget cane.npz -o $ANIM --hold cane:gz=0.88,arc=0.35 --hold-body $BUNDLE
```

```python
from arena_simulation_setup.adapters.arena_animation_retarget import hold
clip, report = hold.apply(clip, hold.Hold.parse("rollator"), body="Common/casual_male_caucasian_old")
```

Text-to-motion constraints, such as Kimodo's hand pins, put a hand only near a handle. The pins are soft, the retarget
changes the body's proportions, and the character a simulator draws differs again from Arena's template: a cane leaves
the fist at the wrong angle, rollator hands hover above the grips. Holding needs millimetres and a few degrees, so this
stage solves the arm last, per clip frame, on the body that will be drawn:

- Arm DOFs only (collar, shoulder, elbow, forearm twist, wrist flexion and tilt); legs, trunk and head stay as retargeted.
- Bounded least squares per frame (scipy), warm-started from the previous frame; the first frame from random starts within
  the limits.
- Terms: the centre of the fist closed on the handle (`arena_runtime` hands.palm_frame(...).grip, the cylinder the
  finger curls close on) on the handle; its axis along it; a loose palm direction; optional comfort ranges for the elbow
  and wrist and an elbow pointing back; staying near the retargeted arm; smoothness to the previous frame.
- Body: `--hold-body` takes a character bundle directory or a model URI. `shared.actor_pose.ActorPose` poses it from the
  bundle's own files the way the simulators draw it: the .rig over the skin skeleton, the neutral clip, soles on the
  floor. It matches Isaac's wire pose provider exactly (`tests/unit/test_actor_pose.py`). Without `--hold-body` the
  stage solves on Arena's template body.

Tasks (body frame, x forward, y left, z up from the soles, m):

| task | hand | target |
|---|---|---|
| `cane` | right (`side=l` mirrors) | fist at (gx, gy, gz); a cane of `length` held `grip_at` below its top, its tip sweeping +-`arc` across the midline in step with the clip's own steps (left at each right heel strike: two-point touch), lifted `tap` between touches |
| `rolling` | right | the same, the tip always on the floor |
| `iv` | left (`side=r`) | round a vertical pole at (gx, gy, gz), thumb up; the pole diagonal across the palm (`angle` 42), the elbow at the side |
| `frame` | both | a pick-up frame's bars `width` apart, following the clip's own two-hand lift and placing, averaged over both hands (rigid frame), held at the bar height `handle` while the frame stands |
| `rollator` | both | the grips at (gx, +-width/2, gz) |

The defaults are the props the stage was made with (IVStand, WalkerFitted, RollatorFitted, a 1.27 m cane) and their
characters. Override them as `TASK:key=value,...`; keys and defaults are in `hold.TASKS` (plus `hold.OPTIONAL`), unknown keys are refused.

The stage also solves the finger curls that close this body's hand on the handle (hands.grasp_on: each finger to first
contact, the thumb by room or, with `wrap`, over the fingers). It writes them into the clip as digit DOFs. A held clip
carries no `body/` track: that track holds the source's own torso, which the arm was not solved against, so players pose
the clip from its angles over the character's stance. A clip that cannot be held (a cane clip without two strides) fails
alone. Each annotation gets a `hold` block, whose `curls` go into an agent type's `pose.hold.<side>.curls`:

```yaml
hold: {task: iv, body: patient_male_caucasian_old, sides: l, curls: {l: [0.57, 1.15, 1.11, 0.94, 0.31]},
       grip_mm: [0.16, 0.34], axis_deg: [0.07, 0.15], radius: 0.008, angle: 42.0}
```

`grip_mm` and `axis_deg` are the mean and p95 residuals over the clip. Five Kimodo assistive walks solved on their characters:

| task | character | grip mm (mean/p95) | axis deg | time |
|---|---|---|---|---|
| iv | patient_male_caucasian_old | 0.16 / 0.34 | 0.07 / 0.15 | 73 s |
| cane | casual_female_caucasian_middleage | 0.01 / 0.03 | 0.01 / 0.03 | 232 s |
| rolling | casual_male_asian_young | 0.00 / 0.01 | 0.01 / 0.03 | 195 s |
| frame | patient_female_asian_old | 0.01 / 0.02 | 0.00 / 0.00 | 332 s |
| rollator | casual_male_caucasian_old | 0.04 / 0.11 | 0.02 / 0.06 | 275 s |

The fidelity report then shows the held arm far from the source's, as intended. A gait profile fitted from a held clip
(`arena_runtime.humans.fit`, 4 harmonics) keeps the hold within a few mm (IV: 4 mm mean, 0.2 deg), since the
fit averages the strides.

## Fidelity report

Every conversion prints a summary (`--check` prints the per-clip table, `--report` writes a CSV). Each clip frame is compared with the nearest source frame on Arena's contract FK, i.e. what RViz draws:
- `*_reach`, `hands_gap`, `feet_gap`, `hands_contact`: in % of limb length.
- `*_line`, `*_upper`/`*_lower`, `*_hand_rot`, `trunk`, `head`: in degrees.
- `step`: the largest frame-to-frame change, which catches flips and jitter.

The axes and chain come from `arena_runtime`'s `humans/pointing/skeleton.py`, which is loaded from its file without rclpy (`$ARENA_SKELETON` overrides the path). `fk_torch.py` is its batched torch port, held to 1e-9 by a test.

## Tests

```bash
cd arena_simulation_setup && python3 -m pytest tests/unit/animation_retarget   # ~2 min; uses the GPU if there is one
```

Coverage:
- input formats and the model skeleton;
- the rest pose, with no shrug, twist or bend;
- agreement with the per-frame reference (`tests/reference.py`);
- IK reach compared with the rotation retarget;
- a synthetic clap that must close;
- hard limits, lying down and continuity;
- batch results equal single-clip results;
- the CLI on a mixed directory;
- pairs (`test_pair.py`): a synthetic handshake whose hands must meet on Arena's bodies, placements, the pair file layouts and `--pair` on the CLI;
- holds (`test_hold.py`): an IV pole held thumb up, rollator grips level, a cane sweeping with the steps, the body track dropped and misspelled keys refused, on the template body.

`test_real_kimodo_collapse` runs when `$ARENA_RETARGET_SAMPLE` points to a Kimodo `.npz`.
