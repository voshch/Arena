---
name: gait-leg-gate
description: Retargeted CMU legs hide three defects invisible to joint metrics; leg gate must pass before presenting any walk
metadata: 
  node_type: memory
  type: project
  originSessionId: d1058176-4dfa-4e43-b9e3-c10599875612
---

Found 2026-07-18 during the 12_01 polish loop (v9-v11), all three invisible to joint-position/mean metrics, visible only to flesh-level checks:

1. **Animated hip helper joints**: CMU rigs animate LHipJoint/RHipJoint, swinging the hip SOCKETS up to 4cm medially around the pelvis (anatomically impossible). Fix: freeze helper local rotation to bind, re-express femur local so femur world orientation is exact; only the socket position returns to the pelvis.
2. **Bone roll convention offset**: retarget leaves ~+-50 deg constant twist about thigh AND shank long axes (CMU vs MakeHuman roll). Rotates the flesh ring medially (thigh voxels cross the midline) and tilts the knee hinge so flexion dives sideways. Fix: per-frame swing-twist decomposition vs bind, zero the twist, compensate child local (world positions below stay exact).
3. **Mean is not trajectory**: v8 stance fix shifted the foot's MEAN lateral to the rail; the +-5cm per-frame wave and knee valgus survived and read as bow/X legs. Fix: per-frame two-stage column lock (knee to rail at socket, ankle to rail at knee, foot world rot restored).

Gate before presenting (scratchpad leg_gate.py + hand_gate.py, both take an npz path argv, rebuild if lost): socket y == bind +-2mm; leg bone twist vs bind < ~5 deg; signed binned thigh-crossing 0 frames (min-vertex-distance CANNOT detect interpenetration; bind baseline -0.86cm). Order in the correction stack: socket freeze -> detwist -> column lock -> foot restore.

IDLE IS LOAD-BEARING (2026-07-18, both-sim "crossed legs" reports): walk-only polish is insufficient. The unpolished idle clip carried sockets adducted to +-0.055, +-45 deg thigh roll, asymmetric feet, 82/82 frames crossing at 11.6cm, and it reaches BOTH renderers: gz blends to idle at low speed, Isaac composes the wire over idle frame 0 as its neutral. Fix: posture.polish_idle = static subset (clavicle, head stabilize, arm hang, hands, socket freeze, detwist, column lock; NO gait-phase arm/elbow rebuild), applied to every non-walk clip in cli.build_actor. GATE EVERY CLIP IN THE BUNDLE (scratchpad bundle_leg_gate.py runs on a clip DAE directly), not just walk, and gate the PRODUCED bundle, not only parity npz. Isaac converter cache digests DAE bytes, so new bundles reconvert automatically.

WIRE BAKE 2026-07-18 (greenlit): GaitGenerator (task_generator gait.py) walk/run now evaluate _WALK_PROFILE, mean + 3 sine harmonics per signal extracted from the polished 12_01 keys (scratchpad extract_gait_profile.py); limb pairs share ONE canonical profile, right side at phi+pi, so antiphase is exact by construction. Antiphase is l(phi)=r(phi+pi), NOT l=-r (nonzero means: hip +11 deg, elbow ~18 deg); test_gait antiphase test updated accordingly (ids 0/180 seed exactly pi apart). Cadence formula unchanged (frozen, gz plugin mirrors it). JOINTS.md gait-synthesis section rewritten. This is what Isaac and rviz render, closing the Isaac-looks-wrong gap (Isaac never plays clips). Head stabilization also in posture.py (HEAD_KEEP 0.3, world-space at the neck; local damping backfires because the actor's neck counters trunk sway), re-spike swapped into cache 2026-07-18 (backup cache_backup_v19spike/).

PORTED 2026-07-18: full stack lives in arena_humans src/arena_humans/posture.py (polish_walk, walk clip ONLY, called in cli.build_actor after rebase_gait); recipe walk switched to cmu/12_01.bvh trim [22, 502]; FINGER_TILT 6 (port FK pins full root to bind like gz, slightly different calibration than the audition's xy-only pin). Parity-verified vs audition gates. 3-cell spike (casual_male_asian_young, casual_female_african_young, office_male_asian_middleage) built ok and swapped into the Arena cache 2026-07-18, v10-spike bundles backed up in session scratchpad cache_backup_v10spike/. Idle clip deliberately NOT polished (arm swing rebuild needs a gait phase). Awaiting user gz verdict before 96-cell regen; arena_humans repo changes uncommitted (user drives).

GATE TOLERANCE FOR GOWNS (grid_v10 sample): narrow-hip female doctor cells (sockets +-0.090..0.094) show 2-5 walk frames at 0.12-0.56cm "crossing" = gown fabric grazing the midline, counted because garment verts skin >0.4 to UpLeg. Under garment thickness, accepted, don't re-flag. The defect class the gate exists for is 10cm+ over most frames.

Body context: naked male flesh WHR ~0.82-0.85 (slim-normal), trochanter width 32.6cm (low-normal), but rig socket separation 21.6cm vs 17-19cm anatomical, so rail-width stance is wider than natural step width.

FOOT HEADING (v19): restoring the clip's foot world orientation after the column lock preserves the heading of the OLD crossed stance = pigeon toes (~4-10 deg inward of bind). Fix in the restore step: pin the foot's lateral axis to horizontal +y (measure yaw from the lateral axis, NOT the projected toe direction, which degenerates at toe-off pitch), which locks heading to bind toe-out (+-5.9 deg) while heel-toe roll passes through.

HAND-IN-THIGH (v12-v18): thumb/finger bones carry their own clip keys that point digits ~10cm medial into the thigh, invisible to palm-direction metrics (they sample joint offsets, not digit bone orientation). Fix stack: forearm medial de-drift + ABDUCT_DEG 7 upper-arm hang + wrist damped to mean and digit locals frozen to bind + mean finger dir aligned down with FINGER_TILT_DEG 4 outward + supinate/damp-align/supinate iteration (forearm twist swings aligned fingers around the forearm axis, so order matters). Gate: hand_gate.py signed bin test, target 0 frames >2mm (residual sub-6mm brush ok, real hands graze trousers).
