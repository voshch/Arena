---
name: project_worker_male_regen
description: "worker_male_african_middleage bundle renders with feminine chest, needs roster regen"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2b0ccdf4-5b9d-4112-a136-490780341021
---

`Common/Human/worker_male_african_middleage` renders in-sim with a feminine chest ("boobs"), a bad body-knob seed from the roster (muscle/breast/body-shape sliders), NOT a pipeline defect. The gz/[[project_human_rig_mirrored_shoulder_axes]] pipeline itself is correct (textures, skeleton, walk all work). Fix later by regenerating this cell in arena_humans (~/dev/arena_humans, see [[project_arena_humans_pipeline]]) with corrected male body params. Flagged 2026-07-04 during gz ped visual QA.
