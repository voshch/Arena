---
name: project_gz_ped_fin_open
description: RESOLVED gz ped fin was long hair skinned across divergent bones; fix = rebind hair to one bone
metadata:
  node_type: memory
  type: project
  originSessionId: 2b0ccdf4-5b9d-4112-a136-490780341021
---

RESOLVED (2026-07-04): the gz ped fin/spike (thin fin from upper torso, only when animated, clean in bind pose) was the LONG HAIR (e.g. long01). MakeHuman auto-weights long hair across every torso bone it drapes over (Head, Spine1, LowerBack, Spine, both Shoulders spanning bind z 0.86..1.53), so under a walk adjacent hair verts ride divergent bones and tear into a spike. Fix: `collada.rebind_hair(dae, hair_material, bone="Head")` collapses every hair-polylist vertex to a single bone so the surface moves coherently; wired in cli.py after prune_influences, gated on recipe.hair (bald cells skip). Recipe now carries `hair` (threaded from cell.hair in roster.cell_recipe). Bisected via headless render, both Head and Neck anchors clean, Head chosen. Wrong earlier hypotheses (ruled out with evidence): bind re-root, merge joint-order remap. NOTE: software/numpy FK does NOT replicate gz zeroing the root bone weight, so it gives false flyers, trust the render oracle only.

Diagnostic tooling (rebuild in-container under a scratch dir, link gz-common5-graphics + gz-rendering8 via PKG_CONFIG_PATH=$(ls -d /opt/ros/jazzy/opt/*_vendor/lib/pkgconfig)):
- render.cc: headless ogre2 render of an animated skin DAE to PNG, THE ground-truth oracle, use before deploying any ped change
- split.py: keep/drop a polylist by material to bisect which mesh piece owns an artifact
- gz_oracle.cc: dumps gz-common AddBvhAnimation aligned per-node keys (validated the Python port in gzalign.py to 4e-3)

Both test cells (casual_female_asian_old, worker_male_african_middleage) regenerated through the full fixed pipeline and deployed to _assets, render-clean. 96-cell regen pending. IDREF vs Name_array: keep converting IDREF->Name once at export in gz_compat, single source of truth, no per-consumer dual readers (user rejected). See [[project_arena_humans_pipeline]], [[project_gz_actor_root_freeze]], [[project_worker_male_regen]].
