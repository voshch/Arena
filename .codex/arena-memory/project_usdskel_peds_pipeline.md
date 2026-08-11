---
name: project_usdskel_peds_pipeline
description: "New UsdSkel-native Isaac ped pipeline (replaces omni.anim.people), architecture facts and knobs"
metadata: 
  node_type: memory
  type: project
  originSessionId: af69523e-6daf-4ccb-ae8a-c988ec14047a
---

Implemented 2026-07-02 (uncommitted): arena_isaac pedestrians are now pure UsdSkel, no omni.anim.* dependency. See [[project_isaac_sensor_publish_fix]] for the sibling sensor work.

- Package: arena_isaac/arena_isaac/peds/ (registry, ped, runtime, write, convert, cache, providers/{base,math,clip,gait,buffer,external,bone_map}). Legacy pedestrian/ package deleted, EXTENSIONS_PEOPLE + navmesh block removed from run_isaacsim.py (omni.graph.nodes kept, moved to the sensor enable tuple).
- Writer: usdrt Fabric writes primary (0.02-0.12 ms/frame at 1-30 chars, measured), plain pxr Set() fallback (0.37-5.0 ms). BOTH render without AnimationGraph, the old "Fabric wall" was graph ownership. Root pose via plain USD xform ops works.
- Assets: ONLY actor is "arenian" (arena_simulation_setup/assets/Common/Pedestrian/arenian/arenian.sdf), skin+clips are remote Fuel DAEs (CC-BY-4.0 Mingfei, NOT CC0 as annotation.yaml claims). Converter (peds/convert.py) parses COLLADA directly, strips walk root advance into stride metadata (1.384 m / 5.79 s), caches under $ARENA_DATA_DIR/peds_usd/<actor>-<digest8>/ with ATTRIBUTION.md + downloaded DAEs (offline rebuilds). meta.json keys are NESTED: meta["clips"]["walk"]["stride_length_m"].
- joint_state override: ROS4HRI 20 semantic scalar angles -> peds/providers/bone_map.py (hand-authored, axes need GUI fine-tuning) -> layered on gait pose with slerp buffer + staleness decay. Partial override supported (e.g. head only).
- task_generator/simulators/human/isaac.py: NVIDIA roster removed, isaac uses base-class model resolution (arenian fallback) like gazebo. isaac_simulator.pedestrian_spawn fills Pedestrian.model_uri.
- Ped count knob: task.random.dynamic.n defaults [0, 0] (no peds!). Set to e.g. [3, 3] via ros2 param, re-read at every reset. models default ['arenian'].
- G2 verified: 3 peds spawn/walk/update 5.3+ min headless, zero errors, robot sensors unaffected. User GUI-verified 2026-07-02: gait movement good.
- Materials (CONVERTER_VERSION 2, 2026-07-02): the arenian DAE has NO textures, six solid-color phong materials per polylist. convert.py parses instance_material -> effect chains, authors UsdGeom.Subset (materialBind, partition) + UsdPreviewSurface (diffuse + Blinn-Phong roughness) per symbol. cache.py hashes CONVERTER_VERSION so old clay caches rebuild. Flat displayColor stays as unbound-face fallback.
- Ped mesh leading the SSOT markers was the humansim engine stamp epoch, see [[project_humansim_engine_stamp_epoch]].
