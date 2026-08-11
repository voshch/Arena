---
name: project_human_rename_stashes
description: "Pedestrian->Human asset class rename: applied on migration/isaac6.0.0 2026-07-03 (stashes consumed), arena_tools deliberately not migrated"
metadata: 
  node_type: memory
  type: project
  originSessionId: af69523e-6daf-4ccb-ae8a-c988ec14047a
---

The Pedestrian -> Human asset-class rename (see [[project_asa_semantic_annotations]]) is APPLIED on migration/isaac6.0.0 as of 2026-07-03: both stashes were popped and dropped, staged in Arena + arena_training awaiting user commit. Includes tree/assets/Human.py (HumanIdentifier/HumanView, _asset_type 'Human'), all task_generator consumers (incl the later-added simulators/human/isaac.py), shared/entities.py, assets/Common/Human/arenian, gazebo.launch.py, docs, training configs, and the arena.repos pin bumped to master@d4d8a13 (branch@hash notation is aspirational, stock vcs import cannot parse it).

- arena_tools deliberately NOT migrated (user: "fuck arena_tools, don't touch that"): ScenarioEditor PedestrianEditor.py:86 still references tree.assets.Pedestrian and is broken on this branch. Do not fix unprompted.
- arenian2 is explicitly NOT Claude's purview (user, twice). Never touch, relocate, or reason about it.
- Runtime vocabulary (arena_people_msgs, arena_peds, PedestrianITF, hunav/humansim, evaluation metrics) intentionally KEPT: it describes walking agents, not the asset class.
- Gazebo consumption of network-fetched Human bundles needed a PedSkeletonPlugin fix (src/gazebo/arena_gz_plugins): actor templates now absolutize skin/clip URIs against the SDF dir via the element DOM (sdf14 Actor has no animation mutators, re-Load from mutated ElementPtr). Fetched bundles are not on GZ_SIM_RESOURCE_PATH and the gz GUI resolves scene URIs itself; arenian only ever rendered because its dir is hardcoded on the resource path in gazebo.launch.py.
- Isaac consumption still blocked by the converter gaps (IDREF_array joints, 8 controllers/geometries, .bvh clips, textures), hit live 2026-07-03: every bundle fails spawn with "has no Name_array". Interim: pin isaac dynamic models to arenian. See [[project_arena_humans_pipeline]].
