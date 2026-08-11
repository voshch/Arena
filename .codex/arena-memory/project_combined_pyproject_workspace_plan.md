---
name: project_combined_pyproject_workspace_plan
description: plan to replace root flat dep list with a derived uv workspace; each ament package declares the pip deps it imports
metadata: 
  node_type: memory
  type: project
  originSessionId: 2c6c4c7b-c6d7-45f2-97dc-4c64f5a1c12f
  modified: 2026-07-31T11:15:38.997Z
---

Ongoing design (finalized 2026-07-02, not yet implemented): turn the hand-maintained ~50-dep flat list in root `pyproject.toml` into a uv workspace where each ament package declares the pip deps it imports; the combined env emerges and is locked via `uv sync --all-packages` behind an `arena uv` helper. Root keeps only "system dependencies".

## Decisions locked with the user
- Members (11, all py3.12): task_generator, arena_bringup, arena_runtime, arena_evaluation, arena_humansim, arena_viz, rerun_utils, human_steering (new 2026-07, ped-gui; pip deps ~none, qt via rosdep), rviz_utils (pip deps ~numpy only), arena_robots (declares own attrs+PyYAML though ament_cmake), arena_simulation_setup/src (real standalone-installable; the outer pyproject stays tool-config-only). Members omit requires-python (root's ==3.12.* drives resolution).
- Never-members (explicit list, conformance lint enforces no [project] on them): arena_training (py3.10), arena_planners/planners/* (py3.8-3.11), arena_planners/arena_planners bridge (dual-use SDK for planner subprocess envs), arena_isaac (runs in Isaac kit python 3.11; has a tool-config-only pyproject at its root since 2026-07, must stay [project]-less).
- Revalidated 2026-07-19 after 2.5 weeks of drift (ped-gui merge etc.): root flat dep list unchanged (only a ruff src path added), dead-dep list still valid (human_steering uses qt, not pyglet), toolchain unchanged; _meta/tools/pull sync line is now line 68.
- Members are uv-VIRTUAL (`[tool.uv] package = false`, no `[build-system]`): uv aggregates deps, colcon owns the build. Build via reduced-shim, see [[project_pyproject_only_ament_vaporware]]: pyproject `[project]` authoritative for name/version/dependencies/[project.scripts]; setup.py shrinks to packages/package_dir/data_files/zip_safe only; setup.cfg (script_dir -> lib/<pkg>) stays.
- Two channels, no cross-duplication: package.xml -> rosdep (ROS/system + rosdep-able pip; venv is --system-site-packages so those ARE importable); pyproject -> uv (pip-only). opencv-python/pyzmq/msgpack/msgpack-numpy leave root (rosdep keys in arena_planners bridge package.xml). NO per-member test extras: pytest/hypothesis are already test_depend rosdep keys in package.xml.
- Root = system deps only: colcon-common-extensions, rosdep, vcstool, pip, setuptools>=61,<80 (window is load-bearing: >=61 for the [project] merge colcon reads, <80 breaks colcon develop), huggingface_hub+PyYAML (_meta features), empy<4 + lark (launch/xacro substrate; <4 is load-bearing for rosidl). ros_introspection/aiomonitor -> dev group. Cross-cutting version policy (empy<4-style, future numpy-ABI pins) goes in root `[tool.uv] constraint-dependencies`, not in members.
- Membership DERIVED: a `colcon list`-based codegen tool writes explicit `[tool.uv.workspace] members` paths into root (no globs, no exclude needed), stamped with a hash of colcon list output; `arena uv` regenerates on mismatch (CI check-mode fails if stale). Predicate: in main colcon graph AND has `[project]`. Planners (py3.8) / training (py3.10) never members. Don't parse CMakeLists.
- Per-member dep lists DERIVED: AST imports + importlib.metadata.packages_distributions over ARENA_VENV, runtime-vs-test split, first-party/ROS filtered out; cross-checked with deptry.
- Guards, not one-shots: deptry per member in pre-commit/CI; shim-conformance lint (setup.py whitelist kwargs only, no [build-system] in virtual members, no PEP621 overlap); `arena uv add --package <member> <dep>` is the canonical way to add deps.
- Deferred to the very end (verify against lock first): drop dead deps defusedxml, netifaces, transforms3d, pathfinding3d, diffusers, rasterio, pyglet, lxml (8, confirmed dead by subagent sweep 2026-07-19). NOT dead, stay: google-genai (task_generator/utils/gpt.py uses `from google import genai` — the form my original grep missed — powers TM_Obstacles.PROMPT) and requests (arena_simulation_setup/scripts/generate_world, outside ruff+setup.py scope). Keep huggingface_hub. sim_setup's setup.py install_requires 'requests' is still a stale declaration to strip in Phase 1 (the live use is the scripts/ file, owned by sim_setup's [project]).

## Adversarial review amendments (4-axis subagent pass, 2026-07-19)
1. arena_planners BRIDGE becomes member #12 (real, has [build-system]). It satisfies the membership predicate anyway (in colcon graph + [project]); excluding it contradicted the predicate AND caused the opencv version cliff: pip-locked opencv-python 4.13 vs apt python3-opencv 4.10+dfsg, with a LIVE main-env import path (task_generator drl adapter -> arena_planners bridge collectors.py lazy `import cv2`). As member, its derived deps (incl. opencv-python, which its [project] currently omits) stay pip-locked. opencv/zmq/msgpack/msgpack-numpy still leave ROOT. Channel-rule amendment: when a live main-env import needs version fidelity, pip/lock wins over rosdep float.
2. Shim spec covers ALL entry-point groups: [project.scripts] AND [project.entry-points.<group>] (arena_bringup registers launch_ros.node_action). Phase-1 gate: diff built egg-info entry_points.txt vs pre-migration, not just colcon-build success.
3. Extractor + deptry scope MUST include scripts/ dirs and setup.py scripts= entries (the requests miss), not just the ruff-scoped tree. Known structural blind spot: arena_bringup/future.py importlib.import_module from launch-XML strings (only stdlib today; conformance lint should flag new users of that mechanism).
4. deptry needs a per-member ignore list for ROS/rosidl imports (*_msgs etc., no pip dist-info) derived from each package.xml's depends — else the guard is noise.
5. Reconciliation rule: dep derivation is Phase-0 bootstrap ONLY; thereafter member pyprojects are source of truth, deptry only checks, regen rewrites nothing but the members block.
6. CI staleness checker uses a stdlib package.xml walker honoring COLCON_IGNORE/AMENT_IGNORE (CI stays ROS-free; lint.yml has no ROS today and standing it up is unbudgeted). `colcon list` remains the in-container codegen source.
7. First-party edges to REAL members get declared: task_generator/arena_robots/arena_runtime -> arena_simulation_setup, task_generator -> arena_planners, via [project].dependencies + [tool.uv.sources] {workspace = true} (idiom already used by arena_training -> rosnav_rl). Virtual->virtual edges stay package.xml-only (uv cannot depend on virtual members).
- Named side item (pre-existing bug, not caused by migration): _meta/features/training/main installs py3.10-pinned arena_training editable into the SHARED py3.12 venv — the "training is isolated" premise is false today; fix separately.

## Quality-delta scorecard (subagent estimates vs status quo)
- Correctness/reproducibility: +0..+1 raw; ~+1.5 with amendments 1+3 (drift-resistance +2 is the anchor; rosdep-channel unlocked-float asymmetry mostly neutralized by amendment 1).
- Maintainability/DX: +1 (ownership legibility +2 is the single biggest win of the whole migration; codegen-in-file friction -1).
- Build/CI/ops: ~-3 raw; ~-1 with amendment 6 (real costs: stale-members hard-fail class, member-pyproject faults now couple uv+colcon; rollback is clean +1, containers unaffected).
- Isolation/ecosystem: +1 (ecosystem alignment +2; first-party-deps hole fixed by amendment 7).

## Feasibility verified statically 2026-07-02 (setuptools 79.0.1, colcon_python_setup_py 0.2.9, uv)
- colcon extraction runs `run_setup(stop_after='config')` -> parse_config_files -> [project] merge happens BEFORE colcon reads metadata; `_unify_entry_points` maps [project].scripts -> dist.entry_points. Shim residue has no PEP621 fields -> zero warnings.
- setuptools overlap: field in both -> warning, pyproject wins; setup.py-only PEP621 field -> _MissingDynamic warning + reset. Neither occurs with the shim.
- colcon_ros ament_python build implicitly creates resource marker + installs package.xml if data_files omit them (deprecated fallback) — keep them declared.
- arena_simulation_setup: colcon builds from the OUTER dir whose pyproject has no [project]; src/pyproject is uv-only.
- `uv sync` alone does NOT install member deps (virtual members can't be depended on) -> helper MUST pass --all-packages.
- arena_viz has no entry points -> its missing setup.cfg is a non-issue.
- No in-container spike needed anymore; post-Phase-1 plain `colcon build` sanity pass suffices.

## Tech checkin 2026-07-31
- Local toolchain unchanged (uv 0.9.26, setuptools 79.0.1, colcon_core 0.21.0): all source-verified semantics still hold. uv workspace membership semantics empirically tested on 0.9.26: explicit missing member paths TOLERATED, empty dirs (uninitialized submodules) and [project]-less pyprojects HARD ERROR, glob discovery errors on any matched non-project dir. Generator regenerating explicit members pre-sync is the tolerance mechanism.
- uv 0.12.0 (2026-07-28): no workspace-membership changes listed, but pre-release resolution default changed (re-lock diffs possible) and pylock/lockfile validation hardened. Re-run the T1-T7 membership acceptance tests before any container uv bump past 0.9.x.
- setuptools wall advancing: >=80 breaks colcon --symlink-install (colcon-core#696, fixed upstream in PR #699, newer than our 0.21.0); 80.1 set Oct 31 deadline for setup.py-install removal, pkg_resources removal Dec. The >=61,<80 pin stays mandatory until a colcon-core bump; pyproject-authoritative manifests are the survival path when colcon goes pip/PEP 660.
- New prior art: colcon-python-project-uv (Briancbn, PROTOTYPE, devel-only) = colcon + uv_build backend + workspace-level venv under install/. Validates the direction, unusable for us (uv_build backend, no ament data_files story). colcon-uv still 0.4.0/per-package.

## Phases
0. Read-only tools (discovery codegen + dep extractor + conformance lint); output proposed members + per-member dep lists. Review gate, no edits.
1. Add [project] + `[tool.uv] package=false` to members; shrink setup.py to shim; reconcile sim_setup (src single source, drop lxml+requests). Inert while root still owns the flat list. Gate: ruff clean; in-container colcon build sanity pass.
2. Flip root to system-deps + generated workspace block; remove opencv/zmq/msgpack/msgpack-numpy; wire `arena uv` (sync --all-packages, UV_PROJECT_ENVIRONMENT, staleness check) into _meta/tools/source:170 + pull:55. Gate: installed dist+version set unchanged (uv pip list diff) minus rosdep-moved.
3. Wire codegen + deptry + conformance lint into pre-commit/CI; add `arena uv add --package`.
4. Dead-dep drop (list above), final grep across launch/configs first, re-lock. Gate: lock diff = exactly dropped dists + now-unshared transitives.
