# arena_simulation_setup

Worlds, assets, environment configs, and the `Identifier` registry for Arena.
Simulators and `task_generator` resolve world layouts, 3D models, materials,
pedestrian agents, and environment templates through the types defined here.

## Guides

- [Identifier registry](src/arena_simulation_setup/tree/README.md): `Identifier`
  ABC, resolver hierarchy, shipped Identifier types, adding a new one.
- [Worlds](worlds/README.md): per-world directory layout, `world.yaml` schema,
  scenario schema, local asset overrides, `WorldIdentifier` resolution.
- [Environment configs](configs/environment/README.md): what an environment
  YAML binds; obstacle group schema; `EnvironmentIdentifier` resolution.
- [Wall presets](configs/walls/README.md): `WallDescription` YAML schema;
  sub-wall types; how `world.yaml` references a wall preset by `kind:`.
- [Authoring a world](AUTHORING.md) — end-to-end guide: create dir, author
  `world.yaml`, generate map, add scenario, validate.
- [Generative worlds](src/arena_simulation_setup/generative/world/README.md):
  the generator registry, the layout IR they compile to, the sketch and letter
  grammars, and the ROS surface the rviz panel drives.
- [Generative scenarios](src/arena_simulation_setup/generative/scenario/README.md):
  the `generate_scenario` verbs, zone roles from the world's KG, the base
  populations, the prompt-driven timeline cases and the benchmark suites over them.

## CLI

| Script | Usage | Effect |
|---|---|---|
| `generate_scenario` | `generate_scenario {zones,classify,populate,promptgen,suite} <world> [--write] ...` | Prints a world's zone roles, asks a model for the missing ones, writes the four base scenarios, realizes prompts on them as timeline cases and writes a benchmark suite over the result. Writing verbs are dry runs without `--write`. See the [generator README](src/arena_simulation_setup/generative/scenario/README.md). |
| `generate_world` | `generate_world "<prompt>" [-e endpoint] [-o outdir]` | Posts a natural-language prompt to a generation server; extracts the returned zip into `worlds/<outdir>/` |
| `model_staging` | `model_staging <install_dir>` | Creates symlinks in `<install_dir>` for all known robot models and writes a `deps` file |
| `nav_at_r` | `nav_at_r <world name or dir> [--robot NAME\|DIR \| --radius M] [--walls-only] [--start X,Y] [--image PNG] [--json]` | Prints the NAV@r reachability of a world per level for a footprint radius: value, traversable components, largest share and cut-off zones. See [AUTHORING.md](AUTHORING.md#reachability-navr). |
| `preload_world` | `preload_world [<world_name>] [--no-scenarios] [--dry-run] [--sounds]` | Resolves every identifier the world and its scenarios reference, downloading what is missing. `--dry-run` reports without transferring. `--sounds` adds the default sound of every sound kind, alone or with a world, and `arena launch` passes it under `acoustics:=arena`. Reached from the CLI as `arena preload`. |
| `touch_world` | `touch_world <world_name> [--all \| --migrate] [--resolution N] [--assets color] ...` | Renders a preview `map.png` + `map.yaml` into the world dir for inspection; `--all` re-saves `world.yaml` as canonical format 2 with `kg.yaml` and `map/<level id>.png`, `--migrate` rewrites a format 1 world in place. The runtime renders its own map in-process, so this is an authoring aid, not required after editing. |

Scripts live under [scripts/](scripts). To fetch a single asset by identifier, use `arena asset pull <kind> <name>`. `arena asset expire [<kind> <name>]` marks cached assets expired, all of them when bare, so each is downloaded again on its next use.

## MCP domains

The package contributes four domains to the `arena_mcp` server, listed with
`python3 -m arena_mcp ls <domain>`.

- `worlds` works offline on the files through the same resolvers and loaders
  the runtime uses: inspect a world with a preview image per level, validate
  (strict format 2, the legacy loader for format 1), migrate a world directory
  or repo world as `touch_world --migrate` does, write, NAV@r with its
  component maps, the KG with its check and the vocabulary packages, scenarios
  (list, show, validate against the world's zones, write), and the task and
  condition grammar parsed and rendered back to text. A world name takes a
  level filter, `name[0,1]`, or a composition, `a,b[1]`. `worlds_write` and
  `worlds_scenario_write` only write into the writable local root
  `$ARENA_ASSETS_DIR_LOCAL/worlds`, a scenario only into a world that lives
  there. `arena://worlds/{name}` serves the inspect summary. Previews render
  at the world generator node's pixel budget.
- `assets` runs the `arena asset` verbs: kinds, find, ls, pull, and push with
  the preflight that refuses references nobody else could resolve (marked
  destructive and open-world, it needs `GCS_ACCESS_TOKEN`), `only` restricting it to
  the files matching a glob and no name publishing every local asset. `assets_ls` and
  `assets_find` with kind `world` list the worlds and every resolver's verdict.
  `assets_import_sources` lists the datasets `arena asset import` converts
  from, `assets_import_items` one dataset's item ids, and `assets_import` runs
  the import of chosen items as a job whose result lists the item paths.
  `assets_retarget_animation` runs the `animation_retarget` tool on SMPL-X
  files as a job whose result lists the clip directories.
  `assets_articulation` resolves one object and lists its links, joints (type,
  parent, child, limits, speed), named states, detents, field families (each
  family with its member joints) and swept box, or says it is rigid.
- `worldgen` runs the world generators in process: preview with diagnostics,
  save under the writable world root (in place over an existing world with
  `overwrite`, as the generator node saves), and queue a world on an env with
  its episode binding.
- `airchitect` drives the external airchitect generation stack through its
  webapp HTTP API, the stack's only published endpoint. The base URL is
  `AIRCHITECT_URL` (default `http://localhost:3000`), reached from the
  container over host networking. The domain is always served. While the stack
  is down every tool fails within seconds with the deploy command
  (`cd ~/airchitect && ./airchitect deploy up --cpu --arena text_kg`), so a
  stack started after the server is picked up without a restart.
  - `airchitect_status` checks readiness: the models per stage with their
    defaults, and backend availability from `models/score`.
  - A session is a uuid in the stack's Redis, kept for one hour after its last
    write, with the stages `prompt`, `floorplan`, `decoration` and `scenario`.
    Stage requests are deferred, so every stage run is a job that polls the
    stage record until the stack stores `completed` or `error`.
  - `airchitect_generate` runs the whole pipeline as one job: prompt, several
    floor plan tries keeping the one with the fewest graph edges left without a
    door and then the fewest `door_auto_*` doors, decoration, the scenario
    stage when a scenario prompt is given, and the zip export. The zip is
    unpacked into the writable world root
    (`$ARENA_ASSETS_DIR_LOCAL/worlds/<name>`), an existing world is replaced
    only with `overwrite`, and the world is checked with the `worlds` verdict,
    NAV@r per level (the `nav_at_r` default radius unless given) and a preview
    per level.
  - Every job keeps its stage requests and responses as JSON, the exported
    `world.zip`, the previews and the NAV@r images under
    `$ARENA_DATA_DIR/mcp/airchitect/<run id>/`, where the run id is the id of
    the job that started it. `resume` takes that id and reuses the completed
    stages in the same session.
  - `airchitect_session`, `airchitect_stage`, `airchitect_kg_edit` and
    `airchitect_export` inspect, rerun, edit and export an existing session.
    `airchitect_ingest` seeds a session from an Arena world, its linked levels
    written out.
  - Job progress names the current stage, and `jobs_status`, `jobs_log`,
    `jobs_wait` and `jobs_cancel` follow a run. Cost is reported as
    unavailable: the stack exposes it only in its SSE events, and only when it
    runs with `COST_DEBUG=1`.

## Human assets and skeletal articulation

### arenian actor

`arenian` is the default pedestrian. Like every other human it is an `arena_humans`
bundle (MakeHuman skin, CMU clips, `cmu_mb` rig) fetched from the asset bucket, not
shipped in the repo. The SDF `<actor>` form is required so Gazebo registers the skinned
mesh and the clips that `PedSkeletonPlugin` poses.

**gpu_lidar implication.** Arena's lidar sensor is `gpu_lidar`, a rendering
sensor that forces a server-side render scene even in headless mode.  Because
`PedSkeletonPlugin` positions and animates the actor in that scene, the robot's
lidar perceives a moving, articulated body, not a static box or capsule, making
the in-sim animation perceptually load-bearing in headless RL training, not only
cosmetic.

`PedSkeletonPlugin` (package `arena_gz_plugins`) is the gz-sim 8 C++ plugin that
subscribes `arena_peds`, positions each actor via `TrajectoryPose`, and poses its
skeleton per bone via `BoneTransforms` from the wire `joint_state`, through the
same bone map as the Isaac provider (`utils/msgs/arena_people_msgs/bone_map.json`), so
both simulators render the same articulated gait.  `BoneTransforms` is an Arena
addition to gz-sim carried by the `gz-sim` overlay in `_meta/repos/gazebo.repos`
and built by `arena feature gazebo update`.  Without the overlay or a bone map the
plugin falls back to scrubbing the `walk` clip via `AnimationTime`.  Enabled by
default via the `ped_skeleton_enabled` launch arg in `arena_bringup`.  The plugin
lives in its own repo, [arena_gz_plugins](https://github.com/voshch/arena_gz_plugins),
pulled into the workspace via `_meta/repos/gazebo.repos`.

## Internals

Key `Identifier` types from `arena_simulation_setup.tree`:

- **`WorldIdentifier`** ([tree/World/view.py](src/arena_simulation_setup/tree/World/view.py)):
  resolves a world name through `ARENA_WORLD_PATH`, the writable local root
  `$ARENA_ASSETS_DIR_LOCAL/worlds`, `ASS_DIR / 'worlds'` and the world buckets, and `.load()` returns
  a `WorldView` that exposes `view.load()` (the `World`), `view.map_yaml(level_id)`
  and `view.scenario`.
- **`ObjectIdentifier`** ([tree/assets/Object.py](src/arena_simulation_setup/tree/assets/Object.py)):
  resolves `<domain>/Object/<name>` to a `ModelWrapper` (SDF + USD).
- **`HumanIdentifier`** ([tree/assets/Human.py](src/arena_simulation_setup/tree/assets/Human.py)):
  resolves `<domain>/Human/<name>` to a `ModelWrapper` (SDF only).
- **`MaterialIdentifier`** ([tree/assets/Material.py](src/arena_simulation_setup/tree/assets/Material.py)):
  resolves `<domain>/Material/<name>` to a `Material`; supports `tint`
  modifier.
- **`WallIdentifier`** ([tree/Wall.py](src/arena_simulation_setup/tree/Wall.py)):
  resolves `<domain>/Wall/<name>/<name>.yaml` to a `WallDescription`;
  realizes to `(WallSegment[], Obstacle[])`.
- **`EnvironmentIdentifier`** ([tree/configs/environment.py](src/arena_simulation_setup/tree/configs/environment.py)):
  resolves `configs/environment/<name>.yaml` to an `EnvironmentDescription`
  (untyped dict).
- **`ParametrizedIdentifier`** ([tree/configs/parametrized.py](src/arena_simulation_setup/tree/configs/parametrized.py)):
  resolves an XML file in `arena_bringup/configs/parametrized/` to a
  `ParametrizedConfig` with `STATIC`, `INTERACTIVE`, and `DYNAMIC` obstacle lists.

See [tree/README.md](src/arena_simulation_setup/tree/README.md) for resolver
order, the `DynamicPaths` singleton, and how to add a new Identifier.

[utils/llm.py](src/arena_simulation_setup/utils/llm.py) holds the LLM client:
`from arena_simulation_setup.utils.llm import genai` (or `openai`) imports the
library on first access, redirected to a local endpoint when `LLM_API_ENDPOINT`
is set and passed through unchanged otherwise. `DEFAULT_MODEL` is
`$ARENA_PROMPT_LLM`, default `gemini-3.5-flash`, the model of
`tm_obstacles:=prompt`.
