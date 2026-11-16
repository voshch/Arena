# task_generator obstacle task modes

`TM_Obstacles` and its shipped subclasses. Each mode's `reset()` returns
`(list[Obstacle], list[DynamicObstacle])`, which `Task.reset_obstacles` writes
into the episode's `ResetPlan` together with the mode's lights. `EpisodeNode` then
spawns them through `EnvironmentManager`.

## `TM_Obstacles` ABC

[`__init__.py`](__init__.py)

```python
class TM_Obstacles(TaskMode):
    async def reset(self, *, seed: int) -> Obstacles:
        return [], []
```

`Obstacles = tuple[list[Obstacle], list[DynamicObstacle]]`. The base
implementation returns empty lists; every subclass overrides `reset`.

`TM_Obstacles` is level-agnostic: it populates the whole compacted map (all loaded
levels at once) via `get_position(s)_on_map` with no `level_id`. Which levels exist is
decided at load time, not per reset: `world:=name` loads every level, `world:=name[0,3]`
loads only levels 0 and 3. Geometry-driven modes read `world_manager.world_compacted()`
(all levels merged into the map frame). Single-level worlds are the degenerate case and
behave exactly as before.

## Shipped modes

| Kind | Class | File | Behavior |
| --- | --- | --- | --- |
| `random` | `TM_Random` | [`random/`](random/) | samples N static and dynamic obstacles from model pools; counts and model lists are ROS params |
| `parametrized` | `TM_Parametrized` | [`parametrized/`](parametrized/) | loads a `ParametrizedConfig` by name from `arena_simulation_setup`; min/max counts per entry |
| `scenario` | `TM_Scenario` | [`scenario/`](scenario/) | reads `static` and `dynamic` lists from a world scenario YAML |
| `environment` | `TM_Environment` | [`environment/`](environment/) | places obstacle groups from an environment config into detected or declared rooms |
| `prompt` | `TM_Prompt` | [`prompt/`](prompt/) | LLM-driven obstacle generation; PROMPT registered per `human` value |

## Package structure

Each `TM_Obstacles` subclass is a package:

- `__init__.py` (eager): declares `_NS` and (optionally) a `ParamGroup` schema, then registers the mode on `OBSTACLES_MODES` (a `TaskModeRegistry` from `tasks/registry.py`) with `namespace=_NS` and `schema=<Group>`. Imported at node startup.
- `impl.py` (lazy): contains the class body. Imported only on first activation.

Parameters live under `task.<mode>.<leaf>` (e.g. `task.random.static.n`).

## Setting per-mode params: staged contract

All `task.*` writes go through `config/queue_episode`. The request carries the mode change (`modes`, or the `tm_obstacles` / `tm_robots` aliases) and either `params` (`rcl_interfaces/Parameter[]`, full names such as `task.random.static.n`) or the leaf-keyed `obstacles_params` / `robots_params` aliases (names **relative to the mode**, no `task.<mode>.` prefix, mapped to full names through the namespace of the mode the queue selects). The server stages them and applies at the next `lifecycle/reset_episode` boundary. Failures warn, never abort. Last-write-wins on duplicate names between resets.

The same request carries `human_params`, which are staged and applied at the same boundary in every obstacle mode. A name is `<backend namespace>.<param>`, e.g. `humansim.global_planner.resolution`, which is also the key of the launch arg. The request is rejected when a name sits under the reserved `human.` prefix or under no known namespace (`Constants.HUMAN_PARAM_NAMESPACES`). The active backend applies the names under its own namespace and skips the rest, so one suite runs unchanged under another backend. A set value stays until a later request overwrites it. See [humans/README.md](../../../../arena_runtime/arena_runtime/arena_runtime/humans/README.md#arena-adapter-humanarena).

A leaf is what's left after stripping `task.<mode>.`. For `task.random.static.n` the leaf is `static.n`; for `task.scenario.file` the leaf is `file`. The active mode is taken from the request's `modes` (`tm_obstacles` / `tm_robots`); sending `task.scenario.file` as a leaf-keyed alias name (full path) results in the server constructing `task.<mode>.task.scenario.file` and dropping it as undeclared.

Because all parameters are forward-declared at startup, raw `SetParameters` also works at any time for the full `task.<mode>.<leaf>` path; no activation ordering constraint.

### `TM_Random` params

Declared under the mode namespace (e.g. `task.random.*`):

| Param | Default | Description |
| --- | --- | --- |
| `static.n` | `[0, 0]` | `[min, max]` static obstacle count |
| `interactive.n` | `[0, 0]` | `[min, max]` interactive obstacle count |
| `dynamic.n` | `[0, 0]` | `[min, max]` dynamic obstacle count |
| `static.models` | `[Warehouse/Object/Barrel_Plastic]` | model name list, empty = all ObjectIdentifiers |
| `interactive.models` | `[Warehouse/Object/Barrel_Plastic]` | model name list, empty = all ObjectIdentifiers |
| `dynamic.models` | `[arenian]` | model name list, empty = all PedestrianIdentifiers |

### `TM_Parametrized` params

Declared under `task.parametrized.*`:

| Param | Default | Description |
| --- | --- | --- |
| `parametrized.file` | `'default.xml'` | `ParametrizedIdentifier` name to resolve |

### `TM_Scenario` params

Declared under `task.scenario.*`:

| Param | Default | Description |
| --- | --- | --- |
| `scenario.file` | first available scenario | scenario name within the active world |

`TM_Scenario` resolves the scenario via
`WorldIdentifier(world_name).resolve_sync().scenario(name).resolve_sync().load()`
and returns `scenario.objects` / `scenario.humans` unchanged on each reset.

### `TM_Environment` params

Declared under `task.environment.*`:

| Param | Default | Description |
| --- | --- | --- |
| `environment.file` | `'office.yaml'` | `EnvironmentIdentifier` name to resolve |

Groups from the environment config are placed into rooms. Rooms are either
taken from `world_manager.world.zones` (explicit zone declarations) or
detected from wall geometry via `_create_rooms_from_walls`.

## Zone references

`TM_Scenario` delegates zone-ref resolution to the `Scenario` loader in
`arena_simulation_setup`. A `pose:` or `waypoints:` entry given as a bare
zone name in a scenario file is resolved to a point in that zone at load time using a seeded
RNG (the seed comes from `_ctx.conf.RNG`), so replaying with the same
seed produces identical placements.

## `obstacles/prompt/`

`TM_Prompt` generates obstacle lists via an LLM. PROMPT registration follows
the `human` param, see
[PROMPT registration](../../../../arena_runtime/arena_runtime/arena_runtime/humans/README.md#prompt-registration),
so which implementation you get follows the `human` arg:

| `human:=` | Implementation | `generation_mode` |
|---|---|---|
| `arena` | [`prompt/arena.py`](prompt/arena.py) | `arena`, `behavior` |
| `hunav` | [`prompt/hunav.py`](prompt/hunav.py) | `behavior_tree`, `crowded_behavior_tree` |

The hunav path needs the optional `arena_hunav_sim_bridge` (BT reference doc
and Chroma DB) and `chromadb`. [`prompt_utils`](prompt/prompt_utils/__init__.py)
serves those names through a module `__getattr__` so importing the package
never requires them — only touching a hunav-only name does, and that raises an
`AttributeError` naming the missing dep rather than an import traceback.
Behaviour-tree prompt context lives in
[`prompt_utils/context_bt.py`](prompt/prompt_utils/context_bt.py), separate from
`context.py`, which carries the arena_humansim schema.

## Adding a new TM_Obstacles mode

1. Create `tasks/obstacles/<name>/` as a package.
2. In `__init__.py`: declare `_NS = _REGISTRY_NAMESPACE("<name>")`, define a `ParamGroup` subclass if the mode has tunable parameters (see the example in [tasks/README.md](../README.md#tm-package-structure)), e.g.

   ```python
   class MyParams(ParamGroup):
       STATIC_N = Param[tuple[float, float]]("static.n", [0, 0], "[min, max] count of static obstacles.", constraint=Pair(), parse=pair)
   ```

   then register the loader with `@OBSTACLES_MODES.register(Modes.TM_Obstacles.<NAME>, namespace=_NS, schema=MyParams)`.
3. In `impl.py`: define the class extending `TM_Obstacles`, build `MyParams(self.node, namespace=self.namespace())` in `__init__` and read `.value`; override `reset` to return `(list[Obstacle], list[DynamicObstacle])`.
4. Add `<NAME> = "<name>"` to `Modes.TM_Obstacles` in [`modes.py`](../../modes.py).
5. `declare_mode_schemas(layer, node)` picks up your schema automatically at node init.
