# task_generator obstacle task modes

`TM_Obstacles` and its shipped subclasses. Each mode's `reset()` returns
`(list[Obstacle], list[DynamicObstacle])` consumed by `EnvironmentManager`.

## `TM_Obstacles` ABC

[`__init__.py:8`](__init__.py#L8)

```python
class TM_Obstacles(TaskMode):
    async def reset(self, **kwargs) -> Obstacles:
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
| `prompt` | `TM_Prompt` | [`prompt/`](prompt/) | LLM-driven obstacle generation; PROMPT registered per `BaseHumanSimulator` subclass |

## Package structure

Each `TM_Obstacles` subclass is a package:

- `__init__.py` (eager): declares `_NS` and (optionally) `_declare_schema`, then registers the mode on `OBSTACLES_MODES` (a `TaskModeRegistry` from `tasks/registry.py`) with `namespace=_NS` and `schema=_declare_schema`. Imported at node startup.
- `impl.py` (lazy): contains the class body. Imported only on first activation.

Parameters live under `task.<mode>.<leaf>` (e.g. `task.random.static.n`).

## Setting per-mode params: staged contract

All `task.*` writes go through `config/queue_episode`. The request carries the mode change and a leaf-keyed `obstacles_params` / `robots_params` payload (`rcl_interfaces/Parameter[]`, names **relative to the mode**, no `task.<mode>.` prefix). The server stages them and applies at the next `lifecycle/reset_episode` boundary. Failures warn, never abort. Last-write-wins on duplicate leaf keys within an axis between resets.

A leaf is what's left after stripping `task.<mode>.`. For `task.random.static.n` the leaf is `static.n`; for `task.scenario.file` the leaf is `file`. The active mode is taken from the request's `tm_obstacles` / `tm_robots`; sending `task.scenario.file` as a param name (full path) results in the server constructing `task.<mode>.task.scenario.file` and dropping it as undeclared.

Because all parameters are forward-declared at startup, raw `SetParameters` also works at any time for the full `task.<mode>.<leaf>` path; no activation ordering constraint.

### `TM_Random` params

Declared under the mode namespace (e.g. `task.random.*`):

| Param | Default | Description |
| --- | --- | --- |
| `static.n` | `[5, 15]` | `[min, max]` static obstacle count |
| `dynamic.n` | `[1, 5]` | `[min, max]` dynamic obstacle count |
| `static.models` | *(all ObjectIdentifiers)* | model name list |
| `dynamic.models` | *(all PedestrianIdentifiers)* | model name list |

### `TM_Parametrized` params

Declared under `task.parametrized.*`:

| Param | Default | Description |
| --- | --- | --- |
| `parametrized.file` | `''` | `ParametrizedIdentifier` name to resolve |

### `TM_Scenario` params

Declared under `task.scenario.*`:

| Param | Default | Description |
| --- | --- | --- |
| `scenario.file` | first available scenario | scenario name within the active world |

`TM_Scenario` resolves the scenario via
`WorldIdentifier(world_name).resolve_sync().scenario(name).resolve_sync().load()`
and returns `scenario.static` / `scenario.dynamic` unchanged on each reset.

### `TM_Environment` params

Declared under `task.environment.*`:

| Param | Default | Description |
| --- | --- | --- |
| `environment.file` | `'default.json'` | `EnvironmentIdentifier` name to resolve |

Groups from the environment config are placed into rooms. Rooms are either
taken from `world_manager.world.zones` (explicit zone declarations) or
detected from wall geometry via `_create_rooms_from_walls`.

## Zone references

`TM_Scenario` delegates zone-ref resolution to the `Scenario` loader in
`arena_simulation_setup`. `pose_ref` and `waypoint_refs` declared in a
scenario file are resolved against named zones at load time using a seeded
RNG (the seed comes from `node.conf.General.RNG`), so replaying with the same
seed produces identical placements.

## `obstacles/prompt/`

`TM_Prompt` ([`prompt/arena.py`](prompt/arena.py), [`prompt/hunav.py`](prompt/hunav.py)) generates obstacle lists
via an LLM. PROMPT registration is per-`BaseHumanSimulator` subclass, see
[PROMPT registration](../../simulators/human/README.md#prompt-registration).

## Adding a new TM_Obstacles mode

1. Create `tasks/obstacles/<name>/` as a package.
2. In `__init__.py`: declare `_NS = _REGISTRY_NAMESPACE("<name>")`, define `_declare_schema(node, ns)` if the mode has tunable parameters (using helpers from [`arena_rclpy_mixins.declarations`](../../../../utils/arena_rclpy_mixins/arena_rclpy_mixins/declarations.py), e.g. `declare_int_pair`, `declare_catalog`), then register the loader with `@OBSTACLES_MODES.register(Constants.TaskMode.TM_Obstacles.<NAME>, namespace=_NS, schema=_declare_schema)`.
3. In `impl.py`: define the class extending `TM_Obstacles`; override `reset` to return `(list[Obstacle], list[DynamicObstacle])`.
4. Add `<NAME> = "<name>"` to `Constants.TaskMode.TM_Obstacles` in [`constants/__init__.py`](../../constants/__init__.py).
5. `walk_schemas` (module-level in `tasks/registry.py`) picks up your schema automatically at node init.
