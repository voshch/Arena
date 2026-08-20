# Task mode configs

Configs under this directory are passed to the task-generator fleet manager
via the `task.config` launch argument.

## Schema

Defined in [`SCHEMA.yaml`](SCHEMA.yaml):

```yaml
task_modes:
  - kind: STRING          # TM_Robots kind - matches Constants.TaskMode.TM_Robots enum value
    produces: GOTO_POSE   # task kind produced; matched against robot.accepts
    assignments:          # list of robot names to pin; [] means pool (greedy allocation)
      - ROBOT_NAME
    config: {}            # arbitrary dict passed through to the TM loader
```

Each entry is a `TaskModeSpec`. The list is processed by the fleet manager in
order (specs earlier in the list get priority during pool allocation).

### Fields

| Field | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | TM_Robots subclass key (`random`, `explore`, `guided`, `scenario`, `demo`, `null`) |
| `produces` | string | no | Task kind this mode emits (default `GOTO_POSE`). Must be in the target robot's `accepts` set |
| `assignments` | list[string] | no | Pin specific robots by name. Empty list = pool (first-fit) |
| `config` | dict | no | Passed through to the TM loader unchanged |

## Allocation rules

See [task_generator/task_generator/tasks/robots/README.md](../../../task_generator/task_generator/tasks/robots/README.md#fleet-manager)
for the full description. Summary:

1. **Pinned first** - robots named in `assignments` are bound to that spec.
   Their `accepts` must include the spec's `produces`; duplicate pins are an
   error.
2. **Pool next** - unpinned robots join the first unpinned spec whose
   `produces` is in their `accepts`. Greedy / first-fit.
3. **Null sink** - a spec with `kind: null` absorbs all still-unallocated
   robots.

## Legacy shorthand

When `task.config` is empty (the default), `task_generator.launch.py` synthesizes a
single-entry config from the `task.robots` arg:

```yaml
task_modes:
  - kind: <task.robots value>
    produces: GOTO_POSE
    assignments: []
    config: {}
```

If both `task.config` and `task.robots` are set explicitly, `task.config` wins.

## Shipped configs

| File | Description |
|---|---|
| [`SCHEMA.yaml`](SCHEMA.yaml) | Schema / template - not a runnable config |
| [`default.yaml`](default.yaml) | Single `explore` mode, pool allocation, no pinning |

## Examples

Single mode, all robots explore:

```yaml
task_modes:
  - kind: explore
    produces: GOTO_POSE
    assignments: []
    config: {}
```

Two-fleet split - jackal_0 follows a scenario, everything else explores:

```yaml
task_modes:
  - kind: scenario
    produces: GOTO_POSE
    assignments: [jackal_0]
    config: {}
  - kind: explore
    produces: GOTO_POSE
    assignments: []
    config: {}
```

Mixed fleet with a null sink for unmatched robots:

```yaml
task_modes:
  - kind: random
    produces: GOTO_POSE
    assignments: []
    config: {}
  - kind: null
    assignments: []
    config: {}
```
