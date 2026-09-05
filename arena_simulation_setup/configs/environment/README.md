# Environment configs

An environment YAML defines named obstacle-group templates that the
`environment` obstacle task mode (`TM_Environment`,
[tasks/obstacles/environment/impl.py](../../../task_generator/task_generator/tasks/obstacles/environment/impl.py))
tiles into a world's free rooms/zones. Each file is a `dict` resolved by
`EnvironmentIdentifier`. This is a separate task mode from `parametrized`
(`TM_Parametrized`), which uses a different, XML-based config under
`arena_bringup/configs`.

## Location and resolution

`EnvironmentIdentifier` is backed by `EnvironmentResolver`, which looks under
`ASS_DIR / 'configs' / 'environment'`
([tree/configs/environment.py:15](../../src/arena_simulation_setup/tree/configs/environment.py#L15)).

`ASS_DIR` is the installed share path of `arena_simulation_setup` (or the
`ASS_DIR` env var when `ament_index_python` is absent).

Name resolution: `EnvironmentIdentifier('hospital')` probes
`configs/environment/hospital.yaml` then `hospital.json`. A name with either
suffix is taken as-is. The `environment` task mode's `<ns>.file` ROS param
takes the same form and defaults to `default`.

## Schema

```yaml
groups:
  - name: office              # group name, used in log/obstacle naming
    size: [1.5, 1.5]          # bounding box of the group [m x m], required
    margin: 0.5               # clearance added around the footprint when tiling [m], default 0.5
    rotations: [0, 180]       # candidate placement rotations [degrees], one drawn per placement, default [0]
    zones: [office_room]      # optional: restrict this group to rooms whose polygon exactly equals one of these named world zones
    entities:
      static:
        - position: [0.0, 0.0, 0.0]   # [x, y, yaw_deg] relative to group origin, required
          model: office_desk           # ObjectIdentifier name, required
        - position: [1, 0, -90]
          model: office_chair
      dynamic:
        - position: [0.0, 0.0, 0.0]   # [x, y, yaw_deg] relative to group origin, required
          model: office_worker         # HumanIdentifier name, required
          waypoints:                   # each entry either a world zone name (str) or an [x, y] coordinate pair
          - office_room
          - [2.0, 1.0]
```

A room without an explicit `zones:` list on any candidate group is filled
from whichever group is eligible for it (every group without `zones:`, plus
any group whose `zones:` contains a zone matching that room by polygon). One
eligible group is chosen at random per room. Groups are tiled repeatedly into
a room, left to right, bottom to top, in a grid stepped by
`size + margin`, until no more copies fit. One `rotations` value is drawn
independently for each placement. `entities.static`/`entities.dynamic`
default to `[]` when omitted.

`EnvironmentDescription` is an untyped `dict` subclass. No validation beyond
top-level `dict` is performed at load time
([tree/configs/environment.py:32](../../src/arena_simulation_setup/tree/configs/environment.py#L32)).
Malformed groups/entities surface as `KeyError`/`TypeError` from the task
mode at reset time, not as a load-time error.

## Shipped files

| File | Groups defined |
|---|---|
| `default.yaml` | `office`, `circle`, `cafeteria`, `residential`, `hospital`, `outdoor` |
| `hospital.yaml` | `hospital_reception` |
| `office.yaml` | `office_quad_short_tables`, `office_horizontal_long_tables`, `office_quad_short_tables_random`, `office_horizontal_long_tables_random` |
| `canteen.yaml` | `canteen_tables`, `canteen_waiting`, `canteen_bar`, `planter_bench`, `BarCabinet` |

## Adding a new environment

Create `configs/environment/<name>.yaml` following the schema above. Reference
it as `<name>` wherever an `EnvironmentIdentifier` is expected, e.g. as the
`environment` task mode's `<ns>.file` param.
