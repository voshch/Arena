# arena_viz

Vendor-neutral visualization contract for Arena. Adapters declare displays
in this vocabulary; each visualizer backend implements a renderer per kind.

This package is **only** the contract: an enum of canonical display kinds
and a small styling dataclass. No ROS deps, no renderers.

## What lives here

- [`DisplayKind`](arena_viz/kinds.py): the canonical enum of display kinds
  (`MAP`, `TF`, `PEDESTRIANS`, `ROBOT_MODEL`, `ODOM`, `LASER_SCAN`,
  `POINTS_3D`, `IMAGE`, `IMU`, `FOOT_CONTACT`, `PATH`, `POSE`, `POLYGON`,
  `TRAJECTORY`, `PLANNING_SCENE`). The vocabulary every adapter speaks.
- [`StyleSpec`](arena_viz/style.py): frozen styling dataclass with viz-neutral
  fields (`color`, `alpha`, `line_width`, `enabled`, `decay`, `latched`) plus an
  `extra` escape hatch keyed by visualizer name for per-viz nudges
  (e.g. `extra={"rviz": {"Color Scheme": "costmap"}}`). `latched=True` marks a
  transient-local topic, backends subscribe with matching QoS so late joiners
  get the stored sample.

## Who uses it

| Producer | What it does |
|---|---|
| [`task_generator/tasks/robots/adapters`](../../task_generator/task_generator/tasks/robots/adapters) | Declares `AdapterDisplayHint(kind=DisplayKind.X, style_json=StyleSpec(...).to_json())` on each adapter. |
| [`task_generator.node._publish_viz_manifest`](../../task_generator/task_generator/node.py) | Emits env-level + per-robot displays as `task_generator_msgs/AdapterVizManifest`. |

| Consumer | Renderer registry |
|---|---|
| [`rviz_utils`](../rviz_utils) | One file per kind under `rviz_utils/renderers/`, returns rviz Displays YAML dict. |
| [`rerun_utils`](../rerun_utils) | One file per kind under `rerun_utils/renderers/`, creates a ROS subscription that calls `rr.log(...)`. |

Each backend's `renderers/__init__.py` asserts at import time that every
`DisplayKind` member has a registered renderer (warn-and-skip variants
count). New backends drop in by mirroring that pattern.

## Adding a new kind

1. Add the member to `DisplayKind`.
2. Add a renderer for it in every backend (`rviz_utils/renderers/`,
   `rerun_utils/renderers/`, ...). The completeness assert will refuse to
   load until all backends have one.
3. Update an adapter to emit the new kind (or update
   `_publish_viz_manifest` for an env-level kind).

`MAP` and `PATH` are good reference renderers in both backends.

## Adding a new viz backend

Mirror `rerun_utils` or `rviz_utils`:

```
utils/<backend>_utils/
  <backend>_utils/
    renderers/
      __init__.py    # imports all renderer modules, REGISTRY assert
      _registry.py   # REGISTRY dict + register decorator
      map.py, tf.py, path.py, ...  # one per DisplayKind
```

The backend subscribes to `{task_generator_node}/state/viz_manifest`,
dispatches each `AdapterDisplay` to `REGISTRY[DisplayKind(d.kind)]`.
