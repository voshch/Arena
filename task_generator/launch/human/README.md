# Human simulator dispatch

Entry point: [`human.launch.py`](human.launch.py).

Called from `task_generator.launch.py`'s OpaqueFunction (once per
environment) with `simulator` (the human-sim key) and `namespace`. Its job
is to select and delegate to one human-simulation backend.

## SelectAction dispatch

The selector expression is `LaunchConfiguration('simulator')`. Registered
keys:

```
simulator key  →  action
──────────────────────────────────────────────────────────────────
dummy          →  empty GroupAction (no nodes)
isaac          →  empty GroupAction (no nodes)
arena          →  arena_humansim/arena_humansim.launch.py
```

The `simulator` `LaunchArgument` is declared after the `SelectAction` is
built; its `choices` are derived from `launch_human_simulator.keys`.

The `human` arg (not the `sim` arg) is passed here as `simulator`. The default
mapping is:
- `sim=dummy` → `human=dummy`
- `sim=gazebo` → `human=arena`
- `sim=isaac` → `human=arena`

This is resolved in `task_generator.launch.py` and can be overridden
explicitly with `human:=<key>`.

## Per-human-sim subdir layout

```
launch/human/
├── human.launch.py        — dispatcher
├── hunav/
│   └── hunav.launch.py             — hunav_sim agent manager
└── arena_humansim/
    └── arena_humansim.launch.py    — arena_humansim node (subsystem mode)
```

`hunav` and `arena` are both registered and selectable; they are alternative
backends, never concurrent — the node resolves exactly one from
`HumanSimulatorRegistry` per the `human` arg. `hunav` additionally requires the
optional `hunav_sim` / `arena_hunav_sim_bridge` deps from `arena.repos`; its
factory imports lazily, so leaving them uninstalled costs nothing until you
select `human:=hunav`.

### arena_humansim/arena_humansim.launch.py

Includes the `arena_humansim` package launch in `subsystem` mode in the given
namespace, so the node is driven externally by the task generator's
`ArenaHumanSimulator` adapter over services and the `world_state` topic.

| Arg | Default | Meaning |
|---|---|---|
| `use_sim_time` | `true` | Passed to the node as a bool parameter |
| `namespace` | (required) | ROS namespace for the arena_humansim node |

## Adding a new human simulator

1. Create `launch/human/<name>/<name>.launch.py`.
2. In `human.launch.py`, call
   `launch_human_simulator.add("<name>", IncludeLaunchDescription(...))`.
3. Update the `human` default expression in `task_generator.launch.py` to map
   the appropriate `sim` values to the new key.
