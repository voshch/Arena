# Auditory simulator dispatch

Entry point: [`auditory.launch.py`](auditory.launch.py).

Called from `task_generator.launch.py`'s OpaqueFunction (once per
environment, next to the human dispatch) with `simulator` (the auditory key),
`namespace`, `environment_namespace`, and every `auditory.*` launch argument.
Its job is to select and delegate to one auditory backend.

## SelectAction dispatch

Keys are `Constants.AuditorySimulator` values:

| Key | Action |
| --- | --- |
| `none` | empty group, no nodes |
| `arena` | includes [`arena/arena.launch.py`](arena/arena.launch.py) |

Selected with `auditory:=<key>` at the top level.

## Node side

`task_generator/simulators/auditory/` mirrors the human registry:
`AuditorySimulatorRegistry` yields a `BaseAuditorySimulator` per key. The
backend class declares what the task generator must provide, today only
`requires_map_server`, which the arena backend sets because the propagation
node builds its acoustic scene from the map topic.

## Layout

```
launch/auditory/
├── auditory.launch.py     dispatcher
└── arena/
    └── arena.launch.py    arena_auditory node stack
```

The nodes themselves live in the `arena_auditory` package, see its
[README](../../../arena_auditory/README.md) for parameters and topics.
