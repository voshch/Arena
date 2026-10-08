# Acoustics simulator dispatch

Entry point: [`acoustics.launch.py`](acoustics.launch.py).

Called from `task_generator.launch.py`'s OpaqueFunction (once per
environment, ahead of the human dispatch, only when `acoustics` is not
`none`) with `simulator` (the acoustics key), `namespace` and
`environment_namespace`. Its job is to select and delegate to one acoustics
backend. Every launch configuration of the parent stays visible to the
backend, so undeclared `auditory.*` keys reach it by prefix.

## SelectAction dispatch

Keys are `Constants.AcousticsSimulator` values:

| Key | Action |
| --- | --- |
| `none` | empty group, no nodes |
| `arena` | includes [`arena/arena.launch.py`](arena/arena.launch.py) |

Selected with `acoustics:=<key>` at the top level.

## arena backend

[`arena/arena.launch.py`](arena/arena.launch.py) is a shim like
`human/arena_humansim/arena_humansim.launch.py`. It fails with the
`arena feature auditory install` hint when the `arena_auditory` package is
missing. Otherwise it strips the `auditory.` prefix from every launch
configuration that carries it and includes `arena_auditory`'s
`launch/arena_auditory.launch.py` in isolation with:

| Arg | Value |
| --- | --- |
| `<key>` | `auditory.<key>` of the parent, forwarded as node param `<key>` to every stack node |
| `debug.map_source` | passed through unchanged when set |
| `namespace` | task generator node namespace, the stack nodes live below it |
| `env.ns` | absolute env namespace |
| `hearing` | `robot.hearing`, `srp` and `seld` default `array.spec` to `four_mic` |

`arena_auditory.launch.py` declares the launch-facing keys itself, with
their descriptions, from `arena_auditory/params.py`. Empty values keep the
node default. The nodes, their parameters and topics are documented in the
[arena_auditory README](../../../arena_auditory/README.md).

## Node side

`task_generator/simulators/acoustics/` mirrors the human registry:
`AcousticsSimulatorRegistry` yields a `BaseAcousticsSimulator` per key.

| Member | Purpose |
| --- | --- |
| `requires_map_server` | the arena backend sets it, the propagation node builds its acoustic scene from the map topic |
| `displays()` | env-level RViz displays |
| `robot_displays(robot)` | per-robot RViz displays, the robot's `_acoustics` manifest entry |
| `plugins()` | RViz panels and tools (`AdapterPlugin`), carried by the viz manifest |
| `recorded_topics()` | `RecordedTopic` rows, published latched on `state/recorded_topics` with the hearing rows |
| `pedestrian_hearing(human)` | a `PedestrianHearing` the backend feeds with `on_heard(agent_id, kind, source_id, audible)` |
| `emit_sounds(sounds)` | the sounds module's `SoundEmission`s of one tick, a no-op by default |

`simulators/acoustics/arena/` is the only task_generator code importing
`arena_auditory` or `arena_auditory_msgs`, always inside methods, so
importing task_generator never needs the feature. It publishes
`ContinuousAudioSourceState` on `continuous_audio_sources`, maps WAV
variants to `wav_loop`, and repeats a sounding source that stopped being
listed `INACTIVE_REPEATS` times as inactive.

## Layout

```
launch/acoustics/
|-- acoustics.launch.py    dispatcher
`-- arena/
    `-- arena.launch.py    install check, shim into arena_auditory's arena_auditory.launch.py
```

The sounds module needs only the sound catalog in arena_simulation_setup.
With `acoustics:=none` sounds from `auditory.static_sounds` or
`task.modules:=sounds` exist as entities but nothing hears them, the launch
warns once.
