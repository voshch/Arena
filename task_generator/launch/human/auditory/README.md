# Auditory Module

The auditory module adds sound events to the Arena human simulation path. It is
enabled with `human:=arena enable_auditory:=true`; with
`enable_auditory:=false`, the same Arena human simulator runs without the
auditory nodes.

## Features

- Human sound events: moving pedestrians emit `footstep`; nearby facing
  pedestrians emit `greeting`.
- Sound-to-material matching: footstep events include floor/material semantic
  tags so playback can choose a matching sample, for example default vs
  walnut-plank footsteps.
- Sound propagation: `sound_propagation_node` converts `SoundEvent` messages
  into `HeardSoundEvent` messages using listener positions, distance loss,
  wall/material attenuation, optional pyroomacoustics RIRs, and one-door
  coupling between adjacent acoustic zones.
- Robot hearing: `robot_hearing_node` listens for `HeardSoundEvent`, discovers
  robots from `state/robots`, republishes per-robot heard events, and publishes
  an RViz text marker when a robot hears a `greeting`.
- Audio playback: `human_sound_playback` plays configured sound assets from
  `config/auditory/acoustic_assets.yaml`.
- Robot motor sound: `robot_sound_node` can publish robot motor `SoundEvent`
  messages from robot odometry.
  
 Expected nodes when enabled include:

- `sound_propagation_node`
- `robot_sound_node`
- `robot_hearing_node`
- `human_sound_playback`
- `sound_propagation_visualizer` (when explicitly enabled)

## Main Topics

- `human_sound_events`: emitted `SoundEvent` stream.
- `heard_sound_events`: propagated `HeardSoundEvent` stream.
- `state/robots`: robot fleet metadata used by propagation, robot sound, and
  robot hearing nodes.
- `<robot_name>/heard_sound`: per-robot heard event output.
- `<robot_name>/heard_sound_marker`: RViz text marker for heard greetings.
- `sound_propagation_markers`: RViz source/portal/listener paths.

## One-door pyroomacoustics coupling

Enable the pyroomacoustics backend and optional RViz path visualizer with:

```bash
arena launch \
  enable_auditory:=true \
  propagation_backend:=pyroomacoustics \
  enable_sound_visualization:=true
```

On world load, Arena pairs each authored door with the acoustic zone touching
the other side and constructs an `AcousticWorldGraph`. A cross-zone RIR is
available only when source and listener zones are directly adjacent through
one paired door. The renderer convolves the source-to-door early response with
the door-to-listener room response and applies `portal_loss_db`. It treats the
authored portal as acoustically open; dynamic door-state coupling is not part
of this one-door implementation.

`HeardSoundEvent.propagation_level` remains the model capability level for
compatibility. Inspect these fields for the actual route:

- `propagation_backend`: `pyroomacoustics_same_room`,
  `pyroomacoustics_one_door`, `level3`, or a legacy path.
- `used_backend_fallback` and `backend_fallback_reason`: whether and why the
  requested pyroomacoustics route could not run.
- `portal_id` and `portal_position`: the paired door used by one-door coupling.

The propagation node logs paired/unpaired doors, acoustic-zone coverage
warnings, and each distinct backend route. Playback logs its independently
verified `playback_backend` and dry/silent fallback reason. Its five-second
diagnostics include portal RIR cache entries, hits, and misses.

The RViz marker colors are cyan for same-room pyroomacoustics, purple for a
one-door route, orange for Level 3, and red for a fallback.

## Tests

Run the auditory ROS tests:

```bash
python3 -m pytest task_generator/tests/ros/test_sound_event.py -q
```

Run only the full auditory round-trip test:

```bash
python3 -m pytest \
  task_generator/tests/ros/test_sound_event.py::test_auditory_round_trip_greeting_reaches_robot_marker \
  -q
```

The round-trip test checks:

1. A synthetic `SoundEvent` with `sound_type="greeting"` is published.
2. `sound_propagation_node` creates a `HeardSoundEvent` for `robot:robot1`.
3. `robot_hearing_node` republishes it on `robot1/heard_sound`.
4. `robot_hearing_node` publishes an RViz text marker on
   `robot1/heard_sound_marker`.
5. The marker text indicates the robot heard a greeting.

## Benchmark

Use the benchmark when you want a same-condition CPU and latency comparison.
The baseline and auditory commands should differ only by `enable_auditory`.

```bash
ros2 run task_generator auditory_benchmark \
  --baseline-cmd "python3 -m arena_bringup.supervisor sim:=gazebo headless:=true human:=arena rviz:=false enable_auditory:=false" \
  --auditory-cmd "python3 -m arena_bringup.supervisor sim:=gazebo headless:=true human:=arena rviz:=false enable_auditory:=true" \
  --duration-sec 120 \
  --startup-delay-sec 20 \
  --output-json /tmp/auditory_benchmark.json \
  --output-csv /tmp/auditory_benchmark.csv
```


The benchmark reports:

- average and max process-tree CPU usage
- observed `SoundEvent` count
- observed `HeardSoundEvent` count
- latency from matching `SoundEvent` to `HeardSoundEvent`

This latency is propagation/message-flow latency. It does not measure physical
speaker-device latency.

## Configuration

- Sound assets: `task_generator/config/auditory/acoustic_assets.yaml`
- Acoustic materials: `task_generator/config/auditory/acoustic_materials.yaml`
- Launch wiring: `task_generator/launch/human/human.launch.py`
- Main nodes: `task_generator/task_generator/auditory/`
- Human event generation: `task_generator/task_generator/simulators/human/`
