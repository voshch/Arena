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
  wall/material attenuation, optional pyroomacoustics RIRs, and cached
  multi-portal coupling across doors and shared open boundaries.
- Robot hearing: `robot_hearing_node` listens for `HeardSoundEvent`, discovers
  robots from `state/robots`, republishes per-robot heard events, and publishes
  an RViz text marker when a robot hears a configured sound type.
- Audio playback: `human_sound_playback` plays configured sound assets from
  `config/auditory/acoustic_assets.yaml`. Its live
  `enable_motor_playback` parameter mutes only workstation motor audio while
  motor emission and ROS propagation continue.
- Robot motor sound: `robot_sound_node` can publish robot motor `SoundEvent`
  messages from robot odometry. In the default `motor_audio_mode:=procedural`,
  Jackals instead publish continuous signed left/right drivetrain state;
  other robot models retain the WAV fallback. Set `motor_audio_mode:=wav` to
  use WAV playback for Jackals as well.
  
 Expected nodes when enabled include:

- `sound_propagation_node`
- `robot_sound_node`
- `robot_hearing_node`
- `human_sound_playback`
- `sound_propagation_visualizer` (enabled by default with the auditory module)

## Main Topics

- `human_sound_events`: emitted `SoundEvent` stream.
- `heard_sound_events`: propagated `HeardSoundEvent` stream.
- `continuous_audio_sources`: persistent procedural source state.
- `continuous_heard_sounds`: listener-specific propagated procedural state.
- `state/robots`: robot fleet metadata used by propagation, robot sound, and
  robot hearing nodes.
- `<robot_name>/heard_sound`: per-robot heard event output.
- `pedestrian_markers/extra`: pedestrian footstep/greeting cones and other
  transient pedestrian overlays.
- `<robot_name>/motor_sound_markers`: robot-local motor arcs.
- `<robot_name>/heard_sound_marker`: RViz text marker for sounds heard by the
  robot.
- `sound_propagation_markers`: RViz source/portal/listener paths.

The generated RViz configuration shows pedestrian cones through
`Arena/Pedestrians/Extra` and places each motor display in the corresponding
`Arena/Robot: <name>` group. Source-to-listener paths, reflections, and door
portals are shown through `Arena/Debug/Sound Propagation`. Heard-sound text is
not added as a separate RViz display.

The Task Generator RViz panel includes `Play robot motor audio on this
workstation`. It reads `human_sound_playback.enable_motor_playback` when the
playback node becomes available, follows external parameter changes, and
rechecks the value on every episode update. The value persists across episode
resets. `enable_motor_playback:=false` sets its initial launch value. This is
separate from `enable_robot_sound`, which controls simulated motor emission.

## Pyroomacoustics portal routing

Enable the pyroomacoustics backend and optional RViz path visualizer with:

```bash
arena launch \
  enable_auditory:=true \
  propagation_backend:=pyroomacoustics \
  enable_sound_visualization:=true
```

On world load, Arena pairs each authored door with the acoustic zone touching
the other side. It also derives an opening portal when both adjacent room
specifications agree that a shared boundary span is open. Explicit doors take
precedence over derived openings.

Each authored zone including a corridor or a rectangular whole world
zone is treated as one ordinary pyroomacoustics room. Enclosed mini-room zones
remain separate rooms. Same-zone sounds use one room-local RIR. Cross-zone
rendering follows a door/opening portal route by default, up to
`max_portal_hops`. A route is rendered by composing room-local RIR segments:
source-to-portal in the source room, portal-to-portal through intermediate
rooms, then portal-to-listener in the listener room. Routes with no connected
portal path use the explicit Level-3/dry fallback. Room and portal-route RIRs
are quantized and cached.

With `pyroom_robot_listeners_only=true` (the default), propagation creates
`HeardSoundEvent` messages only for robot listeners. Pedestrians are not
listeners, so pedestrian-to-pedestrian propagation is not calculated or
published and the RViz propagation visualizer has no corresponding blue paths
to draw. Set it to `false` to add every non-source pedestrian as an
`agent:<id>` listener. Robot-listener events receive the complete route
metadata and are rendered with pyroomacoustics by `human_sound_playback`. The
launch default sets `compute_rir_in_propagation=true`, so the propagation node
constructs the same-room or portal-route RIR and reports the actual
`pyroomacoustics_same_room`, `pyroomacoustics_one_door`, or
`pyroomacoustics_multi_portal` backend in ROS propagation results. Playback
remains the stage that applies the RIR to the waveform.

Dynamic open/closed door state is not currently published by Arena, so
authored doors use `portal_loss_db`. Derived openings use
`opening_portal_loss_db`.

`HeardSoundEvent.propagation_level` remains the model capability level for
compatibility. Inspect these fields for the actual route:

- `propagation_backend`: `pyroomacoustics_same_room`,
  `pyroomacoustics_one_door`, `pyroomacoustics_multi_portal`, `level3`, or a
  legacy path.
- `used_backend_fallback` and `backend_fallback_reason`: whether and why the
  requested pyroomacoustics route could not run.
- `portal_id` and `portal_position`: the paired door used by one-door coupling.
- `portal_ids`, `portal_positions`, `traversed_zones`, `portal_hop_count`, and
  `portal_route_loss_db`: the complete selected route. The singular fields
  retain the first portal for compatibility.

The propagation node logs paired/unpaired doors, acoustic-zone coverage
warnings, and each distinct backend route. Playback logs its independently
verified `playback_backend` and dry/silent fallback reason. Its five-second
diagnostics include room/portal RIR cache entries, hits, misses, mixer stream
state, callback count, voice count, last output peak, and decoded-asset cache
entries, hits, misses, and pending worker loads.

Procedural Jackal audio uses the same playback-side pyroomacoustics RIR lookup
as WAV assets. Its bundled recording-room/microphone transfer is disabled, so
same-room or portal-coupled pyroomacoustics is the only simulated room response.
Propagation constructs the RIR for ROS propagation metadata; playback obtains
the equivalent cached RIR and applies it to the waveform once. Moving-source
RIR lookup uses 0.10 m source/listener position quantization in playback and
0.25 m quantization in propagation. RIR changes use a 100 ms equal-power
crossfade. Continuous convolution uses 1024-frame uniform FFT partitions at
44,100 Hz. The listener signal is mono because it represents one robot
microphone.

The drivetrain broadband noise field is approximately 33 MB. All Jackal voices
in one episode use the episode seed as the field-cache key, so they reference
one shared read-only field instead of allocating approximately 33 MB per robot.
Each robot derives a stable phase and starting-position index from the episode
seed and robot name, preserving deterministic differences between robots. The
shared field is cleared on episode reset; a new episode builds one field for its
new seed.

The asset catalog loads only YAML metadata and validates WAV paths at startup.
The selected WAV is decoded, channel-converted, resampled, and normalized on a
single background worker the first time it is used. The decoded
`CachedSample` is retained for subsequent events; the real-time mixer callback
never performs file I/O or decoding. Configure explicit
`octave_band_levels_db` values for every variant in
`acoustic_assets.yaml` so lazy decoding does not need to perform spectral
analysis. The current bundled assets contain precomputed values.

Docker playback uses the host PulseAudio/PipeWire compatibility socket. The
image installs `libasound2-plugins`, Compose forwards the socket as
`/tmp/pulse/native`, and the launch selects the `pulse` device. This avoids
opening raw `hw:0,0`, which is exclusive and unavailable while the host sound
server owns the analog card. Rebuild/recreate the Arena container after a
Docker audio configuration change.

RViz draws the complete source-to-portals-to-listener line and one cube per
portal. Pedestrian-listener propagation is blue; robot-listener propagation
is purple, on separately controllable marker topics.

Audit every installed world before relying on RIR coverage:

```bash
ros2 run task_generator acoustic_world_audit --stride-cells 10
```

The command reports missing maps, traversable cells outside zones, overlapping
zones, explicit and derived portals, unpaired doors, and graph components. It
returns a non-zero status when map/zone coverage is incomplete.

The current repository audit intentionally reports the remaining world-data
issues instead of fabricating acoustic geometry:

- `hospital_1`: four sampled free border cells below the authored zone bounds.
- `hospital_2`: its two levels do not provide occupancy-map YAML files.
- `map_empty`: free elevator/outside-door cells are outside `empty_zone`.
- `reception`: free map cells above the authored `y=23` room boundaries.
- `three_storied_residential`: some free entry cells are outside its zones.

The residential map YAML files reference the bundled `map.png` files. Multi-
level portal extraction is level-scoped, so geometrically overlapping floors
are never coupled to each other. Correct the remaining world/scenario data when
those locations must receive physically modelled RIRs; until then they retain
an explicit Level-3/dry fallback.

Main routing controls are:

- `derive_opening_portals` (default `true`)
- `minimum_opening_width_m` (default `0.30`)
- `enable_multi_portal_rir` (default `true`)
- `max_portal_hops` (default `4`)
- `portal_loss_db` (default `3.0`)
- `opening_portal_loss_db` (default `0.5`)
- `route_distance_loss_db_per_m` (default `0.05`)
- `portal_source_early_window_sec` (default `0.08`)
- `portal_max_rir_duration_sec` (default `2.0`)
- `pyroom_robot_listeners_only` (default `true`)
- `compute_rir_in_propagation` (default `true`)
- `pyroom_cache_position_quantization_m` (default `0.25`)

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
