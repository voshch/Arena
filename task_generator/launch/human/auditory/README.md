# Auditory Module

The auditory module adds sound events to the Arena human simulation path. It is
enabled by default when the Arena human simulator is selected. Set
`auditory:=none` (the default) runs without the auditory nodes; `auditory:=arena` enables them.

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
- TF microphones: named microphones can attach to any TF frame. Every
  microphone is an independent propagation listener.
- Robot hearing: `robot_hearing_node` listens for `HeardSoundEvent`, discovers
  robots from `state/robots`, republishes per-robot heard events, and publishes
  an RViz text marker when a robot hears a configured sound type.
- Human audio playback: `human_sound_playback` plays human sound assets from
  `config/auditory/acoustic_assets.yaml`.
- Environment audio: scenarios can define a looping radio or one logical
  alarm system backed by any number of fixed speakers. Each speaker keeps its
  own propagation route, delay, attenuation, and RIR while sharing the same
  playback position in the WAV.
- Robot motor sound: `robot_sound_node` can publish robot motor `SoundEvent`
  messages from robot odometry. In the default `auditory.motor:=procedural`,
  Jackals instead publish continuous signed left/right drivetrain state;
  other robot models retain the WAV fallback. Set `auditory.motor:=wav` to
  use WAV playback for Jackals as well. The node also renders and plays robot
  audio. Its live `enable_motor_playback` parameter mutes only workstation
  motor audio while motor emission and ROS propagation continue.

Expected nodes when enabled include:

- `sound_propagation_node`
- `robot_sound_node`
- `robot_hearing_node`
- `human_sound_playback`
- `environment_sound_playback`
- `sound_propagation_visualizer` (enabled by default with the auditory module)

## Main Topics

- `human_sound_events`: emitted `SoundEvent` stream.
- `heard_sound_events`: propagated `HeardSoundEvent` stream.
- `continuous_audio_sources`: persistent procedural source state.
- `continuous_heard_sounds`: listener-specific propagated procedural state.
- `audio_system_states`: transient-local radio and alarm control state.
- `microphone_listeners`: transient-local JSON registry of active microphone
  listener IDs.
- `microphone_markers`: persistent RViz cones and listener ID labels.
- `state/robots`: robot fleet metadata used by propagation, robot sound, and
  robot hearing nodes.
- `<robot_name>/heard_sound`: per-robot heard event output.
- `pedestrian_markers/extra`: pedestrian footstep/greeting cones and other
  transient pedestrian overlays.
- `<robot_name>/motor_sound_markers`: robot-local motor arcs.
- `<robot_name>/heard_sound_marker`: RViz text marker for sounds heard by the
  robot.
- `sound_propagation_markers`: RViz source/portal/listener paths.
- `environment_audio_source_markers`: fixed radio and alarm emitters.

The generated RViz configuration shows pedestrian cones through
`Arena/Pedestrians/Extra` and places each motor display in the corresponding
`Arena/Robot: <name>` group. Source-to-listener paths, reflections, and door
portals are shown through `Arena/Debug/Sound Propagation`. Heard-sound text is
not added as a separate RViz display.

The Auditory RViz panel provides an `Auditory Runtime` group, an
`Audio Playback Microphone` group, an `Environment Audio Sources` table, `Play robot motor
audio on this workstation`, and a live `Motor Sound Tuning` group. The runtime
group independently controls propagation and local radio/alarm playback. The
listener group follows the transient microphone registry and updates human,
robot, and environment playback. Its dropdown selects exactly one
microphone, so workstation audio represents only what that microphone hears.
The controls follow changes made through ROS parameters and
persist across episode resets. `auditory.motor:=off` sets the initial
mute state. This is separate from `auditory.robot_sound`, which controls
simulated motor emission.

The procedural defaults apply a `-9 dB` output trim, reduce the broadband
mechanical-noise layer by `-12 dB`, and use a `1.5` velocity exponent so level
changes are easier to hear as wheel speed changes. Frequency remains directly
driven by signed left and right wheel velocity. The live controls are:

- `motor_volume_db`
- `motor_frequency_scale`
- `motor_tonal_gain_db`
- `motor_broadband_gain_db`
- `motor_speed_exponent`
- `motor_velocity_smoothing_sec`

## Scenario radio and alarm systems

The static-audio Task Generator module is enabled by default and automatically
adds `audio_systems` to `task.modules`. Set
leave `auditory.static_devices` empty to disable it. The selected scenario may
contain a top-level `audio` section:

```yaml
audio:
  systems:
    - name: lobby_radio
      sound_type: music
      asset_id: radio_loop
      loop: true
      initially_active: true
      reference_distance_m: 1.0
      semantic_tags: [radio, background]
      emitters:
        - name: receiver
          entity_ref: lobby_radio_cabinet
          offset: [0.0, 0.0, 0.8]
          source_volume_db: 62.0

    - name: building_alarm
      sound_type: alarm
      asset_id: alarm_loop
      loop: true
      initially_active: false
      semantic_tags: [alarm, emergency]
      emitters:
        - name: east
          position: [3.0, 4.0, 2.6]
          level: level_1
          source_volume_db: 88.0
        - name: west
          position: [12.0, 4.0, 2.6]
          level: level_1
          source_volume_db: 88.0
        - name: upper
          position: [6.0, 8.0, 2.6]
          level: level_2
          source_volume_db: 88.0
```

An emitter uses either `entity_ref` or `position`. `entity_ref` must name one
unique static world entity and its `offset` rotates with that entity. A direct
`position` is level-local. `level` is required for direct positions in a
multi-level world. The Task Generator realizer applies the map origin and the
same multi-level flattening offset used by robots and obstacles before it
publishes the source.

All emitters in one system use one `program_start_time`. This makes the three
alarm speakers play the same point in the alarm WAV. They are still separate
physical sources, so propagation computes three independent speaker-to-listener
paths and playback uses three independent RIR convolvers. A wall, doorway, or
extra distance can therefore delay and attenuate each speaker differently.

The bundled catalog registers `radio_loop.wav` and `alarm_loop.wav`. Custom
looping files should have matching waveform and level at their beginning and
end so the join does not click. A custom music entry has this form:

```yaml
assets:
  radio_loop:
    category: music
    semantic_tags: [radio, background]
    reference_level_db: 62.0
    reference_distance_m: 1.0
    normalization_dbfs: -9.0
    loop: true
    variants:
      - sample_id: radio_loop_01
        file: radio_loop.wav
        tags: [music]
        octave_band_levels_db: auto
```

For a radio or alarm that should be available in any scenario without editing
that scenario, pass the same system schema through `auditory.static_devices`:

```bash
arena launch \
  world:=demo \
  auditory:=arena \
  auditory.static_devices:='[{name: room_radio, sound_type: music, asset_id: radio_loop, loop: true, initially_active: false, emitters: [{name: speaker, position: [5.0, 5.0, 1.2], source_volume_db: 62.0}]}]'
```

Direct positions are local to the named level. Add `level: <level_id>` for
multi-level worlds. Several radios are several list entries. A multi-speaker
alarm is one list entry with several emitters. Scenario and launch-defined
system names must be unique. To keep custom WAV files outside the package,
pass `auditory.assets:=/path/to/acoustic_assets.yaml` and
`auditory.sound_dir:=/path/to/wavs`.

A custom catalog replaces the bundled catalog for every playback node. Keep
the bundled `footstep`, `greeting`, and motor entries in it, and keep their WAV
files in the selected sound directory, alongside the new radio and alarm
assets.

Start or stop one logical system, including all of its emitters, with:

```bash
ros2 service call \
  /arena/env_0/task_generator_node/runtime/set_audio_system \
  task_generator_msgs/srv/SetAudioSystem \
  "{system_id: building_alarm, active: true}"
```

Replace `env_0` when the runtime allocated a different environment.

`initially_active` sets the episode-reset state. The service changes simulated
emission. The launch argument and live `enable_environment_playback` parameter on
`environment_sound_playback` only mutes or unmutes local workstation output,
so propagation and robot hearing continue while it is muted.

The default `auditory.block_size` is 2048 frames. If the host still reports
repeated PulseAudio underflows under a heavy RIR workload, increase it to 4096.
An occasional recovered underrun does not stop propagation or playback.

RViz lists fixed emitters in `Arena/Sound Propagation/Environment Audio
Sources`. Alarm markers are red, other active systems are cyan, and inactive
systems are gray. Emitters use a box marker oriented by the placement drag.
The **Spawn Radio** toolbar tool creates a source at runtime. Set its `Mode`
property to `Music` or `Alarm`, set `Height`, then click and drag in the map.
By default the source starts the corresponding bundled loop immediately.
Enable `Custom Playback` to choose another catalog asset ID, source volume,
loop behavior, or an initially stopped state. The runtime transforms the RViz
Fixed Frame pose into the global map, so placement also works in allocated
environment frames. Use the `Environment Audio Sources` checkbox to start or stop
emission and playback. Select a `runtime_*` row and use `Remove selected runtime
source` to delete it. Runtime sources are cleared on the next episode reset.
The existing robot or pedestrian heard-sound display shows each active
source-to-listener path, portal route, and delay. Set the
visualizer's `continuous_listener_id` parameter when a specific microphone
should own the path display. If it is empty, the first continuous listener is
used.

## Microphones and playback routing

Every robot in `state/robots` automatically creates one microphone named from
the robot instance, for example `robot1_mic` and `robot2_mic`. It follows the
robot base TF frame and appears as a green triangular cone in RViz. This makes
multi-robot microphone testing available without extra launch arguments.

Additional robot-mounted microphones can be configured with
`auditory.microphones`. Each entry names the robot instance, placement,
relative or robot-prefixed TF frame, and stable positive index:

```bash
arena launch \
  auditory:=arena \
  auditory.microphones:='[{owner: robot, robot: jackal, placement: body, frame: base_link, index: 1}, {owner: robot, robot: jackal, placement: front, frame: front_laser, index: 1}]'
```

The robot must exist in `state/robots`. A relative frame is resolved below that
robot's frame prefix. The listener is inactive if the robot is absent or TF
cannot resolve the frame. RViz shows one green triangular cone for each
resolved microphone. The cones follow their TF frames. Entries may name
different active robots, so one launch can expose microphones on several
robots at the same time. Choose one of them in the RViz playback dropdown.

World-mounted microphones are authored in each level's `world.yaml` beside
`zones`:

```yaml
microphones:
  - zone: reception
    placement: ceiling
    frame: map
    position: [4.2, 3.1, 2.9]
    index: 1
  - zone: reception
    placement: ceiling
    frame: map
    position: [7.8, 3.1, 2.9]
    index: 2
```

These become `microphone:zone:reception:ceiling:1` and
`microphone:zone:reception:ceiling:2`. World loading rejects missing zones,
duplicate IDs, map-frame positions outside the declared zone, ceiling
placements in zones without ceilings, and heights that differ from an
explicit `ceiling_height` by more than 5 cm. A non-map TF frame is permitted,
but its resolved runtime position must remain in the declared zone. Ceiling
height falls back to `pyroom_ceiling_height_m` when the zone does not specify
one.

Microphones can also be added during an episode with the RViz **Spawn
Microphone** toolbar tool. Click the desired position and set its `Height` tool
property. Leave `Attach TF Frame` empty for a fixed microphone. Set it to a
resolvable frame such as `env_0/jackal/base_link` to store the clicked offset
in that frame and make the microphone follow it. The runtime transforms the
clicked pose is registered immediately in its RViz or attached TF frame. The
first click creates `microphone1`, followed by `microphone2` and later
increasing IDs. These runtime microphones are cleared and the index restarts
on the next episode or world change.

The new microphone appears immediately in the Auditory panel's **Audio
Playback Microphone** dropdown and as a green triangular cone in
`Arena/Sound Propagation/Microphones`. Select it in **Listen through** to hear
only that microphone's propagated audio.
The spawn service is available at
`<task-generator-namespace>/runtime/spawn_microphone` while auditory simulation
is enabled.

Every finite human or robot clip is published as `SoundEvent` and propagated
to one `HeardSoundEvent` per listener. Procedural drivetrain audio uses
`ContinuousAudioSourceState` and `ContinuousHeardSoundState` because it also
carries wheel velocities, active state, backend, and deterministic seed. Both
heard message types identify the receiving microphone in `listener_id`.
Microphones do not publish separate PCM topics. The listener-specific messages
share `heard_sound_events` and `continuous_heard_sounds`; the playback nodes
filter those streams, render the selected feeds, and send the result to their
configured workstation `audio_device`.

The RViz dropdown applies one microphone ID to propagation,
`human_sound_playback`, `robot_sound_node`, and
`environment_sound_playback`. For a non-RViz workflow, set
`auditory.listener:=robot1_mic` or another registered microphone ID at launch.

When the simulator viewport publishes `/arena/viewport/camera_pose`, two more
listeners appear in the same dropdown:

- `microphone:viewport:projective_center` follows the camera position.
- `microphone:viewport:down_projection` follows the camera x/y position at
  `auditory.viewport_height`, which defaults to 1.6 m.

Selecting either listener detaches workstation playback from the robot
microphone. Viewport microphones are available only when the simulator GUI
publishes the viewport pose.

## Pyroomacoustics portal routing

Enable the pyroomacoustics backend and optional RViz path visualizer with:

```bash
arena launch \
  auditory:=arena \
  auditory.propagation:=pyroomacoustics \
  auditory.viz:=true
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
`HeardSoundEvent` messages only for robot listeners and configured TF
microphones. Pedestrians are not listeners, so pedestrian-to-pedestrian
propagation is not calculated or published and the RViz propagation visualizer
has no corresponding blue paths to draw. Set it to `false` to add every
non-source pedestrian as an `agent:<id>` listener. Robot and microphone
listener events receive the complete route metadata. Human events are rendered
by `human_sound_playback`, and robot events are rendered by
`robot_sound_node`. The launch default sets
`auditory.rir_in_propagation:=true`, so the propagation node
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
The baseline and auditory commands should differ only by `auditory`.

```bash
ros2 run task_generator auditory_benchmark \
  --baseline-cmd "python3 -m arena_bringup.supervisor sim:=gazebo headless:=true human:=arena rviz:=false auditory:=none" \
  --auditory-cmd "python3 -m arena_bringup.supervisor sim:=gazebo headless:=true human:=arena rviz:=false auditory:=arena" \
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
