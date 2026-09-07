# arena_auditory

The auditory simulator adds robot microphones, propagation, hearing, and local
playback as its own axis next to the human simulator. `auditory:=none` (the
default) runs without the auditory nodes; `auditory:=arena` launches this
package's sidecar from `task_generator/launch/auditory/` and makes the task
generator bring up the map server it reads. Human footsteps and greetings are
produced from the human simulator's `arena_peds` topic, whichever backend
publishes it.

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
- Robot hearing: in stereo mode, `robot_hearing_node` consumes the center
  listener. In four-mic mode, `microphone_array_node` first fuses the matching
  FL/FR/RL/RR results and `robot_hearing_node` consumes only that array-derived
  event. It republishes per-robot heard events and RViz text markers.
- Human audio playback: `human_sound_playback` plays human sound assets from
  `config/acoustic_assets.yaml`.
- Environment audio: worlds (or `auditory.static_sounds`) declare standalone
  `sound` semantic entities such as a looping radio or an alarm. Several
  sounds sharing one `sound_on` regime move together as one logical alarm
  while each keeps its own propagation route, delay, attenuation, and RIR.
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
- `microphone_array_node` (with `microphone_mode:=four_mic`)
- `sound_propagation_visualizer` (enabled with `auditory.viz:=true`)

## Main Topics

- `human_sound_events`: emitted `SoundEvent` stream.
- `heard_sound_events`: propagated `HeardSoundEvent` stream.
- `four_mic_heard_sound_events`: one array-fused event per finite sound in
  four-mic mode; this is the input to `robot_hearing_node` in that mode.
- `continuous_audio_sources`: persistent procedural source state.
- `continuous_heard_sounds`: listener-specific propagated procedural state.
- `state/semantics`: latched `SemanticSnapshot` carrying every sound's live
  `sounding`/`volume_db` state, alongside every other semantic kind.
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

## Four-microphone Jackal array

Set `microphone_mode:=four_mic` to replace the legacy 0.20 m side pair with
four independent propagation listeners and start the synchronized raw PCM,
robot-hearing, headphone and TDoA outputs. `microphone_mode:=stereo` preserves
the old behavior.

The implementation is part of the existing ROS 2 auditory stack, not a sensor
plugin tied to a physics simulator. `sound_propagation_node` remains the source
of listener-specific distance, delay, attenuation, occupancy/material
occlusion, reflection metadata and pyroomacoustics portal routing.
`microphone_array_node` renders its four results on one 16 kHz sample clock.
Finite sound events use reliable volatile QoS: current subscribers receive
each event without replaying stale clips to nodes that join later.

The renderer is driven by `/clock`, not by an rcl timer: every clock message
renders the 20 ms blocks it has covered, so sample zero stays anchored to the
sim time of the first block and block stamps never drift behind the clock.
When a clock message finds the render more than `max_catchup_blocks` (10)
behind, the surplus blocks are skipped rather than rendered late: audio shows a
gap, never a growing delay. Skips are counted in the diagnostics and warned
once per lapse. Under `arena lockstep`, the node registers `raw_array` as a
hard channel with one block per window, so the scheduler waits for every block
instead and the skip path never fires. Without sim time (standalone tests) the
renderer falls back to a steady-clock timer.

### Geometry and spacing

The Jackal collision chassis in
`arena_robots/arena_robots/robots/jackal/urdf/jackal.urdf.xacro` is 0.420 m
long, 0.310 m wide and 0.184 m high. Each microphone is inset 0.020 m from its
two nearest horizontal edges and mounted at z=0.220 m:

| Channel | Frame | Position in `base_link` (m) | Inlet yaw |
|---:|---|---:|---:|
| 0 | `mic_front_left` | `(0.190, 0.135, 0.220)` | `+45 deg` |
| 1 | `mic_front_right` | `(0.190, -0.135, 0.220)` | `-45 deg` |
| 2 | `mic_rear_left` | `(-0.190, 0.135, 0.220)` | `+135 deg` |
| 3 | `mic_rear_right` | `(-0.190, -0.135, 0.220)` | `-135 deg` |

Yes, the two principal spacings are deliberately different:

- front-to-back spacing on either side is `0.190 - (-0.190) = 0.380 m`;
- left-to-right spacing at either end is `0.135 - (-0.135) = 0.270 m`;
- diagonal spacing is about `0.466 m`.

This follows the rectangular chassis instead of forcing the receivers into a
smaller square. A square that remained on the chassis would have to reduce the
front-to-back spacing from 0.380 m to 0.270 m, discarding 29% of the available
longitudinal aperture. With speed of sound 343 m/s, the current far-field
maximum path delays are approximately 1,108 us (17.7 samples at 16 kHz)
front-to-back and 787 us (12.6 samples) left-to-right. The larger longitudinal
baseline therefore gives more observable timing information for front/back
classification. TDoA algorithms use the exact coordinates, so equal baselines
are not required. The tradeoff is greater high-frequency spatial aliasing;
the design favors broadband, relatively low-frequency events such as footsteps
and retains the configurable geometry for comparison experiments.

```text
                         +X FRONT

              FL  ↖                 ↗  FR
                    +---------------+
                    |               |
              +Y    |    JACKAL     |    -Y
                    |               |
                    +---------------+
              RL  ↙                 ↘  RR

                         -X REAR
```

The frames are fixed children of `base_link`, so the normal robot TF chain
moves and rotates them rigidly. The arrows represent outward-facing inlet
normals. Propagation is currently omnidirectional because no calibrated
XVF3800 polar response exists in the acoustic API; orientation is retained in
the model, RViz and stream metadata for future directivity models.

Defaults live in `config/jackal_four_mic.yaml` and are configurable
through `mic_array_width`, `mic_array_length`, `mic_height` and
`mic_corner_inset`. The same YAML is loaded by propagation and rendering so
their geometry cannot drift in file-based experiments.

The receiver uses the Seeed reSpeaker XMOS XVF3800 as a technology reference:
four raw PDM MEMS-like channels, -26 dBFS nominal sensitivity, 64 dBA SNR,
120 dB SPL overload and 16 kHz maximum reference-board sampling rate. See the
[XVF3800 data sheet](https://files.seeedstudio.com/Bazaar/product_pdf/114993700.pdf)
and [XVF3800 guide](https://wiki.seeedstudio.com/respeaker_xvf3800_introduction/).
The physical board is circular; Arena retains the synchronized raw-channel
concept but uses Jackal corner positions. It does not apply the hardware's
onboard AEC, AGC, beamforming, noise suppression or phase correction.

### Signal path and topics

```text
sources -> existing sound_propagation_node -> FL / FR / RL / RR results
        -> synchronized raw PCM -> mono diagnostic / spatial stereo / GCC-PHAT
        -> four-channel event fusion -> robot_hearing_node -> jackal/heard_sound
```

Each listener result has its own `direct_delay_sec`, `received_volume_db`,
audibility and occlusion/portal result. Finite events use one source sample and
one common scheduling anchor, then apply each receiver's delay with
fractional-sample interpolation. WAV loops share `program_start_time` while
retaining each receiver's independent delay and level. The renderer never
duplicates one received channel four times.
Procedural Jackal drivetrain state is synthesized once at the array's 16 kHz
rate, then distributed with each microphone's independent propagation gain and
streaming fractional delay. The RViz motor enable and tuning controls are
mirrored to this renderer in four-mic mode.

The legacy human and environment playback nodes are not launched in four-mic
mode; the array is the sole workstation output path. `robot_sound_node` remains
active as the motor-state producer, but its legacy mixer has no local device.
The independent `robot:jackal` center listener is not propagated in four-mic
mode. Once all four microphone results for a finite event arrive, the array
publishes one compatible event using the strongest audible received level, the
earliest audible arrival, and the mean microphone position as the robot-array
center. The existing robot-hearing threshold, delay and marker logic then
publishes `jackal/heard_sound`.

`task_generator_msgs/AudioFrame` contains timestamp, sample rate, encoding,
frame count, fixed ordering, microphone frame IDs, positions, inlet yaws and
interleaved float32 PCM. For a robot named `jackal`, topics end in:

- `jackal/audio/mic_front_left`
- `jackal/audio/mic_front_right`
- `jackal/audio/mic_rear_left`
- `jackal/audio/mic_rear_right`
- `jackal/audio/raw_array` with order FL, FR, RL, RR
- `jackal/audio/hearing/mono`
- `jackal/audio/hearing/energy`
- `jackal/audio/headphones/left`
- `jackal/audio/headphones/right`
- `jackal/audio/headphones/stereo`
- `jackal/audio/diagnostics/tdoa`

`hearing/mono` is a diagnostic signal that selects the highest-RMS raw channel
per block instead of phase-averaging asynchronous signals. It is not the normal
headphone presentation. `hearing/energy` carries linear RMS in
the order FL, FR, RL, RR, hearing, headphone L, headphone R. The canonical
research observation remains the unchanged four-channel `raw_array`.

Headphone monitoring uses
`L=(front_gain*FL + rear_gain*RL)/gain_sum` and the corresponding right-side
expression, with no time alignment. Raw channels keep the -26 dBFS-at-94-dB-SPL
MEMS calibration. A separate `monitor_gain_db` workstation preamp (36 dB by
default) makes those physical levels audible without changing `raw_array`, and
`monitor_limit` bounds headphone peaks. `auditory.playback:=auto` routes the
stereo result to the selected PortAudio device in addition to publishing it.
When `PULSE_SERVER` is present, automatic device selection prefers a stereo
PulseAudio output over an exclusive raw ALSA device.

The PCM renderer runs from a steady wall clock because PortAudio consumes in
wall time even when the simulation real-time factor changes. Its small queue
accepts arbitrary PortAudio callback frame sizes, counts underflow/overflow,
retries a failed device every two seconds, and logs `four-mic audio
diagnostics` at debug level every five seconds. These diagnostics report
received/accepted events, active WAV and drivetrain voices, stream/device
state, queue depth, callback count, peak, rendered/skipped block counts and
the last PortAudio error. A warning is logged once when playback degrades
(stream inactive, or new underflows/overflows in the period) and an info line
once when it recovers, and likewise once when the render falls behind `/clock`
and once when it catches up. `diagnostics/tdoa` is only computed while
something subscribes to it.

### Simulator relationship

The microphone and propagation nodes do not call Gazebo APIs. They consume ROS
interfaces: TF, `state/robots`, the occupancy map and acoustic-world metadata,
plus finite or continuous sound-source messages. Consequently the array is
usable with the repository's Gazebo, Isaac and external/dummy workflows when
those interfaces are present.

Gazebo appears in the concrete commands because it is the repository's default
full physics backend and can run the bundled Jackal, moving pedestrian and
S-bend world together. In that example Gazebo supplies robot/world motion and
the TF/odometry state. It does **not** propagate or render audio; Arena's ROS
auditory nodes do that. The same four-microphone pipeline can be selected with
`sim:=isaac`. A dummy backend is useful for message-level tests or externally
published state, but by itself does not provide the moving physical scenario.

Launch a basic example with either full simulator backend:

```bash
arena launch \
  sim:=gazebo world:=map_empty robot:=jackal \
  human:=arena auditory:=arena auditory.viz:=true \
  microphone_mode:=four_mic auditory.playback:=auto
```

Replace `sim:=gazebo` with `sim:=isaac` when using the Isaac runtime. Backend
world and human-simulation support still determines which complete scenario is
available; it does not change the microphone topic or processing contract.

### Controls, visualization and diagnostics

RViz's **Jackal Four-Mic Hearing** group controls array enable, headphone
enable, mute, master/front/rear gain, FL/FR/RL/RR solo, all-four monitoring,
the monitor preamp, microphone markers, TDoA diagnostics and gain/routing
reset. **Spatial stereo (normal)** maps FL/RL to the left headphone and FR/RR
to the right. **Mono detection preview** duplicates the highest-energy channel
into both ears and exists only for diagnostics. `solo_channel=""` is the
empty-string sentinel for **no solo**, meaning all four channels remain
selected. Solo never changes the canonical four-channel raw or semantic input.

Markers on `microphone_markers` show each position, name, inlet arrow, ON/OFF
state and current dBFS. Solo changes only monitoring products; the raw array
retains all four channels.

Equivalent parameter commands include:

```bash
ros2 param set /arena/env_0/task_generator_node/microphone_array_node mute_all true
ros2 param set /arena/env_0/task_generator_node/microphone_array_node monitor_gain_db 36.0
ros2 param set /arena/env_0/task_generator_node/microphone_array_node solo_channel front_left
ros2 param set /arena/env_0/task_generator_node/microphone_array_node tdoa_enabled true
```

Run the console diagnostic with:

```bash
ros2 run arena_auditory microphone_diagnostic --ros-args \
  -p topic:=/arena/env_0/task_generator_node/jackal/audio/raw_array
```

It reports FL/FR/RL/RR dBFS, GCC-PHAT estimates for FL-FR, RL-RR, FL-RL and
FR-RR, and explicitly labelled coarse energy evidence.

### Four-microphone tests

```bash
python3 -m pytest arena_auditory/tests/unit/test_four_mic_array.py -q
```

The tests cover geometry, rigid motion, channel independence, known
left/right/front/rear arrival ordering, fractional delay, MEMS calibration,
silence, stereo asymmetry, monitor-only amplification, streaming fractional
delay history, and disable/mute/solo/re-enable behavior. At 5 m
from the array center, default geometry produces 786.604 us FR-minus-FL delay
for a left source and 1,107.468 us RL-minus-FL delay for a front source.

Known limitations are:

- Full pyroomacoustics RIR samples are not serialized by `HeardSoundEvent`, so
  raw PCM applies propagated direct/portal delay, calibrated level, occlusion
  and route loss but not the detailed late RIR convolution.
- The simple headphone map is not an HRTF, so front/back perception is weaker
  than the timing information retained in the raw channels.
- NLOS quality depends on authored acoustic zones and connected openings. With
  no valid portal route, propagation reports its explicit Level-3/dry fallback
  instead of fabricating diffraction.

The generated RViz configuration shows pedestrian cones through
`Arena/Pedestrians/Extra` and places each motor display in the corresponding
`Arena/Robot: <name>` group. Source-to-listener paths, reflections, and door
portals are shown through `Arena/Debug/Sound Propagation`. Heard-sound text is
not added as a separate RViz display.

The Auditory RViz panel provides an `Auditory Runtime` group, an
`Audio Playback Microphone` group, a `Sound Entities` table, `Play robot motor
audio on this workstation`, and a live `Motor Sound Tuning` group. The runtime
group independently controls propagation and local radio/alarm playback. The
listener group follows the transient microphone registry and updates human,
robot, and environment playback. Its dropdown selects exactly one
microphone, so workstation audio represents only what that microphone hears.
It is visible in stereo mode and hidden whenever the four-mic array is active;
four-mic monitoring instead uses the array's spatial stereo and diagnostic solo
controls.
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

## World and launch-defined sounds

The `sounds` Task Generator module is added to `task.modules` whenever
`auditory` is not `none`, or `auditory.static_sounds` is non-empty. It renders
every `sound` entity declared in the loaded world, the active scenario, and
the launch configuration.

A static or environment sound is a standalone `sound` semantic entity, not a
container object with sub-emitters. The semantics engine, not the audio
module, owns whether it is playing (`sounding`) and how loud (`volume_db`).
The audio module is a pure renderer: it resolves each sound's placement,
publishes its live state as `ContinuousAudioSourceState` for propagation, and
serves the runtime spawn/remove services.

A `sound` entry takes:

- `name`: unique among all world-, scenario- and launch-defined sounds.
- `asset_id`: a catalog entry from `acoustic_assets.yaml`.
- `position`, `entity_ref` or `frame`, exactly one. `entity_ref` must name
  one unique static world entity, and `offset` then rotates with that
  entity's yaw. A direct `position` is level-local. `level` is required for
  it in a multi-level world and only allowed with it. `frame` names a TF
  frame (env prefix optional, added when missing) and `offset` is local to
  that frame: the renderer publishes the frame, not a map point, and
  propagation re-localizes the source on every update, so a sound on
  `jackal/base_link` moves with the robot. While the frame is not yet in TF
  the source is skipped with a warning, never placed at the origin.
  Pedestrians publish no per-agent TF frame, so `frame` targets robots and
  static frames.
- `loop` (default `true`).
- `reference_distance_m` (default `1.0`), must be positive.
- `semantics`: a `semantics:` list carrying the `sound` preset, which expands
  to a `sounding` predicate and a `volume_db` state.

World sounds are authored per zone, as a sibling list to `schedules:` and
`signals:`:

```yaml
zones:
- name: hallway
  corners: [...]
  sounds:
  - name: hall_siren
    asset_id: alarm_loop
    position: [4.0, 2.3]
    semantics:
    - {preset: sound, params: {sound_on: alarm}}
```

`sound_on` names a regime consulted from another scripted kind, exactly like
a gate's `unlock_on` or a pressure plate's `press_on` (see
[AUTHORING.md](../../../../arena_simulation_setup/AUTHORING.md)). `hall_siren`
stays silent until something asserts the `alarm` regime, for example a
`schedule` entry (`{preset: schedule, params: {windows: [], regime: alarm}}`)
whose `active` predicate a scenario timeline flips. Several `sound` entries
that share one `sound_on` name toggle together as one logical alarm: the
renderer groups them under the same wire `group_id`, so grouped sirens also
share deterministic WAV-variant selection. They stay separate physical
sources, so propagation still computes one independent speaker-to-listener
path per sound and playback uses one independent RIR convolver each. A wall,
doorway, or extra distance can therefore delay and attenuate each speaker
differently.

A world- or scenario-authored sound with no `sound_on` plays only when told
to, and the sounds module warns at load about such a sound, with the service
call that turns it on. An always-on device like a radio is authored with an
initial value in its preset params,
`{preset: sound, params: {sounding: true, volume_db: 62.0}}` (`sounding:` and
`sound_on:` are exclusive). Anything can be toggled later, either from a
scenario timeline entry:

```yaml
timeline:
- at: 0.0
  set:
  - entity: lobby_radio
    field: sounding
    value: "true"
```

or live, through the `SetSemantic` service:

```bash
ros2 service call \
  /arena/env_0/task_generator_node/semantics/set \
  task_generator_msgs/srv/SetSemantic \
  "{entity: env_0/lobby_radio/1, field: sounding, value: 'true'}"
```

Unlike timeline entries, the external service does no bare-name resolution:
`entity` is the realized instance name exactly as published on
`state/semantics` (the bare name wrapped in the env namespace and, for
world-embedded sounds, the level id). Read it off the snapshot topic when in
doubt. Replace `env_0` when the runtime allocated a different environment. The same
service writes `volume_db` (a float literal in `value`). An override on
`sounding` is cleared automatically the next time the sound's natural
(regime-consulted) value changes, matching every other scripted kind. A sound
with no `sound_on` never has a natural transition, so a one-time override
holds for the rest of the episode. Toggling a multi-speaker group this way
needs one `SetSemantic` call per sound name, since each is an independent
entity. Driving a shared `sound_on` regime instead moves the whole group with
one write.

Initial volume defaults to `80.0` dB. To author a different starting volume,
add a separate `volume_db` entry, never a `value:` on the `sound` preset item
itself (a preset-level `value` broadcasts onto every primitive the preset
expands to, which would corrupt the `sounding` predicate too):

```yaml
sounds:
- name: lobby_radio
  asset_id: radio_loop
  entity_ref: lobby_radio_cabinet
  offset: [0.0, 0.0, 0.8]
  semantics:
  - {preset: sound}
  - {state: volume_db, value: 62.0}
```

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

Runtime spawns follow the same rule: `SpawnSound.attach_to_frame` keeps the
request pose in its own frame instead of transforming it to `map` once.

A scenario carries episode-scoped sounds in its own `sounds:` list, same
schema, attached at reset and gone at the next one. Positions may be zone
references like any other scenario placement, and `entity_ref` may name one
of the scenario's `static:` obstacles. The `fire_alarm` scenario of
`three_storied_residential` ships a `bedroom_radio` that plays until the
alarm fires.

For a sound that should be available in any scenario without editing a world,
pass the same schema, as a flat list, through `auditory.static_sounds`:

```bash
arena launch \
  world:=demo \
  auditory:=arena \
  auditory.static_sounds:='[{name: room_radio, asset_id: radio_loop, position: [5.0, 5.0, 1.2], level: level_1, semantics: [{preset: sound, params: {volume_db: 62.0}}]}]'
```

A launch-defined sound plays from the start unless its entry names a
`sound_on` regime or sets `sounding` itself, unlike a world-authored sound,
which stays silent until told otherwise. Direct positions in
`auditory.static_sounds` need `level` in a multi-level world, same as a
world-authored direct `position`. Several radios are several
list entries. A multi-speaker alarm is several list entries sharing one
`sound_on`. World- and launch-defined sound names must be unique. To keep
custom WAV files outside the package, pass
`auditory.assets:=/path/to/acoustic_assets.yaml` and
`auditory.sound_dir:=/path/to/wavs`.

A custom catalog replaces the bundled catalog for every playback node. Keep
the bundled `footstep`, `greeting`, and motor entries in it, and keep their WAV
files in the selected sound directory, alongside the new radio and alarm
assets.

`sound_type` and `semantic_tags` on the wire are derived from the asset
catalog's `category` and `semantic_tags` fields for the sound's `asset_id`,
falling back to the bare `asset_id` with no extra tags when it is not in the
catalog.

`sounding` and `SetSemantic` control simulated emission. The launch argument
and live `enable_environment_playback` parameter on `environment_sound_playback`
only mutes or unmutes local workstation output, so propagation and robot
hearing continue while it is muted.

The default `auditory.block_size` is 2048 frames. If the host still reports
repeated PulseAudio underflows under a heavy RIR workload, increase it to 4096.
An occasional recovered underrun does not stop propagation or playback.

RViz lists rendered sounds in `Arena/Sound Propagation/Environment Audio
Sources`. Alarm-tagged sounds are red, other active sounds are cyan, and
inactive sounds are gray. Sounds use a box marker oriented by the placement
drag. The **Spawn Radio** toolbar tool creates a source at runtime through
`spawn_sound`. Set its `Mode` property to `Music` or `Alarm`, set
`Height`, then click and drag in the map. By default the source starts the
corresponding bundled loop immediately. Enable `Custom Playback` to choose
another catalog asset ID, source volume, loop behavior, or an initially
stopped state. The runtime transforms the RViz Fixed Frame pose into the
global map, so placement also works in allocated environment frames. Use the
`Sound Entities` row toggle to start or stop emission and playback
(a `SetSemantic` write on `sounding`, same as any other sound). Select a
`runtime_*` row and use `Remove selected runtime source` to delete it
(`remove_sound`). Runtime sources are cleared on the next episode
reset. The existing robot or pedestrian heard-sound display shows each active
source-to-listener path, portal route, and delay. Set the
visualizer's `continuous_listener_id` parameter when a specific microphone
should own the path display. If it is empty, the first continuous listener is
used.

## Microphones and playback routing

Every robot in `state/robots` automatically creates a center microphone and a
left/right pair named from the robot instance, for example `robot1_mic`,
`robot1_left_mic`, and `robot1_right_mic`. All three follow the robot base TF
frame. The side microphones are 20 cm apart at 0.35 m height by default, with
left at positive robot y and right at negative robot y. RViz renders the center
microphone green, the left microphone blue, and the right microphone orange.

Both side microphones remain propagation listeners at the same time so their
listener-specific delays and levels can later feed a direction-of-arrival
estimator. The **Left microphone** and **Right microphone** buttons in the
Auditory panel select which one feeds mono workstation playback; the existing
dropdown remains available for every other listener. Playback selection does
not disable propagation to either side microphone.

The propagation-node parameters `robot_side_microphones`,
`robot_side_microphone_separation_m`, `robot_microphone_height_m`, and
`robot_microphone_forward_offset_m` control the automatic pair. Their defaults
are `true`, `0.20`, `0.35`, and `0.0` respectively.

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

The RViz dropdown applies one microphone ID to `human_sound_playback`,
`robot_sound_node`, and `environment_sound_playback`, and keeps that selected
listener in propagation. The robot left/right pair is always propagated in
addition. For a non-RViz workflow, set `auditory.listener:=robot1_mic` or
another registered microphone ID at launch.

### Verify robot side microphones

Start one Jackal, a looping test radio, propagation visualization, and local
playback:

```bash
arena launch \
  world:=map_empty \
  robot:=jackal \
  auditory:=arena \
  auditory.viz:=true \
  auditory.static_sounds:='[{name: mic_test_radio, asset_id: radio_loop, position: [2.0, 2.0, 1.2], semantics: [{preset: sound}]}]'
```

In another shell, confirm registration and RViz marker publication:

```bash
ros2 topic echo \
  /arena/env_0/task_generator_node/microphone_listeners \
  --qos-durability transient_local --once

ros2 topic echo \
  /arena/env_0/task_generator_node/microphone_markers \
  --once --field markers
```

The registry must contain `<robot>_left_mic` and `<robot>_right_mic`. In RViz,
enable `Arena/Sound Propagation/Microphones`; the blue and orange cones must
move and rotate with the robot. Click **Left microphone** and **Right
microphone** under **Legacy Audio Playback Microphone** to compare playback. Confirm
that the selection reached all playback nodes and propagation, replacing
`jackal_left_mic` if the registry shows a different robot name:

```bash
ros2 param get \
  /arena/env_0/task_generator_node/human_sound_playback listener_id
ros2 param get \
  /arena/env_0/task_generator_node/robot_sound_node listener_id
ros2 param get \
  /arena/env_0/task_generator_node/environment_sound_playback listener_id
ros2 param get \
  /arena/env_0/task_generator_node/sound_propagation_node active_microphone_id
```

Finally, confirm that propagation continues to publish both side listeners,
regardless of which playback button is selected:

```bash
ros2 topic echo \
  /arena/env_0/task_generator_node/continuous_heard_sounds \
  --field listener_id
```

The stream must repeatedly contain both `<robot>_left_mic` and
`<robot>_right_mic`. Finite greetings and footsteps can be checked similarly
on `heard_sound_events`.

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

With `ped_hearing=true` (the launch default via `auditory.ped_hearing`),
every non-source pedestrian is an `agent:<id>` listener whose
`continuous_heard_sounds` states feed `BaseHumanSimulator.notify_stimulus`,
edge-triggered on the `audible` bit with the catalog `sound_type` as the
stimulus name. With `false`, pedestrian-to-pedestrian propagation is not
calculated or published and the RViz propagation visualizer has no
corresponding blue paths to draw. Robot and microphone listener events
receive the complete route metadata. Human events are rendered
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
`/tmp/pulse/native`, `auditory.playback:=auto` prefers the `pulse` device
(then `pipewire`, `default`, and the PortAudio default).

RViz draws the complete source-to-portals-to-listener line and one cube per
portal. Pedestrian-listener propagation is blue; robot-listener propagation
is purple, on separately controllable marker topics.

Audit every installed world before relying on RIR coverage:

```bash
ros2 run arena_auditory acoustic_world_audit --stride-cells 10
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
- `ped_hearing` (default `false`, launch sets `true`)
- `compute_rir_in_propagation` (default `true`)
- `pyroom_cache_position_quantization_m` (default `0.25`)

## Tests

Run the auditory ROS tests:

```bash
python3 -m pytest arena_auditory/tests/ros/test_sound_event.py -q
```

Run only the full auditory round-trip test:

```bash
python3 -m pytest \
  arena_auditory/tests/ros/test_sound_event.py::test_auditory_round_trip_greeting_reaches_robot_marker \
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
ros2 run arena_auditory auditory_benchmark \
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

- Sound assets: `arena_auditory/config/acoustic_assets.yaml`
- Acoustic materials: `arena_auditory/config/acoustic_materials.yaml`
- Launch wiring: `task_generator/launch/auditory/` (axis dispatch, `arena/arena.launch.py` node stack)
- Node-side axis: `task_generator/task_generator/simulators/auditory/`
- Nodes and DSP: `arena_auditory/arena_auditory/`
- Human event generation: `arena_auditory/arena_auditory/auditory_events.py`

## Robot-side hearing

`arena_auditory.hearing` is the consumer side. `hearing_belief` turns
`task_generator_msgs/HeardSoundEvent` into a decaying pedestrian-likelihood
grid in the map frame and publishes it as an RViz `OccupancyGrid`
(`hearing/belief_grid`) and a wedge `MarkerArray`. `hearing_policy` reads that
grid and is the single writer of the Nav2 `SpeedFilter` mask
(`hearing/speed_filter_mask`): a listen-then-yield policy that layers a
corner-listen speed cap and a yield hold on top of the belief slowdown. A live
SELDnet front-end turns the 4-mic `AudioFrame` stream into the same event
message, so the belief node is source-agnostic.

### Nodes and entry points

| entry point | what it is |
|---|---|
| `hearing_belief_node` | belief grid (`hearing/belief_node.py`, estimator in `hearing/belief_grid.py`) |
| `hearing_policy` | listen-then-yield, the single writer of the Nav2 speed-filter mask (`hearing/policy_node.py`, pure logic in `hearing/policy.py` and `hearing/corners.py`) |
| `hearing_seld_frontend` | live SELDnet front-end: `AudioFrame` -> `HeardSoundEvent` at 10 Hz (`hearing/seld_frontend_node.py`, model in `hearing/seld.py`) |
| `hearing_audio_replay` | publishes a 4-channel wav as `AudioFrame` blocks, for front-end tests |

### Running in an Arena env

`robot.hearing:=bus` on `arena launch` brings the layer up per env: it includes
`launch/hearing.launch.py` next to the auditory sidecar, merges
`config/hearing/nav2_overlay.yaml` (a `SpeedFilter` on the local costmap and
the controller's `speed_limit_topic`) into the robot's Nav2 parameters through
the nav2 adapter's `params_overlay`, and adds the belief, speed mask and wedge
displays to the RViz that `arena launch` opens:

    export ARENA_WORLD_PATH=$ARENA_WS_DIR/src/Arena/_assets/arena-benchmarks-prod-public/suites/acoustics/worlds
    arena launch sim:=gazebo robot:=jackal world:=acoustics_bend_narrow_O \
        task.robots:=scenario task.obstacles:=scenario \
        task.scenario:=hearing__world-acoustics_bend_narrow_O__robot-moving__pedestrians-1__ends-a-to-b \
        auditory:=arena robot.hearing:=bus

`robot.hearing:=seld` runs the SELDnet front-end instead of the bus and
implies `microphone_mode:=four_mic`. The nodes bind to the robot announced on
`<tg_node>/state/robots`, so `robot:=auto` works; with several robots per env
pass `robot:=<name>` to `hearing.launch.py` directly.

Topics, all under the env namespace `/arena/env_0`:

| direction | topic | note |
|---|---|---|
| in | `task_generator_node/jackal/heard_sound` | bus events from `robot_hearing_node`, BEST_EFFORT |
| in | `task_generator_node/map` | grid geometry, TRANSIENT_LOCAL |
| in | `task_generator_node/state/resetting` | clears the grid on episode reset |
| in | `task_generator_node/jackal/plan` | robot's Nav2 global plan, for blind-bend detection |
| in | tf `map -> env_0/jackal/base_link` | robot pose |
| out | `hearing/speed_filter_mask` | `OccupancyGrid`, TRANSIENT_LOCAL, written by `hearing_policy`, read by the SpeedFilter |
| out | `hearing/policy_state` | `std_msgs/String` JSON: state, dist_to_bend_m, frac_ahead, mass_ahead, mass_behind, mass_total, level_slope_db_s, limit_pct, binding_layer, yield_count, time_yielding_s |
| out | `hearing/policy_markers` | approach lane and hold band |
| out | `hearing/costmap_filter_info` | from `costmap_filter_info_server`, type 1 (percent) |
| out | `hearing/speed_limit` | published by Nav2's SpeedFilter, consumed by `controller_server` |
| out | `hearing/belief_grid`, `hearing/belief_wedges` | RViz |

`source:=seld` starts the front-end on `task_generator_node/jackal/audio/raw_array`
(the renderer's `AudioFrame`, 16 kHz x 4 ch, interleaved float, which exists
only in `microphone_mode:=four_mic`) and consumes
`task_generator_node/jackal/heard_sound_seld` with `bearing_frame: robot` and a
10 Hz nominal event rate. The bus is map-frame and 2 Hz (`source:=bus`, the
default).

### Consumer semantics

Nav2's `SpeedFilter` reads the mask at the robot's own cell and publishes
`speed_limit` when it changes. The belief layer of the mask is the belief
max-filtered over a disc of `reaction_radius_m` (2.0 m) before thresholding:
the robot is slowed while likely pedestrian mass lies within that radius, from
`speed_free_pct` (100) down to `speed_min_pct` (40) as the belief goes from
`belief_threshold` (0.6) to 1, so one footstep does not floor the whole
corridor. Those four now live in `config/hearing/policy.yaml`, read by
`hearing_policy`, not on the belief node. The wedge itself is never narrower
than `min_half_width_m` (0.3 m) so its apex carries full weight.

`use_level_range` (default false) keeps the wedge flat out to `max_range_m`.
On the bus a wall costs about 7 dB, so an occluded pedestrian sounds twice as
far as it is; with the level-derived range the bump sits at 15 m and the
speed limit barely moves before the corner.

### Listen-then-yield

`hearing_policy` composes `hearing/speed_filter_mask` from three layers by
lowest nonzero percentage (0 stays "no limit" and never wins over an actual
cap): belief (the dilated, thresholded `hearing/belief_grid`), listen
(`listen_mps` painted on the approach to a blind bend), and hold (`hold_mps`
on a short band before the bend while yielding). The extra layers exist
because the robot's own drivetrain masks the pedestrian it is trying to hear:
amplitude scales with speed^1 on the Jackal drivetrain, so at 0.5 m/s the
noise floor masks footsteps beyond about 3 m, and slowing to 0.2 m/s pushes
that horizon out to about 8 m.

A plan point is blind when the map line from it to the point `lookahead_m`
further along the plan crosses occupied cells; the bend is the first point
after that where the line clears again. Bends are map-frame points tracked
with `bend_hysteresis_m` hysteresis and a consumed latch (`rearm_after_m`), so
a 1 Hz replan does not re-arm the corner it just cleared.

State machine: cruise, listen once inside `approach_m` of the bend, yield once
the fraction of total belief mass inside the `corner_radius_m` disc around the
bend exceeds `yield_fraction`, release to pass once the mass has moved behind
the robot, faded below `release_fraction`, or the received level has been
falling for `recede_s` (each only after `min_yield_s` of silence), or
unconditionally at `yield_timeout_s`. Pass takes the bend at listen speed,
never full speed, so a pedestrian who yielded in turn is not driven into.

Nav2 reads a 0 % mask as "no limit", so the hold is a `hold_mps` (0.03 m/s)
creep rather than a stop, three times the progress checker's 0.2 m / 20 s
threshold.

`robot.hearing.policy:=belief|listen|full` on `arena launch` (`policy` on
`hearing.launch.py`) picks how many layers are active. Parameters live in
`config/hearing/policy.yaml`: `listen_mps` 0.2, `hold_mps` 0.03,
`lookahead_m` 4.0, `approach_m` 4.0, `hold_len_m` 1.5, `hold_offset_m` 1.0,
`lane_radius_m` 0.6, `corner_radius_m` 2.0, `bend_hysteresis_m` 1.0,
`rearm_after_m` 3.0, `yield_fraction` 0.5, `release_fraction` 0.25,
`min_yield_s` 3.0, `yield_timeout_s` 15.0, `recede_s` 2.0.

v1 has no approach-versus-recede estimate beyond the level trend: presence
near the bend yields regardless of whether the source is closing or
receding.

### Evaluation

The arms are benchmark contestants, `contests/hearing.yaml` in arena_evaluation,
run against the `acoustics` suite from the benchmark bucket:

    arena evaluation benchmark --suite acoustics --contest hearing

`yield_count` and `time_yielding_s` on `hearing/policy_state` are cumulative
per env, so a metric over the recording differences them per episode.

### Front-end checks

    ros2 run arena_auditory hearing_seld_frontend --ros-args -r __ns:=/hearing_test
    ros2 run arena_auditory hearing_audio_replay <4ch.wav> --topic /hearing_test/audio/raw_array

`hearing.seld.SeldFrontend.events_from_wav` reproduces the DCASE per-file output
(410/410 detections on the test clip, azimuths within 1e-5 deg, cuDNN noise).
The streaming path re-runs the model on a 5 s sliding window once per 100 ms
label frame and emits the frame `lookahead_frames` (default 5) before the
window end, since the model is not causal.

The model supplies detection and class. The bearing comes from
`bearing_source`: `gcc` (default) fits the known array geometry to GCC-PHAT
delays over the six mic pairs of the detection frame (`hearing/doa.py`), `seld`
takes the model's azimuth. The shipped checkpoint's DOA head is front-biased:
on the stock footstep rendered through the sim array at 3 m, sources at 0 and
45 deg come back near 12 and 35 deg, but 90 deg left comes back near 2 deg,
270 deg right near -34 deg, and a source directly behind puts a tenth of its
detections within 10 deg of straight ahead. The fit is within 2 deg everywhere
on the same clips (`tests/unit/test_seld_bearing.py`, the checkpoint half runs
when the weights are fetched). With two simultaneous sources the fit follows the
louder one, which is the one thing the model could in principle do better.

### Weights

The front-end needs two files, declared in `weights.yaml` with sha256 pins:
the checkpoint (`.h5`, 5.3 MB) and the feature scaler (`scaler.npz`, mean and
scale per feature bin, fitted on the checkpoint's training split, so the two
only make sense together). They live on Hugging Face and land in
`$ARENA_DATA_DIR/auditory/seld/`:

    ros2 run arena_auditory hearing_setup

The front-end node fetches them itself on first use when they are missing.
The model architecture, SALSA-Lite feature extraction and multi-ACCDOA decode
are in `hearing/dcase.py`, adapted from the DCASE 2023 SELD baseline
(MIT), so no checkout of that repository is needed at runtime.
