# Automated acoustic dataset recording

`record_acoustics_dataset.sh` launches each selected `scenario.yaml` in a fresh
Arena/Gazebo process and records ROS simulation-time data. It records the
Jackal four-microphone raw float stream and the rendered stereo headphone
stream together with `/clock`, episode state, robot odometry, pedestrian
states, TF, the occupancy map, and the door mask.

The default output is:

```text
data/audio_train_set/
└── <scenario-name>/
    ├── <scenario-name>_0001_recording.wav
    ├── <scenario-name>_0001_meta.csv
    ├── <scenario-name>_0001_validation.json
    ├── <scenario-name>_0001_manifest.yaml
    ├── <scenario-name>_0001_audio_timing.parquet
    ├── <scenario-name>_0001_robot_positions.parquet
    ├── <scenario-name>_0001_pedestrian_positions.parquet
    ├── <scenario-name>_0001_frame_labels.parquet
    ├── <scenario-name>_0001_tf_transforms.parquet
    ├── <scenario-name>_0001_episode_events.parquet
    ├── <scenario-name>_0001_sound_events.parquet
    ├── <scenario-name>_0001_continuous_sound_states.parquet
    ├── <scenario-name>_0001_sound_activity.parquet
    ├── <scenario-name>_0001_occupancy_map.npz
    ├── <scenario-name>_0001_door_mask.npz
    ├── scenario.yaml
    └── episode_000/episode_000.mcap
```

The WAV contains the robot's final left/right hearing signal without requiring
an external encoder. Optional FLAC is compact and lossless for its integer PCM
representation. The MCAP remains the source
of truth for the raw propagated float32 microphone values; converting those
arbitrary floats to FLAC would not preserve them exactly.

Audio and labels use the same ROS simulation clock. `AudioFrame.header.stamp`
is set by the microphone-array publisher to the simulation time of the first
sample in the block. A sample at offset `i` therefore has time
`header.stamp + i/sample_rate`. The exporter creates one metadata row per
20 ms audio window and interpolates robot and pedestrian poses at that exact
timestamp. Each row includes sample offset, absolute simulation timestamp,
robot/source pose and velocity, relative Cartesian position, range, bearing,
elevation, radial velocity, occupancy line-of-sight labels, and audio RMS/peak
features. Robot, pedestrian, and TF trajectory exports retain normalized
quaternion `qx,qy,qz,qw` fields in addition to convenient planar yaw values.

`sound_activity.parquet` contains the actual half-open sample intervals placed
on the raw microphone-array clock, grouped across microphone channels. Frame
labels identify active pedestrian/event IDs, sound types, motor activity,
single- versus multiple-pedestrian overlap, and clean single-source windows.
The source `SoundEvent` and continuous propagated state streams are retained in
their own Parquet files for provenance. Dataset recording rejects a new run if
the sample-clock activity annotations are absent.

Build the changed packages on Ubuntu first:

```bash
cd /opt/arena_ws
source src/Arena/_meta/tools/source
arena rebuild task_generator arena_evaluation arena_simulation_setup
source install/setup.bash
sudo apt install ffmpeg  # only needed for optional FLAC export
```

Normalize the generated scenario matrix once after downloading/generating the
acoustics worlds:

```bash
normalize_acoustics_scenarios \
  --worlds-root /opt/arena_ws/src/Arena/arena_simulation_setup/acoustics/worlds \
  --write
```

The listener robot now stays at end A for both direction variants. In
`a-to-b`, pedestrians spawn as a separated left/centre/right cluster near the
robot and initially walk toward B. In `b-to-a`, they spawn as a separated
cluster at B and initially walk toward the robot. Their first waypoint is
always ahead of the spawn pose, including three-pedestrian cases, and routes
reverse at the endpoints so humans continue walking throughout long captures.
The dataset runner also enables scenario lingering, which keeps a moving-robot
episode alive after the robot reaches B until the requested audio window ends.
The same normalization pass changes every acoustics pedestrian model to the
bundled `arenian` default, including older numeric scenarios, so world preload
does not require optional network human assets.

List the deterministic execution order without launching anything:

```bash
arena_simulation_setup/acoustics/record_acoustics_dataset.sh --list
```

Record all worlds and scenarios:

```bash
arena_simulation_setup/acoustics/record_acoustics_dataset.sh
```

Small test runs can select a world, scenario-name pattern, and count:

```bash
arena_simulation_setup/acoustics/record_acoustics_dataset.sh \
  --world-glob 'straight_corridor_O' \
  --scenario-glob '*robot-idle*pedestrians-1*' \
  --max-scenarios 1 \
  --duration 5 \
  --output-relative data/audio_train_set_test
```

Use `--force` to move an incomplete/existing case to a timestamped backup and
repeat it. Completed cases with a valid validation JSON are skipped, so a full
run is resumable. With `--sim gazebo`, Gazebo is visible and windowed by
default. `--gazebo-fullscreen` only asks the desktop window manager to enlarge
the visible window. That optional operation needs `wmctrl`; ordinary visible
recording does not. Pass `--headless` to suppress the Gazebo GUI while keeping
the auditory pipeline, workstation playback, MCAP recording, and export active.
For unattended restart loops, pass `--recover-incomplete`: validated scenarios
remain untouched, while the interrupted scenario directory is moved to a
timestamped backup before that scenario is attempted again.

Playback is enabled by default with `auditory.playback:=auto`, which selects a
PulseAudio/PipeWire-compatible PortAudio output when available. Use
`--playback-device pulse` (or another PortAudio device name) to select one
explicitly. The saved WAV comes from the exact stereo `AudioFrame` supplied
to that output, rather than a sound-card loopback, so it excludes desktop audio
and stays aligned to simulation time even when real-time factor varies.

Validation is performed by the Python waiter/exporter invoked by the Bash
orchestrator, not by Bash syntax alone. A case is accepted only when both audio
streams cover the requested simulation-time interval, the rendered stream is
stereo and non-silent, sample timing is contiguous, robot/pedestrian poses can
be aligned, all four raw microphone channels and their geometry are present,
the episode has a terminal state, and the occupancy map belongs to the same
environment. The waiter owns the episode action, starts the requested
30-second window only on the episode's `RUNNING` event, and cancels it cleanly only
after the requested simulation-time coverage is reached, so the MCAP contains
both RUNNING and terminal episode events. Bash exits on any failed preflight,
action, capture, export, or missing validation file.
