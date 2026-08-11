---
name: project_viewport_camera_itf
description: "ViewportITF viewport-camera control over ROS 2 via reference-frame+pose-stream model; Gazebo done via GUI plugin, Isaac parity pending"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2d01c10b-1fb0-47cc-8baa-6494899e4ee7
---

Viewport (interactive GUI camera) control exposed over ROS 2 under `/arena/viewport/*`.
Redesigned 2026-06-20 from a `follow` service to a reference-frame + pose-stream model.

Endpoints (srv/msg types in arena_runtime_msgs):
- `set_view` (srv ViewportSetView): one-shot eye+target+fov in the current reference frame, snaps immediately.
- `set_reference_frame` (srv ViewportSetReferenceFrame): sets the frame that `set_view` and `cmd_view` are expressed in. Three modes: (a) entity non-empty, tracks that sim_path entity from the ECM each frame; (b) entity empty + has_pose, constant reference at the supplied Pose; (c) entity empty + !has_pose, LATCH, freezes the reference at its current world pose. `mode` field (FULL=0/YAW_ONLY=1/POSITION_ONLY=2) controls how much of the tracked entity's rotation is inherited when tracking.
- `cmd_view` (msg ViewportView, subscribe): a stream of timestamped keyframes (target_time + Pose + world_orientation + fov). The plugin BUFFERS them (deque, QoS best-effort keep-last-64 so none drop) and each render frame interpolates the local pose at clock-now (lerp pos, slerp rot) between the bracketing keyframes. The helper stamps each keyframe now+LEAD (0.3s) so a publish stall up to LEAD is invisible (jitter buffer). This decouples smoothness from publish rate, THE fix for choppiness under CPU load; cost is ~LEAD view latency. Replaced the old per-frame low-pass smoothing.
- `set_projection` (srv ViewportSetProjection): "perspective" or "orthographic". Unchanged.
- `capture` (srv ViewportCapture): request = Pose + world_orientation + fov (current ref frame); response = success + msg + sensor_msgs/Image (rgb8). Snaps to the EXACT composed world pose (reference*local, via the shared ComposeWorld/ResolveReference helpers the live drive also goes through now), renders on the render thread (MaybeCapture, condition-variable handshake, service blocks up to 5s), reads pixels back dropping any alpha. The deterministic-recording primitive.
- `camera_pose` (msg PoseStamped, publish): live camera world pose, ~10 Hz, in the `map` frame.

RECORDING (added 2026-06-21): `Camera.record(out_dir, fps=30)` / CLI `arena cam <name> ... record=<dir> [fps]` renders a shot to a numbered PPM sequence (frame_%05d.ppm, P6 raw RGB so zero encoder dep) via the capture service, one synchronous grab per frame, so output is smooth+deterministic independent of render speed. Segment emits round(duration*fps) frames, discrete look 1 frame, ref/projection verbs 0. CamNode.drive() owns both paths: record (frame-indexed capture) vs live (wall-clock cmd_view stream). CAMERA-locked NOT physics-locked: scene advances at sim rate during capture, pause the sim for full reproducibility. Physics-lockstep (step sim by dt per frame) is the remaining follow-up. Record dir: bare name -> $ARENA_DATA_DIR/recordings/<name>, absolute/slash path verbatim; recording into a NON-EMPTY dir ERRORS (FileExistsError, cli catches -> SystemExit), -f/--force overwrites (clears prior frame_*.ppm). No stamping by choice. record_dir() in record.py validates+prepares before run_main so the error is clean (no ROS start).

ORBIT is SPHERICAL (changed 2026-06-21, was planar): eye on a sphere of `radius` around center looking at it, so radius = true subject distance (constant), vertical knob is `elevation`/`elevation_deg` (0 = level ring, +90 straight down), `height` param removed. elevation=0 == old height=0 so default unchanged. WARN-on-unknown-params (not reject): _Params wrapper in shots.py records which keys a verb's from_params reads, resolve() logs.warning the rest under logger 'arena_cam' (a mistyped param was silently dropped before).

Each render frame the camera world pose is `reference_frame * local_pose`. SIM-PORTABILITY DOCTRINE (decided 2026-06-20): the ONLY irreducibly sim-specific surface is a thin CameraBackend (set/get world pose, set projection/fov, resolve a named entity's world pose). Everything else (keyframe buffer + interpolation, reference composition, drive model, grab-to-release, the ROS services/topics) is sim-agnostic and belongs in a shared ViewportController. The interpolation MUST run in each sim's render loop (that is what makes it jitter-proof), but the logic is written once per language (C++ for gz, Python for Isaac), not per sim. Currently the gz plugin holds it inline (the reference impl); the interpolation core (`SampleBuffer`/`Keyframe`) is already factored, a full ViewportController/CameraBackend extraction is the documented next step.

DRIVE-MODEL INVARIANT (a port re-regressed this once, don't again): `set_view` is one-shot, snaps once then RELEASES to manual orbit, NOT a permanent pin. `drive = oneShot || streaming || (refTargetPose && localSet)`; the track term requires the entity to be RESOLVED in the ECM, so a missing/late track does NOT pin forever. `oneShot` consumed per frame. `streaming` is set per keyframe and CLEARED after kStreamTimeout (400ms, > LEAD) of silence, so a finished/crashed stream releases. A resolved-track shot keeps FOLLOWING the entity after the stream ends, until the user manually moves the camera (its pose drifts from `appliedPose` beyond eps -> drop the reference) or sets something new. An unresolved tracked entity falls back to the world origin while streaming (plays in world frame instead of freezing) and warns once. Spin thread: `spin_once(100ms)` + atomic stop.

Architecture (the non-obvious parts):
- It MUST be a gz **GUI plugin** (`gz::sim::GuiSystem`), not a world/system plugin: the
  user-camera lives in the gz GUI process, the server can't touch it. Plugin is
  `ViewportCamera` in arena_gz_plugins (sibling of PedSkeletonPlugin). It drives the
  user-camera on the render thread (gz Render event), samples reference-frame entity poses
  from the ECM in Update, runs rclcpp on its own spin thread.
- `ViewportITF` is a defaults-only mixin on `BaseSim` (sim/_interface.py) mirroring the
  services in-process (`viewport_set_view`, `viewport_set_reference_frame`,
  `viewport_stream_view`, `viewport_set_projection`, `viewport_camera_pose`);
  GazeboSimulator implements it with short-timeout clients kept OUT of the ready-gate
  (services only exist when the GUI runs, headless has none).
- gazebo.launch.py DERIVES a gui.config at launch from gz's own default
  (~/.gz/sim/8/gui.config): appends the ViewportCamera plugin, pins the render engine,
  passes via the gui-config flag, sets GZ_GUI_PLUGIN_PATH. No vendored full config (gz has
  no loose system default, only the ~/.gz copy it writes on first GUI run; fresh container's
  first launch has no plugin, self-heals from the next). gui-config replaces, not merges.

BUILD GOTCHA (cost a long debug): the ViewportCamera Q_OBJECT class must be moc'd via
explicit `qt5_wrap_cpp(... include/.../ViewportCamera.hh)` in CMakeLists, NOT
`set_target_properties(... AUTOMOC ON)` after add_library, which silently ran the
autogen but produced no moc_ViewportCamera.cpp. Symptom: .so links fine but gz dlopen
fails with `undefined symbol: typeinfo for arena_gz_plugins::ViewportCamera` (vtable/
typeinfo/metaObject never emitted because the Q_OBJECT key function had no moc). Verify
with `nm -C libViewportCamera.so | grep "vtable for ...ViewportCamera"` = V not U.

Isaac had NO ROS viewport API either, so this DEFINED the shared contract. Isaac parity is
a PENDING follow-up: implement the CameraBackend against its viewport camera + mirror the
ViewportController (Python). Built + live-debugged 2026-06-20 on feature/drl-planners: camera
moves, tracking / grab-release work; keyframe buffer added to kill choppiness, user confirmed
silky 2026-06-21. Committed on feature/viewport: arena_gz_plugins 'camera control api' (plugin+msg,
amended) and Arena 'camera control api' (msg) + 'camera scripting' (runtime cam module + arena CLI verb).
THIRD feature (recording, 2026-06-21) is staged-not-committed: Arena gets a 'camera recording' commit
(ViewportCapture.srv + msgs CMake/pkg + cam record path: record.py/drive/capture/CLI) and arena_gz_plugins
a 'camera recording' commit (capture service + MaybeCapture + sensor_msgs dep). C++ NOT yet rebuilt;
the capture pixel-read (camera->Capture/CreateImage on the GUI user-camera) is the untested risk point.
See [[project_arena_swerve_controller]] for the other first-party gz plugin pattern.
