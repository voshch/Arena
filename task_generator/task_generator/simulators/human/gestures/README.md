# gestures

World-frame gesture intents from the `arena_peds` bus become upper-body overlay slots on the `AnimationManager`.

## Wire

`Pedestrian.gesture.kind` names a registered gesture, `Pedestrian.gesture.at` is a world-frame point,
`Pedestrian.gesture.opts` is a JSON object string with per-kind options (empty = defaults).
`BaseHumanSimulator.publish_arena_peds` builds one `GestureRequest(kind, at, pose, moving, opts)` per ped
(opts parsed every tick per ped, invalid JSON warns once per ped and becomes `{}`) and passes it to
`AnimationManager.compute(..., gesture=...)`, which calls the layer through `gesture_hook`.

| kind | opts | effect |
| --- | --- | --- |
| `point` | `{"hand": "auto\|left\|right"}` | forces the pointing arm, a change on a live gesture releases and restarts |
| `look` | none | |

## Slots and companions

Every overlay lives in a named slot on the manager (`set_ped_blend(..., slot="arm")`), each with its own
playhead, envelope and `on_end`. Slots are applied in insertion order. A gesture kind declares the slot it
drives as the primary (`Gesture.slot`, `point` -> `arm`, `look` -> `head`) and optional companions
(`Gesture.companions() -> [(slot, kind)]`, `point` -> `[("head", "look")]`) that follow it at the same target.

The layer keeps one `_Agent` per ped with a `_Slot` per (agent, slot). The primary slot drives the phase
machine, companions follow:

- start: companions install first, the primary waits `GAZE_LEAD_S` (0.15 s) so the gaze leads the arm.
- retarget: the primary retargets when the target moved by more than `RETARGET_EPS_M` and every companion
  retargets to the same target (deferred until it holds if it is mid-transition).
- release: the primary lowers, companions follow `COMPANION_RELEASE_LAG_S` (0.3 s) later.
- a companion whose own reach fails is released or never started, the primary continues alone.
- a generator fault on the primary drops the whole agent, on a companion only that slot.

## Phases

```
requested -> ramp -> hold -> (transition -> hold)* -> release -> gone
```

- `requested`: the generator job is submitted (two worker threads, or inline when `sync` is set). The gait keeps playing.
- `ramp`, `hold`: the clip up to its park frame is an overlay that holds its park pose (or loops a breathing
  segment, see Timing) as long as the intent does.
- `transition`: same kind, target moved: `Gesture.retarget(hold, local, opts)` continues from the held pose without
  dropping the arm. A `ValueError` from `retarget` (for `point`: hand switch) releases and restarts.
- `release`: intent cleared, kind changed, opts changed, or reach lost. Eases from the pose the slot shows now
  into the current clip's lowering arc (`GestureClip.release`) with a `FADE_S` fade-out, then
  drops the transient clips. Never earlier than `HOLD_MIN_S` after the hold is reached unless it is a restart.
  New requests wait until every slot's tail has played.

## Reach and hysteresis

`Gesture.reach(local) -> float` is a metric (radians of azimuth for `point` and `look`) compared against the
kind's `REACH_IN` / `REACH_OUT` (`point` 1.57 / 1.92, `look` 1.05 / 1.22): a slot starts only below
`REACH_IN` and keeps going until the metric exceeds `REACH_OUT`. The flag lives per (agent, slot), so the
gaze can give up while the arm keeps pointing.

## Timing

- Ramp of a fresh `point` = `clamp(swept / OMEGA_RAD_S, MOVE_MIN_S, MOVE_MAX_S)` (3.5 rad/s, 0.3 .. 0.9 s), the
  swept angle being from the hanging arm to the aim. The template's ramp segment is resampled to that length,
  hold and settle are kept as recorded, the release is the recorded lowering arc stretched by `RELEASE_STRETCH`
  (1.25). A retargeted `point` instead lowers from its new hold pose with an ease to rest over
  `move_time(angle from the new aim to down) * RELEASE_STRETCH`, so the arm never swings back through the first target.
- Retarget transitions use the same clamp on the angle between the old and new aim. Below
  `LIGHT_RETARGET_RAD` (20 deg) `PointAtGenerator.retarget_light` keeps the elbow swivel and re-solves only
  the aim, beyond it the full `retarget` with its swivel search runs.
- Breathing: `Gesture.breathing(side)` names joints and amplitudes (`point`: shoulder pitch and elbow of the
  pointing arm, +-1.5 deg). The layer bakes a slow sinusoid (nominal `BREATH_HZ` 0.3, rounded to a whole
  number of frames) into every clip of the slot from the first hold onwards, phase seeded from the agent id
  and continuous across retargets, and appends one period after the park frame as a loop tail
  (`Animation.loop_from`). No wall clock anywhere.
- `look`: closed-form head yaw/pitch from the rest-pose eye point (`skeleton.fk` `eye`, torso assumed upright),
  clamped to the head limits, ease-in 0.25 s, ease-out 0.3 s, retargets ease over 0.25 s.

The world target is converted into the ped frame (x forward, y left, z up, origin on the floor under the ped)
once per clip from the bus pose. Generator reports that flag `ok=False`, `near_target_fallback`, or `relaxed`
are logged once per (ped, kind) and the clip still plays.

Blend joints come from `Gesture.joints(side, moving)`. `point` blends the pointing arm always and the spine
assist only while the ped is idle, so walking gaits keep their torso sway. `look` blends the head triple.
When `moving` flips on a live slot the layer recomputes the set and swaps it in place through
`AnimationManager.set_overlay_joints(..., fade_s=FADE_S)`: leaving joints ramp their weight to 0 and entering
joints ramp to 1, playhead and envelope untouched, no reinstall.

## Registry

`GESTURES` is a `ClassRegistry[str, Callable[[], Gesture]]`. The `Gesture` protocol:

```python
class Gesture(Protocol):
    slot: str
    REACH_IN: float
    REACH_OUT: float
    def joints(self, side: str, moving: bool) -> set[str]: ...
    def reach(self, local: np.ndarray) -> float: ...
    def start(self, local: np.ndarray, opts: dict) -> GestureClip: ...
    def retarget(self, hold: object, local: np.ndarray, opts: dict) -> GestureClip: ...
    def companions(self) -> list[tuple[str, str]]: ...
    def breathing(self, side: str) -> dict[str, float]: ...
```

`GestureClip` carries wire-format frames (`{"angles": {joint: rad}}`, only the blended joints are required),
`fps`, `hold_start`, `hold_end` (the park frame, its pose is what `hold` continues from), `side`, an opaque
`hold` handle, the generator report, and `release` (the lowering arc from the park pose, `None` keeps the
previous one). Generator exceptions drop the gesture with one warning per ped and never
reach the roster publish.

| kind | module | generator |
| --- | --- | --- |
| `point` | [point.py](point.py) | [`pointing.PointAtGenerator`](../pointing/) `point_at` / `retarget` / `retarget_light` |
| `look` | [look.py](look.py) | closed form on [`pointing.skeleton`](../pointing/skeleton.py) |

### Adding a kind

1. `gestures/<kind>.py` with a class satisfying `Gesture`. Reuse `pointing/` for aimed chains, or return frames
   from any clip. Return `[]` from `companions()` and `{}` from `breathing()` unless you want them.
2. Register in `gestures/__init__.py`:

```python
@GESTURES.register("wave")
def _load_wave() -> Callable[[], Gesture]:
    from .wave import WaveGesture
    return WaveGesture
```

3. Scenario authors use `gesture: wave` right away. Nothing in humansim or the messages changes: kinds and
   opts are opaque strings on the wire.

Non-aimed gestures ignore `local` in `start` and can raise `ValueError` from `retarget` to always restart.
