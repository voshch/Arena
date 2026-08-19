# gestures

World-frame attention channels from the `arena_peds` bus become upper-body overlay slots on the `AnimationManager`.

## Wire

Every `Pedestrian.gestures[]` entry is one channel: `slot` names the body part (`head`, `arm`, `arm_l`, `arm_r`),
`at` is a world-frame point, `opts` is a JSON object string with per-slot options (empty = defaults).
`BaseHumanSimulator.publish_arena_peds` builds one `GestureRequest(channels, pose, moving)` per ped with a
`Channel(slot, at, opts)` per entry (opts parsed every tick, invalid JSON warns once per ped and slot and becomes `{}`)
and passes it to `AnimationManager.compute(..., gesture=...)`, which calls the layer through `gesture_hook`.

| slot | kind | opts | hand |
| --- | --- | --- | --- |
| `head` | `look` | none | |
| `arm` | `point` | `{"dominant": "l\|r"}` (default `r`) | resolved once at slot start: `dominant` within the 20 deg midline band, else the target-side arm |
| `arm_l` / `arm_r` | `point` | none | forced left / right |

The engine publishes exactly the active channels each tick. Nothing is implicit: no head channel = the head idles
while the arm points.

## Slots

Every overlay lives in a named slot on the manager (`set_ped_blend(..., slot="arm")`), each with its own
playhead, envelope and `on_end`. Slots are applied in insertion order. `head` drives the `head` overlay, the
three arm channels share the `arm` overlay, so at most one arm points at a time. The layer keeps one `_Agent`
per ped with a `_Slot` per (agent, overlay), and each slot runs its own phase machine, independent of the other:

- start: a channel appears (or reappears). `Gesture.bind(slot, local, opts)` freezes the
  generator options for the life of the slot: for `point` that is the hand, so a chained list keeps its arm
  across the midline and retargets never switch hands. The layer adds `moving` to the opts it binds: a walking
  ped does not blend its torso, so `point` solves its arm against an upright spine, and a `moving` flip on a
  held slot re-solves through a retarget.
- retarget: the channel's target moved by more than `RETARGET_EPS_M` while the slot holds.
- release: the channel disappeared, its `opts` changed, or a different arm channel
  wants the `arm` overlay (that one starts once the tail has played). Restarts (`opts` change, arm switch,
  a `ValueError` from `retarget`) skip `HOLD_MIN_S`.
- a generator fault drops only that slot, with one warning per (ped, kind), and the same channel is not
  resubmitted until its slot, target, or opts change.

## Phases

```
requested -> ramp -> hold -> (transition -> hold)* -> release -> gone
```

- `requested`: the clip is generated inline on the publishing tick. Nothing solves at runtime: `point` swings the
  recorded template onto a hold interpolated from the baked `pointing` table, a few milliseconds of matrix products.
- `ramp`, `hold`: the clip up to its park frame is an overlay that holds its park pose (or loops a breathing
  segment, see Timing) as long as the channel does.
- `transition`: target moved: `Gesture.retarget(hold, local, opts)` continues from the held pose without
  dropping the arm. A `ValueError` from `retarget` releases and restarts.
- `release`: eases from the pose the slot shows now into the current clip's lowering arc (`GestureClip.release`)
  with a `FADE_S` fade-out, then drops the transient clips. Never earlier than `HOLD_MIN_S` after the hold is
  reached unless it is a restart. A request for the same overlay waits until the tail has played.

## Reach

Reach is the engine's call (`arena_humansim.core.behavior.reach`, hysteresis per channel): a channel on the bus is
one the engine wants shown, the layer serves whatever arrives. The baked table covers the full azimuth circle,
so an off-envelope aim yields a flagged hold, never a fault.

## Timing

- Ramp of a fresh `point` = `clamp(swept / OMEGA_RAD_S, MOVE_MIN_S, MOVE_MAX_S)` (3.5 rad/s, 0.3 .. 0.9 s), the
  swept angle being from the hanging arm to the aim. The template's ramp segment is resampled to that length,
  hold and settle are kept as recorded, the release is the recorded lowering arc stretched by `RELEASE_STRETCH`
  (1.25). A retargeted `point` instead lowers from its new hold pose with an ease to rest over
  `move_time(angle from the new aim to down) * RELEASE_STRETCH`, so the arm never swings back through the first target.
- Retarget transitions use the same clamp on the angle between the old and new aim: an eased blend from the
  held pose into the new hold, then the template's settled plateau.
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
(yaw, pitch, roll of waist, spine, chest) only while the ped is idle, so walking gaits keep their torso sway. `look` blends the head triple.
When `moving` flips on a live slot the layer recomputes the set and swaps it in place through
`AnimationManager.set_overlay_joints(..., fade_s=FADE_S)`: leaving joints ramp their weight to 0 and entering
joints ramp to 1, playhead and envelope untouched, no reinstall.

## QA

[qa.py](qa.py) replays scripted cases (single points, bursts, chains, hand switches, grid cells with rests
between, a walker tracking a target) through the real manager and layer and asserts what the sim would show:
no warnings, no wrist/elbow/head snap over `MAX_LINK_STEP_M`, clavicle below `MAX_COLLAR_RAD`, the parked
forearm within `MAX_HOLD_AIM_DEG` of the target, the arm hanging inside every `rest_windows`, and at least
`min_clips` arm clips. `python3 -m task_generator.simulators.human.gestures.qa` renders filmstrips,
`tests/test_gesture_qa.py` runs the same cases.

## Registry

`GESTURES` is a `ClassRegistry[str, Callable[[], Gesture]]`, `SLOT_KIND` maps wire slots to kinds and
`SLOT_OVERLAY` maps them to manager overlays. The `Gesture` protocol:

```python
class Gesture(Protocol):
    def joints(self, side: str, moving: bool) -> set[str]: ...
    def bind(self, slot: str, local: np.ndarray, opts: dict) -> dict: ...
    def start(self, local: np.ndarray, opts: dict) -> GestureClip: ...
    def retarget(self, hold: object, local: np.ndarray, opts: dict) -> GestureClip: ...
    def breathing(self, side: str) -> dict[str, float]: ...
```

`GestureClip` carries wire-format frames (`{"angles": {joint: rad}}`, only the blended joints are required),
`fps`, `hold_start`, `hold_end` (the park frame, its pose is what `hold` continues from), `side`, an opaque
`hold` handle, the generator report, and `release` (the lowering arc from the park pose, `None` keeps the
previous one). Generator exceptions drop the slot with one warning per ped and never reach the roster publish.

| kind | module | generator |
| --- | --- | --- |
| `point` | [point.py](point.py) | [`pointing.BakedPointAt`](../pointing/table.py) `point_at` / `retarget` on the baked hold table, `resolve_hand` wraps `choose_hand` |
| `look` | [look.py](look.py) | closed form on [`pointing.skeleton`](../pointing/skeleton.py) |

### Adding a kind

1. `gestures/<kind>.py` with a class satisfying `Gesture`. Reuse `pointing/` for aimed chains, or return frames
   from any clip. Return `dict(opts)` from `bind()` and `{}` from `breathing()` unless you want them.
2. Register in `gestures/__init__.py` and map a wire slot to it in `SLOT_KIND` / `SLOT_OVERLAY`:

```python
@GESTURES.register("wave")
def _load_wave() -> Callable[[], Gesture]:
    from .wave import WaveGesture
    return WaveGesture
```

3. The engine side then needs a channel that publishes that slot. Slots and opts are opaque strings on the wire.

Non-aimed gestures ignore `local` in `start` and can raise `ValueError` from `retarget` to always restart.
