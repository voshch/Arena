"""Gesture layer: turns world-frame attention channels on the bus into upper-body overlay slots on the AnimationManager."""

from __future__ import annotations

import functools
import json
import math
import typing
from collections.abc import Callable, Sequence

import attrs
import numpy as np
from arena_rclpy_mixins.registry import ClassRegistry

if typing.TYPE_CHECKING:
    from rclpy.impl.rcutils_logger import RcutilsLogger

    from task_generator.simulators.human.animation_mananager import AnimationManager

BODY_HEIGHT = 1.65
RETARGET_EPS_M = 0.15
FADE_S = 0.25
HOLD_MIN_S = 0.6
TRANSITION_S = 0.4
OMEGA_RAD_S = 3.5  # arm/head sweep rate that sets ramp and transition durations
MOVE_MIN_S = 0.3
MOVE_MAX_S = 0.9
RELEASE_STRETCH = 1.25  # lowering arcs play this much slower than the recorded template
BREATH_HZ = 0.3
BREATH_AMP_RAD = math.radians(1.5)

SLOT_KIND = {"head": "look", "arm": "point", "arm_l": "point", "arm_r": "point"}
SLOT_OVERLAY = {"head": "head", "arm": "arm", "arm_l": "arm", "arm_r": "arm"}
OVERLAYS = ("head", "arm")


def move_time(swept_rad: float) -> float:
    """Ramp or transition duration for a sweep of ``swept_rad``."""
    return min(MOVE_MAX_S, max(MOVE_MIN_S, abs(swept_rad) / OMEGA_RAD_S))


def resample(frames: Sequence[dict], n: int) -> list[dict]:
    """``frames`` time-stretched to ``n`` frames by linear interpolation of the angles."""
    if n <= 0 or not frames:
        return []
    if len(frames) == 1 or n == 1:
        return [dict(frames[0]) for _ in range(n)]
    out = []
    span = len(frames) - 1
    for j in range(n):
        s = j * span / (n - 1)
        i = min(int(math.floor(s)), span - 1)
        w = s - i
        a, b = frames[i]["angles"], frames[i + 1]["angles"]
        out.append({**frames[i], "angles": {k: a[k] * (1.0 - w) + b[k] * w for k in a}})
    return out


@attrs.frozen
class Channel:
    """One attention channel of a ped: wire slot, world target, per-slot options."""

    slot: str
    at: tuple[float, float, float]
    opts: dict = attrs.field(factory=dict)


@attrs.frozen
class GestureRequest:
    """One tick of attention channels for one ped, world frame."""

    channels: tuple[Channel, ...]
    pose: tuple[float, float, float]  # x, y, yaw
    moving: bool


@attrs.frozen
class GestureClip:
    """Generator output the layer plays: frames in wire format, the settled hold, and an opaque hold handle for chaining."""

    frames: Sequence[dict]
    fps: float
    hold_start: int
    hold_end: int  # frame the layer parks on, its pose is what ``hold`` continues from
    side: str
    hold: object
    report: dict
    release: Sequence[dict] | None = None  # lowering arc from the park pose, None = keep the previous one


class Gesture(typing.Protocol):
    def joints(self, side: str, moving: bool) -> set[str]: ...

    def bind(self, slot: str, local: np.ndarray, opts: dict) -> dict:
        """Options frozen at slot start and reused for every retarget of that slot."""
        ...

    def start(self, local: np.ndarray, opts: dict) -> GestureClip: ...

    def retarget(self, hold: object, local: np.ndarray, opts: dict) -> GestureClip:
        """Continue from a held pose to a new target. Raise ValueError to force a restart."""
        ...

    def breathing(self, side: str) -> dict[str, float]:
        """Joint -> amplitude (rad) of the slow drift on the parked hold, empty = none."""
        ...


GESTURES: ClassRegistry[str, Callable[[], Gesture]] = ClassRegistry()


@GESTURES.register("point")
def _load_point() -> Callable[[], Gesture]:
    from .point import PointGesture

    return PointGesture


@GESTURES.register("look")
def _load_look() -> Callable[[], Gesture]:
    from .look import LookGesture

    return LookGesture


def world_to_local(at: Sequence[float], pose: Sequence[float]) -> np.ndarray:
    """World xyz to ped-local (x fwd, y left, z up, origin on the floor under the ped)."""
    dx, dy = at[0] - pose[0], at[1] - pose[1]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.array([c * dx + s * dy, -s * dx + c * dy, at[2]], dtype=float)


def blend_into(start: dict[str, float], frames: Sequence[dict], n: int) -> list[dict]:
    """Ease from ``start`` into ``frames`` over the first ``n`` frames."""
    out = []
    for i, frame in enumerate(frames):
        if i >= n:
            out.append(frame)
            continue
        w = 0.5 - 0.5 * math.cos(math.pi * (i + 1) / n)
        angles = dict(frame["angles"])
        for name, value in start.items():
            angles[name] = value * (1.0 - w) + angles[name] * w
        out.append({**frame, "angles": angles})
    return out


def ease_to_rest(park: dict[str, float], n: int) -> list[dict]:
    """``n`` frames easing every joint of ``park`` to 0, the first frame is ``park`` itself."""
    n = max(2, n)
    out = []
    for i in range(n):
        w = 0.5 - 0.5 * math.cos(math.pi * i / (n - 1))
        out.append({"angles": {name: value * (1.0 - w) for name, value in park.items()}})
    return out


def breath_phase(agent_id: int) -> float:
    """Deterministic per-agent breathing phase, spread by the golden ratio."""
    return 2.0 * math.pi * ((agent_id * 0.6180339887498949) % 1.0)


class _Slot:
    """State of one (agent, overlay slot): the channel it serves, the clip it plays, and its phase."""

    __slots__ = (
        "age",
        "bound",
        "breath_ease_s",
        "breath_origin",
        "channel",
        "clip",
        "hold",
        "joints",
        "kind",
        "local",
        "opts",
        "phase",
        "release_frames",
        "release_pending",
        "slot",
        "t",
    )

    def __init__(self, slot: str, channel: str, kind: str, local: np.ndarray, opts: dict, bound: dict) -> None:
        self.slot = slot
        self.channel = channel
        self.kind = kind
        self.local = local
        self.opts = opts
        self.bound = bound
        self.phase = "requested"  # requested | ramp | hold | transition | release
        self.clip: GestureClip | None = None
        self.hold: object = None
        self.joints: set[str] = set()
        self.release_frames: Sequence[dict] = ()
        self.t = 0.0  # seconds since the current clip was installed
        self.age = 0.0  # seconds since the slot was requested
        self.release_pending = False
        self.breath_origin: float | None = None  # age at which the breathing clock started
        self.breath_ease_s = 0.0


class _Agent:
    __slots__ = ("moving", "slots")

    def __init__(self) -> None:
        self.slots: dict[str, _Slot] = {}
        self.moving = False


class GestureLayer:
    """Per-agent gesture state machine, installed as ``AnimationManager.gesture_hook``."""

    def __init__(self, manager: AnimationManager, logger: RcutilsLogger) -> None:
        self._manager = manager
        self._logger = logger
        self._gestures: dict[str, Gesture] = {}
        self._agents: dict[int, _Agent] = {}
        self._warned: dict[int, set[str]] = {}
        self._dropped: dict[tuple[int, str], tuple] = {}  # last channel whose generator failed, not resubmitted until it changes

    def forget(self, agent_id: int) -> None:
        self._agents.pop(agent_id, None)
        self._warned.pop(agent_id, None)
        for key in [k for k in self._dropped if k[0] == agent_id]:
            del self._dropped[key]

    def _warn_once(self, agent_id: int, key: str, msg: str) -> None:
        seen = self._warned.setdefault(agent_id, set())
        if key in seen:
            return
        seen.add(key)
        self._logger.warning(msg)

    def _gesture(self, kind: str) -> Gesture:
        if kind not in self._gestures:
            self._gestures[kind] = GESTURES.get(kind)()
        return self._gestures[kind]

    def _wanted(self, agent_id: int, req: GestureRequest) -> dict[str, Channel]:
        """Overlay slot -> channel; unknown slots warn once, a second arm channel is ignored."""
        wanted: dict[str, Channel] = {}
        for ch in req.channels:
            overlay = SLOT_OVERLAY.get(ch.slot)
            if overlay is None:
                self._warn_once(agent_id, f"slot:{ch.slot}", f"gesture slot {ch.slot!r} for ped {agent_id} is unknown, known: {sorted(SLOT_OVERLAY)}")
                continue
            if overlay in wanted:
                self._warn_once(agent_id, f"dup:{overlay}", f"ped {agent_id} publishes several {overlay} channels, keeping {wanted[overlay].slot!r}")
                continue
            wanted[overlay] = ch
        return wanted

    def __call__(self, agent_id: int, request: object, dt: float) -> None:
        ag = self._agents.get(agent_id)
        req = request if isinstance(request, GestureRequest) else None

        if ag is not None:
            if req is not None and req.moving != ag.moving:
                ag.moving = req.moving
                self._rejoint(agent_id, ag)
            for st in list(ag.slots.values()):
                st.t += dt
                st.age += dt
            self._advance_all(agent_id, ag)
            ag = self._agents.get(agent_id)

        wanted = self._wanted(agent_id, req) if req is not None else {}
        for overlay in OVERLAYS:
            st = ag.slots.get(overlay) if ag is not None else None
            ch = wanted.get(overlay)
            if ch is None:
                if st is not None and st.phase != "release":
                    self._release(agent_id, ag, st)
                continue
            if st is not None and st.phase == "release":
                continue
            assert req is not None
            self._serve(agent_id, overlay, st, ch, req)

    def _serve(self, agent_id: int, overlay: str, st: _Slot | None, ch: Channel, req: GestureRequest) -> None:
        kind = SLOT_KIND[ch.slot]
        gesture = self._gesture(kind)
        local = world_to_local(ch.at, req.pose)
        ag = self._agents.get(agent_id)
        if st is None:
            if self._dropped.get((agent_id, overlay)) != self._drop_key(ch.slot, local, ch.opts):
                self._start_slot(agent_id, overlay, ch, kind, local, req.moving)
            return
        if st.channel != ch.slot or st.opts != ch.opts:
            self._release(agent_id, ag, st, restart=True)
            return
        st.release_pending = False
        if st.phase == "hold" and float(np.linalg.norm(local - st.local)) > RETARGET_EPS_M:
            self._retarget(agent_id, st, local)

    @staticmethod
    def _drop_key(channel: str, local: np.ndarray, opts: dict) -> tuple:
        return (channel, tuple(np.round(local, 1).tolist()), json.dumps(opts, sort_keys=True))

    def _rejoint(self, agent_id: int, ag: _Agent) -> None:
        """Walking flip: crossfade the blend set and re-solve live holds against the torso now shown."""
        for st in list(ag.slots.values()):
            if st.clip is None or st.phase == "release":
                continue
            joints = self._gesture(st.kind).joints(st.clip.side, ag.moving)
            if joints != st.joints:
                st.joints = joints
                self._manager.set_overlay_joints(agent_id, st.slot, joints, FADE_S)
            if st.bound.get("moving") != ag.moving:
                st.bound = {**st.bound, "moving": ag.moving}
                if st.phase == "hold":
                    self._retarget(agent_id, st, st.local)

    # -- transitions ------------------------------------------------------

    def _start_slot(self, agent_id: int, overlay: str, ch: Channel, kind: str, local: np.ndarray, moving: bool) -> None:
        ag = self._agents.get(agent_id)
        if ag is None:
            ag = _Agent()
            ag.moving = moving
            self._agents[agent_id] = ag
        gesture = self._gesture(kind)
        bound = gesture.bind(ch.slot, local, {**ch.opts, "moving": moving})
        st = _Slot(overlay, ch.slot, kind, local, ch.opts, bound)
        ag.slots[overlay] = st
        try:
            clip = gesture.start(local, bound)
        except ValueError as e:
            self._drop(agent_id, ag, st, f"gesture {kind!r} for ped {agent_id} rejected: {e}")
            return
        except Exception as e:  # noqa: BLE001 - generator faults must not take the roster publish down
            self._drop(agent_id, ag, st, f"gesture {kind!r} for ped {agent_id} failed: {e!r}")
            return
        self._install(agent_id, ag, st, clip)

    def _retarget(self, agent_id: int, st: _Slot, local: np.ndarray) -> None:
        gesture = self._gesture(st.kind)
        ag = self._agents[agent_id]
        st.local = local
        try:
            clip = gesture.retarget(st.hold, local, st.bound)
        except ValueError as e:
            self._logger.info(f"gesture {st.kind!r} for ped {agent_id}: {e}, restarting")
            self._release(agent_id, ag, st, restart=True)
            return
        except Exception as e:  # noqa: BLE001 - generator faults must not take the roster publish down
            self._drop(agent_id, ag, st, f"gesture {st.kind!r} for ped {agent_id} failed: {e!r}")
            return
        self._install(agent_id, ag, st, clip)

    def _advance_all(self, agent_id: int, ag: _Agent) -> None:
        for st in list(ag.slots.values()):
            if self._agents.get(agent_id) is not ag:
                return
            self._advance(agent_id, ag, st)

    def _advance(self, agent_id: int, ag: _Agent, st: _Slot) -> None:
        if st.phase in ("ramp", "transition") and st.clip is not None and st.t >= st.clip.hold_start / st.clip.fps:
            st.phase = "hold"
        if st.release_pending and st.phase == "hold" and st.t >= st.clip.hold_start / st.clip.fps + HOLD_MIN_S:
            self._release(agent_id, ag, st)

    def _drop(self, agent_id: int, ag: _Agent, st: _Slot, msg: str) -> None:
        self._warn_once(agent_id, f"drop:{st.kind}", msg)
        self._dropped[(agent_id, st.slot)] = self._drop_key(st.channel, st.local, st.opts)
        if st.clip is not None:
            self._manager.clear_ped_blend(agent_id, FADE_S, slot=st.slot)
        self._end_slot(agent_id, ag, st)

    def _install(self, agent_id: int, ag: _Agent, st: _Slot, clip: GestureClip) -> None:
        fresh = st.clip is None
        self._dropped.pop((agent_id, st.slot), None)
        st.clip = clip
        st.hold = clip.hold
        st.t = 0.0
        st.phase = "ramp" if fresh else "transition"
        if clip.release is not None:
            st.release_frames = clip.release
        self._report(agent_id, st, clip)
        gesture = self._gesture(st.kind)
        st.joints = gesture.joints(clip.side, ag.moving)
        frames, loop_from = self._breathed(agent_id, st, clip, gesture.breathing(clip.side))
        anim = self._manager.register_transient(f"gesture:{st.kind}:{agent_id}:{st.slot}", frames, fps=clip.fps, loop=loop_from is not None, owner=agent_id, loop_from=loop_from or 0)
        self._manager.set_ped_blend(agent_id, anim, blend_joints=st.joints, fade_in_s=FADE_S if fresh else 0.0, slot=st.slot, carry_ramps=not fresh)

    def _breathed(self, agent_id: int, st: _Slot, clip: GestureClip, amps: dict[str, float]) -> tuple[list[dict], int | None]:
        """Frames up to the park pose, with the breathing drift baked in and a one-period park loop appended when the kind breathes."""
        head = [dict(f) for f in clip.frames[: clip.hold_end + 1]]
        if not amps:
            return head, None
        fps = clip.fps
        if st.breath_origin is None:
            st.breath_origin = st.age + clip.hold_start / fps
            st.breath_ease_s = max(clip.hold_end - clip.hold_start, 1) / fps
        period = max(2, int(round(fps / BREATH_HZ)))
        omega = 2.0 * math.pi * fps / period
        phase0 = breath_phase(agent_id)
        t0 = st.age - st.breath_origin

        def drift(k: int, ease: bool) -> float:
            t = t0 + k / fps
            if t <= 0.0:
                return 0.0
            env = min(1.0, t / st.breath_ease_s) if ease else 1.0
            env = env * env * (3.0 - 2.0 * env)
            return env * math.sin(omega * t + phase0)

        out = []
        for k, frame in enumerate(head):
            angles = dict(frame["angles"])
            d = drift(k, True)
            for name, amp in amps.items():
                angles[name] += amp * d
            out.append({**frame, "angles": angles})
        park = clip.frames[clip.hold_end]
        for j in range(period):
            k = clip.hold_end + 1 + j
            angles = dict(park["angles"])
            d = drift(k, False)
            for name, amp in amps.items():
                angles[name] += amp * d
            out.append({**park, "angles": angles})
        return out, clip.hold_end + 1

    def _park(self, st: _Slot, anim_frames: Sequence[dict]) -> dict[str, float]:
        """Angles the slot shows now: the park frame, or wherever the park loop is."""
        clip = st.clip
        assert clip is not None
        k = int(round(st.t * clip.fps))
        n = len(anim_frames)
        loop_from = clip.hold_end + 1
        if k > clip.hold_end and n > loop_from:
            k = loop_from + (k - loop_from) % (n - loop_from)
        return anim_frames[min(k, n - 1)]["angles"]

    def _release(self, agent_id: int, ag: _Agent, st: _Slot, *, restart: bool = False) -> None:
        clip = st.clip
        if clip is None:
            self._end_slot(agent_id, ag, st)
            return
        min_t = clip.hold_start / clip.fps + HOLD_MIN_S
        if st.t < min_t and not restart:
            st.release_pending = True
            return
        st.phase = "release"
        st.release_pending = False
        anim_name = f"gesture:{st.kind}:{agent_id}:{st.slot}"
        current = self._manager.animations.get(anim_name)
        park_all = self._park(st, current.frames) if current is not None else clip.frames[clip.hold_end]["angles"]
        n = min(int(round(TRANSITION_S * clip.fps)), len(st.release_frames) - 1)
        park = {j: park_all[j] for j in park_all if j in st.release_frames[0]["angles"]} if n > 0 else {}
        tail = blend_into(park, st.release_frames, n)
        if len(tail) < 2:
            self._manager.clear_ped_blend(agent_id, FADE_S, slot=st.slot)
            self._end_slot(agent_id, ag, st)
            return
        anim = self._manager.register_transient(f"{anim_name}:release", tail, fps=clip.fps, loop=False, owner=agent_id)
        self._manager.set_ped_blend(agent_id, anim, blend_joints=st.joints, fade_out_s=FADE_S, on_end=functools.partial(self._released, slot=st.slot), slot=st.slot, carry_ramps=True)

    def _released(self, agent_id: int, slot: str) -> None:
        ag = self._agents.get(agent_id)
        if ag is None:
            return
        st = ag.slots.get(slot)
        if st is not None and st.phase == "release":
            self._end_slot(agent_id, ag, st)

    def _end_slot(self, agent_id: int, ag: _Agent, st: _Slot) -> None:
        ag.slots.pop(st.slot, None)
        if not ag.slots:
            self._agents.pop(agent_id, None)

    def _report(self, agent_id: int, st: _Slot, clip: GestureClip) -> None:
        rep = clip.report
        flags = [k for k in ("near_target_fallback", "relaxed") if rep.get(k)]
        if rep.get("ok", True) and not flags:
            return
        self._warn_once(agent_id, f"report:{st.kind}", f"gesture {st.kind!r} for ped {agent_id} at local {np.round(st.local, 2).tolist()}: ok={rep.get('ok')} {' '.join(flags)} clearance={rep.get('clearance_m')}")

