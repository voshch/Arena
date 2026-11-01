"""The `Camera` scripting facade and the camera-verb vocabulary.

Every verb (discrete framing and streamed motion) is a `@primitive`-registered
class that owns its name, its parameter parsing (`from_params`), and its behaviour.
`Camera.add(name, ...)` is the single generic entry point, resolving a name to a
verb or a camera asset (see `shots.py`); the asset YAML and the CLI both route
through it. Adding a verb means writing one decorated class, nothing else. Poses are
expressed in the current reference frame, so an `orbit` authored after `track` runs
in the entity's frame.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Protocol

from arena_rclpy_mixins.spin import run_main

from . import curves
from .client import CameraTake, CamNode, FaceWatches, Steered, TargetSelection
from .curves import FOV_DEFAULT, Quat, Vec3
from .mounts import Mount
from .record import default_name, record_path
from .registry import PRIMITIVES, primitive
from .shots import RigCamera, check_bundle, expand_rig, resolve

# A sampler maps eased progress in [0, 1] to (position, quat, fov).
_Sampler = Callable[[float], tuple[Vec3, Quat, float]]

_WHOLE = (0.0, 1.0)


class _Cursor:
    """The running camera state a segment starts from (resolved at play time)."""

    def __init__(self, pos: Vec3 = (5.0, 5.0, 5.0), quat: Quat | None = None, fov: float = 0.0) -> None:
        self.pos = pos
        self.quat = quat if quat is not None else curves.look_at_quat(pos, (0.0, 0.0, 0.0))
        self.fov = fov


def _known_fov(cursor: _Cursor) -> float:
    return cursor.fov if cursor.fov > 0.0 else FOV_DEFAULT


class _ParamSource(Protocol):
    def __contains__(self, key: str, /) -> bool: ...

    def __getitem__(self, key: str, /) -> object: ...


def _typed[T](params: _ParamSource, key: str, kinds: tuple[type[T], ...]) -> T:
    value = params[key]
    if not isinstance(value, kinds):
        raise ValueError(f"cam param {key!r}: unexpected {type(value).__name__} {value!r}")
    return value


def _vec3(v: Sequence[float]) -> Vec3:
    return (v[0], v[1], v[2])


def _quat(q: Sequence[float]) -> Quat:
    return (q[0], q[1], q[2], q[3])


def _pick(params: _ParamSource, keys: Sequence[str]) -> dict:
    return {k: params[k] for k in keys if k in params}


def _fov_fn(start: float, target: float | None) -> Callable[[float], float]:
    """Per-frame fov: keep (0.0) when target is None, ramp from a known start, else snap."""
    if target is None:
        return lambda _te: 0.0
    if start > 0.0:
        return lambda te: curves.lerp(start, target, te)
    return lambda _te: target


class _Action:
    verb: str = ""
    duration: float = 0.0

    @classmethod
    def from_params(cls, params: _ParamSource) -> _Action:
        raise NotImplementedError

    async def run(self, take: CameraTake, cursor: _Cursor, span: tuple[float, float] = _WHOLE) -> _Cursor:
        raise NotImplementedError

    async def skip(self, take: CameraTake, cursor: _Cursor) -> _Cursor:
        """Apply this action's end state without filming, for a slice that starts past it."""
        return await self.run(take, cursor)

    async def prepare(self, take: CameraTake) -> None:
        """Set up what must exist before a lockstep hold freezes the sim, a no-op for most verbs."""

    def atoms(self) -> list[_Action]:
        return [self]


class _Slice(_Action):
    """The part `start` to `end` s of a flat action list, `end` None for its end."""

    verb = "slice"

    def __init__(self, actions: list[_Action], start: float, end: float | None) -> None:
        self.actions, self.start = actions, start
        total = sum(action.duration for action in actions)
        self.end = total if end is None else min(end, total)
        self.duration = max(0.0, self.end - start)

    async def prepare(self, take: CameraTake) -> None:
        for action in self.actions:
            await action.prepare(take)

    def atoms(self) -> list[_Action]:
        return [atom for action in self.actions for atom in action.atoms()]

    async def skip(self, take: CameraTake, cursor: _Cursor) -> _Cursor:
        for action in self.actions:
            cursor = await action.skip(take, cursor)
        return cursor

    async def run(self, take: CameraTake, cursor: _Cursor, span: tuple[float, float] = _WHOLE) -> _Cursor:
        low, high = self.start + span[0] * self.duration, self.start + span[1] * self.duration
        now = 0.0
        for action in self.actions:
            if not take.ok():
                break
            begin, now = now, now + action.duration
            if begin >= high:
                break
            if now <= low:
                cursor = await action.skip(take, cursor)
            elif action.duration <= 0.0:
                cursor = await action.run(take, cursor)
            else:
                cursor = await action.run(take, cursor, (max(0.0, (low - begin) / action.duration), min(1.0, (high - begin) / action.duration)))
        return cursor


# discrete framing -----------------------------------------------------------


@primitive("look")
@primitive("cut")
class _Look(_Action):
    def __init__(self, eye: Vec3 = (6.0, 6.0, 3.0), target: Vec3 = (0.0, 0.0, 0.5), fov: float = 0.0) -> None:
        self.eye, self.target, self.fov = _vec3(eye), _vec3(target), fov

    @classmethod
    def from_params(cls, params: _ParamSource) -> _Look:
        return cls(**_pick(params, ("eye", "target", "fov")))

    async def run(self, take: CameraTake, cursor: _Cursor, span: tuple[float, float] = _WHOLE) -> _Cursor:
        await take.look(self.eye, self.target, self.fov)
        return await self.skip(take, cursor)

    async def skip(self, take: CameraTake, cursor: _Cursor) -> _Cursor:
        return _Cursor(self.eye, curves.look_at_quat(self.eye, self.target), self.fov or cursor.fov)


class _Ref(_Action):
    """Shared base for the reference-frame verbs (track / world / latch / reference)."""

    def __init__(self, entity: str = "", pose: tuple[Vec3, Quat] | None = None, mode: str = "full") -> None:
        self.entity, self.pose, self.mode = entity, pose, mode

    async def prepare(self, take: CameraTake) -> None:
        await take.prefill(self.entity)

    async def run(self, take: CameraTake, cursor: _Cursor, span: tuple[float, float] = _WHOLE) -> _Cursor:
        await take.set_reference(self.entity, self.pose, self.mode)
        return cursor


@primitive("track")
class _Track(_Ref):
    def __init__(self, entity: str, mode: str = "full") -> None:
        super().__init__(entity=entity, mode=mode)

    @classmethod
    def from_params(cls, params: _ParamSource) -> _Track:
        return cls(entity=_typed(params, "entity", (str,)), **_pick(params, ("mode",)))


@primitive("world")
class _World(_Ref):
    def __init__(self) -> None:
        super().__init__(entity="", pose=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)), mode="full")

    @classmethod
    def from_params(cls, params: _ParamSource) -> _World:
        return cls()


@primitive("latch")
class _Latch(_Ref):
    def __init__(self) -> None:
        super().__init__(entity="", pose=None, mode="full")

    @classmethod
    def from_params(cls, params: _ParamSource) -> _Latch:
        return cls()


@primitive("reference")
class _Reference(_Ref):
    def __init__(self, pose: tuple[Vec3, Quat] = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)), mode: str = "full") -> None:
        pos, quat = pose
        super().__init__(entity="", pose=(_vec3(pos), _quat(quat)), mode=mode)

    @classmethod
    def from_params(cls, params: _ParamSource) -> _Reference:
        return cls(**_pick(params, ("pose", "mode")))


@primitive("projection")
class _Projection(_Action):
    def __init__(self, projection: str = "perspective") -> None:
        self.projection = projection

    @classmethod
    def from_params(cls, params: _ParamSource | str) -> _Projection:
        return cls(params) if isinstance(params, str) else cls(**_pick(params, ("projection",)))

    async def run(self, take: CameraTake, cursor: _Cursor, span: tuple[float, float] = _WHOLE) -> _Cursor:
        await take.set_projection(self.projection)
        return cursor


# streamed motion ------------------------------------------------------------


class _Segment(_Action):
    """A timed move streamed as cmd_view at the node's frame rate."""

    def __init__(self, duration: float, ease: str, world_orientation: bool) -> None:
        self.duration = max(0.0, float(duration))
        self.ease = ease
        self.world_orientation = world_orientation

    def plan(self, cursor: _Cursor) -> tuple[_Sampler, _Cursor]:
        """Build the per-frame sampler and the end cursor from the start cursor."""
        raise NotImplementedError

    async def run(self, take: CameraTake, cursor: _Cursor, span: tuple[float, float] = _WHOLE) -> _Cursor:
        sampler, end = self.plan(cursor)

        def frame_at(t: float) -> tuple[Vec3, Quat, float]:
            return sampler(curves.ease(self.ease, t))

        await take.drive(self.duration, self.world_orientation, frame_at, span)
        return end if span[1] >= 1.0 else _Cursor(*frame_at(span[1]))

    async def skip(self, take: CameraTake, cursor: _Cursor) -> _Cursor:
        return self.plan(cursor)[1]


@primitive("hold")
class _Hold(_Segment):
    def __init__(self, duration: float = 2.0) -> None:
        super().__init__(duration, "linear", False)

    @classmethod
    def from_params(cls, params: _ParamSource) -> _Hold:
        return cls(**_pick(params, ("duration",)))

    def plan(self, cursor: _Cursor) -> tuple[_Sampler, _Cursor]:
        pose = (cursor.pos, cursor.quat, cursor.fov)
        return (lambda _te: pose), cursor


@primitive("move_to")
class _MoveTo(_Segment):
    def __init__(
        self,
        eye: Vec3 = (6.0, 6.0, 3.0),
        target: Vec3 = (0.0, 0.0, 0.5),
        fov: float | None = None,
        duration: float = 4.0,
        ease: str = "inout",
        world_orientation: bool = False,
    ) -> None:
        super().__init__(duration, ease, world_orientation)
        self.eye, self.target, self.fov = _vec3(eye), _vec3(target), fov

    @classmethod
    def from_params(cls, params: _ParamSource) -> _MoveTo:
        return cls(**_pick(params, ("eye", "target", "fov", "duration", "ease", "world_orientation")))

    def plan(self, cursor: _Cursor) -> tuple[_Sampler, _Cursor]:
        p0, q0 = cursor.pos, cursor.quat
        p1 = self.eye
        q1 = curves.look_at_quat(self.eye, self.target)
        fov = _fov_fn(cursor.fov, self.fov)

        def sample(te: float) -> tuple[Vec3, Quat, float]:
            return curves.vlerp(p0, p1, te), curves.slerp(q0, q1, te), fov(te)

        end_fov = self.fov if self.fov is not None else cursor.fov
        return sample, _Cursor(p1, q1, end_fov)


@primitive("orbit")
class _Orbit(_Segment):
    """Sweep azimuth on a sphere of `radius` around `center`, looking at it. `elevation` is
    the angle above horizontal (0 = level ring, +pi/2 = straight down). `radius`, `elevation`
    and `start_angle` default (None) to the current camera's pose around `center`."""

    def __init__(
        self,
        radius: float | None = None,
        elevation: float | None = None,
        center: Vec3 = (0.0, 0.0, 0.0),
        sweep: float = 2.0 * math.pi,
        start_angle: float | None = None,
        duration: float = 8.0,
        ease: str = "inout",
        look_height: float = 0.0,
        fov: float | None = None,
        world_orientation: bool = False,
    ) -> None:
        super().__init__(duration, ease, world_orientation)
        self.radius, self.elevation, self.center = radius, elevation, _vec3(center)
        self.sweep, self.start_angle = sweep, start_angle
        self.look_height, self.fov = look_height, fov

    @classmethod
    def from_params(cls, params: _ParamSource) -> _Orbit:
        kwargs = _pick(params, ("radius", "elevation", "center", "sweep", "start_angle", "duration", "ease", "look_height", "fov", "world_orientation"))
        if "sweep_deg" in params:
            kwargs["sweep"] = math.radians(_typed(params, "sweep_deg", (int, float)))
        if "start_deg" in params:
            kwargs["start_angle"] = math.radians(_typed(params, "start_deg", (int, float)))
        if "elevation_deg" in params:
            kwargs["elevation"] = math.radians(_typed(params, "elevation_deg", (int, float)))
        return cls(**kwargs)

    def plan(self, cursor: _Cursor) -> tuple[_Sampler, _Cursor]:
        cx, cy, cz = self.center
        look = (cx, cy, cz + self.look_height)
        offset = curves.vsub(cursor.pos, self.center)
        dist = curves.vlen(offset)
        radius = self.radius if self.radius is not None else dist
        if self.elevation is not None:
            elevation = self.elevation
        else:
            elevation = math.asin(max(-1.0, min(1.0, offset[2] / dist))) if dist > 0.0 else 0.0
        ring = radius * math.cos(elevation)  # horizontal radius at this elevation
        rise = radius * math.sin(elevation)
        a0 = self.start_angle if self.start_angle is not None else math.atan2(offset[1], offset[0])
        fov = _fov_fn(cursor.fov, self.fov)

        def sample(te: float) -> tuple[Vec3, Quat, float]:
            a = a0 + self.sweep * te
            eye = (cx + ring * math.cos(a), cy + ring * math.sin(a), cz + rise)
            return eye, curves.look_at_quat(eye, look), fov(te)

        pos_end, quat_end, _ = sample(1.0)
        end_fov = self.fov if self.fov is not None else cursor.fov
        return sample, _Cursor(pos_end, quat_end, end_fov)


@primitive("dolly")
class _Dolly(_Segment):
    def __init__(self, distance: float = 2.0, duration: float = 2.0, ease: str = "inout") -> None:
        super().__init__(duration, ease, False)
        self.distance = distance

    @classmethod
    def from_params(cls, params: _ParamSource) -> _Dolly:
        return cls(**_pick(params, ("distance", "duration", "ease")))

    def plan(self, cursor: _Cursor) -> tuple[_Sampler, _Cursor]:
        p0, q, f = cursor.pos, cursor.quat, cursor.fov
        p1 = curves.vadd(p0, curves.vscale(curves.quat_forward(q), self.distance))

        def sample(te: float) -> tuple[Vec3, Quat, float]:
            return curves.vlerp(p0, p1, te), q, f

        return sample, _Cursor(p1, q, f)


@primitive("dolly_zoom")
class _DollyZoom(_Segment):
    """Vertigo: change fov while dollying so `target` keeps its apparent size (from_fov None = current fov)."""

    def __init__(
        self,
        target: Vec3 = (0.0, 0.0, 0.5),
        from_fov: float | None = None,
        to_fov: float = 1.2,
        duration: float = 3.0,
        ease: str = "inout",
    ) -> None:
        super().__init__(duration, ease, False)
        self.target, self.from_fov, self.to_fov = _vec3(target), from_fov, to_fov

    @classmethod
    def from_params(cls, params: _ParamSource) -> _DollyZoom:
        return cls(**_pick(params, ("target", "from_fov", "to_fov", "duration", "ease")))

    def plan(self, cursor: _Cursor) -> tuple[_Sampler, _Cursor]:
        from_fov = self.from_fov if self.from_fov is not None else _known_fov(cursor)
        d0 = curves.vlen(curves.vsub(cursor.pos, self.target))
        out = curves.vnorm(curves.vsub(cursor.pos, self.target))

        def sample(te: float) -> tuple[Vec3, Quat, float]:
            fov = curves.lerp(from_fov, self.to_fov, te)
            # keep subject size: d * tan(fov/2) constant
            d = d0 * math.tan(from_fov / 2.0) / math.tan(fov / 2.0)
            eye = curves.vadd(self.target, curves.vscale(out, d))
            return eye, curves.look_at_quat(eye, self.target), fov

        pos_end, quat_end, fov_end = sample(1.0)
        return sample, _Cursor(pos_end, quat_end, fov_end)


@primitive("flyby")
class _Flyby(_Segment):
    def __init__(
        self,
        eyes: Sequence[Vec3] = ((6.0, -6.0, 3.0), (6.0, 6.0, 3.0), (-6.0, 6.0, 3.0)),
        target: Vec3 = (0.0, 0.0, 0.5),
        fov: float | None = None,
        duration: float = 8.0,
        ease: str = "inout",
        world_orientation: bool = False,
    ) -> None:
        super().__init__(duration, ease, world_orientation)
        self.eyes = [_vec3(e) for e in eyes]
        self.target, self.fov = _vec3(target), fov

    @classmethod
    def from_params(cls, params: _ParamSource) -> _Flyby:
        return cls(**_pick(params, ("eyes", "target", "fov", "duration", "ease", "world_orientation")))

    def plan(self, cursor: _Cursor) -> tuple[_Sampler, _Cursor]:
        pts = [cursor.pos, *self.eyes]
        fov = _fov_fn(cursor.fov, self.fov)

        def sample(te: float) -> tuple[Vec3, Quat, float]:
            eye = curves.catmull_rom(pts, te)
            return eye, curves.look_at_quat(eye, self.target), fov(te)

        pos_end, quat_end, _ = sample(1.0)
        end_fov = self.fov if self.fov is not None else cursor.fov
        return sample, _Cursor(pos_end, quat_end, end_fov)


@primitive("pan")
class _Pan(_Segment):
    """Rotate the view in place horizontally: eye fixed, yaw sweeps, pitch held."""

    def __init__(self, sweep: float = math.pi / 2.0, duration: float = 4.0, ease: str = "inout") -> None:
        super().__init__(duration, ease, False)
        self.sweep = sweep

    @classmethod
    def from_params(cls, params: _ParamSource) -> _Pan:
        kwargs = _pick(params, ("sweep", "duration", "ease"))
        if "sweep_deg" in params:
            kwargs["sweep"] = math.radians(_typed(params, "sweep_deg", (int, float)))
        return cls(**kwargs)

    def plan(self, cursor: _Cursor) -> tuple[_Sampler, _Cursor]:
        eye, f = cursor.pos, cursor.fov
        fwd = curves.quat_forward(cursor.quat)
        yaw0 = math.atan2(fwd[1], fwd[0])
        pitch0 = -math.asin(max(-1.0, min(1.0, fwd[2])))

        def sample(te: float) -> tuple[Vec3, Quat, float]:
            yaw = yaw0 + self.sweep * te
            d = (math.cos(pitch0) * math.cos(yaw), math.cos(pitch0) * math.sin(yaw), -math.sin(pitch0))
            return eye, curves.look_at_quat(eye, curves.vadd(eye, d)), f

        pos_end, quat_end, _ = sample(1.0)
        return sample, _Cursor(pos_end, quat_end, f)


@primitive("tilt")
class _Tilt(_Segment):
    """Rotate the view in place vertically: eye fixed, pitch sweeps, yaw held."""

    def __init__(self, sweep: float = math.pi / 6.0, duration: float = 4.0, ease: str = "inout") -> None:
        super().__init__(duration, ease, False)
        self.sweep = sweep

    @classmethod
    def from_params(cls, params: _ParamSource) -> _Tilt:
        kwargs = _pick(params, ("sweep", "duration", "ease"))
        if "sweep_deg" in params:
            kwargs["sweep"] = math.radians(_typed(params, "sweep_deg", (int, float)))
        return cls(**kwargs)

    def plan(self, cursor: _Cursor) -> tuple[_Sampler, _Cursor]:
        eye, f = cursor.pos, cursor.fov
        fwd = curves.quat_forward(cursor.quat)
        yaw0 = math.atan2(fwd[1], fwd[0])
        pitch0 = -math.asin(max(-1.0, min(1.0, fwd[2])))

        def sample(te: float) -> tuple[Vec3, Quat, float]:
            pitch = max(-1.5533, min(1.5533, pitch0 + self.sweep * te))
            d = (math.cos(pitch) * math.cos(yaw0), math.cos(pitch) * math.sin(yaw0), -math.sin(pitch))
            return eye, curves.look_at_quat(eye, curves.vadd(eye, d)), f

        pos_end, quat_end, _ = sample(1.0)
        return sample, _Cursor(pos_end, quat_end, f)


@primitive("zoom")
class _Zoom(_Segment):
    """Change fov only, eye and aim fixed (from_fov None = current fov)."""

    def __init__(self, from_fov: float | None = None, to_fov: float = 0.6, duration: float = 2.0, ease: str = "inout") -> None:
        super().__init__(duration, ease, False)
        self.from_fov, self.to_fov = from_fov, to_fov

    @classmethod
    def from_params(cls, params: _ParamSource) -> _Zoom:
        return cls(**_pick(params, ("from_fov", "to_fov", "duration", "ease")))

    def plan(self, cursor: _Cursor) -> tuple[_Sampler, _Cursor]:
        pos, quat = cursor.pos, cursor.quat
        from_fov = self.from_fov if self.from_fov is not None else _known_fov(cursor)

        def sample(te: float) -> tuple[Vec3, Quat, float]:
            return pos, quat, curves.lerp(from_fov, self.to_fov, te)

        return sample, _Cursor(pos, quat, self.to_fov)


@primitive("pov")
class _Pov(_Action):
    """Look through a pedestrian's face frame for `duration` s, then latch where the face last was."""

    def __init__(self, face: str | float, duration: float = 10.0, fov: float | None = None, clip_near: float | None = None) -> None:
        self.face = str(int(face)) if isinstance(face, float) and face.is_integer() else str(face)
        self.duration = max(0.0, float(duration))
        self.fov = fov
        self.clip_near = clip_near
        self._prepared = False
        self._watches: FaceWatches | None = None

    @classmethod
    def from_params(cls, params: _ParamSource) -> _Pov:
        return cls(face=_typed(params, "face", (str, int, float)), **_pick(params, ("duration", "fov", "clip_near")))

    async def prepare(self, take: CameraTake) -> None:
        self._watches = await take.watch_faces(self.face)
        self._prepared = True

    async def run(self, take: CameraTake, cursor: _Cursor, span: tuple[float, float] = _WHOLE) -> _Cursor:
        watches = self._watches if self._prepared else await take.watch_faces(self.face)
        self._watches, self._prepared = None, False
        if watches is None:
            return cursor
        try:
            fov = self.fov if self.fov is not None else watches.fov
            take.get_logger().info(f"cam: pov {sorted(set(watches.frames.values()))}, fov {math.degrees(fov):.0f} deg")
            pose = ((0.0, 0.0, 0.0), curves.look_at_quat((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), fov)
            take.clip_near = self.clip_near if self.clip_near is not None else watches.clip_near
            await take.set_reference("", None, "full", frames=watches.frames)
            await take.drive(self.duration, False, lambda _t: pose, span)
            take.clip_near = -1.0
            take.stream(pose[0], pose[1], False, fov)
            await take.set_reference("", None, "full")
        finally:
            take.clip_near = 0.0
            watches.close()
        return _Cursor(*pose)

    async def skip(self, take: CameraTake, cursor: _Cursor) -> _Cursor:
        if self._watches is not None:
            self._watches.close()
        self._watches, self._prepared = None, False
        return cursor


class _Steer(_Action):
    verb = "steer"

    def __init__(self, next_frame: Callable[[float], Steered | None]) -> None:
        self.next_frame = next_frame

    async def run(self, take: CameraTake, cursor: _Cursor, span: tuple[float, float] = _WHOLE) -> _Cursor:
        await take.steer(self.next_frame)
        return cursor


class Camera:
    """Author a camera shot as a chain of `add(verb, ...)` calls or load a camera asset, then `play()` or `record()` it."""

    def __init__(self, targets: TargetSelection) -> None:
        self._targets = targets
        self._actions: list[_Action] = []
        self._rig: list[RigCamera] = []
        self._projection: str | None = None
        self._live = ""

    def add(self, name: str, params: object = None, /, **kwargs: object) -> Camera:
        """Queue a verb or camera asset by name, an asset resolves through its global tiers. Params come as a positional dict/scalar (YAML) or as kwargs (Python).

        An asset expands eagerly to its atomic actions, so the queue stays flat and
        a verb and an asset are indistinguishable at this call site.
        """
        spec = params if params is not None else kwargs
        if self._rig:
            raise ValueError(f"cannot add {name!r} to a rig, a rig is a root only")
        if name in PRIMITIVES:
            self._actions.extend(resolve(name, spec))
            return self
        return self.load(name, _global_bundle(name), spec)

    def load(self, root: str, bundle: dict[str, dict], params: object = None) -> Camera:
        """Queue the camera asset `root` of a bundle, a rig replaces the queue."""
        check_bundle(bundle)
        if "cameras" not in bundle[root]:
            self._actions.extend(resolve(root, params or {}, (), bundle))
            return self
        if self._actions:
            raise ValueError(f"rig {root!r} is a root only, it cannot follow other steps")
        rig = expand_rig(root, bundle[root], params, bundle)
        self._rig, self._projection, self._live = rig.cameras, rig.projection, rig.live
        return self

    def steer(self, next_frame: Callable[[float], Steered | None]) -> Camera:
        """Queue an open-ended segment whose poses come from live input (the drive panel), record only."""
        self._actions.append(_Steer(next_frame))
        return self

    @property
    def projection(self) -> str | None:
        return self._projection

    def cameras(self, *, recording: bool) -> list[RigCamera]:
        """The cameras a take runs: every rig camera when recording, else the live one."""
        if not self._rig:
            return [RigCamera(actions=self._actions)]
        return self._rig if recording else [camera for camera in self._rig if camera.name == self._live]

    def _run(self, record: tuple[str, float] | None = None, *, force: bool = False, lockstep: bool = False, fps: float = 30.0, mounts: Sequence[Mount] = (), env: int = 0) -> bool:
        nodes: list[CamNode] = []

        def make() -> CamNode:
            nodes.append(CamNode(timeline=self, targets=self._targets, record=record, force=force, lockstep=lockstep, fps=fps, mounts=mounts, env=env))
            return nodes[-1]

        run_main(make)
        return bool(nodes) and nodes[0].ran

    def play(self, *, lockstep: bool = False, fps: float = 30.0, env: int = 0) -> bool:
        """Connect to the live sim, run the shot, disconnect. Blocks until done, False when no viewport ran it.

        `lockstep` gates playback frame by frame: with a run active the cam rides it as a
        hard channel gated at 1/fps, with no run it takes its own hold and steps the sim by
        1/fps per frame. Without it playback runs on the wall clock, with no sync to the sim.
        """
        return self._run(lockstep=bool(lockstep), fps=float(fps), env=env)

    def record(self, out: str | None = None, fps: float = 30.0, force: bool = False, lockstep: bool = False, mounts: Sequence[Mount] = (), env: int = 0) -> bool:
        """Render the shot to a video, one capture per frame, encoded by ffmpeg.

        A bare `out` name lands under `$ARENA_DATA_DIR/recordings/`, an absolute
        or slash-bearing path is used verbatim, and the suffix picks the container
        (none means .mp4). `None` names the take `take_<YYYYmmdd-HHMMSS>`. With
        several cameras selected each gets its own file, tagged `-sim` / `-viz<env>`,
        and each rig camera its own file tagged by its name.
        An existing file is refused with an error log unless `force`. Raises
        `FileNotFoundError` without ffmpeg. Blocks until done.

        `lockstep` makes the recording frame-exact: with a run active
        the cam rides it as a hard channel gated at 1/fps, with no run it takes its
        own hold and steps the sim by 1/fps between frames. False when no viewport ran the shot
        or a recording failed: a hold, step, capture (after its retries) or encode that failed
        stops the take there.

        Each of `mounts` films from its TF frame through the sim camera at every recorded
        frame's sim time, into its own file tagged by the frame. Mounts need `lockstep`
        and the sim camera, and a timeline that films next to them cannot `latch`. A rig
        needs `lockstep` too.
        """
        if mounts and any(atom.verb == "latch" for action in self._actions for atom in action.atoms()):
            raise ValueError("latch cannot run next to mounts, a shared surface cannot hold a latched reference")
        path = record_path(out or default_name("take"))
        return self._run((str(path), float(fps)), force=bool(force), lockstep=bool(lockstep), mounts=mounts, env=env)

    async def prepare(self, take: CameraTake) -> None:
        """Let each queued action of `take`'s camera set up against a connected node before the timeline runs."""
        for action in take.camera.actions:
            await action.prepare(take)

    async def run(self, take: CameraTake) -> None:
        """Execute the actions of `take`'s camera against a connected node (called by CamNode)."""
        cursor = _Cursor()
        seeded = take.camera_pose()
        if seeded is not None:
            cursor = _Cursor(seeded[0], seeded[1], 0.0)
        for action in take.camera.actions:
            if not take.ok():
                break
            take.get_logger().info(f"cam: {action.verb}")
            cursor = await action.run(take, cursor)


def _global_bundle(name: str) -> dict[str, dict]:
    """A bundle of the camera asset `name` from the global tiers, a ValueError naming them when no tier holds it."""
    from arena_simulation_setup.tree import Identifier
    from arena_simulation_setup.tree.assets.Camera import collect, find_spec

    async def gather() -> dict[str, dict] | None:
        spec = await find_spec(name)
        return None if spec is None else await collect(name, spec)

    bundle = Identifier._run_sync(gather())
    if bundle is None:
        raise ValueError(f"unknown camera name: {name!r} (verbs: {sorted(PRIMITIVES)}, shots: {global_cameras()})")
    return bundle


def global_spec(name: str) -> dict | None:
    """The spec of the camera asset `name` from the global tiers, None when no tier holds it."""
    from arena_simulation_setup.tree import Identifier
    from arena_simulation_setup.tree.assets.Camera import find_spec

    return Identifier._run_sync(find_spec(name))


def global_cameras() -> list[str]:
    """The camera assets of the global tiers, the names `arena cam` plays with no world loaded."""
    from arena_simulation_setup.tree import Identifier
    from arena_simulation_setup.tree.assets.Camera import candidates

    return [identifier.name for identifier in Identifier._run_sync(candidates())]
