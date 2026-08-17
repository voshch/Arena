"""The `Camera` scripting facade and the camera-verb vocabulary.

Every verb (discrete framing and streamed motion) is a `@primitive`-registered
class that owns its name, its parameter parsing (`from_params`), and its behaviour.
`Camera.add(name, ...)` is the single generic entry point, resolving a name to a
verb or a named shot (see `shots.py`); the shot YAML and the CLI both route through
it. Adding a verb means writing one decorated class, nothing else. Poses are
expressed in the current reference frame, so an `orbit` authored after `track` runs
in the entity's frame. Timing is wall-clock (LIVE) only.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from . import curves
from .client import CamNode, TargetSelection
from .curves import Quat, Vec3
from .record import record_dir
from .registry import primitive
from .shots import resolve

# A sampler maps eased progress in [0, 1] to (position, quat, fov).
_Sampler = Callable[[float], tuple[Vec3, Quat, float]]


FOV_DEFAULT = 1.047


class _Cursor:
    """The running camera state a segment starts from (resolved at play time)."""

    def __init__(self, pos: Vec3 = (5.0, 5.0, 5.0), quat: Quat | None = None, fov: float = 0.0) -> None:
        self.pos = pos
        self.quat = quat if quat is not None else curves.look_at_quat(pos, (0.0, 0.0, 0.0))
        self.fov = fov


def _known_fov(cursor: _Cursor) -> float:
    return cursor.fov if cursor.fov > 0.0 else FOV_DEFAULT


def _pick(params: dict, keys: Sequence[str]) -> dict:
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

    @classmethod
    def from_params(cls, params: object) -> _Action:
        raise NotImplementedError

    async def run(self, node: CamNode, cursor: _Cursor) -> _Cursor:
        raise NotImplementedError


# discrete framing -----------------------------------------------------------


@primitive("look")
@primitive("cut")
class _Look(_Action):
    def __init__(self, eye: Vec3 = (6.0, 6.0, 3.0), target: Vec3 = (0.0, 0.0, 0.5), fov: float = 0.0) -> None:
        self.eye, self.target, self.fov = tuple(eye), tuple(target), fov

    @classmethod
    def from_params(cls, params: dict) -> _Look:
        return cls(**_pick(params, ("eye", "target", "fov")))

    async def run(self, node: CamNode, cursor: _Cursor) -> _Cursor:
        await node.look(self.eye, self.target, self.fov)
        return _Cursor(self.eye, curves.look_at_quat(self.eye, self.target), self.fov or cursor.fov)


class _Ref(_Action):
    """Shared base for the reference-frame verbs (track / world / latch / reference)."""

    def __init__(self, entity: str = "", pose: tuple[Vec3, Quat] | None = None, mode: str = "full") -> None:
        self.entity, self.pose, self.mode = entity, pose, mode

    async def run(self, node: CamNode, cursor: _Cursor) -> _Cursor:
        await node.set_reference(self.entity, self.pose, self.mode)
        return cursor


@primitive("track")
class _Track(_Ref):
    def __init__(self, entity: str, mode: str = "full") -> None:
        super().__init__(entity=entity, mode=mode)

    @classmethod
    def from_params(cls, params: dict) -> _Track:
        return cls(entity=params["entity"], **_pick(params, ("mode",)))


@primitive("world")
class _World(_Ref):
    def __init__(self) -> None:
        super().__init__(entity="", pose=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)), mode="full")

    @classmethod
    def from_params(cls, params: dict) -> _World:
        return cls()


@primitive("latch")
class _Latch(_Ref):
    def __init__(self) -> None:
        super().__init__(entity="", pose=None, mode="full")

    @classmethod
    def from_params(cls, params: dict) -> _Latch:
        return cls()


@primitive("reference")
class _Reference(_Ref):
    def __init__(self, pose: tuple[Vec3, Quat] = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)), mode: str = "full") -> None:
        pos, quat = pose
        super().__init__(entity="", pose=(tuple(pos), tuple(quat)), mode=mode)

    @classmethod
    def from_params(cls, params: dict) -> _Reference:
        return cls(**_pick(params, ("pose", "mode")))


@primitive("projection")
class _Projection(_Action):
    def __init__(self, projection: str = "perspective") -> None:
        self.projection = projection

    @classmethod
    def from_params(cls, params: object) -> _Projection:
        return cls(params) if isinstance(params, str) else cls(**_pick(params, ("projection",)))

    async def run(self, node: CamNode, cursor: _Cursor) -> _Cursor:
        await node.set_projection(self.projection)
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

    async def run(self, node: CamNode, cursor: _Cursor) -> _Cursor:
        sampler, end = self.plan(cursor)

        def frame_at(t: float) -> tuple[Vec3, Quat, float]:
            return sampler(curves.ease(self.ease, t))

        await node.drive(self.duration, self.world_orientation, frame_at)
        return end


@primitive("hold")
class _Hold(_Segment):
    def __init__(self, duration: float = 2.0) -> None:
        super().__init__(duration, "linear", False)

    @classmethod
    def from_params(cls, params: dict) -> _Hold:
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
        self.eye, self.target, self.fov = tuple(eye), tuple(target), fov

    @classmethod
    def from_params(cls, params: dict) -> _MoveTo:
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
        self.radius, self.elevation, self.center = radius, elevation, tuple(center)
        self.sweep, self.start_angle = sweep, start_angle
        self.look_height, self.fov = look_height, fov

    @classmethod
    def from_params(cls, params: dict) -> _Orbit:
        kwargs = _pick(params, ("radius", "elevation", "center", "sweep", "start_angle", "duration", "ease", "look_height", "fov", "world_orientation"))
        if "sweep_deg" in params:
            kwargs["sweep"] = math.radians(params["sweep_deg"])
        if "start_deg" in params:
            kwargs["start_angle"] = math.radians(params["start_deg"])
        if "elevation_deg" in params:
            kwargs["elevation"] = math.radians(params["elevation_deg"])
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
    def from_params(cls, params: dict) -> _Dolly:
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
        self.target, self.from_fov, self.to_fov = tuple(target), from_fov, to_fov

    @classmethod
    def from_params(cls, params: dict) -> _DollyZoom:
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
        self.eyes = [tuple(e) for e in eyes]
        self.target, self.fov = tuple(target), fov

    @classmethod
    def from_params(cls, params: dict) -> _Flyby:
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
    def from_params(cls, params: dict) -> _Pan:
        kwargs = _pick(params, ("sweep", "duration", "ease"))
        if "sweep_deg" in params:
            kwargs["sweep"] = math.radians(params["sweep_deg"])
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
    def from_params(cls, params: dict) -> _Tilt:
        kwargs = _pick(params, ("sweep", "duration", "ease"))
        if "sweep_deg" in params:
            kwargs["sweep"] = math.radians(params["sweep_deg"])
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
    def from_params(cls, params: dict) -> _Zoom:
        return cls(**_pick(params, ("from_fov", "to_fov", "duration", "ease")))

    def plan(self, cursor: _Cursor) -> tuple[_Sampler, _Cursor]:
        pos, quat = cursor.pos, cursor.quat
        from_fov = self.from_fov if self.from_fov is not None else _known_fov(cursor)

        def sample(te: float) -> tuple[Vec3, Quat, float]:
            return pos, quat, curves.lerp(from_fov, self.to_fov, te)

        return sample, _Cursor(pos, quat, self.to_fov)


class Camera:
    """Author a camera shot as a chain of `add(verb, ...)` calls, then `play()` it."""

    def __init__(self, targets: TargetSelection) -> None:
        self._targets = targets
        self._actions: list[_Action] = []

    def add(self, name: str, params: object = None, /, **kwargs: object) -> Camera:
        """Queue a verb or shot by name. Params come as a positional dict/scalar (YAML) or as kwargs (Python).

        A shot expands eagerly to its atomic actions, so the queue stays flat and
        a verb and a shot are indistinguishable at this call site.
        """
        spec = params if params is not None else kwargs
        self._actions.extend(resolve(name, spec))
        return self

    def play(self) -> None:
        """Connect to the live sim, run the shot, disconnect. Blocks until done."""
        CamNode.run_main(timeline=self, targets=self._targets)

    def record(self, out_dir: str, fps: float = 30.0, force: bool = False, lockstep: bool = False) -> None:
        """Render the shot to a numbered PPM sequence, one capture per frame.

        A bare `out_dir` name lands under `$ARENA_DATA_DIR/recordings/`; an absolute
        or slash-bearing path is used verbatim. Raises `FileExistsError` if the
        directory is not empty unless `force`. Blocks until done.

        `lockstep` makes the recording frame-exact (Gazebo only): with a run active
        the cam rides it as a hard channel gated at 1/fps, with no run it takes its
        own hold and steps the sim by 1/fps between frames.
        """
        path = record_dir(out_dir, force)
        CamNode.run_main(timeline=self, targets=self._targets, record=(str(path), float(fps)), lockstep=bool(lockstep))

    async def run(self, node: CamNode) -> None:
        """Execute the queued actions against a connected node (called by CamNode.setup)."""
        cursor = _Cursor()
        seeded = node.camera_pose()
        if seeded is not None:
            cursor = _Cursor(seeded[0], seeded[1], 0.0)
        for action in self._actions:
            if not node.ok():
                break
            node.get_logger().info(f"cam: {action.verb}")
            cursor = await action.run(node, cursor)
