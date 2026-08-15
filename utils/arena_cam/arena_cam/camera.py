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


class _Cursor:
    """The running camera state a segment starts from (resolved at play time)."""

    def __init__(self, pos: Vec3 = (5.0, 5.0, 5.0), quat: Quat | None = None, fov: float = 0.0) -> None:
        self.pos = pos
        self.quat = quat if quat is not None else curves.look_at_quat(pos, (0.0, 0.0, 0.0))
        self.fov = fov


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
    def __init__(self, eye: Vec3, target: Vec3, fov: float = 0.0) -> None:
        self.eye, self.target, self.fov = tuple(eye), tuple(target), fov

    @classmethod
    def from_params(cls, params: dict) -> _Look:
        return cls(tuple(params["eye"]), tuple(params["target"]), params.get("fov", 0.0))

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
    @classmethod
    def from_params(cls, params: dict) -> _Track:
        return cls(entity=params["entity"], mode=params.get("mode", "full"))


@primitive("world")
class _World(_Ref):
    @classmethod
    def from_params(cls, params: dict) -> _World:
        return cls(entity="", pose=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)), mode="full")


@primitive("latch")
class _Latch(_Ref):
    @classmethod
    def from_params(cls, params: dict) -> _Latch:
        return cls(entity="", pose=None, mode="full")


@primitive("reference")
class _Reference(_Ref):
    @classmethod
    def from_params(cls, params: dict) -> _Reference:
        pos, quat = params["pose"]
        return cls(entity="", pose=(tuple(pos), tuple(quat)), mode=params.get("mode", "full"))


@primitive("projection")
class _Projection(_Action):
    def __init__(self, projection: str) -> None:
        self.projection = projection

    @classmethod
    def from_params(cls, params: object) -> _Projection:
        return cls(params if isinstance(params, str) else params["projection"])

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
    def __init__(self, duration: float) -> None:
        super().__init__(duration, "linear", False)

    @classmethod
    def from_params(cls, params: dict) -> _Hold:
        return cls(params["duration"])

    def plan(self, cursor: _Cursor) -> tuple[_Sampler, _Cursor]:
        pose = (cursor.pos, cursor.quat, cursor.fov)
        return (lambda _te: pose), cursor


@primitive("move_to")
class _MoveTo(_Segment):
    def __init__(self, eye: Vec3, target: Vec3, fov: float | None, duration: float, ease: str, world_orientation: bool) -> None:
        super().__init__(duration, ease, world_orientation)
        self.eye, self.target, self.fov = tuple(eye), tuple(target), fov

    @classmethod
    def from_params(cls, params: dict) -> _MoveTo:
        return cls(
            tuple(params["eye"]),
            tuple(params["target"]),
            params.get("fov"),
            params["duration"],
            params.get("ease", "inout"),
            params.get("world_orientation", False),
        )

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
    """Sweep azimuth on a sphere of `radius` around `center`, looking at it. The eye
    keeps a constant distance to the subject; `elevation` is the angle above the
    horizontal (0 = level ring, +pi/2 = straight down)."""

    def __init__(
        self,
        radius: float,
        elevation: float,
        center: Vec3,
        sweep: float,
        start_angle: float | None,
        duration: float,
        ease: str,
        look_height: float,
        fov: float | None,
        world_orientation: bool,
    ) -> None:
        super().__init__(duration, ease, world_orientation)
        self.radius, self.elevation, self.center = radius, elevation, tuple(center)
        self.sweep, self.start_angle = sweep, start_angle
        self.look_height, self.fov = look_height, fov

    @classmethod
    def from_params(cls, params: dict) -> _Orbit:
        sweep = math.radians(params["sweep_deg"]) if "sweep_deg" in params else params.get("sweep", 2.0 * math.pi)
        start = math.radians(params["start_deg"]) if "start_deg" in params else params.get("start_angle")
        elev = math.radians(params["elevation_deg"]) if "elevation_deg" in params else params.get("elevation", 0.0)
        return cls(
            radius=params["radius"],
            elevation=elev,
            center=tuple(params.get("center", (0.0, 0.0, 0.0))),
            sweep=sweep,
            start_angle=start,
            duration=params.get("duration", 8.0),
            ease=params.get("ease", "inout"),
            look_height=params.get("look_height", 0.0),
            fov=params.get("fov"),
            world_orientation=params.get("world_orientation", False),
        )

    def plan(self, cursor: _Cursor) -> tuple[_Sampler, _Cursor]:
        cx, cy, cz = self.center
        look = (cx, cy, cz + self.look_height)
        ring = self.radius * math.cos(self.elevation)  # horizontal radius at this elevation
        rise = self.radius * math.sin(self.elevation)
        a0 = self.start_angle if self.start_angle is not None else math.atan2(cursor.pos[1] - cy, cursor.pos[0] - cx)
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
    def __init__(self, distance: float, duration: float, ease: str) -> None:
        super().__init__(duration, ease, False)
        self.distance = distance

    @classmethod
    def from_params(cls, params: dict) -> _Dolly:
        return cls(params["distance"], params.get("duration", 2.0), params.get("ease", "inout"))

    def plan(self, cursor: _Cursor) -> tuple[_Sampler, _Cursor]:
        p0, q, f = cursor.pos, cursor.quat, cursor.fov
        p1 = curves.vadd(p0, curves.vscale(curves.quat_forward(q), self.distance))

        def sample(te: float) -> tuple[Vec3, Quat, float]:
            return curves.vlerp(p0, p1, te), q, f

        return sample, _Cursor(p1, q, f)


@primitive("dolly_zoom")
class _DollyZoom(_Segment):
    def __init__(self, target: Vec3, from_fov: float, to_fov: float, duration: float, ease: str) -> None:
        super().__init__(duration, ease, False)
        self.target, self.from_fov, self.to_fov = tuple(target), from_fov, to_fov

    @classmethod
    def from_params(cls, params: dict) -> _DollyZoom:
        return cls(
            tuple(params["target"]),
            params["from_fov"],
            params["to_fov"],
            params.get("duration", 3.0),
            params.get("ease", "inout"),
        )

    def plan(self, cursor: _Cursor) -> tuple[_Sampler, _Cursor]:
        d0 = curves.vlen(curves.vsub(cursor.pos, self.target))
        out = curves.vnorm(curves.vsub(cursor.pos, self.target))

        def sample(te: float) -> tuple[Vec3, Quat, float]:
            fov = curves.lerp(self.from_fov, self.to_fov, te)
            # keep subject size: d * tan(fov/2) constant
            d = d0 * math.tan(self.from_fov / 2.0) / math.tan(fov / 2.0)
            eye = curves.vadd(self.target, curves.vscale(out, d))
            return eye, curves.look_at_quat(eye, self.target), fov

        pos_end, quat_end, fov_end = sample(1.0)
        return sample, _Cursor(pos_end, quat_end, fov_end)


@primitive("flyby")
class _Flyby(_Segment):
    def __init__(self, eyes: Sequence[Vec3], target: Vec3, fov: float | None, duration: float, ease: str, world_orientation: bool) -> None:
        super().__init__(duration, ease, world_orientation)
        self.eyes = [tuple(e) for e in eyes]
        self.target, self.fov = tuple(target), fov

    @classmethod
    def from_params(cls, params: dict) -> _Flyby:
        return cls(
            [tuple(e) for e in params["eyes"]],
            tuple(params["target"]),
            params.get("fov"),
            params.get("duration", 8.0),
            params.get("ease", "inout"),
            params.get("world_orientation", False),
        )

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

    def __init__(self, sweep: float, duration: float, ease: str) -> None:
        super().__init__(duration, ease, False)
        self.sweep = sweep

    @classmethod
    def from_params(cls, params: dict) -> _Pan:
        sweep = math.radians(params["sweep_deg"]) if "sweep_deg" in params else params.get("sweep", math.pi / 2.0)
        return cls(sweep, params.get("duration", 4.0), params.get("ease", "inout"))

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

    def __init__(self, sweep: float, duration: float, ease: str) -> None:
        super().__init__(duration, ease, False)
        self.sweep = sweep

    @classmethod
    def from_params(cls, params: dict) -> _Tilt:
        sweep = math.radians(params["sweep_deg"]) if "sweep_deg" in params else params.get("sweep", math.pi / 6.0)
        return cls(sweep, params.get("duration", 4.0), params.get("ease", "inout"))

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
    """Change fov only (eye and aim fixed). from_fov is explicit since fov can't be read back."""

    def __init__(self, from_fov: float, to_fov: float, duration: float, ease: str) -> None:
        super().__init__(duration, ease, False)
        self.from_fov, self.to_fov = from_fov, to_fov

    @classmethod
    def from_params(cls, params: dict) -> _Zoom:
        return cls(params["from_fov"], params["to_fov"], params.get("duration", 2.0), params.get("ease", "inout"))

    def plan(self, cursor: _Cursor) -> tuple[_Sampler, _Cursor]:
        pos, quat = cursor.pos, cursor.quat

        def sample(te: float) -> tuple[Vec3, Quat, float]:
            return pos, quat, curves.lerp(self.from_fov, self.to_fov, te)

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
