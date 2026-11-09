from __future__ import annotations

import math
import typing
from collections.abc import Iterator, Mapping, Sequence

import attrs
import numpy as np
from typing_extensions import Self

from arena_simulation_setup.utils.cattrs import Idempotent, Parseable
from arena_simulation_setup.utils.resolution import resolve_zone_point

try:
    import geometry_msgs
    import geometry_msgs.msg
except ImportError:
    if not typing.TYPE_CHECKING:

        class _uninstanceable:
            """
            class that cannot be instantiated
            """

            def __new__(cls, *args: object, **kwargs: object) -> object:
                raise TypeError(f"installation of geometry_msgs is required to use {cls.__name__}")

            def __init__(self, *args: object, **kwargs: object) -> None:
                raise TypeError(f"installation of geometry_msgs is required to use {self.__class__.__name__}")

            def __getattribute__(self, name: str):
                raise TypeError(f"installation of geometry_msgs is required to use {self.__class__.__name__}.{name}")

        class geometry_msgs:
            """
            polyfill geometry_msgs.msg
            """

            class msg:
                class Point(_uninstanceable): ...

                class Quaternion(_uninstanceable): ...

                class Pose(_uninstanceable): ...


assert geometry_msgs  # type: ignore

EulerOrder = typing.Literal['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx']
_EulerIndices: dict[str, tuple[int, int, int]] = {
    'xyz': (0, 1, 2),
    'xzy': (0, 2, 1),
    'yxz': (1, 0, 2),
    'yzx': (1, 2, 0),
    'zxy': (2, 0, 1),
    'zyx': (2, 1, 0),
}


AXES: dict[str, float] = {
    '+x': 0.0,
    '+y': math.pi / 2,
    '-x': math.pi,
    '-y': -math.pi / 2,
    'xy': math.pi / 4,
}


def _yaw_gap(a: float, b: float) -> float:
    return abs(math.remainder(a - b, 2 * math.pi))


@attrs.define
class Vector3:
    x: float = attrs.field(converter=float)
    y: float = attrs.field(converter=float)
    z: float = attrs.field(converter=float, default=0.0)

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y
        yield self.z

    def __add__(self, other: Vector3) -> Position:
        return Position(x=self.x + other.x, y=self.y + other.y, z=self.z + other.z)

    def __sub__(self, other: Vector3) -> Position:
        return Position(x=self.x - other.x, y=self.y - other.y, z=self.z - other.z)

    def __mul__(self, other: float) -> Position:
        return Position(x=self.x * other, y=self.y * other, z=self.z * other)

    def __rmul__(self, other: float) -> Position:
        return self * other

    def __truediv__(self, other: float) -> Position:
        return Position(x=self.x / other, y=self.y / other, z=self.z / other)

    def norm(self, n: float = 2) -> float:
        """
        return norm of position vector
        """
        return (self.x**n + self.y**n + self.z**n) ** (1 / n)

    def normalized(self) -> Position:
        return self / (self.norm() or 1)

    def to_orientation(self) -> Orientation:
        """
        return orientation of vector
        """
        yaw = math.atan2(self.y, self.x)
        pitch = math.atan2(-self.z, math.sqrt(self.x * self.x + self.y * self.y))
        return Orientation.from_euler((0, pitch, yaw), order='xyz')


@attrs.define
class Position(Parseable, Idempotent, Vector3):
    """
    3D position
    """

    @classmethod
    def parse(cls, value: geometry_msgs.msg.Point | Sequence[float] | Mapping[str, float]) -> Self:
        """
        parse value into Position
        formats: [x,y], [x,y,z], {x,y[,z]}, zone ref name
        """
        if isinstance(value, str):
            point = resolve_zone_point(value)
            return cls(x=point.x, y=point.y, z=point.z)

        if isinstance(value, geometry_msgs.msg.Point):
            return cls.from_msg(value)

        if isinstance(value, Mapping):
            if not {"x", "y"} <= set(value) or not set(value) <= {"x", "y", "z"}:
                raise ValueError(f"Translation mapping must have keys x, y[, z], got {value}")
            return cls(x=value["x"], y=value["y"], z=value.get("z", 0.0))

        if len(value) == 3:
            return cls(*value)

        if len(value) == 2:
            return cls(value[0], value[1], 0.0)

        raise ValueError(f"Translation must be [x,y] or [x,y,z], got {value}")

    @classmethod
    def from_msg(cls, point: geometry_msgs.msg.Point) -> Self:
        """
        parse geometry_msgs.msg.Point
        """
        return cls(x=point.x, y=point.y, z=point.z)

    def to_msg(self) -> geometry_msgs.msg.Point:
        """
        return self as geometry_msgs.msg.Point
        """
        return geometry_msgs.msg.Point(x=self.x, y=self.y, z=self.z)


@attrs.define
class Orientation(Parseable, Idempotent):
    """
    3D orientation
    """

    w: float = attrs.field(converter=float)
    x: float = attrs.field(converter=float)
    y: float = attrs.field(converter=float)
    z: float = attrs.field(converter=float)

    @classmethod
    def parse(cls, value: geometry_msgs.msg.Quaternion | Sequence[float] | float | str) -> Self:
        """
        parse value into Orientation
        formats: yaw, axis (+x, +y, -x, -y, xy), [roll, pitch, yaw], [w,x,y,z]
        """

        if isinstance(value, geometry_msgs.msg.Quaternion):
            return cls.from_msg(typing.cast(geometry_msgs.msg.Quaternion, value))

        if isinstance(value, str):
            if value not in AXES:
                raise ValueError(f"orientation axis must be one of {', '.join(AXES)}, got {value!r}")
            return cls.from_euler((0, 0, AXES[value]), order='xyz')

        if isinstance(value, Sequence):
            if len(value) == 4:
                return cls(*value)

            if len(value) == 3:
                return cls.from_euler((value[0], value[1], value[2]))

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return cls.from_euler((0, 0, float(value)), order='xyz')

        raise ValueError(f"could not parse Orientation from {value}")

    @classmethod
    def identity(cls) -> Orientation:
        return cls(1.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_msg(cls, quaternion: geometry_msgs.msg.Quaternion) -> Self:
        """
        parse geometry_msgs.msg.Quaternion
        """
        return cls(w=quaternion.w, x=quaternion.x, y=quaternion.y, z=quaternion.z)

    def to_msg(self) -> geometry_msgs.msg.Quaternion:
        """
        return self as geometry_msgs.msg.Quaternion
        """
        return geometry_msgs.msg.Quaternion(w=self.w, x=self.x, y=self.y, z=self.z)

    @classmethod
    def from_xyzw(cls, quaternion: Sequence[float] | np.ndarray) -> Self:
        """Orientation of an x y z w quaternion."""
        x, y, z, w = quaternion
        return cls(w=w, x=x, y=y, z=z)

    def to_xyzw(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.w])

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> Self:
        """Orientation of a rotation matrix (its top-left 3x3), w >= 0."""
        m = np.asarray(matrix, dtype=float)
        trace = m[0, 0] + m[1, 1] + m[2, 2]
        if trace > 0.0:
            s = 2.0 * math.sqrt(trace + 1.0)
            w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
        n = math.copysign(math.sqrt(w * w + x * x + y * y + z * z), w)
        return cls(w=w / n, x=x / n, y=y / n, z=z / n)

    def to_matrix(self) -> np.ndarray:
        """3x3 rotation matrix of the normalized quaternion."""
        w, x, y, z = self.normalized()
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ]
        )

    def normalized(self) -> Orientation:
        """Unit quaternion of the same rotation, identity for the zero quaternion."""
        n = math.sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z)
        return Orientation.identity() if n == 0.0 else Orientation(w=self.w / n, x=self.x / n, y=self.y / n, z=self.z / n)

    def __pow__(self, exponent: float) -> Orientation:
        """The rotation turned `exponent` of the way along its shorter arc."""
        sign = 1.0 if self.w >= 0.0 else -1.0
        w, x, y, z = sign * self.w, sign * self.x, sign * self.y, sign * self.z
        s = math.sqrt(x * x + y * y + z * z)
        if s < 1e-12:
            return Orientation.identity()
        half = exponent * math.atan2(s, w)
        k = math.sin(half) / s
        return Orientation(w=math.cos(half), x=x * k, y=y * k, z=z * k)

    @classmethod
    def from_euler(cls, angles: tuple[float, float, float], order: EulerOrder = 'xyz') -> Self:
        x, y, z = (angles[index] for index in _EulerIndices[order])

        cr = math.cos(x * 0.5)
        sr = math.sin(x * 0.5)
        cp = math.cos(y * 0.5)
        sp = math.sin(y * 0.5)
        cy = math.cos(z * 0.5)
        sy = math.sin(z * 0.5)

        return cls(
            w=cr * cp * cy + sr * sp * sy,
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy,
        )

    def to_euler(self, order: EulerOrder = 'xyz') -> tuple[float, float, float]:
        """
        return euler angles
        """
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x * self.x + self.y * self.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (self.w * self.y - self.z * self.x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y * self.y + self.z * self.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        result: list[float] = [0, 0, 0]
        for index, value in zip(_EulerIndices[order], (roll, pitch, yaw), strict=False):
            result[index] = value

        return (result[0], result[1], result[2])

    @classmethod
    def from_yaw(cls, yaw: float) -> Orientation:
        """
        return Orientation from yaw angle
        """
        return cls.from_euler((0, 0, yaw), 'xyz')

    def to_yaw(self) -> float:
        """
        return yaw angle in radians
        """
        return self.to_euler()[2]

    def inverse(self) -> Orientation:
        """
        return the inverse rotation of a unit quaternion
        """
        return Orientation(w=self.w, x=-self.x, y=-self.y, z=-self.z)

    @typing.overload
    def __mul__(self, other: Orientation) -> Self: ...

    @typing.overload
    def __mul__(self, other: Vector3) -> Vector3: ...

    def __mul__(self, other: Orientation | Vector3) -> Orientation | Vector3:
        if isinstance(other, Vector3):
            return Vector3(
                x=(1 - 2 * (self.y * self.y + self.z * self.z)) * other.x + (2 * (self.x * self.y - self.w * self.z)) * other.y + (2 * (self.x * self.z + self.w * self.y)) * other.z,
                y=(2 * (self.x * self.y + self.w * self.z)) * other.x + (1 - 2 * (self.x * self.x + self.z * self.z)) * other.y + (2 * (self.y * self.z - self.w * self.x)) * other.z,
                z=(2 * (self.x * self.z - self.w * self.y)) * other.x + (2 * (self.y * self.z + self.w * self.x)) * other.y + (1 - 2 * (self.x * self.x + self.y * self.y)) * other.z,
            )
        if isinstance(other, Orientation):
            return Orientation(
                w=self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
                x=self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
                y=self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
                z=self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            )
        raise ValueError(f"cannot multiply Orientation with {other}")

    def __iter__(self) -> Iterator[float]:
        yield self.w
        yield self.x
        yield self.y
        yield self.z


@attrs.define
class Pose(Parseable, Idempotent):
    """
    3D pose
    """

    position: Position = attrs.field(converter=Position.converter, factory=lambda: Position(0, 0))
    orientation: Orientation = attrs.field(converter=Orientation.converter, factory=lambda: Orientation(1, 0, 0, 0))

    @classmethod
    def parse(cls, value: geometry_msgs.msg.Pose | Sequence[float] | Sequence[Sequence[float]] | Mapping[str, float]) -> Self:
        """
        parse value into Pose
        formats: [x,y], [x,y,yaw], [x,y,z,roll,pitch,yaw], [x,y,z,w,x,y,z], [[*position], [*orientation]], {x,y[,z][,roll][,pitch][,yaw]}, zone ref name
        """
        if isinstance(value, str):
            return cls(position=resolve_zone_point(value), orientation=Orientation.identity())

        if isinstance(value, geometry_msgs.msg.Pose):
            return cls.from_msg(value)

        if isinstance(value, Mapping) and not {"position", "orientation"} & set(value):
            if not {"x", "y"} <= set(value) or not set(value) <= {"x", "y", "z", "roll", "pitch", "yaw"}:
                raise ValueError(f"Pose mapping must have keys x, y[, z][, roll][, pitch][, yaw], got {value}")
            return cls(
                position=Position(x=value["x"], y=value["y"], z=value.get("z", 0.0)),
                orientation=Orientation.from_euler((value.get("roll", 0.0), value.get("pitch", 0.0), value.get("yaw", 0.0))) if {"roll", "pitch", "yaw"} & set(value) else Orientation.identity(),
            )

        # direct sequence
        if all(isinstance(v, (int, float)) for v in value):
            value = typing.cast(typing.Sequence[float], value)

            if len(value) == 2:
                return cls(Position(x=value[0], y=value[1], z=0.0), Orientation(w=1.0, x=0.0, y=0.0, z=0.0))

            if len(value) == 3:
                return cls(Position(x=value[0], y=value[1], z=0.0), Orientation.from_yaw(value[2]))

            if len(value) in (6, 7):
                return cls(position=Position.parse(value[:3]), orientation=Orientation.parse(value[3:]))

        # split sequence
        if len(value) == 2 and all(isinstance(v, Sequence) and all(isinstance(n, (int, float)) for n in v) for v in value):
            value = typing.cast(typing.Sequence[typing.Sequence[float]], value)
            return cls(position=Position.parse(value[0]), orientation=Orientation.parse(value[1]))

        raise ValueError(f"could not parse Pose from {value}")

    @classmethod
    def from_msg(cls, pose: geometry_msgs.msg.Pose) -> Self:
        """
        parse geometry_msgs.msg.Pose
        """
        return cls(position=Position.from_msg(pose.position), orientation=Orientation.from_msg(pose.orientation))

    def to_msg(self) -> geometry_msgs.msg.Pose:
        """
        return self as geometry_msgs.msg.Pose
        """
        return geometry_msgs.msg.Pose(position=self.position.to_msg(), orientation=self.orientation.to_msg())

    def to_2d(self) -> tuple[float, float, float]:
        """
        return self as (x, y, yaw)
        """
        return (self.position.x, self.position.y, self.orientation.to_yaw())

    def inverse(self) -> Pose:
        """
        return the pose that composes with this one to the identity
        """
        orientation = self.orientation.inverse()
        return Pose(position=Position(0, 0, 0) - orientation * self.position, orientation=orientation)

    @typing.overload
    def __mul__(self, other: Pose) -> Pose: ...

    @typing.overload
    def __mul__(self, other: Vector3) -> Position: ...

    def __mul__(self, other: Pose | Vector3) -> Pose | Position:
        """
        compose with a pose given in this pose's frame, or carry a point out of this pose's frame
        """
        if isinstance(other, Pose):
            return Pose(position=self * other.position, orientation=self.orientation * other.orientation)
        if isinstance(other, Vector3):
            return self.position + self.orientation * other
        raise ValueError(f"cannot multiply Pose with {other}")


@attrs.define
class Facing(Parseable, Idempotent):
    """
    allowed facing directions, any one of its orientations
    """

    orientations: list[Orientation] = attrs.field(
        converter=lambda value: [item if isinstance(item, Orientation) else Orientation.parse(item) for item in value],
        validator=attrs.validators.min_len(1),
    )

    @classmethod
    def parse(cls, value: Sequence[float | str] | float | str) -> Self:
        """
        parse value into Facing
        formats: an orientation as yaw or axis, or a list of them
        """
        if isinstance(value, (str, int, float)):
            return cls([Orientation.parse(value)])
        if isinstance(value, Sequence):
            return cls([Orientation.parse(item) for item in value])
        raise ValueError(f"could not parse Facing from {value}")

    @property
    def yaws(self) -> list[float]:
        """
        yaw angles in radians
        """
        return [orientation.to_yaw() for orientation in self.orientations]

    def contains(self, yaw: float, tolerance: float = 1e-6) -> bool:
        """
        whether yaw is one of the allowed directions
        """
        return any(_yaw_gap(yaw, allowed) <= tolerance for allowed in self.yaws)

    def nearest(self, yaw: float) -> Orientation:
        """
        allowed orientation closest to yaw
        """
        return min(self.orientations, key=lambda orientation: _yaw_gap(yaw, orientation.to_yaw()))

    def sample(self, rng: np.random.Generator) -> Orientation:
        """
        one allowed orientation, uniformly
        """
        return self.orientations[int(rng.integers(len(self.orientations)))]

    def serialize(self) -> float | str | list[float | str]:
        """
        yaws, an exact axis as its name, a single one unwrapped
        """
        items = [next((axis for axis, angle in AXES.items() if _yaw_gap(yaw, angle) < 1e-9), yaw) for yaw in self.yaws]
        return items[0] if len(items) == 1 else items


@attrs.define
class PositionRadius(Position, Idempotent):
    radius: float = attrs.field(converter=float, default=1.0)

    @classmethod
    def parse(cls, value: geometry_msgs.msg.Point | Sequence[float] | Mapping[str, float]) -> Self:
        """
        parse value into PositionRadius
        formats: [x,y], [x,y,radius], [x,y,z,radius]
        """
        if isinstance(value, geometry_msgs.msg.Point):
            return cls(value.x, value.y, value.z, 1.0)

        if isinstance(value, Mapping):
            raise TypeError(f"PositionRadius must be [x,y], [x,y,radius] or [x,y,z,radius], got {value}")

        if len(value) == 2:
            return cls(value[0], value[1])

        if len(value) == 3:
            return cls(value[0], value[1], radius=value[2])

        if len(value) == 4:
            return cls(*value)

        raise ValueError(f"PositionRadius must be [x,y], [x,y,radius] or [x,y,z,radius], got {value}")

    def __iter__(self) -> Iterator[float]:
        yield from super().__iter__()
        yield self.radius


@attrs.define
class Scale(Parseable, Idempotent):
    x: float = attrs.field(converter=float, default=1.0)
    y: float = attrs.field(converter=float, default=1.0)
    z: float = attrs.field(converter=float, default=1.0)

    @classmethod
    def parse(cls, value: geometry_msgs.msg.Point | Sequence[float]) -> Self:
        """
        parse value into Position
        formats: [x,y,z]
        """
        if isinstance(value, geometry_msgs.msg.Point):
            return cls.from_msg(value)

        if len(value) == 3:
            return cls(*value)

        raise ValueError(f"{cls.__name__} must be [x,y,z], got {value}")

    @classmethod
    def from_msg(cls, point: geometry_msgs.msg.Point) -> Self:
        """
        parse geometry_msgs.msg.Point
        """
        return cls(x=point.x, y=point.y, z=point.z)

    def to_msg(self) -> geometry_msgs.msg.Point:
        """
        return self as geometry_msgs.msg.Point
        """
        return geometry_msgs.msg.Point(x=self.x, y=self.y, z=self.z)


def sample_point_in_polygon(
    vertices: list[Position],
    rng: np.random.Generator,
    *,
    is_valid: typing.Callable[[Position], bool] | None = None,
    max_retries: int = 50,
) -> Position:
    n = len(vertices)
    if n < 3:
        return vertices[0] if vertices else Position(0, 0)

    # Fan triangulation from first vertex
    triangles = []
    areas = []
    v0 = vertices[0]
    for i in range(1, n - 1):
        v1, v2 = vertices[i], vertices[i + 1]
        area = abs((v1.x - v0.x) * (v2.y - v0.y) - (v2.x - v0.x) * (v1.y - v0.y)) / 2.0
        triangles.append((v0, v1, v2))
        areas.append(area)

    total = sum(areas)
    if total <= 0:
        return Position(v0.x, v0.y)

    def _sample_once() -> Position:
        r = float(rng.random()) * total
        cumulative = 0.0
        tri = triangles[0]
        for _tri, area in zip(triangles, areas, strict=False):
            cumulative += area
            if cumulative >= r:
                tri = _tri
                break

        u = float(rng.random())
        v = float(rng.random())
        if u + v > 1.0:
            u, v = 1.0 - u, 1.0 - v
        w = 1.0 - u - v

        a, b, c = tri
        return Position(
            w * a.x + u * b.x + v * c.x,
            w * a.y + u * b.y + v * c.y,
        )

    if is_valid is None:
        return _sample_once()

    last = _sample_once()
    for _ in range(max_retries):
        if is_valid(last):
            return last
        last = _sample_once()
    if is_valid(last):
        return last

    import warnings

    warnings.warn(
        f"sample_point_in_polygon: no valid sample after {max_retries} retries; returning last candidate",
        stacklevel=2,
    )
    return last


@attrs.define
class PointResolver:
    """Resolves zone/door/elevator names to a sampled point within their polygon."""

    lookup: typing.Callable[[str], list[Position] | None]
    rng: np.random.Generator
    is_valid: typing.Callable[[Position], bool] | None = None
    candidates: typing.Callable[[], typing.Iterable[str]] | None = None

    def resolve(self, name: str) -> Position:
        polygon = self.lookup(name)
        if polygon is None:
            known = sorted(self.candidates()) if self.candidates is not None else []
            hint = f"; known refs: {', '.join(known)}" if known else ""
            raise ValueError(f"zone ref {name!r} not found in world{hint}")
        return sample_point_in_polygon(polygon, self.rng, is_valid=self.is_valid)


def angle_diff(a: float, b: float) -> float:
    """
    returns difference of angles
    """
    A = (a - b) % 2 * np.pi
    B = (b - a) % 2 * np.pi
    return -A if A < B else B
