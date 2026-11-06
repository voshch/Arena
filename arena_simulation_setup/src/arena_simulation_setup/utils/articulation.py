"""Articulated objects: the links of an object's meshes, the joints between them, their named states and detents, the link poses at given joint values and the semantic fields they read as."""

from __future__ import annotations

import enum
import itertools
import math
import re
import typing
from collections.abc import Iterable, Mapping, Sequence

import attrs
import numpy as np

from arena_simulation_setup.utils.geometry import Orientation, Pose, Position

Side = tuple[float, float]
Box = tuple[Side, Side, Side]
Values = dict[str, float]
JointsSpec = str | float | Mapping[str, float | str] | Mapping[str, Mapping[str, float | str]]

NAME = re.compile(r"^[a-z0-9_]+$")
STATE_TOLERANCE = 1e-3
SWEEP_SAMPLES = 4096
SWEEP_STEPS = 9
DEFAULT_ANGULAR_VELOCITY = 1.0
DEFAULT_LINEAR_VELOCITY = 0.25
FAMILIES = ("openness", "pos", "angle", "setting", "q")
WRITABLE_FAMILIES = ("openness", "pos", "setting", "q")
RESERVED_STATES = frozenset({"state", "target", "moving", "blocked", *FAMILIES})
ROLES = ("door", "drawer", "lid", "cover", "hatch", "flap", "window", "hinge", "handle", "knob", "button", "switch", "latch", "lock")


class JointType(enum.StrEnum):
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    CONTINUOUS = "continuous"
    FIXED = "fixed"


def _name(kind: str, value: object) -> str:
    if not isinstance(value, str) or not NAME.match(value):
        raise ValueError(f"{kind} name must match [a-z0-9_]+, got {value!r}")
    return value


def _unreserved(kind: str, value: object) -> str:
    name = _name(kind, value)
    if name in RESERVED_STATES:
        raise ValueError(f"{kind} name {name!r} is reserved, avoid {', '.join(sorted(RESERVED_STATES))}")
    return name


def _number(field: str, value: object, accepted: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{field} takes {accepted}, got {value!r}")
    try:
        number = float(value)
    except ValueError:
        raise ValueError(f"{field} takes {accepted}, got {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{field} takes {accepted}, got {value!r}")
    return number


def _nonnegative(field: str, value: object) -> float | None:
    if value is None:
        return None
    number = _number(field, value, "a number >= 0")
    if number < 0.0:
        raise ValueError(f"{field} takes a number >= 0, got {value!r}")
    return number


def _role(value: object) -> str | None:
    if value is None:
        return None
    if value not in ROLES:
        raise ValueError(f"joint role must be one of {', '.join(ROLES)}, got {value!r}")
    return typing.cast(str, value)


def _strict(kind: str, cls: type, value: object) -> dict[str, typing.Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{kind} must be a mapping, got {value!r}")
    fields = {field.name for field in attrs.fields(cls)}
    if unknown := sorted(set(value) - fields):
        raise ValueError(f"{kind} has unknown keys {unknown}, known: {sorted(fields)}")
    return dict(value)


def _rounded(instance: object) -> dict[str, object]:
    return {key: round(value, 6) + 0.0 if isinstance(value, float) else value for key, value in attrs.asdict(instance, recurse=False).items() if value is not None}


@attrs.define(kw_only=True)
class Spring:
    """A passive spring pulling a joint toward its rest value, stiffness in N/m or N m/rad."""

    stiffness: float = attrs.field(converter=lambda value: typing.cast(float, _nonnegative("spring stiffness", value)))
    rest: float = attrs.field(default=0.0, converter=lambda value: _number("spring rest", value, "a joint value"))

    @classmethod
    def parse(cls, value: Spring | Mapping[str, typing.Any]) -> Spring:
        return value if isinstance(value, Spring) else cls(**_strict("spring", cls, value))


@attrs.define(kw_only=True)
class Dynamics:
    """Passive physics of a joint as its asset authors it, each part optional: dry friction in N or N m, or a PhysX friction coefficient scaling the joint's load, viscous damping in N s/m or N m s/rad, armature in kg or kg m^2, and a spring."""

    friction: float | None = attrs.field(default=None, converter=lambda value: _nonnegative("friction", value))
    friction_coefficient: float | None = attrs.field(default=None, converter=lambda value: _nonnegative("friction_coefficient", value))
    damping: float | None = attrs.field(default=None, converter=lambda value: _nonnegative("damping", value))
    armature: float | None = attrs.field(default=None, converter=lambda value: _nonnegative("armature", value))
    spring: Spring | None = attrs.field(default=None, converter=attrs.converters.optional(Spring.parse))

    def __attrs_post_init__(self) -> None:
        if self.friction is not None and self.friction_coefficient is not None:
            raise ValueError("dynamics take friction or friction_coefficient, not both")

    @classmethod
    def parse(cls, value: Dynamics | Mapping[str, typing.Any]) -> Dynamics:
        return value if isinstance(value, Dynamics) else cls(**_strict("dynamics", cls, value))

    def unstructure(self) -> dict[str, object]:
        result = _rounded(self)
        if self.spring is not None:
            result["spring"] = _rounded(self.spring)
        return result


@attrs.define(kw_only=True)
class Material:
    """Contact properties of a link's surfaces as its asset authors them: Coulomb friction coefficients and restitution."""

    static_friction: float | None = attrs.field(default=None, converter=lambda value: _nonnegative("static_friction", value))
    dynamic_friction: float | None = attrs.field(default=None, converter=lambda value: _nonnegative("dynamic_friction", value))
    restitution: float | None = attrs.field(default=None, converter=lambda value: None if value is None else _fraction("restitution", value))

    @classmethod
    def parse(cls, value: Material | Mapping[str, typing.Any]) -> Material:
        return value if isinstance(value, Material) else cls(**_strict("material", cls, value))

    def unstructure(self) -> dict[str, object]:
        return _rounded(self)


def _fraction(field: str, value: object) -> float:
    number = _number(field, value, "a number in [0, 1]")
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} takes a number in [0, 1], got {value!r}")
    return number


@attrs.define(kw_only=True)
class Latch:
    """A catch holding its joint at its closed value while the `by` joint sits short of `release`, a fraction of its range."""

    by: str = attrs.field(converter=lambda value: _name("joint", value))
    release: float = attrs.field(default=0.5, converter=lambda value: _fraction("latch release", value))

    @classmethod
    def parse(cls, value: Latch | Mapping[str, typing.Any]) -> Latch:
        return value if isinstance(value, Latch) else cls(**_strict("latch", cls, value))

    def unstructure(self) -> dict[str, object]:
        return _rounded(self)


def family_members[V](fields: Mapping[str, V], family: str) -> dict[str, V]:
    """The `<family>.<member>` entries of one entity's fields, by member."""
    prefix = f"{family}."
    return {field.removeprefix(prefix): value for field, value in fields.items() if field.startswith(prefix)}


def openness(value: float, closed: float, opened: float) -> float:
    """How far open from closed toward open in [0, 1], read open whichever way a two-sided joint swings."""
    return min(max(abs(value - closed) / abs(opened - closed), 0.0), 1.0)


def position(value: float, limit: Side) -> float:
    """Place within the limit in [0, 1], 0 for a zero-width limit."""
    low, high = limit
    if high <= low:
        return 0.0
    return min(max((value - low) / (high - low), 0.0), 1.0)


def angle(value: float) -> float:
    """Angle wrapped into [0, 2 pi)."""
    wrapped = value % (2 * math.pi)
    return 0.0 if wrapped >= 2 * math.pi else wrapped


def setting(value: float, detents: Mapping[str, float], wraps: bool = False) -> str:
    """The first detent within STATE_TOLERANCE of the value, a full turn apart counting as equal when it wraps, empty between detents."""
    return next((name for name, detent in detents.items() if abs(math.remainder(value - detent, 2 * math.pi) if wraps else value - detent) <= STATE_TOLERANCE), "")


def _vector(value: object) -> Position:
    if isinstance(value, Position):
        return value
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) != 3:
        raise ValueError(f"expected [x, y, z], got {value!r}")
    return Position(*(float(component) for component in value))


def _axis(value: object) -> Position:
    axis = _vector(value)
    if axis.norm() < 1e-9:
        raise ValueError(f"joint axis must not be zero, got {list(axis)}")
    return axis.normalized()


def _limit(value: object) -> Side | None:
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"joint limit must be [lower, upper], got {value!r}")
    lower, upper = (float(bound) for bound in value)
    if not (math.isfinite(lower) and math.isfinite(upper)) or lower > upper:
        raise ValueError(f"joint limit needs finite lower <= upper, got {value!r}")
    return (lower, upper)


def rotation(axis: Position, angle: float) -> Orientation:
    """Rotation by angle (radians) about a unit axis."""
    half = angle / 2.0
    s = math.sin(half)
    return Orientation(w=math.cos(half), x=axis.x * s, y=axis.y * s, z=axis.z * s)


def matrix(pose: Pose) -> np.ndarray:
    """4x4 homogeneous transform of a pose."""
    w, x, y, z = pose.orientation
    result = np.eye(4)
    result[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    result[:3, 3] = list(pose.position)
    return result


@attrs.define(kw_only=True)
class Joint:
    """A joint moving its child link relative to its parent, origin and axis in the model frame at all joint values 0."""

    name: str = attrs.field(converter=lambda value: _name("joint", value))
    type: JointType = attrs.field(converter=JointType)
    parent: str = attrs.field(converter=lambda value: _name("link", value))
    child: str = attrs.field(converter=lambda value: _name("link", value))
    origin: Position = attrs.field(factory=lambda: Position(0.0, 0.0, 0.0), converter=_vector)
    axis: Position = attrs.field(factory=lambda: Position(0.0, 0.0, 1.0), converter=_axis)
    limit: Side | None = attrs.field(default=None, converter=_limit)
    velocity: float | None = attrs.field(default=None, converter=attrs.converters.optional(float))
    role: str | None = attrs.field(default=None, converter=_role)
    dynamics: Dynamics | None = attrs.field(default=None, converter=attrs.converters.optional(Dynamics.parse))
    latch: Latch | None = attrs.field(default=None, converter=attrs.converters.optional(Latch.parse))

    def __attrs_post_init__(self) -> None:
        if self.parent == self.child:
            raise ValueError(f"joint {self.name} connects link {self.child} to itself")
        if self.type in (JointType.REVOLUTE, JointType.PRISMATIC) and self.limit is None:
            raise ValueError(f"{self.type} joint {self.name} needs a limit: [lower, upper]")
        if self.type in (JointType.CONTINUOUS, JointType.FIXED) and self.limit is not None:
            raise ValueError(f"{self.type} joint {self.name} takes no limit")
        if self.velocity is not None and not self.velocity > 0.0:
            raise ValueError(f"joint {self.name} velocity must be > 0, got {self.velocity}")
        if self.dynamics is not None and self.type is JointType.FIXED:
            raise ValueError(f"fixed joint {self.name} takes no dynamics")
        if self.latch is not None and self.type is JointType.FIXED:
            raise ValueError(f"fixed joint {self.name} takes no latch")
        if self.latch is not None and self.limit is None:
            raise ValueError(f"{self.type} joint {self.name} takes a latch only with a limit")
        if self.latch is not None and self.latch.by == self.name:
            raise ValueError(f"joint {self.name} cannot latch on itself")

    @property
    def movable(self) -> bool:
        return self.type is not JointType.FIXED

    @property
    def speed(self) -> float:
        """Kinematic speed in rad/s or m/s."""
        if self.velocity is not None:
            return self.velocity
        return DEFAULT_LINEAR_VELOCITY if self.type is JointType.PRISMATIC else DEFAULT_ANGULAR_VELOCITY

    @property
    def bounds(self) -> Side:
        """Reachable values, one full turn for a continuous joint and 0 for a fixed one."""
        if self.limit is not None:
            return self.limit
        if self.type is JointType.CONTINUOUS:
            return (-math.pi, math.pi)
        return (0.0, 0.0)

    def clamp(self, value: float) -> float:
        if self.limit is None:
            return math.remainder(value, 2 * math.pi) if self.type is JointType.CONTINUOUS else 0.0
        return min(max(value, self.limit[0]), self.limit[1])

    def motion(self, value: float) -> Pose:
        """Model-frame motion of the child at this joint value, relative to its parent."""
        if self.type in (JointType.REVOLUTE, JointType.CONTINUOUS):
            turn = rotation(self.axis, value)
            return Pose(position=self.origin - turn * self.origin, orientation=turn)
        if self.type is JointType.PRISMATIC:
            return Pose(position=self.axis * value, orientation=Orientation.identity())
        return Pose(position=Position(0.0, 0.0, 0.0), orientation=Orientation.identity())

    @classmethod
    def parse(cls, value: Joint | Mapping[str, typing.Any]) -> Joint:
        if isinstance(value, Joint):
            return value
        if not isinstance(value, Mapping):
            raise ValueError(f"joint must be a mapping, got {value!r}")
        fields = {field.name for field in attrs.fields(cls)}
        if unknown := sorted(set(value) - fields):
            raise ValueError(f"joint {value.get('name')!r} has unknown keys {unknown}, known: {sorted(fields)}")
        if missing := sorted({"name", "type", "parent", "child"} - set(value)):
            raise ValueError(f"joint {value.get('name')!r} is missing {missing}")
        try:
            return cls(**value)
        except ValueError as e:
            raise ValueError(f"joint {value.get('name')!r}: {e}") from None

    def unstructure(self) -> dict[str, object]:
        result: dict[str, object] = {
            "name": self.name,
            "type": str(self.type),
            "parent": self.parent,
            "child": self.child,
            "origin": [round(component, 6) + 0.0 for component in self.origin],
            "axis": [round(component, 6) + 0.0 for component in self.axis],
        }
        if self.limit is not None:
            result["limit"] = [round(bound, 6) + 0.0 for bound in self.limit]
        if self.velocity is not None:
            result["velocity"] = self.velocity
        if self.role is not None:
            result["role"] = self.role
        if self.dynamics is not None:
            result["dynamics"] = self.dynamics.unstructure()
        if self.latch is not None:
            result["latch"] = self.latch.unstructure()
        return result


def _links(value: object) -> dict[str, list[str]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"links must map each link name to its source node names, got {value!r}")
    links: dict[str, list[str]] = {}
    for link, nodes in value.items():
        if nodes is not None and (isinstance(nodes, str) or not isinstance(nodes, Iterable)):
            raise ValueError(f"link {link!r} takes a list of source node names, got {nodes!r}")
        links[_name("link", link)] = [str(node) for node in nodes or ()]
    return links


def _joints(value: object) -> list[Joint]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"joints must be a list, got {value!r}")
    return [Joint.parse(joint) for joint in value]


def _states(value: object) -> dict[str, Values]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"states must map each state name to joint values, got {value!r}")
    states: dict[str, Values] = {}
    for state, values in value.items():
        if not isinstance(values, Mapping) or not values:
            raise ValueError(f"state {state!r} must map joint names to values, got {values!r}")
        states[_name("state", state)] = {_name("joint", joint): float(position) for joint, position in values.items()}
    return states


def _detents(value: object) -> dict[str, Values]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"detents must map each joint name to named values, got {value!r}")
    detents: dict[str, Values] = {}
    for joint, values in value.items():
        if not isinstance(values, Mapping) or not values:
            raise ValueError(f"detents of joint {joint!r} must map detent names to values, got {values!r}")
        detents[_name("joint", joint)] = {_unreserved("detent", detent): float(number) for detent, number in values.items()}
    return detents


def _materials(value: object) -> dict[str, Material]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"materials must map link names to materials, got {value!r}")
    return {_name("link", link): Material.parse(material) for link, material in value.items()}


def _box(value: object) -> Box | None:
    if value is None:
        return None
    sides = [tuple(float(bound) for bound in side) for side in typing.cast("Iterable[Iterable[float]]", value)]
    if len(sides) != 3 or any(len(side) != 2 or side[0] > side[1] for side in sides):
        raise ValueError(f"swept must be [[x_min, x_max], [y_min, y_max], [z_min, z_max]], got {value!r}")
    return typing.cast(Box, tuple(sides))


@attrs.define(kw_only=True)
class Articulation:
    """Links, joints, named states and per-joint detents of an articulated object, in the model frame at all joint values 0."""

    links: dict[str, list[str]] = attrs.field(converter=_links)
    joints: list[Joint] = attrs.field(converter=_joints)
    states: dict[str, Values] = attrs.field(factory=dict, converter=_states)
    detents: dict[str, Values] = attrs.field(factory=dict, converter=_detents)
    swept: Box | None = attrs.field(default=None, converter=_box)
    materials: dict[str, Material] = attrs.field(factory=dict, converter=_materials)

    def __attrs_post_init__(self) -> None:
        names = [joint.name for joint in self.joints]
        if duplicates := sorted({name for name in names if names.count(name) > 1}):
            raise ValueError(f"joint names must be unique, duplicated: {', '.join(duplicates)}")
        if not any(joint.movable for joint in self.joints):
            raise ValueError("an articulation needs at least one joint that is not fixed")
        for joint in self.joints:
            if joint.name in RESERVED_STATES:
                raise ValueError(f"joint name {joint.name!r} is reserved, avoid {', '.join(sorted(RESERVED_STATES))}")
            for link in (joint.parent, joint.child):
                if link not in self.links:
                    raise ValueError(f"joint {joint.name} names link {link!r}, which links does not list (links: {', '.join(self.links)})")
        children = [joint.child for joint in self.joints]
        if twice := sorted({link for link in children if children.count(link) > 1}):
            raise ValueError(f"a link has one parent joint, {', '.join(twice)} has more")
        roots = [link for link in self.links if link not in children]
        if len(roots) != 1:
            raise ValueError(f"an articulation has exactly one root link (no parent joint), found {', '.join(roots) or 'none'}")
        if len(self.order()) != len(self.joints):
            raise ValueError("joints form a cycle")
        if unreached := sorted(set(self.links) - {roots[0], *children}):
            raise ValueError(f"links {', '.join(unreached)} hang on no joint")
        if strays := sorted(set(self.materials) - set(self.links)):
            raise ValueError(f"materials name links {', '.join(strays)}, which links does not list (links: {', '.join(self.links)})")
        nodes = [node for link_nodes in self.links.values() for node in link_nodes]
        if repeated := sorted({node for node in nodes if nodes.count(node) > 1}):
            raise ValueError(f"source nodes belong to one link each, {', '.join(repeated)} is listed twice")
        by_name = self.joint_map
        for joint in self.joints:
            if joint.latch is None:
                continue
            by = by_name.get(joint.latch.by)
            if by is None or by.limit is None:
                raise ValueError(f"joint {joint.name} latches on {joint.latch.by!r}, which is no limited joint, limited joints: {', '.join(j.name for j in self.joints if j.limit is not None)}")
        for state, values in self.states.items():
            if state in RESERVED_STATES or state in by_name:
                raise ValueError(f"state name {state!r} is taken, avoid {', '.join(sorted(RESERVED_STATES))} and the joint names")
            for joint_name, value in values.items():
                joint = by_name.get(joint_name)
                if joint is None:
                    raise ValueError(f"state {state} names joint {joint_name!r}, joints: {', '.join(by_name)}")
                if not joint.movable:
                    raise ValueError(f"state {state} sets fixed joint {joint_name}")
                if joint.limit is not None and not joint.limit[0] - STATE_TOLERANCE <= value <= joint.limit[1] + STATE_TOLERANCE:
                    raise ValueError(f"state {state} sets {joint_name} to {value}, outside its limit {list(joint.limit)}")
        for joint_name, values in self.detents.items():
            joint = by_name.get(joint_name)
            if joint is None or not joint.movable:
                raise ValueError(f"detents name {joint_name!r}, which is no movable joint, joints: {', '.join(joint.name for joint in self.movable)}")
            for detent, value in values.items():
                if joint.limit is not None and not joint.limit[0] - STATE_TOLERANCE <= value <= joint.limit[1] + STATE_TOLERANCE:
                    raise ValueError(f"detent {detent} of {joint_name} at {value} is outside its limit {list(joint.limit)}")

    @property
    def joint_map(self) -> dict[str, Joint]:
        return {joint.name: joint for joint in self.joints}

    @property
    def root(self) -> str:
        children = {joint.child for joint in self.joints}
        return next(link for link in self.links if link not in children)

    @property
    def movable(self) -> list[Joint]:
        return [joint for joint in self.joints if joint.movable]

    def order(self) -> list[Joint]:
        """Joints parents first, the joints a cycle cuts off left out."""
        children = {joint.child for joint in self.joints}
        reached = {link for link in self.links if link not in children}
        ordered: list[Joint] = []
        pending = list(self.joints)
        while pending:
            ready = [joint for joint in pending if joint.parent in reached]
            if not ready:
                break
            for joint in ready:
                ordered.append(joint)
                reached.add(joint.child)
                pending.remove(joint)
        return ordered

    def initial(self) -> Values:
        """The `closed` state where there is one, else every joint at 0 clamped into its limit."""
        values = {joint.name: joint.clamp(0.0) for joint in self.movable}
        values.update(self.states.get("closed", {}))
        return values

    def resolve(self, spec: JointsSpec | None, base: Mapping[str, float] | None = None) -> Values:
        """Joint values of a state name, an openness of every openness joint, joint values (numbers or state names) or family mappings of member joints, unnamed joints keep base or the initial values."""
        values = dict(base) if base is not None else self.initial()
        if spec is None:
            return values
        if isinstance(spec, str):
            return self._set("target", spec, values)
        if not isinstance(spec, Mapping):
            return self._set("openness", spec, values)
        keys = set(spec) & set(FAMILIES)
        if keys and keys != set(spec):
            raise ValueError(f"joints mixes family keys {sorted(keys)} with joint names {sorted(set(spec) - keys)}, use one or the other")
        if keys:
            for family, members in spec.items():
                if family not in WRITABLE_FAMILIES:
                    raise ValueError(f"joints family {family!r} is read-only, set {', '.join(WRITABLE_FAMILIES)}")
                if not isinstance(members, Mapping):
                    raise ValueError(f"joints family {family} maps its member joints to values, got {members!r}")
                for name, value in members.items():
                    values = self._set(f"{family}.{name}", value, values)
            return values
        joints = self.joint_map
        for name, value in spec.items():
            joint = joints.get(name)
            if joint is None or not joint.movable:
                raise ValueError(f"unknown movable joint {name!r}, joints: {', '.join(joint.name for joint in self.movable)}")
            if isinstance(value, str):
                if name not in self.states.get(value, {}):
                    raise ValueError(f"state {value!r} sets no value for joint {name}")
                value = self.states[value][name]
            values[name] = joint.clamp(_number(f"joint {name}", value, "a number or a state name"))
        return values

    def target_for(self, field: str, value: str, base: Mapping[str, float]) -> Values:
        """Joint targets after a timeline sets field to value, joints the field does not move keep base."""
        return self._set(field, value, dict(base))

    def _set(self, field: str, value: object, values: Values) -> Values:
        if field == "target":
            if not isinstance(value, str) or value.strip() not in self.states:
                raise ValueError(f"unknown state {value!r}, states: {', '.join(self.states) or 'none'}, target takes a state name")
            values.update(self.states[value.strip()])
            return values
        families = self.families()
        if field == "openness":
            if "openness" not in families:
                raise ValueError("openness needs a joint the closed and open states set apart, this articulation has none")
            fraction = _fraction(field, value)
            for name in families["openness"]:
                values[name] = self._joint_value("openness", self.joint_map[name], fraction)
            return values
        family, _, name = field.partition(".")
        if family not in WRITABLE_FAMILIES or not name:
            raise ValueError(f"field {field!r} is not writable, writable: {', '.join(sorted(self.writable_fields()))}")
        if name not in families.get(family, ()):
            raise ValueError(f"field {field!r} names no {family} joint, {family} joints: {', '.join(families.get(family, ())) or 'none'}")
        values[name] = self._joint_value(family, self.joint_map[name], value)
        return values

    def _joint_value(self, family: str, joint: Joint, value: object) -> float:
        field = f"{family}.{joint.name}"
        if family == "openness":
            closed, opened = self.states["closed"][joint.name], self.states["open"][joint.name]
            return closed + _fraction(field, value) * (opened - closed)
        if family == "pos":
            low, high = joint.bounds
            return low + _fraction(field, value) * (high - low)
        if family == "setting":
            detents = self.detents[joint.name]
            if not isinstance(value, str) or value.strip() not in detents:
                raise ValueError(f"{field} takes a detent name of {joint.name}: {', '.join(detents)}, got {value!r}")
            return joint.clamp(detents[value.strip()])
        return joint.clamp(_number(field, value, "a number"))

    def _family(self, joint: Joint) -> str:
        closed, opened = self.states.get("closed", {}), self.states.get("open", {})
        if joint.name in closed and joint.name in opened and abs(opened[joint.name] - closed[joint.name]) > STATE_TOLERANCE:
            return "openness"
        return "pos" if joint.limit is not None else "angle"

    def families(self) -> dict[str, tuple[str, ...]]:
        """Member joints of each non-empty field family, sorted."""
        members: dict[str, list[str]] = {family: [] for family in FAMILIES}
        for joint in self.movable:
            members[self._family(joint)].append(joint.name)
            members["q"].append(joint.name)
            if self.detents.get(joint.name):
                members["setting"].append(joint.name)
        return {family: tuple(sorted(names)) for family, names in members.items() if names}

    def fields(self, values: Mapping[str, float]) -> tuple[dict[str, float], dict[str, str]]:
        """Continuous `openness.`, `pos.`, `angle.` and `q.<joint>` fields and discrete `setting.<joint>` fields at the joint values."""
        continuous: dict[str, float] = {}
        discrete: dict[str, str] = {}
        joints = self.joint_map
        for family, members in self.families().items():
            for name in members:
                value = values.get(name, 0.0)
                field = f"{family}.{name}"
                if family == "openness":
                    continuous[field] = openness(value, self.states["closed"][name], self.states["open"][name])
                elif family == "pos":
                    continuous[field] = position(value, joints[name].bounds)
                elif family == "angle":
                    continuous[field] = angle(value)
                elif family == "setting":
                    discrete[field] = setting(value, self.detents[name], joints[name].type is JointType.CONTINUOUS)
                else:
                    continuous[field] = value
        return continuous, discrete

    def field_names(self) -> frozenset[str]:
        """Every field of this articulation's semantic entity: state, target, the predicates and the family fields."""
        return frozenset({"state", "target", "moving", "blocked", *self.states, *(f"{family}.{name}" for family, members in self.families().items() for name in members)})

    def writable_fields(self) -> frozenset[str]:
        """Every field a timeline may set: target, openness with openness joints, and the writable family fields."""
        families = self.families()
        return frozenset({"target", *(("openness",) if "openness" in families else ()), *(f"{family}.{name}" for family in WRITABLE_FAMILIES for name in families.get(family, ()))})

    def state_of(self, values: Mapping[str, float], tolerance: float = STATE_TOLERANCE) -> str:
        """The first state whose joint values all match, empty between states."""
        for state, targets in self.states.items():
            if all(abs(values.get(joint, 0.0) - target) <= tolerance for joint, target in targets.items()):
                return state
        return ""

    def latch_holds(self, joint: Joint, values: Mapping[str, float]) -> bool:
        """Whether the joint's latch is engaged at the values: its `by` joint short of the release fraction of its range."""
        if joint.latch is None:
            return False
        by = self.joint_map[joint.latch.by]
        return position(values.get(by.name, 0.0), by.bounds) < joint.latch.release

    def link_poses(self, values: Mapping[str, float]) -> dict[str, Pose]:
        """Model-frame pose of each link's geometry at the joint values, unnamed joints at 0."""
        poses = {self.root: Pose(position=Position(0.0, 0.0, 0.0), orientation=Orientation.identity())}
        for joint in self.order():
            poses[joint.child] = poses[joint.parent] * joint.motion(values.get(joint.name, 0.0))
        return poses

    def transformed(self, transform: np.ndarray) -> Articulation:
        """This articulation carried through a rigid transform with uniform scale, as the build applies to stand a model on the floor."""
        transform = np.asarray(transform, dtype=np.float64)
        linear = transform[:3, :3]
        scale = float(np.cbrt(np.linalg.det(linear)))
        if not scale > 0.0 or not np.allclose(linear.T @ linear, scale * scale * np.eye(3), atol=1e-6 * max(1.0, scale * scale)):
            raise ValueError(f"articulations follow rigid transforms with uniform scale only, got {transform.tolist()}")
        joints = []
        for joint in self.joints:
            origin = linear @ np.array(list(joint.origin)) + transform[:3, 3]
            axis = linear @ np.array(list(joint.axis))
            linear_joint = joint.type is JointType.PRISMATIC
            dynamics = joint.dynamics
            if dynamics is not None and dynamics.spring is not None and linear_joint:
                dynamics = attrs.evolve(dynamics, spring=attrs.evolve(dynamics.spring, rest=dynamics.spring.rest * scale))
            joints.append(
                attrs.evolve(
                    joint,
                    origin=Position(*origin),
                    axis=Position(*axis),
                    limit=None if joint.limit is None else tuple(bound * scale for bound in joint.limit) if linear_joint else joint.limit,
                    velocity=None if joint.velocity is None else joint.velocity * scale if linear_joint else joint.velocity,
                    dynamics=dynamics,
                )
            )
        prismatic = {joint.name for joint in self.joints if joint.type is JointType.PRISMATIC}
        states = {state: {joint: value * scale if joint in prismatic else value for joint, value in values.items()} for state, values in self.states.items()}
        detents = {joint: {detent: value * scale if joint in prismatic else value for detent, value in values.items()} for joint, values in self.detents.items()}
        return Articulation(links=self.links, joints=joints, states=states, detents=detents, materials=self.materials)

    def configurations(self, samples: int = SWEEP_SAMPLES, steps: int = SWEEP_STEPS) -> Iterable[Values]:
        """Joint values covering every joint's reachable range: the full grid of steps per joint when it fits in samples, else every joint's bounds with the others at 0 plus seeded random draws."""
        movable = self.movable
        grids = [np.linspace(*joint.bounds, steps) for joint in movable]
        if steps ** len(movable) <= samples:
            for combination in itertools.product(*grids):
                yield dict(zip((joint.name for joint in movable), combination, strict=True))
            return
        for joint, grid in zip(movable, grids, strict=True):
            for value in grid:
                yield {joint.name: float(value)}
        rng = np.random.default_rng(0)
        for _ in range(samples):
            yield {joint.name: float(rng.uniform(*joint.bounds)) for joint in movable}

    def sweep(self, link_boxes: Mapping[str, Box], samples: int = SWEEP_SAMPLES) -> Box:
        """Bounding box of the link boxes over every configuration, links without a box left out."""
        corners = {link: np.array(list(itertools.product(*box))) for link, box in link_boxes.items()}
        low = np.full(3, np.inf)
        high = np.full(3, -np.inf)
        for values in self.configurations(samples):
            for link, pose in self.link_poses(values).items():
                if link not in corners:
                    continue
                transform = matrix(pose)
                points = corners[link] @ transform[:3, :3].T + transform[:3, 3]
                low = np.minimum(low, points.min(axis=0))
                high = np.maximum(high, points.max(axis=0))
        if not np.all(np.isfinite(low)):
            raise ValueError("no link has a box to sweep")
        return typing.cast(Box, tuple((round(float(lo), 4) + 0.0, round(float(hi), 4) + 0.0) for lo, hi in zip(low, high, strict=True)))

    @classmethod
    def parse(cls, value: Articulation | Mapping[str, typing.Any]) -> Articulation:
        if isinstance(value, Articulation):
            return value
        if not isinstance(value, Mapping):
            raise ValueError(f"articulation must be a mapping of links, joints, states and detents, got {value!r}")
        if unknown := sorted(set(value) - {"links", "joints", "states", "detents", "swept", "materials"}):
            raise ValueError(f"articulation has unknown keys {unknown}, known: links, joints, states, detents, swept, materials")
        if missing := sorted({"links", "joints"} - set(value)):
            raise ValueError(f"articulation is missing {missing}")
        return cls(**value)

    def unstructure(self) -> dict[str, object]:
        result: dict[str, object] = {"links": {link: list(nodes) for link, nodes in self.links.items()}, "joints": [joint.unstructure() for joint in self.joints]}
        if self.states:
            result["states"] = {state: {joint: round(value, 6) + 0.0 for joint, value in values.items()} for state, values in self.states.items()}
        if self.detents:
            result["detents"] = {joint: {detent: round(value, 6) + 0.0 for detent, value in values.items()} for joint, values in self.detents.items()}
        if self.swept is not None:
            result["swept"] = [list(side) for side in self.swept]
        if self.materials:
            result["materials"] = {link: material.unstructure() for link, material in self.materials.items()}
        return result


def structure_articulation(value: Articulation | Mapping[str, typing.Any] | None) -> Articulation | None:
    return None if value is None else Articulation.parse(value)


def unstructure_articulation(value: Articulation | None) -> dict[str, object] | None:
    return None if value is None else value.unstructure()
