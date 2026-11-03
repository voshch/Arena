"""Articulated objects in every simulator: physics bodies with driven joints where the simulator has them, else one obstacle per link bundle moved kinematically on the mechanism tick."""

from __future__ import annotations

import asyncio
import math
import typing
from collections.abc import Awaitable, Callable, Mapping, Sequence

import attrs
import geometry_msgs.msg
import numpy as np
from arena_simulation_setup.tree.assets.Object import ObjectIdentifier, ObjectView
from arena_simulation_setup.utils.articulation import Articulation, Joint, JointType, Material, Values

from arena_runtime.shared import Obstacle, Pose, Position

if typing.TYPE_CHECKING:
    from arena_simulation_setup.shared import Light
    from arena_simulation_setup.utils.geometry import Scale

    from ._interface import MechanismITF

LINK_SEPARATOR = '__'
JOINT_TOLERANCE = 1e-6
GRAVITY = 9.81
DENSITY = 150.0
MASS_RANGE = (0.2, 20.0)
BARE_LINK_MASS = 0.05
MIN_INERTIA = 1e-4
HOLD_MARGIN = 1.2
EFFORT_PER_FRICTION = 3.0
LEAD = 0.25
LAG_BLOCKED = 0.9
MOTION_TOLERANCE = 1e-3
SURFACE_FRICTION = 0.5
JOINT_CODES = {JointType.REVOLUTE: 0, JointType.PRISMATIC: 1, JointType.CONTINUOUS: 2, JointType.FIXED: 3}

Disc = tuple[str, tuple[float, float], float]
Point = tuple[float, float]


@attrs.frozen
class DriveTuning:
    """Passive and drive constants of one joint kind, the friction and effort floors before gravity raises them."""

    damping: float
    friction: float
    effort: float
    saturation: float
    settle: float
    armature: float = 0.01


ANGULAR = DriveTuning(damping=0.5, friction=0.2, effort=10.0, saturation=0.1, settle=0.05)
LINEAR = DriveTuning(damping=5.0, friction=2.0, effort=40.0, saturation=0.02, settle=0.01)
ANGULAR_ROLES = {
    "door": (0.5, 2.0),
    "window": (0.5, 2.0),
    "lid": (0.3, 1.0),
    "cover": (0.3, 1.0),
    "hatch": (0.3, 1.0),
    "flap": (0.3, 1.0),
    "hinge": (0.3, 1.0),
    "knob": (0.02, 0.01),
    "switch": (0.02, 0.01),
    "handle": (0.05, 0.02),
    "latch": (0.05, 0.02),
    "lock": (0.05, 0.02),
}
"""Dry friction (N m) and damping (N m s/rad) of a hinge by role, where its asset authors none."""
LINEAR_ROLES = {
    "drawer": (6.0, 15.0),
    "door": (10.0, 20.0),
    "window": (10.0, 20.0),
    "button": (0.5, 0.5),
}
"""Dry friction (N) and damping (N s/m) of a slide by role, where its asset authors none."""


@attrs.frozen
class LinkInertia:
    """Mass, center of mass in the model frame and principal moments along the model axes of one link."""

    mass: float
    center: tuple[float, float, float]
    moments: tuple[float, float, float]


@attrs.frozen
class JointDynamics:
    """Passive damping, friction, armature and spring of a joint, its drive's effort limit, stiffness and damping, and the gap within which a drive snaps onto its target."""

    damping: float
    friction: float
    armature: float
    effort: float
    stiffness: float
    drive_damping: float
    settle: float
    spring_stiffness: float = 0.0
    spring_rest: float = 0.0


@attrs.frozen
class JointCommand:
    """One object's drive references, the joints whose drives are on, and the joints set onto their reference at rest."""

    runtime: ArticulatedObject
    reference: Values
    drive: frozenset[str]
    place: frozenset[str]


def joint_gap(joint: Joint, current: float, target: float) -> float:
    """Signed distance from current to target, the short way round for a continuous joint."""
    if joint.type is JointType.CONTINUOUS:
        return math.remainder(target - current, 2 * math.pi)
    return target - current


def scaled(pose: Pose, scale: Scale | None) -> Pose:
    """A model-frame link pose with its translation scaled componentwise."""
    if scale is None:
        return pose
    position = pose.position
    return Pose(position=Position(position.x * scale.x, position.y * scale.y, position.z * scale.z), orientation=pose.orientation)


def hull(points: Sequence[Point]) -> list[Point]:
    """Convex hull, counter-clockwise."""
    ordered = sorted(set(points))
    if len(ordered) < 3:
        return ordered

    def cross(o: Point, a: Point, b: Point) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[Point] = []
    upper: list[Point] = []
    for chain, sequence in ((lower, ordered), (upper, reversed(ordered))):
        for point in sequence:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], point) <= 0.0:
                chain.pop()
            chain.append(point)
    return lower[:-1] + upper[:-1]


def disc_hits(polygon: Sequence[Point], center: Point, radius: float) -> bool:
    """Whether a disc overlaps a convex counter-clockwise polygon."""
    if not polygon:
        return False
    x, y = center
    inside = len(polygon) >= 3
    for i, (ax, ay) in enumerate(polygon):
        bx, by = polygon[(i + 1) % len(polygon)]
        dx, dy = bx - ax, by - ay
        length = dx * dx + dy * dy
        t = 0.0 if length <= 0.0 else min(1.0, max(0.0, ((x - ax) * dx + (y - ay) * dy) / length))
        if math.hypot(x - (ax + t * dx), y - (ay + t * dy)) < radius:
            return True
        inside = inside and dx * (y - ay) - dy * (x - ax) >= 0.0
    return inside


@attrs.define
class ArticulatedObject:
    """One articulated obstacle: its link bundles and its initial, current and target joint values."""

    obstacle: Obstacle
    name: str
    sim_path: str
    entity: str
    articulation: Articulation
    links: dict[str, ObjectView]
    initial: Values
    current: Values
    target: Values
    blocked: bool = False
    placed: bool = True
    physical: bool = False
    inertias: dict[str, LinkInertia] = attrs.field(factory=dict)
    dynamics: dict[str, JointDynamics] = attrs.field(factory=dict)
    driving: set[str] = attrs.field(factory=set)
    reference: Values = attrs.field(factory=dict)
    motion: bool = False

    @property
    def moving(self) -> bool:
        if self.physical:
            return bool(self.driving) or self.motion
        return any(abs(joint_gap(joint, self.current[joint.name], self.target[joint.name])) > JOINT_TOLERANCE for joint in self.articulation.movable)

    def command(self, target: Values) -> None:
        """New joint targets, the joints they move driven toward them from where they are."""
        for joint in self.articulation.movable:
            name = joint.name
            if abs(joint_gap(joint, self.target[name], target[name])) > JOINT_TOLERANCE and name not in self.driving:
                self.driving.add(name)
                self.reference[name] = self.current[name]
        self.target = dict(target)

    def reset(self) -> None:
        """Current and target back to the initial values, the links placed again on the next tick."""
        self.current = dict(self.initial)
        self.target = dict(self.initial)
        self.blocked = False
        self.placed = False
        self.driving.clear()
        self.reference = dict(self.initial)
        self.motion = False

    def advance(self, dt: float) -> Values:
        """Joint values one step of dt toward the target at each joint's speed."""
        values = dict(self.current)
        for joint in self.articulation.movable:
            current, target = self.current[joint.name], self.target[joint.name]
            gap = joint_gap(joint, current, target)
            step = joint.speed * dt
            values[joint.name] = target if abs(gap) <= step else joint.clamp(current + math.copysign(step, gap))
        return values

    def link_poses(self, values: Values) -> dict[str, Pose]:
        """Map-frame pose of every link bundle at the joint values."""
        poses = self.articulation.link_poses(values)
        return {link: self.obstacle.pose * scaled(poses[link], self.obstacle.scale) for link in self.links}

    def link_obstacle(self, link: str, pose: Pose) -> Obstacle:
        name = f'{self.name}{LINK_SEPARATOR}{link}'
        part = Obstacle(
            name=name,
            pose=pose,
            model=ObjectIdentifier.inline(self.links[link], name=f'{self.obstacle.model.name}/links/{link}'),
            level=self.obstacle.level,
            scale=self.obstacle.scale,
        )
        part.sim_path = f'{self.sim_path}{LINK_SEPARATOR}{link}'
        return part

    def link_obstacles(self) -> list[Obstacle]:
        """One obstacle per link bundle at the current joint values."""
        return [self.link_obstacle(link, pose) for link, pose in self.link_poses(self.current).items()]

    def footprint(self, link: str, pose: Pose) -> list[Point]:
        """Map-frame xy hull of a link's annotated box at a link pose, empty without a box."""
        box = self.links[link].extent
        if box is None:
            return []
        scale = self.obstacle.scale
        sx, sy, sz = (scale.x, scale.y, scale.z) if scale is not None else (1.0, 1.0, 1.0)
        corners = (pose * Position(x * sx, y * sy, z * sz) for x in box[0] for y in box[1] for z in box[2])
        return hull([(corner.x, corner.y) for corner in corners])

    def tree(self) -> list[tuple[str, Joint | None]]:
        """Every link with the joint to its parent, parents first, the root with None."""
        return [(self.articulation.root, None), *((joint.child, joint) for joint in self.articulation.order())]

    def make_physical(self) -> None:
        """Derive the link inertias and joint dynamics a physics simulator builds the object from."""
        self.physical = True
        self.inertias = link_inertias(self.articulation, self.links, self.obstacle.scale)
        self.dynamics = joint_dynamics(self.articulation, self.inertias, self.obstacle.scale)
        self.reference = dict(self.current)


def _scales(scale: Scale | None) -> np.ndarray:
    return np.ones(3) if scale is None else np.array([scale.x, scale.y, scale.z], dtype=np.float64)


def link_inertias(articulation: Articulation, links: Mapping[str, ObjectView], scale: Scale | None) -> dict[str, LinkInertia]:
    """Each link a solid box of its scaled annotated extent at DENSITY, mass clamped into MASS_RANGE, a link without a box a small point mass at its joint."""
    factors = _scales(scale)
    origins = {joint.child: np.array(list(joint.origin)) * factors for joint in articulation.joints}
    result: dict[str, LinkInertia] = {}
    for link in articulation.links:
        view = links.get(link)
        box = view.extent if view is not None else None
        if box is None:
            center = origins.get(link, np.zeros(3))
            result[link] = LinkInertia(BARE_LINK_MASS, (float(center[0]), float(center[1]), float(center[2])), (MIN_INERTIA, MIN_INERTIA, MIN_INERTIA))
            continue
        low, high = np.sort(np.array(box, dtype=np.float64).T * factors, axis=0)
        size = high - low
        mass = float(np.clip(DENSITY * np.prod(size), *MASS_RANGE))
        x, y, z = (float(side) ** 2 for side in size)
        center = (low + high) / 2
        result[link] = LinkInertia(
            mass,
            (float(center[0]), float(center[1]), float(center[2])),
            (max(mass / 12 * (y + z), MIN_INERTIA), max(mass / 12 * (x + z), MIN_INERTIA), max(mass / 12 * (x + y), MIN_INERTIA)),
        )
    return result


def _subtree(articulation: Articulation, link: str) -> list[str]:
    links = [link]
    for joint in articulation.order():
        if joint.parent in links:
            links.append(joint.child)
    return links


def _samples(joint: Joint) -> list[float]:
    low, high = joint.bounds
    return [float(value) for value in np.linspace(low, high, 9)]


def joint_dynamics(articulation: Articulation, inertias: Mapping[str, LinkInertia], scale: Scale | None) -> dict[str, JointDynamics]:
    """Per movable joint: the friction, damping, armature and spring its asset authors, else its role's or kind's friction raised to hold the worst static gravity load over its range, an effort beating friction, spring and damping at speed, a drive saturating at the tuning's gap and critically damped on the moved inertia."""
    factors = _scales(scale)
    weight = np.array([0.0, 0.0, -GRAVITY])
    result: dict[str, JointDynamics] = {}
    for joint in articulation.movable:
        tuning = LINEAR if joint.type is JointType.PRISMATIC else ANGULAR
        subtree = _subtree(articulation, joint.child)
        mass = sum(inertias[link].mass for link in subtree)
        load = 0.0
        moved = 0.0
        for value in _samples(joint):
            poses = {link: scaled(pose, scale) for link, pose in articulation.link_poses({joint.name: value}).items()}
            parent = poses[joint.parent]
            axis = np.array(list(parent.orientation * joint.axis))
            if joint.type is JointType.PRISMATIC:
                load = max(load, mass * GRAVITY * abs(axis[2]))
                moved = mass
                continue
            origin = np.array(list(parent * Position(*(np.array(list(joint.origin)) * factors))))
            torque = 0.0
            inertia = 0.0
            for link in subtree:
                body = inertias[link]
                center = np.array(list(poses[link] * Position(*body.center)))
                lever = center - origin
                torque += float(np.dot(np.cross(lever, body.mass * weight), axis))
                along = np.array(list(poses[link].orientation.inverse() * Position(*axis)))
                inertia += body.mass * float(np.dot(lever, lever) - np.dot(lever, axis) ** 2) + float(np.dot(along**2, body.moments))
            load = max(load, abs(torque))
            moved = max(moved, inertia)
        roles = LINEAR_ROLES if joint.type is JointType.PRISMATIC else ANGULAR_ROLES
        friction, damping = roles.get(joint.role or "", (tuning.friction, tuning.damping))
        friction = max(friction, HOLD_MARGIN * load)
        armature = tuning.armature
        spring_stiffness = spring_rest = 0.0
        if (authored := joint.dynamics) is not None:
            if authored.friction is not None:
                friction = authored.friction
            elif authored.friction_coefficient is not None:
                friction = authored.friction_coefficient * mass * GRAVITY
            damping = damping if authored.damping is None else authored.damping
            armature = armature if authored.armature is None else authored.armature
            if authored.spring is not None:
                spring_stiffness, spring_rest = authored.spring.stiffness, authored.spring.rest
        stretch = max(abs(bound - spring_rest) for bound in joint.bounds)
        effort = max(tuning.effort, EFFORT_PER_FRICTION * (friction + spring_stiffness * stretch + damping * joint.speed))
        stiffness = effort / tuning.saturation
        result[joint.name] = JointDynamics(
            damping=damping,
            friction=friction,
            armature=armature,
            effort=effort,
            stiffness=stiffness,
            drive_damping=2.0 * math.sqrt(stiffness * (moved + armature)),
            settle=tuning.settle,
            spring_stiffness=spring_stiffness,
            spring_rest=spring_rest,
        )
    return result


def link_light_pose(runtime: ArticulatedObject, light: Light) -> tuple[Position, float] | None:
    """Map position and yaw turn of a light riding on a link of the object at its current joint values, None off the moving links."""
    articulation = runtime.articulation
    if light.pose is None or not light.link or light.link == articulation.root or light.link not in articulation.links:
        return None
    origin = runtime.obstacle.pose
    motion = scaled(articulation.link_poses(runtime.current)[light.link], runtime.obstacle.scale)
    placed = origin * motion * (origin.inverse() * light.pose)
    return (placed.position, (placed.orientation * light.pose.orientation.inverse()).to_yaw())


async def resolve_articulated(obstacle: Obstacle, entity: str) -> ArticulatedObject | None:
    """The runtime of an obstacle whose object has an articulation and link bundles, its semantic entity named entity, None for a plain one."""
    try:
        view = await obstacle.model.resolve()
    except FileNotFoundError:
        return None
    articulation = view.articulation
    if articulation is None:
        return None
    links: dict[str, ObjectView] = {}
    for link in articulation.links:
        try:
            links[link] = view.link(link)
        except FileNotFoundError:
            continue
    if not links:
        return None
    initial = articulation.resolve(obstacle.joints)
    return ArticulatedObject(
        obstacle=obstacle,
        name=obstacle.name,
        sim_path=obstacle.sim_path,
        entity=entity,
        articulation=articulation,
        links=links,
        initial=initial,
        current=dict(initial),
        target=dict(initial),
    )


async def _dispatch(
    verb: Callable[[Sequence[Obstacle]], Awaitable[Sequence[bool]]],
    groups: Sequence[Sequence[Obstacle] | None],
) -> list[bool]:
    """Send every group in one batch, a group succeeds when all its obstacles do, None fails."""
    batch = [obstacle for group in groups if group is not None for obstacle in group]
    oks = list(await verb(batch)) if batch else []
    results: list[bool] = []
    start = 0
    for group in groups:
        if group is None:
            results.append(False)
            continue
        results.append(all(oks[start : start + len(group)]))
        start += len(group)
    return results


async def spawn_objects(mech: MechanismITF, obstacles: Sequence[Obstacle]) -> list[bool]:
    from ._mechanism_shim import _ensure_loop

    logger = mech.node.get_logger()

    async def resolve(obstacle: Obstacle) -> ArticulatedObject | Exception | None:
        try:
            return await resolve_articulated(obstacle, mech._realizer.prefix(obstacle.name))
        except ValueError as e:
            return e

    runtimes = await asyncio.gather(*map(resolve, obstacles))
    groups: list[Sequence[Obstacle] | None] = []
    physical: list[ArticulatedObject] = []
    for obstacle, runtime in zip(obstacles, runtimes, strict=True):
        if isinstance(runtime, Exception):
            logger.error(f"articulated object {obstacle.name!r} not spawned: {runtime}")
            groups.append(None)
        elif runtime is None:
            groups.append([obstacle])
        elif mech.ARTICULATED_PHYSICS:
            runtime.make_physical()
            physical.append(runtime)
            groups.append(())
        else:
            groups.append(runtime.link_obstacles())
    results = await _dispatch(mech.obstacle_spawn, groups)
    if physical:
        spawned = dict(zip((id(runtime) for runtime in physical), await mech._articulated_spawn(physical), strict=True))
        results = [spawned.get(id(runtime), ok) for runtime, ok in zip(runtimes, results, strict=True)]
    for runtime, ok in zip(runtimes, results, strict=True):
        if not isinstance(runtime, ArticulatedObject) or (runtime.physical and not ok):
            continue
        if (replaced := mech._articulated.get(runtime.name)) is not None:
            mech._semantics.detach(replaced.entity)
        mech._articulated[runtime.name] = runtime
        mech._semantics.attach('articulated', runtime.entity)
        _ensure_loop(mech)
    return results


async def move_objects(mech: MechanismITF, obstacles: Sequence[Obstacle]) -> list[bool]:
    groups: list[Sequence[Obstacle] | None] = []
    physical: list[ArticulatedObject] = []
    for obstacle in obstacles:
        runtime = mech._articulated.get(obstacle.name)
        if runtime is None:
            groups.append([obstacle])
            continue
        try:
            runtime.initial = runtime.articulation.resolve(obstacle.joints)
        except ValueError as e:
            mech.node.get_logger().error(f"articulated object {obstacle.name!r} keeps its initial joint values: {e}")
        runtime.obstacle = obstacle
        runtime.reset()
        if runtime.physical:
            groups.append([obstacle])
            physical.append(runtime)
            continue
        runtime.placed = True
        groups.append(runtime.link_obstacles())
    results = await _dispatch(mech.obstacle_move, groups)
    if physical:
        await _drive(mech, [place(runtime) for runtime in physical])
    return results


async def delete_objects(mech: MechanismITF, obstacles: Sequence[Obstacle]) -> list[bool]:
    groups: list[Sequence[Obstacle] | None] = []
    for obstacle in obstacles:
        runtime = mech._articulated.pop(obstacle.name, None)
        if runtime is None:
            groups.append([obstacle])
            continue
        mech._semantics.detach(runtime.entity)
        groups.append([obstacle] if runtime.physical else runtime.link_obstacles())
    return await _dispatch(mech.obstacle_delete, groups)


def link_message[M](message: M, runtime: ArticulatedObject, link: str, joint: Joint | None, dynamics: Callable[..., typing.Any]) -> M:
    """Fill a simulator's ArticulatedLink message with a link's name, inertia, surface material and the scaled joint to its parent, dynamics building its JointDynamics."""
    inertia = runtime.inertias[link]
    fields = typing.cast(typing.Any, message)
    fields.name = link
    fields.mass = inertia.mass
    fields.center = geometry_msgs.msg.Point(x=inertia.center[0], y=inertia.center[1], z=inertia.center[2])
    fields.moments = geometry_msgs.msg.Vector3(x=inertia.moments[0], y=inertia.moments[1], z=inertia.moments[2])
    if (material := runtime.articulation.materials.get(link)) is not None:
        fields.has_material = True
        fields.static_friction, fields.dynamic_friction, fields.restitution = surface(material)
    if joint is None:
        return message
    origin = np.array(list(joint.origin)) * _scales(runtime.obstacle.scale)
    fields.parent = joint.parent
    fields.joint = joint.name
    fields.type = JOINT_CODES[joint.type]
    fields.origin = geometry_msgs.msg.Point(x=float(origin[0]), y=float(origin[1]), z=float(origin[2]))
    fields.axis = geometry_msgs.msg.Vector3(x=joint.axis.x, y=joint.axis.y, z=joint.axis.z)
    fields.lower, fields.upper = joint.bounds
    fields.value = runtime.current.get(joint.name, 0.0)
    if (values := runtime.dynamics.get(joint.name)) is not None:
        fields.dynamics = dynamics(
            damping=values.damping,
            friction=values.friction,
            armature=values.armature,
            effort=values.effort,
            stiffness=values.stiffness,
            drive_damping=values.drive_damping,
            spring_stiffness=values.spring_stiffness,
            spring_rest=values.spring_rest,
        )
    return message


def surface(material: Material) -> tuple[float, float, float]:
    """Static and dynamic friction coefficients and restitution of a link material, an unset friction taking the other's value, else SURFACE_FRICTION, an unset restitution 0."""
    static = material.static_friction if material.static_friction is not None else material.dynamic_friction
    dynamic = material.dynamic_friction if material.dynamic_friction is not None else static
    return (
        SURFACE_FRICTION if static is None else static,
        SURFACE_FRICTION if dynamic is None else dynamic,
        0.0 if material.restitution is None else material.restitution,
    )


def command_fields(command: JointCommand) -> dict[str, list]:
    """A simulator JointCommand message's per-joint lists over the object's movable joints."""
    joints = [joint.name for joint in command.runtime.articulation.movable]
    return {
        "joints": joints,
        "reference": [command.reference[joint] for joint in joints],
        "drive": [joint in command.drive for joint in joints],
        "place": [joint in command.place for joint in joints],
    }


def place(runtime: ArticulatedObject) -> JointCommand:
    """Every joint set onto its current value at rest, drives off."""
    runtime.placed = True
    runtime.reference = dict(runtime.current)
    return JointCommand(runtime, dict(runtime.current), frozenset(), frozenset(joint.name for joint in runtime.articulation.movable))


def drive(runtime: ArticulatedObject, dt: float) -> JointCommand:
    """Ramp each driven joint's reference toward its target at the joint's speed, at most LEAD seconds ahead of the joint, a joint within its settle gap released onto the target."""
    reference = dict(runtime.current)
    driven: set[str] = set()
    settled: set[str] = set()
    for joint in runtime.articulation.movable:
        name = joint.name
        if name not in runtime.driving:
            continue
        current, target = runtime.current[name], runtime.target[name]
        if abs(joint_gap(joint, current, target)) <= runtime.dynamics[name].settle:
            runtime.driving.discard(name)
            reference[name] = target
            settled.add(name)
            continue
        step = joint.speed * dt
        previous = runtime.reference.get(name, current)
        gap = joint_gap(joint, previous, target)
        ramped = target if abs(gap) <= step else previous + math.copysign(step, gap)
        lead = joint.speed * LEAD
        ahead = joint_gap(joint, current, ramped)
        reference[name] = joint.clamp(current + max(-lead, min(lead, ahead)))
        driven.add(name)
    runtime.reference = reference
    return JointCommand(runtime, reference, frozenset(driven), frozenset(settled))


def take_reading(runtime: ArticulatedObject, values: Values) -> None:
    """Measured joint values in, clamped into each joint's range, the targets of joints no drive holds following them, blocked when a drive lags its reference by the whole lead."""
    movable = [joint for joint in runtime.articulation.movable if joint.name in values]
    measured = {joint.name: joint.clamp(values[joint.name]) for joint in movable}
    runtime.motion = any(abs(joint_gap(joint, runtime.current[joint.name], measured[joint.name])) > MOTION_TOLERANCE for joint in movable)
    runtime.current = {**runtime.current, **measured}
    for joint in movable:
        if joint.name not in runtime.driving:
            runtime.target[joint.name] = measured[joint.name]
    runtime.blocked = any(abs(joint_gap(joint, measured[joint.name], runtime.reference[joint.name])) >= LAG_BLOCKED * joint.speed * LEAD for joint in movable if joint.name in runtime.driving)


async def _drive(mech: MechanismITF, commands: Sequence[JointCommand]) -> None:
    readings = await mech._articulated_drive(commands)
    for command, reading in zip(commands, readings, strict=True):
        if reading is not None:
            take_reading(command.runtime, reading)
        elif command.place:
            command.runtime.placed = False


async def step_articulated(mech: MechanismITF, dt: float, discs: Sequence[Disc]) -> None:
    """Move every articulated object one tick toward its target: drive a physical one's joints and read them back, move a kinematic one's links unless they would enter an agent disc."""
    batch: list[Obstacle] = []
    commands: list[JointCommand] = []
    for runtime in list(mech._articulated.values()):
        if runtime.physical:
            commands.append(drive(runtime, dt) if runtime.placed else place(runtime))
            continue
        if not runtime.placed:
            runtime.placed = True
            batch.extend(runtime.link_obstacles())
            continue
        if not runtime.moving:
            runtime.blocked = False
            continue
        values = runtime.advance(dt)
        before = runtime.link_poses(runtime.current)
        after = runtime.link_poses(values)
        moved = [link for link, pose in after.items() if pose != before[link]]
        footprints = [runtime.footprint(link, after[link]) for link in moved]
        if any(disc_hits(footprint, xy, radius) for footprint in footprints for _name, xy, radius in discs):
            runtime.blocked = True
            continue
        runtime.blocked = False
        runtime.current = values
        batch.extend(runtime.link_obstacle(link, after[link]) for link in moved)
    if batch:
        await mech.obstacle_move(batch)
    if commands:
        await _drive(mech, commands)
