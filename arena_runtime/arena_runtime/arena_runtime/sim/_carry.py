"""Carry policy for every simulator: an object a closed gripper holds is welded to the hand by the simulator, or follows the gripper's TF frame on the mechanism tick without a weld."""

from __future__ import annotations

import math
import typing
from collections.abc import Mapping, Sequence

import attrs
import yaml

from arena_runtime.shared import WORLD_ENTITY_PREFIX, Obstacle, Orientation, Pose, Position, Robot
from arena_runtime.tf_frames import frames_below

from ._interface import resolve_obstacle_box

if typing.TYPE_CHECKING:
    from ._interface import MechanismITF

CARRY_REACH = 0.05
POINT_REACH = 0.3
CARRY_EPSILON = 1e-3
SUPPORT_SLACK = 0.02
SUPPORT_RADIUS = 3.0


@attrs.define
class Carry:
    """One gripper's held object: its authored name, the obstacle as last placed, its pose in the hand frame and whether the simulator welds it."""

    robot: str
    sim_path: str
    joint: str
    obj: str
    hand: str
    obstacle: Obstacle
    offset: Pose
    welded: bool = False


def box_gap(point: Position, pose: Pose, box: tuple[tuple[float, float, float], tuple[float, float, float]]) -> float:
    """Distance from a map point to the oriented box (size, center) given in the frame of pose, 0 inside."""
    local = pose.inverse() * point
    size, center = box
    outside = [max(abs(p - c) - s / 2, 0.0) for p, c, s in zip(local, center, size, strict=True)]
    return math.hypot(*outside)


async def object_gap(points: Sequence[Position], obstacle: Obstacle) -> float:
    """Distance from the nearest map point to the obstacle's annotated box, or to its origin less the point reach slack without one."""
    box = await resolve_obstacle_box(obstacle)
    if box is None:
        return min(max(math.dist(tuple(point), tuple(obstacle.pose.position)) - (POINT_REACH - CARRY_REACH), 0.0) for point in points)
    return min(box_gap(point, obstacle.pose, box) for point in points)


async def _bbox(obstacle: Obstacle) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    try:
        view = await obstacle.model.resolve()
    except FileNotFoundError:
        return None
    return view.bbox


def _robot_floor(mech: MechanismITF, hand: str) -> float:
    """Map height of the TF root of the robot that owns the hand frame, 0 when no robot frame lies above the hand."""
    buffer = mech._mechanism_tf_buffer
    if buffer is None:
        return 0.0
    parents = {str(name): str(info.get('parent', '')) for name, info in (yaml.safe_load(buffer.all_frames_as_yaml()) or {}).items()}
    prefix = f"{hand.rsplit('/', 1)[0]}/"
    root = hand
    while parents.get(root, '').startswith(prefix):
        root = parents[root]
    pose = None if root == hand else mech.frame_transform(root)
    return 0.0 if pose is None else pose.position.z


async def settle(mech: MechanismITF, obstacle: Obstacle, floor: float) -> Obstacle:
    """The obstacle upright at its yaw, resting on the highest annotated top under its box center that is not above its bottom, or on the floor."""
    upright = Pose(obstacle.pose.position, Orientation.from_yaw(obstacle.pose.orientation.to_yaw()))
    box = await _bbox(obstacle)
    center = upright.position if box is None else upright * Position(*box[1])
    bottom = upright.position.z if box is None else center.z - box[0][2] / 2
    rest = floor
    for other in list(mech._carry_objects.values()):
        if other.name == obstacle.name or math.hypot(other.pose.position.x - center.x, other.pose.position.y - center.y) > SUPPORT_RADIUS:
            continue
        support = await _bbox(other)
        if support is None:
            continue
        (sx, sy, sz), (cx, cy, cz) = support
        local = other.pose.inverse() * center
        if abs(local.x - cx) > sx / 2 or abs(local.y - cy) > sy / 2:
            continue
        top = other.pose.position.z + cz + sz / 2
        if rest < top <= bottom + SUPPORT_SLACK:
            rest = top
    position = upright.position
    return attrs.evolve(obstacle, pose=Pose(Position(position.x, position.y, position.z + rest - bottom), upright.orientation))


def _moved(a: Pose, b: Pose) -> bool:
    turn = abs(sum(x * y for x, y in zip(a.orientation, b.orientation, strict=True)))
    return math.dist(tuple(a.position), tuple(b.position)) > CARRY_EPSILON or 1.0 - turn > CARRY_EPSILON


def remember(mech: MechanismITF, obstacles: Sequence[Obstacle]) -> None:
    """Record where each obstacle was spawned or moved to."""
    for obstacle in obstacles:
        mech._carry_objects[obstacle.name] = obstacle


def forget(mech: MechanismITF, obstacles: Sequence[Obstacle]) -> None:
    """Drop deleted obstacles and the carries holding them."""
    names = {obstacle.name for obstacle in obstacles}
    for name in names:
        mech._carry_objects.pop(name, None)
        mech._carry_resting.pop(name, None)
    for key, carry in list(mech._carried.items()):
        if carry.obstacle.name in names:
            del mech._carried[key]


def forget_robot(mech: MechanismITF, sim_path: str) -> None:
    """Drop the carries of a deleted robot, leaving its objects where they are."""
    for key, carry in list(mech._carried.items()):
        if carry.sim_path == sim_path:
            del mech._carried[key]


async def attach(mech: MechanismITF, robot: Robot, joint: str, obj: str, hand: str) -> bool:
    """Carry the nearest registered obstacle named obj within CARRY_REACH of the hand frame, False without one or without the frame."""
    from ._mechanism_shim import _ensure_loop

    logger = mech.node.get_logger()
    tip = mech.frame_transform(hand) if hand else None
    if tip is None:
        logger.warning(f"carry: {robot.name!r} has no TF frame {hand!r} in map to grasp {obj!r} with")
        return False
    held = {carry.obstacle.name for carry in mech._carried.values()}
    below = [hand] if mech._mechanism_tf_buffer is None else frames_below(mech._mechanism_tf_buffer, hand)
    points = [pose.position for frame in below if (pose := mech.frame_transform(frame)) is not None]
    best: tuple[float, Obstacle] | None = None
    gaps: dict[str, float] = {}
    for name in (f'{WORLD_ENTITY_PREFIX}{obj}', obj):
        obstacle = mech._carry_objects.get(name)
        if obstacle is None or name in held:
            continue
        gap = gaps[name] = await object_gap(points, obstacle)
        if gap <= CARRY_REACH and (best is None or gap < best[0]):
            best = (gap, obstacle)
    if best is None:
        hand_at = tuple(round(v, 3) for v in tip.position)
        if gaps:
            places = {name: tuple(round(v, 3) for v in mech._carry_objects[name].pose.position) for name in gaps}
            logger.warning(f"carry: {obj!r} out of reach of {hand!r} at {hand_at} and its {len(points) - 1} frames below: gaps {gaps}, objects at {places}, reach {CARRY_REACH} m")
        else:
            logger.warning(f"carry: no spawned obstacle named {obj!r} or {WORLD_ENTITY_PREFIX + obj!r} ({len(mech._carry_objects)} known), hand {hand!r} at {hand_at}")
        return False
    obstacle = best[1]
    mech._carry_resting.setdefault(obstacle.name, obstacle)
    carry = mech._carried[(robot.name, joint)] = Carry(robot=robot.name, sim_path=robot.sim_path, joint=joint, obj=obj, hand=hand, obstacle=obstacle, offset=tip.inverse() * obstacle.pose)
    carry.welded = await mech._carry_weld(carry)
    if not carry.welded:
        await mech._carry_lift([obstacle])
    logger.info(f"carry: {robot.name!r} holds {obstacle.name!r} {'welded' if carry.welded else 'on the mechanism tick'}")
    _ensure_loop(mech)
    return True


async def _unweld(mech: MechanismITF, carry: Carry) -> None:
    """Release a welded carry where it is, its obstacle at the pose the hand frame gives, else where the simulator reports it."""
    pose = await mech._carry_unweld(carry)
    if (tip := mech.frame_transform(carry.hand)) is not None:
        pose = tip * carry.offset
    if pose is not None:
        carry.obstacle = attrs.evolve(carry.obstacle, pose=pose)


async def detach(mech: MechanismITF, robot: Robot, joint: str) -> None:
    """Set what the gripper joint carries down below where it is now, on the support under it or the robot's floor."""
    carry = mech._carried.pop((robot.name, joint), None)
    if carry is None:
        return
    if carry.welded:
        await _unweld(mech, carry)
    carry.obstacle = await settle(mech, carry.obstacle, _robot_floor(mech, carry.hand))
    mech._carry_objects[carry.obstacle.name] = carry.obstacle
    await mech._carry_drop([carry.obstacle])
    mech.node.get_logger().info(f"carry: {robot.name!r} sets {carry.obstacle.name!r} down at {tuple(round(v, 3) for v in carry.obstacle.pose.position)}")


def held(mech: MechanismITF) -> Mapping[str, frozenset[str]]:
    out: dict[str, set[str]] = {}
    for carry in mech._carried.values():
        out.setdefault(carry.robot, set()).add(carry.obj)
    return {robot: frozenset(objects) for robot, objects in out.items()}


def offsets(mech: MechanismITF, robot: str) -> Mapping[str, tuple[str, Pose]]:
    return {carry.obj: (carry.hand, carry.offset) for carry in mech._carried.values() if carry.robot == robot}


async def follow(mech: MechanismITF) -> None:
    """Move every carried obstacle to its hand frame pose, those that moved since the last tick."""
    moved: list[Obstacle] = []
    for carry in mech._carried.values():
        if carry.welded:
            continue
        tip = mech.frame_transform(carry.hand)
        if tip is None:
            continue
        target = tip * carry.offset
        if not _moved(carry.obstacle.pose, target):
            continue
        carry.obstacle = attrs.evolve(carry.obstacle, pose=target)
        mech._carry_objects[carry.obstacle.name] = carry.obstacle
        moved.append(carry.obstacle)
    if moved:
        await mech._carry_follow(moved)


async def release_all(mech: MechanismITF) -> None:
    """Drop every carry and put each obstacle carried since the last call back where it rested before."""
    for carry in [carry for carry in mech._carried.values() if carry.welded]:
        await _unweld(mech, carry)
    mech._carried.clear()
    resting = list(mech._carry_resting.values())
    mech._carry_resting.clear()
    for obstacle in resting:
        mech._carry_objects[obstacle.name] = obstacle
    if resting:
        await mech._carry_drop(resting)
