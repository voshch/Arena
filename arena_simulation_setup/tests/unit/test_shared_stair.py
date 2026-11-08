from __future__ import annotations

import itertools
import math
import typing

import attrs
import pytest
from arena_simulation_setup.shared import Placement, Stairs, stair_height
from arena_simulation_setup.shared.world import STAIRS_BASE, STAIRS_BRUSH_DEPTH, STAIRS_CLEARANCE, STAIRS_GAP, STAIRS_HANDRAIL_BAR, STAIRS_LANDING_BACKOFF, STAIRS_NOSING, STAIRS_PLATE, STAIRS_SEAM_INSET, STAIRS_SINK, StairsBox, StairsModel, StairsRail, StairsSide
from arena_simulation_setup.tree.assets.Material import Material
from arena_simulation_setup.tree.assets.Object import ObjectIdentifier
from arena_simulation_setup.utils.geometry import Pose, Position, Vector3


Side = typing.Literal['+x', '-x', '+y', '-y']


def _half(mouth_side: Side = '-x', *, top: bool = False, kind: typing.Literal['stairs', 'escalator'] = 'stairs', to: str = '', style: typing.Literal['steps', 'belt'] = 'steps', speed: float = 0.5, rise: float = 3.0) -> Stairs:
    return Stairs(name='half', position=Position(10.3, 2.0), mouth_side=mouth_side, top=top, kind=kind, to=to, style=style, speed=speed, rise=rise)


def _face(box: StairsBox, x: float, z: float) -> Vector3:
    """Map-frame point of a box at local x along its length and local z."""
    _, pose = box
    offset = pose.orientation * Vector3(x, 0.0, z)
    return Vector3(pose.position.x + offset.x, pose.position.y + offset.y, pose.position.z + offset.z)


def _box_along(stair: Stairs, box: StairsBox) -> float:
    return stair.along(box[1].position.x, box[1].position.y)


def _centers(boxes: list[StairsBox]) -> list[tuple[float, float, float]]:
    return [(pose.position.x, pose.position.y, pose.position.z) for _, pose in boxes]


@pytest.mark.parametrize('dest_side', ['+x', '-y', '+y'])
@pytest.mark.parametrize(('across', 'heading'), [(0.4, 0.0), (-0.3, 0.25), (0.0, -0.4)])
def test_landing_mirrors_the_side_offset_and_keeps_the_heading_relative_to_the_seam(dest_side: Side, across: float, heading: float) -> None:
    source = _half()
    dest = Stairs(name='dest', position=Position(3.7, 2.0), mouth_side=dest_side, top=True)
    point = source.local(source.length - 0.2, across)
    x, y, yaw, _dz = source.landing(dest, point.x, point.y, source.yaw + heading)
    limit = dest.length - STAIRS_SEAM_INSET - STAIRS_LANDING_BACKOFF
    assert dest.across(x, y) == pytest.approx(-across)
    assert limit - dest.pitch < dest.along(x, y) <= limit
    assert math.remainder(yaw - (dest.yaw + math.pi) - heading, math.tau) == pytest.approx(0.0, abs=1e-9)


def test_landing_keeps_the_offset_on_the_walker_side() -> None:
    source = _half()
    dest = Stairs(name='dest', position=Position(3.7, 2.0), mouth_side='-y', top=True)
    point = source.local(source.length - 0.2, 0.4)
    x, y, yaw, _dz = source.landing(dest, point.x, point.y, source.yaw)
    seam = dest.seam
    left = -math.sin(yaw) * (x - seam.x) + math.cos(yaw) * (y - seam.y)
    assert left == pytest.approx(0.4)


@pytest.mark.parametrize(('kind', 'style'), [('stairs', 'steps'), ('escalator', 'steps'), ('escalator', 'belt')])
@pytest.mark.parametrize('upwards', [True, False])
def test_landing_puts_the_walker_on_the_same_spot_of_a_tread_at_the_same_clearance(kind: typing.Literal['stairs', 'escalator'], style: typing.Literal['steps', 'belt'], upwards: bool) -> None:
    low = _half(kind=kind, style=style, to='high' if upwards else '')
    high = Stairs(name='high', position=Position(3.7, 2.0), mouth_side='+x', top=True, kind=kind, style=style, to='' if upwards else 'half')
    source, dest = (low, high) if upwards else (high, low)
    travel, dest_travel = 0.37, 0.91
    along = source.length - STAIRS_SEAM_INSET + 0.05
    point = source.local(along, 0.0)
    x, y, _yaw, dz = source.landing(dest, point.x, point.y, source.yaw, travel, dest_travel)
    landed = dest.along(x, y)
    assert dz == pytest.approx(dest.surface(landed, dest_travel) - source.surface(along, travel))
    assert not dest.crossed(x, y)
    if style == 'belt':
        return
    (start, length), *_ = [span for span in source.spans(travel) if span[0] <= along <= span[0] + span[1]]
    matches = [dest_start + dest_length - landed for dest_start, dest_length in dest.spans(dest_travel) if dest_start <= landed <= dest_start + dest_length]
    assert along - start == pytest.approx(matches[0])
    assert length == pytest.approx(dest.pitch)


def test_crossed_starts_at_the_trigger_line_inside_the_half() -> None:
    half = _half()
    trigger = half.length - STAIRS_SEAM_INSET
    for along, across, expected in [(trigger - 0.01, 0.0, False), (trigger + 0.01, 0.0, True), (half.length, 0.69, True), (trigger + 0.01, 0.71, False), (half.length + 0.01, 0.0, False)]:
        point = half.local(along, across)
        assert half.crossed(point.x, point.y) is expected, (along, across)


@pytest.mark.parametrize('top', [False, True])
def test_height_reaches_half_the_rise_at_the_seam(top: bool) -> None:
    half = _half(top=top, rise=3.0)
    sign = -1.0 if top else 1.0
    assert (half.height(0.0), half.height(half.length), half.height(half.length * 2)) == pytest.approx((0.0, sign * 1.5, sign * 1.5))
    assert math.tan(half.slope) == pytest.approx(sign * 1.5 / half.length)


def _corners(box: StairsBox) -> list[Vector3]:
    (sx, sy, sz), _ = box
    return [_face_3d(box, x, y, z) for x in (-sx / 2.0, sx / 2.0) for y in (-sy / 2.0, sy / 2.0) for z in (-sz / 2.0, sz / 2.0)]


def _face_3d(box: StairsBox, x: float, y: float, z: float) -> Vector3:
    _, pose = box
    offset = pose.orientation * Vector3(x, y, z)
    return Vector3(pose.position.x + offset.x, pose.position.y + offset.y, pose.position.z + offset.z)


@pytest.mark.parametrize('style', ['steps', 'belt'])
@pytest.mark.parametrize('whole', [False, True])
def test_every_tread_stays_on_the_stair_and_a_wrapping_one_slides_out_from_under_its_plate(style: typing.Literal['steps', 'belt'], whole: bool) -> None:
    half = _half(kind='escalator', to='other', style=style)
    stair = half
    if whole:
        half.joined = True
        stair = half.whole_stair()
    for travel in (0.0, 0.13, 0.71, 2.5 * stair.pitch):
        starts = stair.tread_starts(travel)
        assert len(starts) == stair.tread_count
        for start in starts:
            assert -stair.pitch - 1e-9 <= start <= stair.length - STAIRS_GAP + 1e-9
        hidden = [start for start in starts if stair.under_plate(start)]
        for start in starts:
            assert stair.under_plate(start) == (start < STAIRS_GAP + (stair.pitch / 2.0 if whole else 0.0) - 1e-9 or start + stair.pitch > stair.length - STAIRS_GAP - (stair.pitch / 2.0 if whole else -STAIRS_GAP) + 1e-9)
        assert len(hidden) <= 2
        covered = sorted((start, start + stair.pitch) for start in starts)
        assert all(b[0] - a[1] < 1e-9 for a, b in itertools.pairwise(covered))
    run = stair.run
    assert _centers(stair.treads(0.4 + run)) == [pytest.approx(center) for center in _centers(stair.treads(0.4))]


def _base_top(half: Stairs, along: float) -> float:
    tops = []
    for box in half.base():
        (span, _, thickness), _ = box
        low, high = _face(box, -span / 2.0, thickness / 2.0), _face(box, span / 2.0, thickness / 2.0)
        a0, a1 = half.along(low.x, low.y), half.along(high.x, high.y)
        if min(a0, a1) - 1e-9 <= along <= max(a0, a1) + 1e-9:
            tops.append(low.z + (along - a0) * (high.z - low.z) / (a1 - a0))
    return max(tops, default=-math.inf)


def _profile(half: Stairs, box: StairsBox) -> list[tuple[float, float]]:
    (sx, _, sz), _ = box
    return [(half.along(point.x, point.y), point.z) for point in (_face(box, x, z) for x, z in ((-sx / 2.0, -sz / 2.0), (sx / 2.0, -sz / 2.0), (sx / 2.0, sz / 2.0), (-sx / 2.0, sz / 2.0)))]


def _profiles_overlap(half: Stairs, a: StairsBox, b: StairsBox) -> bool:
    """Whether two boxes' side profiles intersect, by separating axes."""
    pa, pb = _profile(half, a), _profile(half, b)
    for poly in (pa, pb):
        for (x0, z0), (x1, z1) in itertools.pairwise([*poly, poly[0]]):
            nx, nz = z0 - z1, x1 - x0
            ra, rb = [x * nx + z * nz for x, z in pa], [x * nx + z * nz for x, z in pb]
            if max(ra) <= min(rb) + 1e-12 or max(rb) <= min(ra) + 1e-12:
                return False
    return True


@pytest.mark.parametrize('top', [False, True])
@pytest.mark.parametrize('style', ['steps', 'belt'])
def test_moving_treads_never_touch_the_base_the_floor_or_each_other(top: bool, style: typing.Literal['steps', 'belt']) -> None:
    half = Stairs(name='half', position=Position(10.3, 2.0), top=top, kind='escalator', to='other', style=style, size=[3.4, 1.0])
    for step in range(12):
        travel = step * half.pitch / 5.0
        boxes = half.treads(travel)
        ordered = sorted(boxes, key=lambda box: half.along(box[1].position.x, box[1].position.y))
        assert not any(_profiles_overlap(half, a, b) for a, b in itertools.pairwise(ordered))
        for box, start in zip(boxes, half.tread_starts(travel), strict=True):
            for corner in _corners(box):
                assert corner.z > _base_top(half, half.along(corner.x, corner.y))
                if half.under_plate(start):
                    assert corner.z <= half.height(0.0) - STAIRS_SINK + 1e-9


@pytest.mark.parametrize('top', [False, True])
@pytest.mark.parametrize('style', ['steps', 'belt'])
def test_escalator_treads_and_planks_run_level_with_the_floor_over_the_landing(top: bool, style: typing.Literal['steps', 'belt']) -> None:
    half = Stairs(name='half', position=Position(10.3, 2.0), top=top, kind='escalator', to='other', style=style, size=[3.4, 1.0])
    assert half.flat == half.landing_length == 0.8
    assert (half.height(0.0), half.height(half.flat), half.height(half.length)) == pytest.approx((0.0, 0.0, -1.5 if top else 1.5))
    assert math.tan(abs(half.slope)) == pytest.approx(1.5 / (half.length - half.flat))
    level = 0
    for box, start in zip(half.treads(0.37), half.tread_starts(0.37), strict=True):
        if not half.under_plate(start) and start + half.pitch <= half.flat:
            level += 1
            assert half.surface(start + half.pitch / 2.0, 0.37) == pytest.approx(0.0)
            assert max(corner.z for corner in _corners(box)) == pytest.approx(0.0)
    assert level
    assert _half().flat == 0.0


@pytest.mark.parametrize('top', [False, True])
def test_treads_and_belt_planks_carry_the_walking_surface(top: bool) -> None:
    steps = _half(kind='escalator', to='other', top=top)
    for (size, pose), start in zip(steps.treads(0.3), steps.tread_starts(0.3), strict=True):
        if steps.under_plate(start):
            assert pose.position.z + size[2] / 2.0 == pytest.approx(max(steps.height(start), steps.height(start + steps.pitch)) - STAIRS_SINK)
            continue
        assert size == pytest.approx((steps.pitch - STAIRS_GAP, steps.width - 2.0 * STAIRS_CLEARANCE, steps.riser - STAIRS_GAP))
        assert pose.position.z + size[2] / 2.0 == pytest.approx(steps.surface(start + steps.pitch / 2.0, 0.3))
    belt = _half(kind='escalator', to='other', style='belt', top=top)
    for box, start in zip(belt.treads(0.3), belt.tread_starts(0.3), strict=True):
        if belt.under_plate(start):
            continue
        back = _face(box, -box[0][0] / 2.0, belt.plank / 2.0)
        assert box[0][2] == belt.plank
        assert belt.along(back.x, back.y) == pytest.approx(start)
        assert back.z == pytest.approx(belt.surface(start))


def _quaternion(pose: Pose) -> tuple[float, float, float, float]:
    return pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w


@pytest.mark.parametrize('top', [False, True])
@pytest.mark.parametrize('style', ['steps', 'belt'])
@pytest.mark.parametrize('boards', [True, False])
def test_tread_velocities_match_the_motion_of_the_treads(top: bool, style: typing.Literal['steps', 'belt'], boards: bool) -> None:
    half = Stairs(name='half', position=Position(10.3, 2.0), mouth_side='+y', top=top, kind='escalator', to='other' if boards else '', style=style, speed=0.6, size=[3.4, 1.0])
    dt = 1e-3
    for travel in (0.13, 0.71, 1.93):
        before, after = half.treads(travel), half.treads(travel + half.belt * dt)
        moving = zip(before, after, half.tread_velocities(travel), half.tread_starts(travel), half.tread_starts(travel + half.belt * dt), strict=True)
        checked = 0
        for (size, pose), (resized, moved), velocity, start, then in moving:
            if abs(then - start) > 1.0 or half.under_plate(start) != half.under_plate(then):
                continue
            if size != pytest.approx(resized, abs=1e-12) or _quaternion(pose) != pytest.approx(_quaternion(moved), abs=1e-12):
                continue
            checked += 1
            step = ((moved.position.x - pose.position.x) / dt, (moved.position.y - pose.position.y) / dt, (moved.position.z - pose.position.z) / dt)
            assert velocity == pytest.approx(step, abs=1e-6)
            assert (velocity[0], velocity[1]) == pytest.approx((0.0, half.belt * half.axis[1]))
        assert checked


@pytest.mark.parametrize('top', [False, True])
@pytest.mark.parametrize('mouth_side', ['-x', '+y'])
@pytest.mark.parametrize('style', ['steps', 'belt'])
def test_base_runs_under_the_ramp_and_the_landing_from_mouth_to_seam(top: bool, mouth_side: Side, style: typing.Literal['steps', 'belt']) -> None:
    half = Stairs(name='half', position=Position(10.3, 2.0), mouth_side=mouth_side, top=top, kind='escalator', style=style, size=[3.4, 1.0])
    slabs = half.base()
    assert len(slabs) == 2
    drop = half.drop
    ends = []
    for index, box in enumerate(slabs):
        (span, width, thickness), _ = box
        assert (width, thickness) == (half.width, STAIRS_BASE)
        for x in (-span / 2.0, 0.0, span / 2.0):
            point = _face(box, x, thickness / 2.0)
            along = half.along(point.x, point.y)
            ends.append(along)
            assert abs(half.across(point.x, point.y)) == pytest.approx(0.0, abs=1e-9)
            expected = (along - half.flat) * math.tan(half.slope) - drop if index == 0 else -drop
            assert point.z == pytest.approx(expected, abs=1e-6)
    assert min(ends) <= 1e-6
    assert max(ends) == pytest.approx(half.length)


def test_belt_runs_towards_the_seam_on_the_boarding_half_only() -> None:
    assert _half(to='other').belt == 0.0
    assert _half(kind='escalator', to='other', speed=0.6).belt == 0.6
    assert _half(kind='escalator', speed=0.6).belt == -0.6


@pytest.mark.parametrize('top', [False, True])
def test_steps_rise_or_sink_to_the_seam(top: bool) -> None:
    half = _half(top=top)
    tops = [pose.position.z + size[2] / 2.0 for size, pose in half.steps()]
    assert tops == pytest.approx([half.height((i if top else i + 1) * half.pitch) for i in range(len(tops))])
    assert len(tops) == half.tread_count
    assert [half.surface((i + 0.5) * half.pitch) for i in range(len(tops))] == pytest.approx(tops)


@pytest.mark.parametrize('top', [False, True])
def test_walls_line_the_sides_under_the_walking_surface_raise_side_walls_above_it_and_close_the_seam(top: bool) -> None:
    half = _half(top=top)
    *sides, seam = half.walls()
    assert all(abs(abs(half.across(point.x, point.y)) - half.width / 2.0) < 1e-9 for wall in sides for point in (wall.start, wall.end))
    lining = [wall for wall in sides if wall.bottom is not None]
    above = [wall for wall in sides if wall.bottom is None]
    for wall in lining:
        assert wall.material is half.material_of('fixture')
        assert wall.top == 0.0
        assert wall.bottom == (-half.rise / 2.0 if top else pytest.approx(-(half.rise / 2.0 + half.drop + STAIRS_BASE)))
        assert all(point.z == pytest.approx(0.0 if top else half.height(half.along(point.x, point.y))) for point in (wall.start, wall.end))
    for across in (half.width / 2.0, -half.width / 2.0):
        side = [wall for wall in above if half.across(wall.start.x, wall.start.y) == pytest.approx(across)]
        assert half.along(side[0].start.x, side[0].start.y) == pytest.approx(0.0)
        assert half.along(side[-1].end.x, side[-1].end.y) == pytest.approx(half.length)
        for wall in side:
            assert wall.height == 1.0
            assert all(point.z == pytest.approx(0.0 if top else half.height(half.along(point.x, point.y))) for point in (wall.start, wall.end))
    assert (seam.start, seam.end) == (Position(11.6, 1.3), Position(11.6, 2.7))
    assert seam.top == pytest.approx(half.height(half.length) + 1.0)


def test_a_joined_half_drops_its_seam_wall_and_runs_its_treads_to_the_seam() -> None:
    half = _half(kind='escalator', to='other')
    open_walls = len(half.walls())
    half.joined = True
    assert len(half.walls()) == open_walls - 1
    assert half.run == pytest.approx(half.length - 2.0 * STAIRS_GAP)
    starts = [start for start in half.tread_starts(0.0) if start is not None]
    assert max(starts) + half.pitch == pytest.approx(half.length - STAIRS_GAP)


@pytest.mark.parametrize('travel', [0.0, 0.07, 0.15, 0.23])
def test_a_joined_escalator_carries_its_riders_on_the_treads_of_its_whole_body_across_the_seam(travel: float) -> None:
    low = Stairs(name='low', group='esc', kind='escalator', position=Position(10.3, 2.0), to='high', rise=3.0)
    high = Stairs(name='high', group='esc', kind='escalator', position=Position(12.9, 2.0), mouth_side='+x', top=True, rise=3.0)
    low.joined = high.joined = True
    body = low.whole_stair()
    moved = {'low': travel, 'high': 0.5}
    for along in [low.flat + 0.05 + 0.02 * i for i in range(int((body.length - 2.0 * low.flat - 0.1) / 0.02))]:
        x = low.mouth.x + along
        level = 0.0 if along <= low.length + low.reach else low.rise
        assert stair_height((low, high), x, 2.0, moved) + level >= body.height(along) - 1e-9
        assert stair_height((low, high), x, 2.0, moved) + level == pytest.approx(body.footing(along, travel))


def test_stair_geometry_rides_on_the_position_height() -> None:
    low = _half(kind='escalator', to='other')
    high = Stairs(name='half', position=Position(10.3, 2.0, 3.0), kind='escalator', to='other')
    for (_, a), (_, b) in zip(low.treads(0.4), high.treads(0.4), strict=True):
        assert b.position.z - a.position.z == pytest.approx(3.0)
    assert [box[1].position.z + 3.0 for box in _half().steps()] == pytest.approx([box[1].position.z for box in Stairs(name='s', position=Position(10.3, 2.0, 3.0)).steps()])


def test_placement_shifts_each_level_from_its_abstract_origin_to_its_offset() -> None:
    placement = Placement(origins={'0': (0.0, 0.0), '1': (17.0, 0.0)}, offsets={'0': (0.0, 0.0, 0.0), '1': (9.2, 0.0, 3.0)}, bounds={'0': (0.0, 0.0, 14.0, 10.0), '1': (0.0, 0.0, 14.0, 10.0)}, env=(5.0, 5.0))
    assert placement.stacked
    assert placement.level_at(5.0 + 17.0 + 3.0, 5.0 + 2.0) == '1'
    assert placement.level_at(5.0 + 3.0, 5.0 + 2.0) == '0'
    assert placement.level_at(5.0 + 15.5, 5.0 + 2.0) == ''
    abstract = Position(5.0 + 17.0 + 3.0, 7.0, 0.3)
    placed = placement.place(abstract)
    assert (placed.x, placed.y, placed.z) == pytest.approx((5.0 + 9.2 + 3.0, 7.0, 3.3))
    back = placement.unplace(placed, '1')
    assert (back.x, back.y, back.z) == pytest.approx((abstract.x, abstract.y, abstract.z))
    assert placement.place(Position(8.0, 7.0, 0.0), '0') == Position(8.0, 7.0, 0.0)
    spread = Placement(origins={'0': (0.0, 0.0), '1': (17.0, 0.0)}, offsets={'0': (0.0, 0.0, 0.0), '1': (17.0, 0.0, 0.0)})
    assert not spread.stacked


def test_tread_depth_sets_the_step_count_and_the_riser() -> None:
    shallow = Stairs(name='s', position=Position(10.3, 2.0), size=[3.6, 1.4], tread=0.3)
    assert shallow.tread_count == 12
    assert shallow.pitch == pytest.approx(0.3)
    assert shallow.riser == pytest.approx(1.5 / 12)
    assert len(shallow.steps()) == 12


def test_side_height_and_plank_thickness_follow_their_fields() -> None:
    sides = {'left': StairsSide(height=1.2), 'right': StairsSide(height=0.0)}
    low = Stairs(name='s', position=Position(10.3, 2.0), sides=sides)
    above = [wall for wall in low.walls()[:-1] if wall.bottom is None]
    assert above and all(wall.height == 1.2 and low.across(wall.start.x, wall.start.y) > 0.0 for wall in above)
    assert low.walls()[-1].top == pytest.approx(low.height(low.length) + 1.2)
    assert [wall.height for wall in Stairs(name='w', position=Position(10.3, 2.0), top=True, sides=sides).walls() if wall.bottom is None] == [1.2]
    belt = Stairs(name='b', position=Position(10.3, 2.0), kind='escalator', style='belt', to='o', plank=0.05)
    assert {box[0][2] for box in belt.treads(0.2)} == {0.05}
    assert belt.surface(1.0) == pytest.approx(belt.height(1.0))


@pytest.mark.parametrize('style', ['steps', 'belt'])
def test_landing_length_sets_the_level_run_of_an_escalator_only(style: typing.Literal['steps', 'belt']) -> None:
    escalator = Stairs(name='e', position=Position(10.3, 2.0), kind='escalator', style=style, size=[4.0, 1.0], landing_length=1.2)
    assert escalator.flat == 1.2
    assert escalator.height(1.2) == 0.0
    assert escalator.height(4.0) == pytest.approx(1.5)
    assert Stairs(name='s', position=Position(10.3, 2.0), landing_length=1.2).flat == 0.0


@pytest.mark.parametrize(('kind', 'style'), [('stairs', 'steps'), ('escalator', 'steps'), ('escalator', 'belt')])
def test_a_whole_stair_runs_the_joined_pair_as_one_body_from_mouth_to_mouth(kind: typing.Literal['stairs', 'escalator'], style: typing.Literal['steps', 'belt']) -> None:
    bottom = Stairs(name='f_bottom', position=Position(7.3, 2.0), kind=kind, style=style, to='f_top')
    bottom.joined = True
    body = bottom.whole_stair()
    assert (body.whole, body.length, body.mouth) == (True, 2.0 * bottom.length, bottom.mouth)
    assert body.height(body.length) == pytest.approx(bottom.rise)
    for along in (0.3, 1.0, 2.0, bottom.length - 0.1):
        assert body.height(along) == pytest.approx(bottom.height(along))
    assert all(abs(abs(body.across(wall.start.x, wall.start.y)) - body.width / 2.0) < 1e-9 for wall in body.walls())
    if kind == 'stairs':
        tops = [pose.position.z + size[2] / 2.0 for size, pose in body.steps()]
        assert tops[-1] == pytest.approx(bottom.rise)
        assert all(b > a for a, b in itertools.pairwise(tops))
        return
    starts = [start for start in body.tread_starts(0.37) if start is not None]
    assert min(starts) >= -body.pitch - 1e-9
    assert max(starts) + body.pitch <= body.length - STAIRS_GAP + body.pitch / 2.0 + 1e-9
    assert len(body.treads(0.37)) == body.tread_count
    assert body.height(body.length - body.flat) == pytest.approx(bottom.rise)
    assert len(body.base()) == 3


def test_landing_plates_lie_flush_with_the_floor_over_the_pits_and_the_wrap() -> None:
    half = Stairs(name='e', position=Position(10.3, 2.0), kind='escalator', size=[3.4, 1.0], to='o')
    ((size, pose),) = half.plates()
    reach = half.pitch + STAIRS_GAP
    assert size == pytest.approx((reach + STAIRS_GAP, 1.0, STAIRS_PLATE))
    assert half.along(pose.position.x, pose.position.y) == pytest.approx((STAIRS_GAP - reach) / 2.0)
    assert pose.position.z + size[2] / 2.0 == pytest.approx(half.height(0.0))
    half.joined = True
    body = half.whole_stair()
    near, far = body.plates()
    assert near[1].position.z + STAIRS_PLATE / 2.0 == pytest.approx(0.0)
    assert far[1].position.z + STAIRS_PLATE / 2.0 == pytest.approx(body.height(body.length))
    assert body.along(far[1].position.x, far[1].position.y) + far[0][0] / 2.0 == pytest.approx(body.length + body.pitch + STAIRS_GAP)
    assert Stairs(name='s', position=Position(10.3, 2.0)).plates() == []


@pytest.mark.parametrize('top', [False, True])
def test_rails_follow_the_walking_surface_and_only_solid_ones_block_the_floor_plan(top: bool) -> None:
    rails = [StairsRail(across=0.62, material=Material.default('wall')), StairsRail(across=0.0, height=1.0, solid=True), StairsRail(across=-0.62, kind='steel_handrail')]
    half = Stairs(name='e', position=Position(10.3, 2.0), kind='escalator', size=[3.4, 1.4], top=top, rails=rails)
    rails_open = [wall for wall in half.open_walls() if wall.material is not half.material_of('brushes')]
    bar, preset = rails_open[0], rails_open[-1]
    solid = [wall for wall in half.walls() if wall.top == 1.0 and abs(half.across(wall.start.x, wall.start.y)) < 1e-9]
    assert len(rails_open) == 2 * len(solid) == 2 * len(half._pieces())
    assert (bar.top, bar.bottom, preset.height, preset.bottom, preset.kind) == (0.9, pytest.approx(0.9 - STAIRS_HANDRAIL_BAR), 0.9, None, 'steel_handrail')
    assert half.across(bar.start.x, bar.start.y) == pytest.approx(0.62)
    for wall in [*rails_open, *solid]:
        assert all(point.z == pytest.approx(half.height(half.along(point.x, point.y))) for point in (wall.start, wall.end))


def test_steps_split_into_a_tread_plate_on_a_riser_only_when_risers_take_their_own_material() -> None:
    plain = _half()
    assert [(name, box) for name, _material, box in plain.static_boxes()] == [(f'step_{i}', box) for i, box in enumerate(plain.steps())]
    oak, white = Material.default('floor'), Material.default('wall')
    dressed = attrs.evolve(plain, surfaces={'steps': oak, 'risers': white})
    boxes = dressed.static_boxes()
    assert [name for name, _material, _box in boxes] == [name for i in range(len(plain.steps())) for name in (f'step_{i}', f'riser_{i}')]
    for (size, pose), (_, tread_material, (tread_size, tread_pose)), (_, riser_material, (riser_size, riser_pose)) in zip(plain.steps(), boxes[::2], boxes[1::2], strict=True):
        assert (tread_material, riser_material) == (oak, white)
        assert tread_size[2] == pytest.approx(STAIRS_NOSING)
        assert tread_size[2] + riser_size[2] == pytest.approx(size[2])
        assert tread_pose.position.z + tread_size[2] / 2.0 == pytest.approx(pose.position.z + size[2] / 2.0)
        assert riser_pose.position.z - riser_size[2] / 2.0 == pytest.approx(pose.position.z - size[2] / 2.0)


def test_an_escalator_dresses_its_base_as_fixture_and_its_plates_as_landings() -> None:
    steel, granite = Material.default('wall'), Material.default('floor')
    half = Stairs(name='e', position=Position(10.3, 2.0), kind='escalator', size=[3.4, 1.0], to='o', surfaces={'fixture': steel, 'landings': granite})
    boxes = half.static_boxes()
    assert [box for name, _material, box in boxes if name.startswith('base_')] == half.base()
    assert [box for name, _material, box in boxes if name.startswith('plate_')] == half.plates()
    assert {name.split('_')[0]: material for name, material, _box in boxes} == {'base': steel, 'plate': granite}
    assert half.material_of('steps') is half.material


@pytest.mark.parametrize('style', ['steps', 'belt'])
@pytest.mark.parametrize('whole', [False, True])
def test_a_tread_under_a_plate_slides_out_one_millimetre_below_the_floor(style: typing.Literal['steps', 'belt'], whole: bool) -> None:
    half = Stairs(name='e', position=Position(10.3, 2.0), kind='escalator', style=style, size=[3.4, 1.0], to='o')
    if whole:
        half.joined = True
        half = half.whole_stair()
    assert STAIRS_SINK == STAIRS_GAP == 0.001
    under = [box for travel in (0.0, half.pitch / 3.0, half.pitch / 2.0) for box, start in zip(half.treads(travel), half.tread_starts(travel), strict=True) if half.under_plate(start)]
    assert under
    for box in under:
        level = half.height(0.0) if half.along(box[1].position.x, box[1].position.y) < half.length / 2.0 else half.height(half.length)
        assert max(corner.z for corner in _corners(box)) == pytest.approx(level - STAIRS_SINK)


@pytest.mark.parametrize('style', ['steps', 'belt'])
def test_escalator_brushes_run_along_both_inner_sides_above_the_treads_and_stairs_have_none(style: typing.Literal['steps', 'belt']) -> None:
    half = Stairs(name='e', position=Position(10.3, 2.0), kind='escalator', style=style, size=[3.4, 1.0], to='o')
    brushes = half.open_walls()
    assert len(brushes) == 2 * len(half._pieces())
    assert all(wall.material is half.material_of('brushes') and wall.material.name == 'Carpet_Charcoal' for wall in brushes)
    assert {round(half.across(wall.start.x, wall.start.y), 9) for wall in brushes} == {round(0.5 - STAIRS_BRUSH_DEPTH, 9), round(STAIRS_BRUSH_DEPTH - 0.5, 9)}
    for travel in (0.0, half.pitch / 3.0):
        for size, pose in half.treads(travel):
            along = half.along(pose.position.x, pose.position.y)
            assert pose.position.z + size[2] / 2.0 < half.height(along) + brushes[0].bottom
    assert Stairs(name='s', position=Position(10.3, 2.0)).open_walls() == []


def test_a_surface_model_sits_on_top_of_every_box_of_that_surface() -> None:
    capture = ObjectIdentifier('tread_capture')
    plain = _half()
    dressed = attrs.evolve(plain, surface_models={'steps': capture})
    placed = dressed.static_models()
    assert [name for name, _model, _pose in placed] == [f'model_step_{i}' for i in range(len(plain.steps()))]
    for (_name, model, pose), (size, box) in zip(placed, plain.steps(), strict=True):
        assert model is capture
        assert pose.position.z == pytest.approx(box.position.z + size[2] / 2.0)
        assert (pose.position.x, pose.position.y) == pytest.approx((box.position.x, box.position.y))
    escalator = Stairs(name='e', position=Position(10.3, 2.0), kind='escalator', size=[3.4, 1.0], to='o', surface_models={'steps': capture, 'fixture': capture})
    travel = escalator.pitch / 3.0
    assert [pose for pose in escalator.tread_models(travel)] == [Pose(position=box.position + box.orientation * Position(0.0, 0.0, size[2] / 2.0), orientation=box.orientation) for size, box in escalator.treads(travel)]
    assert len([name for name, _model, _pose in escalator.static_models() if name.startswith('model_base_')]) == len(escalator.base())
    assert _half().tread_models(0.0) == []


def test_staircase_models_fall_on_the_half_they_stand_over_and_all_on_a_whole_staircase() -> None:
    models = [StairsModel(model='low_sign', at=(1.0, 0.2, 1.5)), StairsModel(model='high_sign', at=(4.0, 0.2, 2.5), yaw=0.5)]
    bottom = attrs.evolve(_half(to='half_top'), models=models)
    top = attrs.evolve(_half('+x', top=True, to='half'), models=models)
    assert [name for name, _model, _pose in bottom.static_models()] == ['model_0']
    ((name, _model, pose),) = top.static_models()
    assert name == 'model_1'
    assert top.along(pose.position.x, pose.position.y) == pytest.approx(2.0 * top.length - 4.0)
    assert top.across(pose.position.x, pose.position.y) == pytest.approx(-0.2)
    assert pose.position.z == pytest.approx(2.5 - top.rise)
    bottom.joined = True
    whole = bottom.whole_stair()
    placed = {name: pose for name, _model, pose in whole.static_models()}
    assert set(placed) == {'model_0', 'model_1'}
    assert whole.along(placed['model_1'].position.x, placed['model_1'].position.y) == pytest.approx(4.0)
    assert placed['model_1'].position.z == pytest.approx(2.5)
