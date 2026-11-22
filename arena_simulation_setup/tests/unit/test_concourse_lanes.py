"""Unit tests for concourse flow lane blocks written into the scenario."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from itertools import permutations
from typing import Any

import numpy as np
import pydantic
import pytest
import yaml
from arena_humansim.utils.lanes import build_track
from arena_humansim.utils.types import Lane, LaneSide, Track
from arena_simulation_setup.generative.world.concourse import ConcourseConfiguration, WorldGeneratorConcourse
from arena_simulation_setup.tree.World.lane_check import build_scenario_lanes
from numpy.typing import NDArray


def _scenario(config: dict[str, Any]) -> tuple[WorldGeneratorConcourse, dict[str, Any]]:
    gen = WorldGeneratorConcourse(config, random.Random(42))
    gen.compute()
    files = gen.files()
    return gen, yaml.safe_load(files["scenarios/default/scenario.yaml"])


def _lane_of(scenario: dict[str, Any], region: str) -> dict[str, Any]:
    lane = scenario["regions"][region]["config"].get("lane")
    assert lane is not None
    return lane


def test_flow_through_two_branches_has_three_centerline_segments() -> None:
    config = {
        "length": 30.0,
        "width": 4.0,
        "branches": [
            {"at_s": 8.0, "side": "left", "width": 3.0, "length": 6.0},
            {"at_s": 20.0, "side": "right", "width": 3.0, "length": 6.0},
        ],
        "flows": [
            {
                "name": "cross",
                "source_dock": "dock_branch_0_left",
                "sink_dock": "dock_branch_1_right",
                "lane": {"side": "right"},
            }
        ],
    }
    _gen, scenario = _scenario(config)
    lane = _lane_of(scenario, "dock_branch_0_left_source_cross")
    assert len(lane["segments"]) == 3


def test_lane_block_without_a_side_is_rejected() -> None:
    try:
        ConcourseConfiguration.model_validate({"flows": [{"name": "no_side", "lane": {"offset": 0.5}}]})
    except pydantic.ValidationError:
        return
    raise AssertionError("a lane block without a side was accepted")


def test_opposed_flows_with_lanes_are_marked_two_way() -> None:
    config = {
        "length": 30.0,
        "width": 4.0,
        "flows": [
            {"name": "east", "source_dock": "dock_start", "sink_dock": "dock_end", "lane": {"side": "right"}},
            {"name": "west", "source_dock": "dock_end", "sink_dock": "dock_start", "lane": {"side": "right"}},
        ],
    }
    _gen, scenario = _scenario(config)
    assert _lane_of(scenario, "dock_start_source_east")["two_way"] is True
    assert _lane_of(scenario, "dock_end_source_west")["two_way"] is True


def test_partly_shared_opposite_stretch_marks_both_flows_two_way() -> None:
    config = {
        "length": 30.0,
        "width": 4.0,
        "branches": [
            {"at_s": 8.0, "side": "left", "width": 3.0, "length": 6.0},
            {"at_s": 20.0, "side": "right", "width": 3.0, "length": 6.0},
        ],
        "flows": [
            {"name": "cross", "source_dock": "dock_branch_0_left", "sink_dock": "dock_branch_1_right", "lane": {"side": "right"}},
            {"name": "west", "source_dock": "dock_end", "sink_dock": "dock_start", "lane": {"side": "right"}},
        ],
    }
    _gen, scenario = _scenario(config)
    assert _lane_of(scenario, "dock_branch_0_left_source_cross")["two_way"] is True
    assert _lane_of(scenario, "dock_end_source_west")["two_way"] is True


def test_partly_shared_stretch_in_the_same_direction_stays_one_way() -> None:
    config = {
        "length": 30.0,
        "width": 4.0,
        "branches": [
            {"at_s": 8.0, "side": "left", "width": 3.0, "length": 6.0},
            {"at_s": 20.0, "side": "right", "width": 3.0, "length": 6.0},
        ],
        "flows": [
            {"name": "cross", "source_dock": "dock_branch_0_left", "sink_dock": "dock_branch_1_right", "lane": {"side": "right"}},
            {"name": "east", "source_dock": "dock_start", "sink_dock": "dock_end", "lane": {"side": "right"}},
        ],
    }
    _gen, scenario = _scenario(config)
    assert _lane_of(scenario, "dock_branch_0_left_source_cross")["two_way"] is False
    assert _lane_of(scenario, "dock_start_source_east")["two_way"] is False


def test_flow_passing_a_colonnade_gets_its_column_spacing_as_sweep_length() -> None:
    config = {
        "length": 30.0,
        "width": 4.0,
        "colonnades": [{"start_s": 5.0, "end_s": 20.0, "spacing": 4.0}],
        "flows": [{"name": "through", "source_dock": "dock_start", "sink_dock": "dock_end", "lane": {"side": "right"}}],
    }
    _gen, scenario = _scenario(config)
    assert _lane_of(scenario, "dock_start_source_through")["sweep_length"] == 4.0


def test_lane_centerlines_start_and_end_at_the_dock_nav_poses() -> None:
    config = {
        "length": 30.0,
        "width": 4.0,
        "flows": [{"name": "through", "source_dock": "dock_start", "sink_dock": "dock_end", "lane": {"side": "center"}}],
    }
    gen, scenario = _scenario(config)
    segments = _lane_of(scenario, "dock_start_source_through")["segments"]
    first = segments[0]["centerline"][0]
    last = segments[-1]["centerline"][-1]
    start_nav = gen._dock_metadata["dock_start"]["nav_pose"]
    end_nav = gen._dock_metadata["dock_end"]["nav_pose"]
    assert math.dist((first[0], first[1]), (start_nav[0], start_nav[1])) < 0.05
    assert math.dist((last[0], last[1]), (end_nav[0], end_nav[1])) < 0.05


def test_flow_without_a_lane_block_writes_no_lane() -> None:
    config = {"length": 30.0, "width": 4.0, "flows": [{"name": "plain", "source_dock": "dock_start", "sink_dock": "dock_end"}]}
    _gen, scenario = _scenario(config)
    assert "lane" not in scenario["regions"]["dock_start_source_plain"]["config"]


HALLWAY_LENGTH = 30.0
BRANCH_LENGTH = 6.0
BRANCH_WIDTH = 3.0
BRANCH_GAP = 12.0
GRID_QS = np.linspace(-1.0, 1.0, 21)
SPAWN_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
CORNER_TOLERANCE = 1e-6
TURN_TOLERANCE = 1e-9
OPPOSED_DOT = -0.5


@dataclass(frozen=True)
class _Corner:
    """A vertex where a lane turns, with its leg directions and the reach of its inside."""

    index: int
    vertex: NDArray[np.float64]
    t_in: NDArray[np.float64]
    t_out: NDArray[np.float64]
    turn: float
    reach: float


def _branch_layout(width: float, first_at: float, first_side: str, second_side: str, hallway_flow: str) -> dict[str, Any]:
    """Hallway with two side branches, a cross flow between them and a hallway flow, west or east."""
    hallway_docks = {"west": ("dock_end", "dock_start"), "east": ("dock_start", "dock_end")}[hallway_flow]
    return {
        "length": HALLWAY_LENGTH,
        "width": width,
        "branches": [
            {"at_s": first_at, "side": first_side, "width": BRANCH_WIDTH, "length": BRANCH_LENGTH},
            {"at_s": first_at + BRANCH_GAP, "side": second_side, "width": BRANCH_WIDTH, "length": BRANCH_LENGTH},
        ],
        "flows": [
            {
                "name": "cross",
                "source_dock": f"dock_branch_0_{first_side}",
                "sink_dock": f"dock_branch_1_{second_side}",
                "lane": {"side": "right"},
            },
            {"name": hallway_flow, "source_dock": hallway_docks[0], "sink_dock": hallway_docks[1], "lane": {"side": "right"}},
        ],
    }


def _layout_cases() -> list[tuple[str, dict[str, Any], bool]]:
    """Each branch layout as (name, config, whether the hallway flow runs against the cross flow)."""
    cases: list[tuple[str, dict[str, Any], bool]] = []
    for width in (4.0, 5.0):
        for first_at in (6.0, 10.0):
            for first_side in ("left", "right"):
                for second_side in ("left", "right"):
                    for hallway_flow in ("west", "east"):
                        name = f"w{width}_at{first_at}_{first_side}_{second_side}_{hallway_flow}"
                        config = _branch_layout(width, first_at, first_side, second_side, hallway_flow)
                        cases.append((name, config, hallway_flow == "west"))
    return cases


LAYOUT_CASES = _layout_cases()
LAYOUTS = [pytest.param(config, id=name) for name, config, _ in LAYOUT_CASES]
LAYOUT_EXPECTATIONS = [pytest.param(config, expect, id=name) for name, config, expect in LAYOUT_CASES]


def _family_lanes(config: dict[str, Any]) -> dict[str, Lane]:
    """Lane of each flow of a branch layout, keyed by flow name, built from the generated scenario."""
    gen = WorldGeneratorConcourse(copy.deepcopy(config), random.Random(42))
    world = gen.compute()
    lanes, failures = build_scenario_lanes(gen.scenarios()["default"].scenario["regions"], world)
    assert not failures
    return {flow["name"]: lanes[f"{flow['source_dock']}_source_{flow['name']}"] for flow in config["flows"]}


def _sample(start: NDArray[np.float64], end: NDArray[np.float64], step: float) -> NDArray[np.float64]:
    count = max(2, math.ceil(float(np.linalg.norm(end - start)) / step) + 1)
    return start + np.linspace(0.0, 1.0, count)[:, None] * (end - start)


def _walked_path(track: Track, step: float) -> NDArray[np.float64]:
    """Each leg up to its arrival switch, then the switch path to the waypoint after the next one."""
    points = np.array([[p.x, p.y] for p in track.points])
    pieces = []
    for j in range(1, len(points) - 1):
        start, corner, end = points[j - 1], points[j], points[j + 1]
        leg = float(np.linalg.norm(corner - start))
        fraction = max(0.0, 1.0 - track.radii[j] / leg) if leg > 0.0 else 0.0
        switch = start + (corner - start) * fraction
        pieces.extend([_sample(start, switch, step), _sample(switch, end, step)])
    return np.vstack(pieces)


def _in_band(lane: Lane, points: np.ndarray) -> np.ndarray:
    """Whether each point lies inside the lane's band, measured against the nearest station of the lane."""
    stations = np.column_stack([lane.x, lane.y])
    nearest = np.argmin(((points[:, None, :] - stations[None, :, :]) ** 2).sum(axis=2), axis=1)
    lateral = (points[:, 0] - lane.x[nearest]) * lane.nx[nearest] + (points[:, 1] - lane.y[nearest]) * lane.ny[nearest]
    return np.abs(lateral - lane.center[nearest]) <= lane.half_width[nearest]


def _cross(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Z component of the cross product of 2D vectors, taken row by row."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _leg_directions(lane: Lane, index: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Unit directions of the legs entering and leaving a vertex."""
    vertices = np.column_stack([lane.vertex_x, lane.vertex_y])
    t_in = vertices[index] - vertices[index - 1]
    t_out = vertices[index + 1] - vertices[index]
    return t_in / np.linalg.norm(t_in), t_out / np.linalg.norm(t_out)


def _miter(t_in: NDArray[np.float64], t_out: NDArray[np.float64]) -> NDArray[np.float64]:
    """Vector from a vertex to where its two offset lines meet at unit lateral distance."""
    normals = np.array([[-t_in[1], t_in[0]], [-t_out[1], t_out[0]]])
    return np.linalg.solve(normals, np.ones(2))


def _offset_corner(lane: Lane, index: int, lateral: float) -> NDArray[np.float64]:
    """Point where the offset lines of the two legs at a vertex meet, at the given lateral distance."""
    t_in, t_out = _leg_directions(lane, index)
    return np.array([lane.vertex_x[index], lane.vertex_y[index]]) + lateral * _miter(t_in, t_out)


def _corners(lane: Lane) -> list[_Corner]:
    """Vertices where the lane turns, each with the reach of its inside."""
    corners: list[_Corner] = []
    for index in range(1, len(lane.vertex_s) - 1):
        t_in, t_out = _leg_directions(lane, index)
        cross = float(_cross(t_in, t_out))
        if abs(cross) <= TURN_TOLERANCE:
            continue
        s_vertex = float(lane.vertex_s[index])
        center = float(np.interp(s_vertex, lane.s, lane.center))
        half = float(np.interp(s_vertex, lane.s, lane.half_width))
        miter_length = float(np.linalg.norm(_miter(t_in, t_out)))
        reach = max(abs(center - half), abs(center + half)) * miter_length + lane.settings.arrival_radius
        corners.append(
            _Corner(
                index=index,
                vertex=np.array([lane.vertex_x[index], lane.vertex_y[index]]),
                t_in=t_in,
                t_out=t_out,
                turn=math.copysign(1.0, cross),
                reach=reach,
            )
        )
    return corners


def _is_outer(lane: Lane, corner: _Corner) -> bool:
    """Whether the lane's band lies on the outside of the turn at this corner."""
    if lane.settings.side is LaneSide.CENTER:
        return True
    return (lane.settings.side is LaneSide.RIGHT) == (corner.turn > 0.0)


def _corner_goal_misses(lane: Lane) -> list[tuple[int, float, float]]:
    """Goals at the corners that a track must pass, as (vertex index, q, s0)."""
    misses: list[tuple[int, float, float]] = []
    goals = lane.goal_s
    for corner in _corners(lane):
        if not np.any(np.isclose(goals, lane.vertex_s[corner.index], rtol=0.0, atol=1e-9)):
            continue
        s_vertex = float(lane.vertex_s[corner.index])
        center = float(np.interp(s_vertex, lane.s, lane.center))
        half = float(np.interp(s_vertex, lane.s, lane.half_width))
        for fraction in SPAWN_FRACTIONS:
            s0 = fraction * lane.settings.spawn_length
            if s0 >= s_vertex:
                continue
            for q in GRID_QS:
                goal = _offset_corner(lane, corner.index, center + q * half)
                xy = np.array([[p.x, p.y] for p in build_track(lane, float(q), s0).points])
                if float(np.min(np.linalg.norm(xy - goal, axis=1))) >= CORNER_TOLERANCE:
                    misses.append((corner.index, float(q), s0))
    return misses


def _violations(walker: Lane, other: Lane) -> int:
    """Walked samples inside the other lane's band and on the inside of one of the walker's corners."""
    corners = _corners(walker)
    if not corners:
        return 0
    count = 0
    for q in GRID_QS:
        for fraction in SPAWN_FRACTIONS:
            track = build_track(walker, float(q), fraction * walker.settings.spawn_length)
            walked = _walked_path(track, walker.settings.chord_tolerance)
            in_other = _in_band(other, walked)
            for corner in corners:
                offset = walked - corner.vertex
                inside = (np.linalg.norm(offset, axis=1) <= corner.reach) & (corner.turn * _cross(corner.t_in, offset) > 0.0) & (corner.turn * _cross(corner.t_out, offset) > 0.0)
                count += int(np.count_nonzero(inside & in_other))
    return count


def _opposed_pairs(a: Lane, b: Lane) -> int:
    """Station pairs of two lanes within the chord tolerance whose travel directions oppose each other."""
    distance = np.hypot(a.x[:, None] - b.x[None, :], a.y[:, None] - b.y[None, :])
    ta_x, ta_y = a.ny[:, None], -a.nx[:, None]
    tb_x, tb_y = b.ny[None, :], -b.nx[None, :]
    dot = ta_x * tb_x + ta_y * tb_y
    return int(np.count_nonzero((distance <= a.settings.chord_tolerance) & (dot < OPPOSED_DOT)))


def _opposed(a: Lane, b: Lane) -> bool:
    """Whether two lanes share a station pair that travels in opposite directions."""
    return _opposed_pairs(a, b) > 0


@pytest.mark.parametrize("layout", LAYOUTS)
def test_every_track_passes_the_goal_corners_of_a_branch_layout(layout: dict[str, Any]) -> None:
    for lane in _family_lanes(layout).values():
        misses = _corner_goal_misses(lane)
        assert not misses, f"{lane.flow}: {len(misses)} corner goal misses, e.g. {misses[:3]}"


@pytest.mark.parametrize(("layout", "expect_opposed"), LAYOUT_EXPECTATIONS)
def test_goal_corners_lie_on_outside_corners_of_opposed_lanes(layout: dict[str, Any], expect_opposed: bool) -> None:
    for lane in _family_lanes(layout).values():
        outside = [float(lane.vertex_s[corner.index]) for corner in _corners(lane) if lane.settings.two_way and _is_outer(lane, corner)]
        for value in lane.goal_s:
            assert np.any(np.isclose(outside, value, rtol=0.0, atol=1e-9)), f"{lane.flow}: goal at {value:.2f} m is not an outside corner"
        if not expect_opposed:
            assert lane.goal_s.size == 0


@pytest.mark.parametrize("layout", LAYOUTS)
def test_no_walked_track_enters_an_opposed_band_inside_a_corner(layout: dict[str, Any]) -> None:
    for walker, other in permutations(_family_lanes(layout).values(), 2):
        if _opposed(walker, other):
            found = _violations(walker, other)
            assert found == 0, f"{walker.flow} walks into the opposed band of {other.flow} at {found} samples"


@pytest.mark.parametrize(("layout", "expect_opposed"), LAYOUT_EXPECTATIONS)
def test_opposed_detection_matches_the_hallway_direction(layout: dict[str, Any], expect_opposed: bool) -> None:
    lanes = _family_lanes(layout)
    hallway = layout["flows"][1]["name"]
    assert _opposed(lanes["cross"], lanes[hallway]) == expect_opposed
