"""EgoBench scenario generators: placement per encounter type in corridor and open worlds."""

import math

import pytest
import shapely
from arena_simulation_setup.generative.egobench import scenario_corridor, scenario_open
from arena_simulation_setup.generative.egobench.encounter import FIXED_PLANNER, MIN_RELEASE_M, Generated, Rejected
from arena_simulation_setup.generative.egobench.manifest import Encounter, Pedestrian, Space
from arena_simulation_setup.generative.egobench.sampler import Sample
from arena_simulation_setup.generative.egobench.scenario_corridor import runs
from arena_simulation_setup.generative.world import WorldGenerator, WorldGeneratorType
from arena_simulation_setup.generative.world.layout import Area, Layout, Segment
from arena_simulation_setup.tree.World.Scenario import parse_scenario


def sample(encounter: Encounter, space: Space = Space.CORRIDOR, pedestrian: Pedestrian = Pedestrian.FIXED, **overrides: float) -> Sample:
    values = {"lateral_offset": 1.2, "robot_speed": 0.5, "ped_speed": 1.2}
    values.update(overrides)
    return Sample(space=space, world={}, world_seed=0, encounter=encounter, pedestrian=pedestrian, agent_type="adult", **values)


def corridor(width: float = 3.0, arm: float = 0.0) -> Layout:
    """A 20 m corridor along +x, with a side branch `arm` m long leaving its middle towards +y."""
    segments = [Segment(a=(0.0, 0.0), b=(10.0, 0.0), width=width), Segment(a=(10.0, 0.0), b=(20.0, 0.0), width=width)]
    if arm:
        segments.append(Segment(a=(10.0, 0.0), b=(10.0, arm), width=width))
    return Layout(segments=segments)


def open_field(side: float = 20.0) -> Layout:
    return Layout(areas=[Area(polygon=shapely.box(0, 0, side, side))])


def generated(result: Generated | Rejected) -> Generated:
    assert isinstance(result, Generated), result
    return result


def design(result: Generated | Rejected) -> dict:
    return generated(result).provenance["design"]


def lane_distance(d: dict) -> float:
    """Distance of the pedestrian's spawn from the line of the robot's leg."""
    (sx, sy, _), (gx, gy, _) = d["robot_start"], d["robot_goal"]
    px, py = d["ped_route"][0]
    return abs((gx - sx) * (sy - py) - (sx - px) * (gy - sy)) / math.dist((sx, sy), (gx, gy))


def test_collinear_segments_join_into_one_run_with_its_branch():
    found = runs(corridor(arm=6.0))
    assert found[0].leg.length == pytest.approx(20.0)
    assert [(a.arc, a.direction) for a in found[0].arms] == [(pytest.approx(10.0), (0.0, 1.0))]
    assert found[1].leg.length == pytest.approx(6.0)


def test_an_offset_wider_than_the_corridor_is_rejected():
    result = scenario_corridor.generate(corridor(width=2.0), sample(Encounter.ONCOMING))
    assert isinstance(result, Rejected)
    assert "does not fit a 2.00 m corridor" in result.reason


def test_an_oncoming_pedestrian_walks_the_lane_at_the_lateral_offset():
    d = design(scenario_corridor.generate(corridor(), sample(Encounter.ONCOMING)))
    assert lane_distance(d) == pytest.approx(1.2)
    robot_dx = d["robot_goal"][0] - d["robot_start"][0]
    ped_dx = d["ped_route"][1][0] - d["ped_route"][0][0]
    assert robot_dx * ped_dx < 0


def test_a_negative_offset_puts_the_pedestrian_on_the_right():
    d = design(scenario_corridor.generate(corridor(), sample(Encounter.STANDING, lateral_offset=-1.2)))
    (sx, sy, theta), (px, py) = d["robot_start"], d["ped_route"][0]
    assert math.cos(theta) * (py - sy) - math.sin(theta) * (px - sx) < 0


def ped_type(result: Generated) -> dict:
    return result.types[result.scenario["humans"][0]["agent"]["agent_type"].removeprefix("./")]


def test_a_reactive_pedestrian_is_released_when_the_robot_has_driven_its_lead():
    result = generated(scenario_corridor.generate(corridor(), sample(Encounter.ONCOMING, pedestrian=Pedestrian.REACTIVE)))
    assert result.provenance["design"]["release_arc"] >= MIN_RELEASE_M
    (entry,) = result.scenario["timeline"]
    assert entry["when"].startswith("robot within ")
    assert entry["stimulus"] == [{"agents": ["ped_0"], "need": "go"}]
    assert "local_planner" not in ped_type(result)


def test_a_fixed_pedestrian_starts_on_the_clock_when_a_nominal_robot_reaches_the_release():
    result = generated(scenario_corridor.generate(corridor(), sample(Encounter.ONCOMING)))
    design = result.provenance["design"]
    (entry,) = result.scenario["timeline"]
    assert "when" not in entry
    assert entry["at"] == pytest.approx(design["release_arc"] / 0.5, abs=0.01)
    assert design["release_time"] == entry["at"]


def test_a_fixed_pedestrian_ignores_the_robot():
    assert ped_type(generated(scenario_corridor.generate(corridor(), sample(Encounter.ONCOMING))))["local_planner"] == FIXED_PLANNER


def test_both_pedestrian_kinds_walk_the_same_route():
    fixed, reactive = (design(scenario_corridor.generate(corridor(), sample(Encounter.ONCOMING, pedestrian=kind))) for kind in Pedestrian)
    assert fixed["ped_route"] == reactive["ped_route"]
    assert fixed["robot_start"] == reactive["robot_start"]


def test_a_standing_pedestrian_faces_the_robot_and_is_never_released():
    result = generated(scenario_corridor.generate(corridor(), sample(Encounter.STANDING)))
    assert "timeline" not in result.scenario
    robot_theta = result.provenance["design"]["robot_start"][2]
    ped_theta = result.scenario["humans"][0]["pose"][2]
    assert math.cos(ped_theta - robot_theta) == pytest.approx(-1.0)


def test_a_slower_robot_cannot_overtake():
    result = scenario_corridor.generate(corridor(), sample(Encounter.OVERTAKE))
    assert isinstance(result, Rejected)
    assert "is not overtaken" in result.reason


def test_an_overtaking_pedestrian_starts_behind_the_robot():
    d = design(scenario_corridor.generate(corridor(), sample(Encounter.OVERTAKEN)))
    heading = d["robot_start"][2]
    behind = (d["ped_route"][0][0] - d["robot_start"][0]) * math.cos(heading) + (d["ped_route"][0][1] - d["robot_start"][1]) * math.sin(heading)
    assert behind < 0


def test_a_followed_pedestrian_walks_from_the_start():
    result = generated(scenario_corridor.generate(corridor(), sample(Encounter.FOLLOW)))
    assert result.scenario["timeline"] == [{"at": 0.0, "stimulus": [{"agents": ["ped_0"], "need": "go"}]}]


def test_a_crossing_comes_out_of_the_side_branch():
    d = design(scenario_corridor.generate(corridor(arm=8.0), sample(Encounter.CROSSING_LEFT)))
    spawn = d["ped_route"][0]
    assert spawn[0] == pytest.approx(10.0)
    assert spawn[1] > 1.5


def test_a_crossing_from_the_right_drives_the_run_the_other_way():
    d = design(scenario_corridor.generate(corridor(arm=8.0), sample(Encounter.CROSSING_RIGHT)))
    assert d["robot_start"][0] > d["robot_goal"][0]
    assert d["ped_route"][0][1] > 1.5


def test_a_crossing_without_a_side_branch_is_rejected():
    result = scenario_corridor.generate(corridor(), sample(Encounter.CROSSING_LEFT))
    assert isinstance(result, Rejected)
    assert "no side branch" in result.reason


def test_an_open_crossing_is_square_to_the_robot_and_from_its_left():
    d = design(scenario_open.generate(open_field(), sample(Encounter.CROSSING_LEFT, Space.OPEN)))
    (sx, sy, theta), ((ax, ay), (bx, by)) = d["robot_start"], d["ped_route"]
    assert (bx - ax) * math.cos(theta) + (by - ay) * math.sin(theta) == pytest.approx(0.0, abs=1e-2)
    assert math.cos(theta) * (ay - sy) - math.sin(theta) * (ax - sx) > 0


def test_an_open_field_too_small_for_a_chord_is_rejected():
    result = scenario_open.generate(open_field(side=5.0), sample(Encounter.ONCOMING, Space.OPEN))
    assert isinstance(result, Rejected)
    assert "no straight chord" in result.reason


@pytest.mark.parametrize("encounter", list(Encounter))
def test_every_encounter_fits_an_open_field_and_parses(encounter: Encounter):
    speed = 0.3 if encounter is Encounter.OVERTAKE else 1.2
    result = generated(scenario_open.generate(open_field(), sample(encounter, Space.OPEN, ped_speed=speed)))
    parsed = parse_scenario(result.scenario)
    assert [h.name for h in parsed.humans] == ["ped_0"]
    agent_type = result.scenario["humans"][0]["agent"]["agent_type"]
    assert agent_type.removeprefix("./") in result.types


def test_a_generated_corridor_world_places_an_oncoming_pedestrian():
    generator = WorldGenerator(WorldGeneratorType.CORRIDOR, {"branches": 4, "corridor_width": 3.0}, seed=3)
    generator.compute()
    layout = generator.layout()
    assert layout is not None
    generated(scenario_corridor.generate(layout, sample(Encounter.ONCOMING)))
