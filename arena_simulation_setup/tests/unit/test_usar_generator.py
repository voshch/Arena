"""USAR task generator: hidden callers placed by corner count in rooms worlds, written as search scenarios with a held-out test split."""

import math
import random
from pathlib import Path

import pydantic
import pytest
import yaml
from arena_simulation_setup.generative.egobench.encounter import Rejected
from arena_simulation_setup.generative.scenario.materialize import is_generated
from arena_simulation_setup.generative.usar import cli
from arena_simulation_setup.generative.usar.manifest import CallLevel, Hiding, Manifest
from arena_simulation_setup.generative.usar.placement import Hideout, corners_of, hide
from arena_simulation_setup.generative.usar.sampler import held_out, tasks
from arena_simulation_setup.generative.world import WorldGenerator, WorldGeneratorType
from arena_simulation_setup.tree.World import World
from arena_simulation_setup.tree.World.Scenario import ScenarioView

MVP = Path(__file__).parents[2] / "configs" / "usar" / "mvp.yaml"


def manifest(**overrides: object) -> Manifest:
    data: dict[str, object] = {"name": "unit", "world": {"rows": 3, "cols": 3}, "world_variants": 2, "tasks_per_world": 3, "corners": [0, 1, 2]}
    data.update(overrides)
    return Manifest.model_validate(data)


def rooms(seed: int) -> World:
    return WorldGenerator(WorldGeneratorType.ROOMS, {"rows": 3, "cols": 4, "loops": 1}, seed).compute()


def placed(world: World, corners: int, hiding: Hiding, seed: int = 0) -> Hideout:
    hideout = hide(world, corners, hiding, random.Random(seed))
    assert not isinstance(hideout, Rejected), hideout
    return hideout


def test_a_straight_route_has_no_corner():
    assert corners_of([(0.0, 0.0), (2.0, 0.0), (5.0, 0.0)]) == 0


def test_a_right_angle_is_one_corner_and_a_slight_bend_none():
    assert corners_of([(0.0, 0.0), (2.0, 0.0), (2.0, 3.0)]) == 1
    assert corners_of([(0.0, 0.0), (2.0, 0.0), (4.0, 2.0 * math.tan(math.radians(20.0)))]) == 0


@pytest.mark.parametrize("corners", [0, 1, 2])
def test_the_person_hides_at_the_requested_corner_count(corners: int):
    hideout = placed(rooms(1), corners, Hiding.RANDOM)
    assert corners_of(hideout.route) == hideout.corners == corners
    assert hideout.route[0] == hideout.start and hideout.route[-1] == hideout.person


def test_an_adversarial_spot_misleads_at_least_as_much_as_any_random_one():
    world = rooms(2)
    adversarial = placed(world, 1, Hiding.ADVERSARIAL)
    assert all(adversarial.misdirection >= placed(world, 1, Hiding.RANDOM, seed).misdirection - 1e-9 for seed in range(10))


def test_an_unreachable_corner_count_is_rejected():
    assert isinstance(hide(rooms(0), 40, Hiding.RANDOM, random.Random(0)), Rejected)


def test_the_task_axes_rotate_across_worlds_with_corners_fastest():
    items = tasks(manifest(call_levels=["weak"], tasks_per_world=2))
    assert [item.corners for item in items] == [0, 1, 2, 0]
    assert {item.call_level for item in items} == {CallLevel.WEAK}


def test_the_test_split_follows_from_the_world_keys_alone():
    keys = ["usar_rooms_b", "usar_rooms_a", "usar_rooms_c"]
    assert held_out(keys, 1) == held_out(list(reversed(keys)), 1) == {"usar_rooms_a"}


def test_holding_out_every_world_is_refused():
    with pytest.raises(pydantic.ValidationError, match="test worlds"):
        manifest(test_worlds=2)


def test_an_unknown_world_axis_is_refused():
    with pytest.raises(pydantic.ValidationError, match="no configuration field"):
        manifest(world={"floors": 2})


def test_a_dry_run_writes_nothing(tmp_path: Path):
    report = cli.run(manifest(), out=tmp_path)
    assert report.placed and not report.worlds and not any(tmp_path.iterdir())


def test_a_written_task_runs_the_search_mode_on_a_marked_scenario(tmp_path: Path):
    report = cli.run(manifest(tasks_per_world=1), out=tmp_path, write=True)
    outcome = next(o for o in report.outcomes if o.reason is None)
    scenario_dir = report.worlds[outcome.task.world_key] / "scenarios" / outcome.task.scenario_key
    assert is_generated(scenario_dir / "scenario.yaml")
    scenario = ScenarioView(scenario_dir).load()
    assert [robot.phases for robot in scenario.robots] == [[]]
    assert [human.name for human in scenario.humans] == ["caller"]
    assert [sound.name for sound in scenario.sounds] == ["call"]
    provenance = yaml.safe_load((scenario_dir / "provenance.yaml").read_text())
    assert provenance["run"]["task.robots"] == "search"
    assert provenance["design"]["corners"] == outcome.task.corners


def test_the_index_puts_each_world_in_exactly_one_split(tmp_path: Path):
    report = cli.run(manifest(test_worlds=1), out=tmp_path, write=True)
    splits = report.index(manifest(test_worlds=1))["worlds"]
    assert isinstance(splits, dict)
    assert len(splits["test"]) == 1 and not set(splits["test"]) & set(splits["tune"])


def test_the_mvp_manifest_places_at_least_45_of_its_50_tasks():
    report = cli.run(Manifest.load(MVP))
    assert len(report.outcomes) == 50
    assert report.placed >= 45
