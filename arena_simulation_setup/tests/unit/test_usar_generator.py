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
    data: dict[str, object] = {"name": "unit", "world": {"rows": 3, "cols": 3}, "world_variants": 2, "spots_per_world": 3, "corners": [0, 1, 2]}
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


@pytest.mark.parametrize("seed", range(10))
def test_an_adversarial_spot_misleads_at_least_as_much_as_a_random_one_from_the_same_spots(seed: int):
    world = rooms(2)
    assert placed(world, 1, Hiding.ADVERSARIAL, seed).misdirection >= placed(world, 1, Hiding.RANDOM, seed).misdirection - 1e-9


def test_an_unreachable_corner_count_is_rejected():
    assert isinstance(hide(rooms(0), 40, Hiding.RANDOM, random.Random(0)), Rejected)


def test_the_spot_axes_rotate_across_worlds_with_corners_fastest():
    items = tasks(manifest(call_levels=["weak"], spots_per_world=2))
    assert [item.corners for item in items] == [0, 1, 2, 0]
    assert {item.call_level for item in items} == {CallLevel.WEAK}


def test_every_spot_comes_once_per_call_level():
    items = tasks(manifest(spots_per_world=2))
    assert [(item.corners, item.call_level) for item in items[:4]] == [(0, CallLevel.WEAK), (0, CallLevel.LOUD), (1, CallLevel.WEAK), (1, CallLevel.LOUD)]
    assert items[0].spot_key == items[1].spot_key != items[2].spot_key
    assert items[0].scenario_key != items[1].scenario_key


def test_one_spot_per_combination_covers_every_combination_on_every_world():
    items = tasks(manifest(spots_per_world=6, hiding=["random", "adversarial"], call_levels=["weak"]))
    by_world: dict[str, set[tuple[int, Hiding]]] = {}
    for item in items:
        by_world.setdefault(item.world_key, set()).add((item.corners, item.hiding))
    assert all(combos == {(c, h) for c in (0, 1, 2) for h in Hiding} for combos in by_world.values())


def without_volume(scenario: dict) -> dict:
    sounds = [{**sound, "semantics": [{**semantic, "params": {k: v for k, v in semantic["params"].items() if k != "volume_db"}} for semantic in sound["semantics"]]} for sound in scenario["sounds"]]
    return {**scenario, "sounds": sounds}


def test_the_call_levels_of_a_spot_share_the_person_and_differ_only_in_volume(tmp_path: Path):
    report = cli.run(manifest(spots_per_world=2), out=tmp_path, write=True)
    pairs: dict[tuple[str, str], dict[CallLevel, dict]] = {}
    for outcome in report.outcomes:
        if outcome.reason is not None:
            continue
        root = report.worlds[outcome.task.world_key] / "scenarios" / outcome.task.scenario_key
        docs: dict = {name: yaml.safe_load((root / f"{name}.yaml").read_text()) for name in ("scenario", "provenance")}
        docs["types"] = {str(path.relative_to(root)): path.read_bytes() for path in (root / "types").rglob("*") if path.is_file()}
        pairs.setdefault((outcome.task.world_key, outcome.task.spot_key), {})[outcome.task.call_level] = docs
    assert pairs
    for levels in pairs.values():
        weak, loud = levels[CallLevel.WEAK], levels[CallLevel.LOUD]
        assert without_volume(weak["scenario"]) == without_volume(loud["scenario"])
        assert weak["types"] and weak["types"] == loud["types"]
        assert (weak["scenario"]["sounds"][0]["semantics"][0]["params"]["volume_db"], loud["scenario"]["sounds"][0]["semantics"][0]["params"]["volume_db"]) == (50.0, 70.0)
        assert {k: v for k, v in weak["provenance"]["design"].items() if k != "call_db"} == {k: v for k, v in loud["provenance"]["design"].items() if k != "call_db"}
        assert (weak["provenance"]["design"]["call_db"], loud["provenance"]["design"]["call_db"]) == (50.0, 70.0)
        assert weak["provenance"]["task"]["spot"] == loud["provenance"]["task"]["spot"]


def test_the_test_split_follows_from_the_world_keys_alone():
    keys = ["usar_rooms_b", "usar_rooms_a", "usar_rooms_c"]
    assert held_out(keys, 1) == held_out(list(reversed(keys)), 1) == {"usar_rooms_a"}


def test_a_limited_run_keeps_the_full_runs_split():
    full = {o.task.world_key: o.split for o in cli.run(manifest(world_variants=3, test_worlds=1)).outcomes}
    for limit in range(1, len(tasks(manifest(world_variants=3, test_worlds=1))) + 1):
        report = cli.run(manifest(world_variants=3, test_worlds=1), limit=limit)
        assert {o.task.world_key: o.split for o in report.outcomes}.items() <= full.items()


@pytest.mark.parametrize("limit", [1, 2, 5])
def test_a_limit_counts_spots_and_keeps_every_call_level_of_each(limit: int):
    levels: dict[tuple[str, str], list[CallLevel]] = {}
    for outcome in cli.run(manifest(), limit=limit).outcomes:
        levels.setdefault((outcome.task.world_key, outcome.task.spot_key), []).append(outcome.task.call_level)
    assert len(levels) == limit
    assert all(found == [CallLevel.WEAK, CallLevel.LOUD] for found in levels.values())


def test_a_repeated_call_level_is_refused():
    with pytest.raises(pydantic.ValidationError, match="call_levels"):
        manifest(call_levels=["weak", "weak"])


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
    report = cli.run(manifest(spots_per_world=1), out=tmp_path, write=True)
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


def test_a_rejected_spot_is_rejected_at_every_call_level():
    by_spot: dict[tuple[str, str], set[bool]] = {}
    for outcome in cli.run(manifest(spots_per_world=6)).outcomes:
        by_spot.setdefault((outcome.task.world_key, outcome.task.spot_key), set()).add(outcome.reason is None)
    assert all(len(placed) == 1 for placed in by_spot.values())


def test_the_mvp_manifest_places_all_of_its_120_scenarios():
    report = cli.run(Manifest.load(MVP))
    assert len(report.outcomes) == 120
    assert report.placed == 120
