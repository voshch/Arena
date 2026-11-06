"""EgoBench world generator and batch CLI: one sample's world with its scenario, and a manifest written as worlds plus an index."""

import random
from pathlib import Path

import pydantic
import pytest
import yaml
from arena_simulation_setup.generative.egobench import cli
from arena_simulation_setup.generative.egobench.manifest import Encounter, Manifest, Pedestrian, Space
from arena_simulation_setup.generative.egobench.meta import Unplaceable, WorldGeneratorEgoBench
from arena_simulation_setup.generative.egobench.sampler import Sample, sample
from arena_simulation_setup.generative.scenario.materialize import is_generated
from arena_simulation_setup.generative.world import WorldGenerator, WorldGeneratorType
from arena_simulation_setup.generative.world.world_generator import save_generated
from arena_simulation_setup.tree.World import WorldView
from arena_simulation_setup.tree.World.Scenario import ScenarioView

MVP = Path(__file__).parents[2] / "configs" / "egobench" / "mvp.yaml"


def one(encounter: Encounter = Encounter.ONCOMING, space: Space = Space.CORRIDOR, **world: float) -> Sample:
    return Sample(space=space, world=world or {"corridor_width": 3.0}, world_seed=1, encounter=encounter, pedestrian=Pedestrian.FIXED, lateral_offset=1.2, robot_speed=0.5, ped_speed=1.2, agent_type="adult")


def manifest(**overrides: object) -> Manifest:
    data: dict[str, object] = {"name": "unit", "worlds": {"open": {"obstacles": 0}}, "encounters": ["oncoming", "standing"], "lateral_offset": [1.0, 1.4]}
    data.update(overrides)
    return Manifest.model_validate(data)


def test_the_generator_is_registered_and_runs_on_its_defaults():
    assert WorldGeneratorType.EGOBENCH in WorldGenerator.available()
    generator = WorldGenerator(WorldGeneratorType.EGOBENCH, {}, 0)
    assert generator.compute().all_levels[0].zones
    assert generator.params()["modes"]["obstacles"] == "scenario"


def test_an_unknown_encounter_is_rejected_by_the_configuration():
    with pytest.raises(pydantic.ValidationError):
        WorldGeneratorEgoBench({"encounter": "dancing"}, random.Random(0))


def test_an_unknown_pedestrian_kind_is_rejected_by_the_configuration():
    with pytest.raises(pydantic.ValidationError):
        WorldGeneratorEgoBench({"pedestrian": "scripted"}, random.Random(0))


def test_a_world_field_of_another_space_is_rejected_by_the_configuration():
    with pytest.raises(ValueError, match="open has no configuration field corridor_width"):
        WorldGeneratorEgoBench({"space": "open", "world": '{"corridor_width": 3.0}'}, random.Random(0))


def test_the_generator_carries_the_pedestrian_kind_into_its_scenario():
    generator = WorldGeneratorEgoBench({"pedestrian": "reactive", "world": '{"corridor_width": 3.0, "branches": 4}', "world_seed": 3}, random.Random(0))
    assert generator.sample.pedestrian is Pedestrian.REACTIVE
    assert "__reactive__" in generator.params()["params"]["robots"]["file"]


def test_an_encounter_that_does_not_fit_fails_the_world():
    generator = WorldGeneratorEgoBench.of(one(corridor_width=2.0))
    with pytest.raises(Unplaceable, match="does not fit a 2.00 m corridor"):
        generator.compute()


def test_the_episode_binding_names_the_sample_scenario():
    item = one()
    params = WorldGeneratorEgoBench.of(item).params()
    assert params["params"]["robots"] == params["params"]["obstacles"] == {"file": item.scenario_key}
    assert params["modes"] == {"robots": "scenario", "obstacles": "scenario"}


def test_a_saved_world_carries_its_marked_scenario_types_and_provenance(tmp_path: Path):
    item = one(Encounter.CROSSING_LEFT, Space.OPEN, obstacles=0)
    generator = WorldGeneratorEgoBench.of(item)
    save_generated(WorldView(tmp_path / "w"), generator.compute(), generator.files())
    scenario_dir = tmp_path / "w" / "scenarios" / item.scenario_key
    assert is_generated(scenario_dir / "scenario.yaml")
    scenario = ScenarioView(scenario_dir).load()
    assert [h.name for h in scenario.humans] == ["ped_0"]
    assert len(scenario.robots) == 1
    assert (scenario_dir / scenario.humans[0].agent["agent_type"].removeprefix("./")).is_file()
    provenance = yaml.safe_load((scenario_dir / "provenance.yaml").read_text())
    assert provenance["sample"] == item.to_dict()


def test_a_dry_run_writes_nothing(tmp_path: Path):
    report = cli.run(manifest(), out=tmp_path / "out")
    assert report.placed == 4
    assert not (tmp_path / "out").exists()


def test_samples_on_one_world_are_written_into_it(tmp_path: Path):
    report = cli.run(manifest(), out=tmp_path, write=True)
    (world,) = report.worlds.values()
    assert sorted(p.name for p in (world / "scenarios").iterdir()) == sorted(o.sample.scenario_key for o in report.outcomes)


def test_the_cli_and_the_generator_write_the_same_scenario(tmp_path: Path):
    m = manifest()
    item = sample(m)[0]
    report = cli.run(m, out=tmp_path / "batch", write=True)
    generator = WorldGeneratorEgoBench.of(item)
    save_generated(WorldView(tmp_path / "single"), generator.compute(), generator.files())
    batch = report.worlds[item.world_key] / "scenarios" / item.scenario_key / "scenario.yaml"
    single = tmp_path / "single" / "scenarios" / item.scenario_key / "scenario.yaml"
    assert batch.read_text() == single.read_text()


def test_the_index_lists_every_sample_with_its_status(tmp_path: Path):
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump({"name": "mix", "worlds": {"corridor": {"corridor_width": 2.0, "branches": 3}}, "encounters": ["standing"], "lateral_offset": [0.6, 1.4]}))
    assert cli.main([str(path), "--out", str(tmp_path), "--write"]) == 0
    index = yaml.safe_load((tmp_path / "mix.index.yaml").read_text())
    assert [s["status"] for s in index["samples"]] == ["ok", "rejected"]
    assert "does not fit a 2.00 m corridor" in index["samples"][1]["reason"]


def test_the_cli_fails_when_nothing_is_placed(tmp_path: Path):
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump({"name": "none", "worlds": {"corridor": {"corridor_width": 2.0}}, "encounters": ["oncoming"], "lateral_offset": 1.5}))
    assert cli.main([str(path), "--out", str(tmp_path)]) == 1


def test_the_mvp_manifest_spans_three_encounters_at_five_offsets_for_both_pedestrian_kinds():
    samples = sample(Manifest.load(MVP))
    assert {s.pedestrian for s in samples} == set(Pedestrian)
    assert {s.encounter for s in samples} == {Encounter.ONCOMING, Encounter.CROSSING_LEFT, Encounter.CROSSING_RIGHT}
    assert sorted({s.lateral_offset for s in samples}) == [0.6, 0.8, 1.0, 1.2, 1.4]
