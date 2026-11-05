"""EgoBench manifest and sampler: axes, validation, grid size and determinism."""

import random
from pathlib import Path

import pydantic
import pytest
from arena_simulation_setup.generative.egobench.manifest import Encounter, Manifest, Pedestrian, Space, Span
from arena_simulation_setup.generative.egobench.sampler import Sample, sample


def manifest(**overrides: object) -> Manifest:
    data: dict[str, object] = {
        "name": "unit",
        "worlds": {"corridor": {"corridor_width": [2.0, 3.0], "branches": 1}, "open": {"obstacles": 0}},
        "encounters": ["oncoming", "standing"],
        "lateral_offset": {"min": 1.2, "max": 1.5, "step": 0.1},
    }
    data.update(overrides)
    return Manifest.model_validate(data)


def test_a_stepped_span_includes_both_ends():
    assert Span(min=1.2, max=1.5, step=0.1).values() == [1.2, 1.3, 1.4, 1.5]


def test_an_unstepped_span_draws_inside_its_bounds():
    rng = random.Random(3)
    assert all(1.2 <= Span(min=1.2, max=1.5).draw(rng) <= 1.5 for _ in range(100))


def test_a_reversed_span_is_rejected():
    with pytest.raises(pydantic.ValidationError, match="below its min"):
        Span(min=2.0, max=1.0)


def test_the_grid_is_the_product_of_every_axis():
    # corridor: 2 widths x 2 encounters x 4 offsets, open: 1 x 2 x 4
    assert len(sample(manifest())) == 16 + 8


def test_world_variants_multiply_the_grid():
    assert len(sample(manifest(world_variants=3))) == 3 * 24


def test_the_sample_count_caps_a_grid():
    assert len(sample(manifest(samples=5))) == 5


def test_random_sampling_draws_the_requested_count():
    drawn = sample(manifest(sampling="random", samples=40, lateral_offset={"min": 1.2, "max": 1.5}))
    assert len(drawn) == 40
    assert {s.space for s in drawn} == {Space.CORRIDOR, Space.OPEN}
    assert all(1.2 <= s.lateral_offset <= 1.5 for s in drawn)


def test_the_same_seed_gives_the_same_samples():
    first = sample(manifest(sampling="random", samples=20, seed=7))
    assert first == sample(manifest(sampling="random", samples=20, seed=7))
    assert first != sample(manifest(sampling="random", samples=20, seed=8))


def test_random_sampling_without_a_count_is_rejected():
    with pytest.raises(pydantic.ValidationError, match="sample count"):
        manifest(sampling="random")


def test_a_grid_over_an_unstepped_span_is_rejected():
    with pytest.raises(pydantic.ValidationError, match="needs a step"):
        manifest(lateral_offset={"min": 1.2, "max": 1.5})


def test_an_unknown_world_field_is_named():
    with pytest.raises(pydantic.ValidationError, match="no configuration field hallway_height"):
        manifest(worlds={"corridor": {"hallway_height": 3.0}})


def test_an_invalid_world_value_fails_at_sampling():
    with pytest.raises(pydantic.ValidationError):
        sample(manifest(worlds={"corridor": {"corridor_width": -1.0}}))


def test_samples_on_one_world_share_its_key():
    samples = sample(manifest(worlds={"corridor": {"corridor_width": 2.0}}))
    assert len({s.world_key for s in samples}) == 1
    assert len({s.scenario_key for s in samples}) == len(samples)


def test_a_sample_survives_its_dict_form():
    original = sample(manifest(pedestrians=["reactive"]))[0]
    assert Sample.from_dict(original.to_dict()) == original
    assert original.encounter is Encounter.ONCOMING
    assert original.pedestrian is Pedestrian.REACTIVE


def test_pedestrians_default_to_a_fixed_route():
    assert {s.pedestrian for s in sample(manifest())} == {Pedestrian.FIXED}


def test_both_pedestrian_kinds_double_the_grid_and_name_their_scenarios():
    samples = sample(manifest(pedestrians=["fixed", "reactive"]))
    assert len(samples) == 2 * 24
    assert {s.scenario_key.split("__")[1] for s in samples} == {"fixed", "reactive"}


def test_an_unknown_pedestrian_kind_is_rejected():
    with pytest.raises(pydantic.ValidationError, match="pedestrians"):
        manifest(pedestrians=["scripted"])


def test_a_manifest_loads_from_yaml(tmp_path: Path):
    path = tmp_path / "m.yaml"
    path.write_text("name: file\nworlds:\n  open: {width: [10.0, 20.0]}\nencounters: [crossing_left]\n")
    assert len(sample(Manifest.load(path))) == 2


def test_an_unknown_pedestrian_profile_is_named():
    with pytest.raises(pydantic.ValidationError, match="no pedestrian profile robot"):
        manifest(agent_type=["adult", "robot"])


@pytest.mark.parametrize(("field", "axis"), [("robot_speed", 0.0), ("ped_speed", [1.2, -0.5]), ("ped_speed", {"min": 0.0, "max": 1.2, "step": 0.6})])
def test_a_speed_at_or_below_zero_is_rejected(field: str, axis: object):
    with pytest.raises(pydantic.ValidationError, match=f"{field} must be above 0"):
        manifest(**{field: axis})


@pytest.mark.parametrize("overrides", [{"seed": -1}, {"world_variants": 1001}])
def test_world_seeds_stay_deterministic_and_distinct(overrides: dict[str, int]):
    with pytest.raises(pydantic.ValidationError):
        manifest(**overrides)
