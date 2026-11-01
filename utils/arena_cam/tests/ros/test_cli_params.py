from __future__ import annotations

import pytest

from arena_cam.cli import is_path, main, parse_params


def test_params_coerce_numbers_tuples_bools_and_strings() -> None:
    assert parse_params(["duration=2", "eye=1,2,3", "loop=true", "entity=jackal"]) == {"duration": 2.0, "eye": (1.0, 2.0, 3.0), "loop": True, "entity": "jackal"}


def test_a_token_without_equals_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="expected key=value, got 'oops'"):
        parse_params(["oops"])


def test_the_cli_exits_with_the_param_error() -> None:
    with pytest.raises(SystemExit, match="expected key=value, got 'oops'"):
        main(["hold", "oops"])


def test_shot_files_are_told_apart_from_names() -> None:
    assert is_path("tour.yaml")
    assert is_path("shots/tour")
    assert not is_path("orbit")


def test_a_mount_without_lockstep_is_refused() -> None:
    with pytest.raises(SystemExit):
        main(["hold", "--record", "take", "--mount", "env_0/jackal/base_link"])


def test_a_mount_without_the_sim_camera_is_refused() -> None:
    with pytest.raises(SystemExit):
        main(["hold", "--viz", "0", "--record", "take", "--lockstep", "--mount", "env_0/jackal/base_link"])


def test_a_malformed_mount_is_refused() -> None:
    with pytest.raises(SystemExit):
        main(["hold", "--record", "take", "--lockstep", "--mount", "env_0/jackal/base_link@0,0"])
