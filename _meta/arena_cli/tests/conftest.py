from collections.abc import Callable
from pathlib import Path

import pytest
from arena_cli.features.robots import robot_submodules

_ARENA = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def require_installed() -> Callable[..., None]:
    """Skips the test unless every submodule tagged for each robot or component is checked out."""
    submodules = robot_submodules(_ARENA)

    def require(*names: str) -> None:
        for name in names:
            if any(not (_ARENA / path).is_dir() or not any((_ARENA / path).iterdir()) for path in submodules.get(name, [])):
                pytest.skip(f"{name} is not installed, run `arena feature robots add {name}`")

    return require
