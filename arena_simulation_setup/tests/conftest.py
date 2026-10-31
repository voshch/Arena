from __future__ import annotations

import importlib.util

import pytest


def pytest_configure(config: pytest.Config) -> None:
    if importlib.util.find_spec("arena_simulation_setup") is not None:
        from arena_simulation_setup.testing import install_fake_assets

        install_fake_assets()


@pytest.fixture
def rclpy_context():
    rclpy = pytest.importorskip("rclpy")
    rclpy.init()
    try:
        yield rclpy
    finally:
        rclpy.shutdown()


@pytest.fixture(scope="session")
def default_sounds():
    from arena_simulation_setup.tree.assets.sound_catalog import SoundLibrary

    library = SoundLibrary.default()
    try:
        for kind in library.kinds().values():
            if kind.default_asset:
                library.asset(kind.default_asset)
    except KeyError as exc:
        pytest.skip(f"default sound assets do not resolve: {exc}")
    return library
