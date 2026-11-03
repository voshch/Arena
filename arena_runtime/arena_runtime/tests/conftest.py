from __future__ import annotations

import functools
import importlib.util

import pytest

_ROS_SKIP_REASON = "ROS2 not discoverable, source install/setup.bash to enable"


def pytest_configure(config: pytest.Config) -> None:
    if importlib.util.find_spec("arena_simulation_setup") is not None:
        from arena_simulation_setup.testing import install_fake_assets

        install_fake_assets()


@pytest.fixture(scope="session")
def point_table() -> None:
    from arena_runtime.humans.pointing.table import bake

    bake(az_step=30.0, el_step=15.0, el_max=75.0, workers=4)


@pytest.fixture(autouse=True)
def _baked_point_table(request: pytest.FixtureRequest) -> None:
    if any("point_at_template" in marker.args for marker in request.node.iter_markers("clips")):
        request.getfixturevalue("point_table")


@functools.cache
def _clip_error(name: str) -> str | None:
    try:
        from arena_simulation_setup.tree.assets.Animation import AnimationIdentifier

        _ = AnimationIdentifier.parse(name).resolve_sync().clip
    except Exception as exc:
        return f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
    return None


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Clips are assets (bucket or local asset dirs), not repo files: skip tests whose clips cannot be resolved."""
    for marker in item.iter_markers("clips"):
        missing = {name: err for name in marker.args if (err := _clip_error(name)) is not None}
        if missing:
            pytest.skip(f"animation clips unavailable ({', '.join(missing)}), try `arena asset pull animation <name>`: {next(iter(missing.values()))}")


def _ros_available() -> bool:
    try:
        import rclpy  # noqa: F401
    except ImportError:
        return False
    return True


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _ros_available():
        return
    skip = pytest.mark.skip(reason=_ROS_SKIP_REASON)
    for item in items:
        path = str(item.path)
        if "/tests/ros/" in path or "/tests/integration/" in path:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def rclpy_context():
    try:
        import rclpy
        import rclpy.node
    except ImportError:
        yield None
        return
    rclpy.init()
    node = rclpy.node.Node("pytest_host")
    try:
        yield rclpy
    finally:
        node.destroy_node()
        rclpy.shutdown()


@pytest.fixture()
def requires_hearing() -> None:
    if importlib.util.find_spec("arena_hearing") is None:
        pytest.skip("arena_hearing not installed: arena feature hearing install")


@pytest.fixture()
def fake_task_layer(monkeypatch: pytest.MonkeyPatch) -> object:
    import importlib.metadata
    from types import SimpleNamespace

    from arena_runtime.env.task_layer import DEFAULT_TASK_LAYER, TASK_LAYER_GROUP
    from arena_runtime.testing.env_host import FakeTaskLayer

    fake = FakeTaskLayer()
    real = importlib.metadata.entry_points

    def entry_points(**kwargs: object) -> object:
        if kwargs.get("group") == TASK_LAYER_GROUP:
            return [SimpleNamespace(name=DEFAULT_TASK_LAYER, load=lambda: fake)]
        return real(**kwargs)

    monkeypatch.setattr(importlib.metadata, "entry_points", entry_points)
    return fake


@pytest.fixture()
def requires_auditory() -> None:
    if importlib.util.find_spec("arena_auditory") is None:
        pytest.skip("arena_auditory not installed: arena feature auditory install")
