from __future__ import annotations

import importlib.util

import pytest

_ROS_SKIP_REASON = "ROS2 not discoverable: source install/setup.bash to enable"


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


def _skip_without(package: str, feature: str) -> None:
    if importlib.util.find_spec(package) is None:
        pytest.skip(f"{package} not installed: arena feature {feature} install")


@pytest.fixture()
def requires_auditory() -> None:
    _skip_without("arena_auditory", "auditory")


@pytest.fixture()
def requires_hearing() -> None:
    _skip_without("arena_hearing", "hearing")


@pytest.fixture()
def radio_loop():
    from arena_simulation_setup.tree.assets.sound_catalog import SoundLibrary

    try:
        return SoundLibrary.default().asset("radio_loop")
    except KeyError as exc:
        pytest.skip(f"sound asset radio_loop does not resolve: {exc}")
