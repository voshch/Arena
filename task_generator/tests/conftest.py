from __future__ import annotations

import functools

import pytest

_ROS_SKIP_REASON = "ROS2 not discoverable: source install/setup.bash to enable"


def _ros_available() -> bool:
    try:
        import rclpy  # noqa: F401
    except ImportError:
        return False
    return True


@functools.cache
def _clip_error(name: str) -> str | None:
    try:
        from arena_simulation_setup.tree.assets.Animation import AnimationIdentifier

        AnimationIdentifier.parse(name).resolve_sync().clip  # noqa: B018
    except Exception as exc:
        return f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
    return None


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Clips are assets (bucket or local asset dirs), not repo files: skip tests whose clips cannot be resolved."""
    for marker in item.iter_markers("clips"):
        missing = {name: err for name in marker.args if (err := _clip_error(name)) is not None}
        if missing:
            pytest.skip(f"animation clips unavailable ({', '.join(missing)}), try `arena asset pull animation <name>`: {next(iter(missing.values()))}")


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
