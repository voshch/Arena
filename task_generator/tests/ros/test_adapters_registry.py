from __future__ import annotations

import pytest
from arena_rclpy_mixins.registry import ClassRegistry


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


def test_register_and_get_round_trip():
    reg: ClassRegistry[str, type] = ClassRegistry()

    class _Foo:
        pass

    @reg.register("_test_foo")
    def _load():
        return _Foo

    assert reg.get("_test_foo") is _Foo


def test_register_raises_on_duplicate():
    reg: ClassRegistry[str, type] = ClassRegistry()

    @reg.register("_test_dup")
    def _load_a():
        return object

    with pytest.raises(ValueError):

        @reg.register("_test_dup")
        def _load_b():
            return object


def test_get_raises_key_error_unknown_kind():
    reg: ClassRegistry[str, type] = ClassRegistry()

    @reg.register("_test_known")
    def _load():
        return object

    with pytest.raises(KeyError) as exc_info:
        reg.get("_no_such_kind")
    assert "known" in str(exc_info.value)


def test_lazy_loading():
    reg: ClassRegistry[str, type] = ClassRegistry()

    @reg.register("_test_poison")
    def _poison():
        raise RuntimeError("should not be called")

    class _Bar:
        pass

    @reg.register("_test_bar")
    def _load():
        return _Bar

    assert reg.get("_test_bar") is _Bar


def test_adapter_kind_classvar_matches_registry_key():
    """Convention guard: each adapter's `kind` ClassVar must equal its registry key."""
    from task_generator.tasks.robots.adapters import ADAPTERS

    for cap, reg in ADAPTERS.items():
        for kind in reg.keys():
            cls = reg.get(kind)
            assert cls.kind == kind, f"{cls.__name__}.kind={cls.kind!r} != registry key {kind!r} (cap={cap!r})"


def test_adapter_meta_attached_on_every_adapter():
    from task_generator.tasks.robots.adapters import ADAPTERS, AdapterMeta

    for reg in ADAPTERS.values():
        for kind in reg.keys():
            cls = reg.get(kind)
            assert isinstance(cls._adapter_meta, AdapterMeta), f"{cls.__name__} missing _adapter_meta"


def test_every_mobile_adapter_subclasses_mobile_adapter():
    from task_generator.tasks.robots.adapters import ADAPTERS
    from task_generator.tasks.robots.adapters.mobile import MobileAdapter

    for kind in ADAPTERS["mobile"].keys():
        cls = ADAPTERS["mobile"].get(kind)
        assert issubclass(cls, MobileAdapter), f"{cls.__name__} (cap=mobile) must inherit MobileAdapter"


class TestAdapterMetaConverters:
    def test_accepts_set_coerces_to_frozenset(self):
        from arena_robots.bringup.mobile.nav2 import Nav2Bringup
        from arena_robots.clients.goto_pose import GotoPoseClient
        from arena_robots.task_kinds import TaskKind
        from task_generator.tasks.robots.adapters import AdapterMeta

        meta = AdapterMeta(
            accepts={TaskKind.GOTO_POSE},
            bringup=Nav2Bringup,
            cap="mobile",
            client=GotoPoseClient,
        )
        assert isinstance(meta.accepts, frozenset)
        assert meta.accepts == frozenset({TaskKind.GOTO_POSE})

    def test_displays_list_coerces_to_tuple(self):
        from arena_robots.bringup.mobile.nav2 import Nav2Bringup
        from arena_robots.clients.goto_pose import GotoPoseClient
        from arena_robots.task_kinds import TaskKind
        from arena_viz import DisplayKind
        from task_generator.tasks.robots.adapters import AdapterDisplayHint, AdapterMeta

        hint = AdapterDisplayHint(name="X", topic="{ns}/x", kind=DisplayKind.POSE)
        meta = AdapterMeta(
            accepts={TaskKind.GOTO_POSE},
            bringup=Nav2Bringup,
            cap="mobile",
            client=GotoPoseClient,
            displays=[hint],
        )
        assert isinstance(meta.displays, tuple)
        assert meta.displays == (hint,)
