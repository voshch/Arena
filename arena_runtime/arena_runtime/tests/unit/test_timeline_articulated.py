from __future__ import annotations

import asyncio
import random
import types
import typing
import uuid
from pathlib import Path

import pytest
import yaml

DT = 0.25
CABINET_LINKS = {
    "body": [[-0.5, 0.5], [-0.3, 0.3], [0.0, 1.0]],
    "door": [[-0.5, 0.5], [-0.32, -0.3], [0.0, 0.5]],
    "drawer": [[-0.45, 0.45], [-0.3, 0.25], [0.6, 0.8]],
}
CABINET = {
    "links": {"body": ["body"], "door": ["door"], "drawer": ["drawer"], "knob": []},
    "joints": [
        {"name": "door", "type": "revolute", "parent": "body", "child": "door", "origin": [0.5, -0.3, 0.0], "axis": [0, 0, 1], "limit": [0.0, 1.6]},
        {"name": "drawer", "type": "prismatic", "parent": "body", "child": "drawer", "axis": [0, -1, 0], "limit": [0.0, 0.4], "velocity": 0.2},
        {"name": "knob", "type": "continuous", "parent": "door", "child": "knob", "origin": [0.0, -0.32, 0.25], "axis": [0, 1, 0]},
    ],
    "states": {"closed": {"door": 0.0, "drawer": 0.0}, "open": {"door": 1.5, "drawer": 0.4}},
    "swept": [[-0.5, 0.52], [-1.32, 0.3], [0.0, 1.0]],
}


@pytest.fixture
def assets(tmp_path: Path):
    """Asset root with an articulated cabinet, the model name unique per test."""
    from arena_simulation_setup.tree import DynamicPaths

    cabinet = tmp_path / "Kitchen" / "Object" / f"Cabinet{uuid.uuid4().hex[:8]}"
    for link, box in CABINET_LINKS.items():
        (cabinet / "links" / link).mkdir(parents=True)
        (cabinet / "links" / link / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": box}))
    (cabinet / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": CABINET_LINKS["body"], "articulation": CABINET}))
    previous = DynamicPaths.ARENA.path
    DynamicPaths.ARENA.path = tmp_path
    yield {"cabinet": f"Kitchen/{cabinet.name}"}
    DynamicPaths.ARENA.path = previous


def _realizer():
    from arena_runtime.env.realizer import Realizer

    return Realizer(Realizer._Configuration(x=0.0, y=0.0, prefix="env_0"))


def _obstacle(name: str, model: str):
    from arena_runtime.shared import Obstacle, Orientation, Pose, Position

    return _realizer().realize(Obstacle(name=name, model=model, pose=Pose(position=Position(0.0, 0.0, 0.0), orientation=Orientation.from_yaw(0.0))))


@pytest.fixture
def on_dummy_sim():

    def run(body: typing.Callable[..., typing.Awaitable[None]]) -> None:
        from arena_rclpy_mixins import ArenaMixinNode
        from arena_rclpy_mixins.shared import Namespace
        from arena_runtime.sim.dummy_simulator import DummySimulator

        async def main() -> None:
            suffix = uuid.uuid4().hex[:8]
            node = ArenaMixinNode(f"timeline_{suffix}", namespace=f"/timeline_{suffix}")
            sim = DummySimulator(node=node, namespace=Namespace(f"/timeline_{suffix}"), realizer=_realizer())
            try:
                await body(sim)
            finally:
                await sim.stop_mechanisms()
                node.destroy_node()

        asyncio.run(main())

    return run


def test_a_timeline_set_drives_the_object_and_the_judge_reads_its_state(assets, on_dummy_sim):
    from arena_simulation_setup.shared.conditions import parse_atom
    from arena_simulation_setup.shared.judge import Sample, atom_holds
    from arena_simulation_setup.tree.World.Scenario import TimelineEntry
    from arena_runtime.sim import _mechanism_shim as shim
    from arena_runtime.env.episode_node import EpisodeNode

    class Logger:
        def warning(self, msg: str) -> None:
            raise AssertionError(msg)

    async def body(sim):
        await sim.spawn_objects([_obstacle("cabinet", assets["cabinet"])])
        node = types.SimpleNamespace(_semantic_names={}, _realizer=_realizer(), _simulator=sim, _sound_levels={}, _timeline_state=[{"rng": random.Random("7:0")}], get_logger=Logger)
        for method in ("_resolve_semantic_entity", "_apply_semantic_checked", "_set_semantic", "_resolve_timeline_value", "_semantic_value"):
            setattr(node, method, getattr(EpisodeNode, method).__get__(node))
        node._norm_token = EpisodeNode._norm_token
        node._stringify_float = EpisodeNode._stringify_float

        def holds(atom: str) -> bool | None:
            return atom_holds(parse_atom(atom), Sample(t=0.0, field=node._semantic_value), {})

        assert holds("cabinet.state == closed")
        EpisodeNode._fire_timeline_entry(node, 0, TimelineEntry(set=[{"entity": "cabinet", "field": "target", "value": "open"}], at=0.0))
        assert holds("cabinet.target == open")
        assert not holds("cabinet.state == open")
        for _ in range(8):
            await shim._tick(sim, DT)
        assert holds("cabinet.state == open")
        assert node._semantic_value("cabinet", "q.door") == "1.5"
        assert node._semantic_value("cabinet", "openness.door") == "1.0"
        assert holds("cabinet.moving == false")

    on_dummy_sim(body)
