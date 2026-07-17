from __future__ import annotations

import asyncio
import types

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")
    pytest.importorskip("arena_humansim_msgs.msg")


def _fake_obstacle(sim_path: str) -> object:
    pose = types.SimpleNamespace(
        position=types.SimpleNamespace(x=0.0, y=0.0),
        orientation=types.SimpleNamespace(to_yaw=lambda: 0.0),
    )
    return types.SimpleNamespace(
        sim_path=sim_path,
        pose=pose,
        velocity=1.0,
        waypoints=[],
    )


class _FakeSpawnClient:
    def __init__(self) -> None:
        self.last_request = None

    async def call_timeout(self, request: object) -> object:
        from arena_humansim_msgs.srv import SpawnAgents

        self.last_request = request
        response = SpawnAgents.Response()
        response.success = True
        response.spawned_ids = [a.agent_id for a in request.agents]
        return response


def _make_adapter(driver_set: tuple[str, ...], seed: int) -> tuple[object, _FakeSpawnClient]:
    from task_generator.constants.rng import EpisodeRng
    from task_generator.simulators.human.arena_humansim import arena_humansim as mod

    adapter = object.__new__(mod.ArenaHumanSimulator)
    adapter._next_id = 0
    adapter._bridge_agent_ids = set()
    adapter._agent_names = {}

    rng = EpisodeRng()
    rng.reseed(seed)
    conf = types.SimpleNamespace(
        Human=types.SimpleNamespace(DRIVER_SET=driver_set),
        General=types.SimpleNamespace(RNG=rng),
    )
    _noop_logger = types.SimpleNamespace(
        info=lambda *a, **k: None, error=lambda *a, **k: None, get_child=lambda *a, **k: _noop_logger,
    )
    # `node` is a read-only property backed by NodeInterface's name-mangled
    # `__node` slot; set it directly since there is no public setter.
    adapter._NodeInterface__node = types.SimpleNamespace(conf=conf, get_logger=lambda: _noop_logger)

    client = _FakeSpawnClient()
    adapter._spawn_client = client

    return adapter, client


def _drawn_policies(driver_set: tuple[str, ...], seed: int, n_agents: int = 2) -> list[str]:
    from task_generator.simulators.human.arena_humansim import arena_humansim as mod

    adapter, client = _make_adapter(driver_set, seed)
    obstacles = [_fake_obstacle(f"ped_{i}") for i in range(n_agents)]

    # ArenaHumanDynamicObstacle resolution is independent of the driver sampler
    # and requires real Pedestrian assets; force the "no per-agent params" path.
    original = mod.ArenaHumanDynamicObstacle.from_dynamic_obstacle
    mod.ArenaHumanDynamicObstacle.from_dynamic_obstacle = classmethod(lambda cls, obstacle: None)
    try:
        asyncio.run(adapter._spawn_dynamic_obstacles_impl(obstacles))
    finally:
        mod.ArenaHumanDynamicObstacle.from_dynamic_obstacle = original

    assert client.last_request is not None
    return [a.policy for a in client.last_request.agents]


def test_driver_set_empty_leaves_policy_unset():
    policies = _drawn_policies(driver_set=(), seed=1)
    assert policies == ["", ""]


def test_driver_drawn_once_per_episode_applies_to_all_agents():
    policies = _drawn_policies(driver_set=("orca", "sfm"), seed=1)
    assert len(set(policies)) == 1
    assert policies[0] in ("orca", "sfm")


def test_driver_draw_reproducible_for_same_seed():
    first = _drawn_policies(driver_set=("orca", "sfm"), seed=42)
    second = _drawn_policies(driver_set=("orca", "sfm"), seed=42)
    assert first == second


def test_driver_stable_across_multiple_spawn_calls_in_one_episode():
    """The impl can fire more than once per episode (per-obstacle extend() path).
    Every call within one episode must yield the SAME driver — RNG.once() guarantees
    it (draw is memoized until the next reseed), so the crowd never mixes drivers."""
    from task_generator.simulators.human.arena_humansim import arena_humansim as mod

    adapter, client = _make_adapter(driver_set=("orca", "sfm"), seed=7)
    original = mod.ArenaHumanDynamicObstacle.from_dynamic_obstacle
    mod.ArenaHumanDynamicObstacle.from_dynamic_obstacle = classmethod(lambda cls, obstacle: None)
    try:
        drivers = []
        for call in range(3):  # three separate spawn batches, no reseed between
            asyncio.run(adapter._spawn_dynamic_obstacles_impl([_fake_obstacle(f"ped_{call}")]))
            drivers.append(client.last_request.agents[0].policy)
    finally:
        mod.ArenaHumanDynamicObstacle.from_dynamic_obstacle = original
    assert len(set(drivers)) == 1  # identical across all three calls
