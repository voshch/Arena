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
        # Mirror the production ROSParam wrapper: the adapter must read `.value`
        # (a raw tuple here would mask the object-vs-value bug caught 2026-07-18).
        Human=types.SimpleNamespace(DRIVER_SET=types.SimpleNamespace(value=driver_set)),
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


def _drawn_policies(episode_driver: str, n_agents: int = 2) -> list[str]:
    """Impl-level: the driver draw itself now lives in BaseHumanSimulator.spawn_dynamic_obstacles
    (see base-layer tests below); _spawn_dynamic_obstacles_impl merely reads `self._episode_driver`."""
    from task_generator.simulators.human.arena_humansim import arena_humansim as mod

    adapter, client = _make_adapter(driver_set=(), seed=1)
    adapter._episode_driver = episode_driver
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
    policies = _drawn_policies(episode_driver="", n_agents=2)
    assert policies == ["", ""]


def test_episode_driver_applies_to_all_agents():
    policies = _drawn_policies(episode_driver="sfm", n_agents=2)
    assert policies == ["sfm", "sfm"]


def test_sfm_detuned_maps_to_sfm_policy_with_offset_params():
    """driver_set entry 'sfm_detuned' runs the plain 'sfm' physics policy but shifts
    each agent's per-agent relaxation_time/repulsion_strength/repulsion_range by the
    fixed offsets in arena_humansim.arena_humansim._SFM_DETUNED_OFFSETS (~2x SFM's own
    ParamDist std, away from HSFM's inherited identical defaults)."""
    from task_generator.simulators.human.arena_humansim import arena_humansim as mod

    base_lp = {"relaxation_time": 0.5, "repulsion_strength": 2.1, "repulsion_range": 0.3}
    fake_perception = types.SimpleNamespace(vision_range=5.0, vision_fov=180.0)
    fake_params = types.SimpleNamespace(
        desired_velocity=1.1,
        agent_radius=0.35,
        perception=fake_perception,
        local_planner_params=dict(base_lp),
    )
    fake_parsed = types.SimpleNamespace(
        agent_type="adult",
        sample_params=lambda rng: fake_params,
        waypoints_msg=None,
    )
    from arena_humansim_msgs.msg import Waypoints as WaypointsMsg

    fake_parsed.waypoints_msg = WaypointsMsg()

    adapter, client = _make_adapter(driver_set=(), seed=1)
    adapter._episode_driver = "sfm_detuned"
    original = mod.ArenaHumanDynamicObstacle.from_dynamic_obstacle
    mod.ArenaHumanDynamicObstacle.from_dynamic_obstacle = classmethod(lambda cls, obstacle: fake_parsed)
    try:
        asyncio.run(adapter._spawn_dynamic_obstacles_impl([_fake_obstacle("ped_0")]))
    finally:
        mod.ArenaHumanDynamicObstacle.from_dynamic_obstacle = original

    agent = client.last_request.agents[0]
    assert agent.policy == "sfm"
    assert agent.relaxation_time == pytest.approx(base_lp["relaxation_time"] + mod._SFM_DETUNED_OFFSETS["relaxation_time"])
    assert agent.repulsion_strength == pytest.approx(base_lp["repulsion_strength"] + mod._SFM_DETUNED_OFFSETS["repulsion_strength"])
    assert agent.repulsion_range == pytest.approx(base_lp["repulsion_range"] + mod._SFM_DETUNED_OFFSETS["repulsion_range"])


# --- Base-layer tests: BaseHumanSimulator.spawn_dynamic_obstacles ----------------
#
# The per-episode driver draw and the driver-change respawn logic both live in the
# shared base class (task_generator.simulators.human.BaseHumanSimulator), so pedestrians
# routed to the "already spawned" move path also pick up a new driver when it changes.


def _fake_dynamic_obstacle(name: str) -> object:
    pose = types.SimpleNamespace(
        position=types.SimpleNamespace(x=0.0, y=0.0),
        orientation=types.SimpleNamespace(to_yaw=lambda: 0.0),
    )
    return types.SimpleNamespace(name=name, sim_path=name, pose=pose, velocity=1.0, waypoints=[])


class _FakeRNGOnce:
    """Stands in for `conf.General.RNG`: each `.once()` call returns the next preset
    driver, ignoring `factory` — simulates one draw per episode without depending on
    the real ParamDist RNG's choice() distribution."""

    def __init__(self, drivers: list[str]) -> None:
        self._drivers = drivers
        self._idx = 0

    def once(self, *namespace: str, factory: object) -> str:
        value = self._drivers[min(self._idx, len(self._drivers) - 1)]
        self._idx += 1
        return value

    def stream(self, *namespace: str) -> object:
        raise AssertionError("stream() should not be reached; once() is faked directly")


def _make_base_sim(driver_set: tuple[str, ...], drivers: list[str] | None = None) -> object:
    from task_generator.simulators.human import BaseHumanSimulator, KnownObstacles

    class _ConcreteHumanSim(BaseHumanSimulator):
        async def _spawn_obstacles_impl(self, obstacles): return obstacles
        async def _spawn_dynamic_obstacles_impl(self, obstacles): return list(obstacles)
        async def _spawn_walls_impl(self, walls): return True
        async def _spawn_doors_impl(self, doors): return True
        async def _remove_obstacles_impl(self, names): return True
        async def _remove_pedestrians_impl(self): return True
        async def _remove_walls_impl(self, names): return True
        async def _remove_doors_impl(self, names): return True
        async def _spawn_robot_impl(self, robots): return (True,) * len(robots)
        async def _remove_robot_impl(self, robots): return (True,) * len(robots)
        async def _move_robot_impl(self, robots): return (True,) * len(robots)
        async def _add_regions_impl(self, regions): return True
        async def _remove_regions_impl(self, regions): return True

    sim = object.__new__(_ConcreteHumanSim)
    sim._known_obstacles = KnownObstacles()
    sim._episode_driver = ""
    sim._spawned_driver = None
    sim._ensure_spawnable = lambda obstacles: _identity(obstacles)

    rng = _FakeRNGOnce(drivers if drivers is not None else [])
    conf = types.SimpleNamespace(
        Human=types.SimpleNamespace(DRIVER_SET=types.SimpleNamespace(value=driver_set)),
        General=types.SimpleNamespace(RNG=rng),
    )
    _noop_logger = types.SimpleNamespace(
        debug=lambda *a, **k: None, info=lambda *a, **k: None,
        warning=lambda *a, **k: None, error=lambda *a, **k: None,
        get_child=lambda *a, **k: _noop_logger,
    )
    sim._NodeInterface__node = types.SimpleNamespace(conf=conf, get_logger=lambda: _noop_logger)

    async def _ok(obstacles):
        return [True] * len(obstacles)

    sim._simulator = types.SimpleNamespace(pedestrian_move=_ok, pedestrian_delete=_ok, pedestrian_spawn=_ok)
    return sim


async def _identity(obstacles):
    return obstacles


def test_episode_driver_draw_goes_through_rng_once():
    """The draw is delegated to `conf.General.RNG.once("humansim", "driver", ...)` —
    the production RNG memoizes that per episode. Here we assert the base layer calls
    it exactly once per `spawn_dynamic_obstacles` invocation (non-empty obstacles),
    and skips it entirely when obstacles is empty."""
    sim = _make_base_sim(driver_set=("orca", "sfm"), drivers=["orca", "sfm"])
    calls = sim._NodeInterface__node.conf.General.RNG

    asyncio.run(sim.spawn_dynamic_obstacles([_fake_dynamic_obstacle("ped_0")]))
    assert calls._idx == 1
    assert sim._episode_driver == "orca"

    asyncio.run(sim.spawn_dynamic_obstacles([]))  # empty obstacles: no draw
    assert calls._idx == 1
    assert sim._episode_driver == "orca"

    asyncio.run(sim.spawn_dynamic_obstacles([_fake_dynamic_obstacle("ped_1")]))
    assert calls._idx == 2
    assert sim._episode_driver == "sfm"


def test_driver_change_triggers_respawn_of_known_obstacles():
    sim = _make_base_sim(driver_set=("orca", "sfm"), drivers=["orca", "sfm"])

    spawn_calls: list[list[object]] = []

    async def _spawn_impl(obstacles):
        spawn_calls.append(list(obstacles))
        return list(obstacles)

    sim._spawn_dynamic_obstacles_impl = _spawn_impl

    despawn_calls: list[list[object]] = []

    async def _delete(obstacles):
        despawn_calls.append(list(obstacles))
        return [True] * len(obstacles)

    move_calls: list[list[object]] = []

    async def _move(obstacles):
        move_calls.append(list(obstacles))
        return [True] * len(obstacles)

    sim._simulator.pedestrian_delete = _delete
    sim._simulator.pedestrian_move = _move

    obstacle = _fake_dynamic_obstacle("ped_0")

    # Episode 1: new obstacle, driver "orca" — registers/spawns via impl, no despawn.
    asyncio.run(sim.spawn_dynamic_obstacles([obstacle]))
    assert sim._episode_driver == "orca"
    assert len(spawn_calls) == 1
    assert despawn_calls == []
    assert move_calls == []

    # Episode 2: driver flips to "sfm" — known+spawned obstacle must be despawned and
    # re-registered (re-spawned via impl) instead of taking the move path.
    asyncio.run(sim.spawn_dynamic_obstacles([obstacle]))
    assert sim._episode_driver == "sfm"
    assert len(despawn_calls) == 1
    assert despawn_calls[0] == [obstacle]
    assert len(spawn_calls) == 2
    assert move_calls == []


def test_empty_driver_set_no_despawn_move_path_preserved():
    """When driver_set is unconfigured, behavior is byte-identical to before the
    driver-DR fix: known+spawned obstacles always take the move path, never despawned."""
    sim = _make_base_sim(driver_set=(), drivers=[])

    despawn_calls: list[list[object]] = []

    async def _delete(obstacles):
        despawn_calls.append(list(obstacles))
        return [True] * len(obstacles)

    move_calls: list[list[object]] = []

    async def _move(obstacles):
        move_calls.append(list(obstacles))
        return [True] * len(obstacles)

    sim._simulator.pedestrian_delete = _delete
    sim._simulator.pedestrian_move = _move

    obstacle = _fake_dynamic_obstacle("ped_0")

    asyncio.run(sim.spawn_dynamic_obstacles([obstacle]))  # episode 1: registers
    asyncio.run(sim.spawn_dynamic_obstacles([obstacle]))  # episode 2: same obstacle

    assert sim._episode_driver == ""
    assert despawn_calls == []
    assert len(move_calls) == 1
    assert move_calls[0] == [obstacle]
