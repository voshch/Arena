"""A bare TM_EdgeCase carrying fakes for everything a reset touches. Shared by the ROS tests."""

from __future__ import annotations

from types import SimpleNamespace


class _Param:
    def __init__(self, value):
        self.value = value


class _Logger:
    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    def warn(self, msg):
        self.messages.append(("warn", str(msg)))

    def info(self, msg):
        self.messages.append(("info", str(msg)))

    def error(self, msg):
        self.messages.append(("error", str(msg)))

    def debug(self, msg):
        self.messages.append(("debug", str(msg)))

    def text(self, level=None):
        return "\n".join(m for lvl, m in self.messages if level is None or lvl == level)


class _Robot:
    """Minimal RobotManager stand-in: `route_goal` is abstract-frame like `start_pos`."""

    def __init__(self, start, goal):
        from task_generator.shared import Pose, Position

        self.start_pos = Pose(Position(*start))
        self.route_goal = Pose(Position(*goal))
        self.goal_pos = Pose(Position(goal[0] + 16.0, goal[1] + 13.0))
        self.pose = None
        self.namespace = "/pytest_robot"


class _Zone:
    def __init__(self, name, corners):
        self.name = name
        self.corners = [SimpleNamespace(x=x, y=y) for x, y in corners]


class _Driver:
    """Stands in for the waypoint driver: records what was armed."""

    def __init__(self):
        self.entries = []
        self.origin = None

    def arm(self, entries, origin=None):
        self.entries = list(entries)
        self.origin = origin

    def outcomes(self):
        return []


class _Base:
    def __init__(self, static, dynamic):
        self._static, self._dynamic = static, dynamic
        self.pending_regions = []

    async def reset(self, **kwargs):
        return list(self._static), list(self._dynamic)

    async def teardown(self):
        pass


class _Env:
    """Environment-manager stand-in: records removals and respawns."""

    def __init__(self):
        self.removed: list[list[str]] = []
        self.spawned: list[list] = []
        self.updated: list[list] = []
        self.update_ok = True
        self.set_agent_waypoints = None

    async def update_agents(self, obstacles):
        self.updated.append(list(obstacles))
        return self.update_ok

    def ezilear(self, p):
        return p

    def realize(self, p):
        return p

    async def remove_obstacles_by_id(self, ids):
        self.removed.append(list(ids))
        return list(ids), []

    async def spawn_dynamic_obstacles(self, obstacles):
        self.spawned.append(list(obstacles))


class _RosParams:
    def __init__(self, values):
        self._values = values
        self.sets: list[tuple[str, object]] = []

    def __getitem__(self, _type):
        return self

    def get(self, name, default=None):
        return self._values.get(name, default)

    def set(self, name, value):
        self._values[name] = value
        self.sets.append((name, value))
        return True


def obstacle(name, start, waypoints, agent_type="adult", speed=1.0, model="arenian"):
    from task_generator.shared import DynamicObstacle, Pose, Position

    return DynamicObstacle(
        name=name, model=model, pose=Pose(Position(*start)),
        waypoints=[Position(*w) for w in waypoints],
        extra={"agent": {"agent_type": agent_type, "desired_velocity": speed}},
    )


def _testable_cls():
    from task_generator.tasks.obstacles.edge_case.impl import TM_EdgeCase

    return type("_TestableEdgeCase", (TM_EdgeCase,), {"node": None, "_logger": None})


def _mode(
    tmp_path, *, robots=None, safe_dist=0.0, score=False, score_scope="auto", objects="",
    scenario="case", zones=(), population=(), block=None, world_map=None, prompt="", prompt_base="",
):
    tm = object.__new__(_testable_cls())
    tm._scorer = None
    tm._score_broken = False
    tm._pending_case = None
    tm._type_cache = {}
    tm._route_cache = None
    tm._objects = None
    tm._objects_broken = False
    tm._timeline_cache = None
    tm._waypoints = None
    tm._waypoints_broken = False
    tm._block = block
    tm._last_block_path = ""
    tm._prompt_scenarios = {}
    tm._population = {}
    tm._logger = _Logger()
    tm.tmp_dir = type("_Tmp", (), {"name": str(tmp_path / "generated")})()
    tm._config = type("_Cfg", (), {
        "read_scenario_block": _Param(True),
        "prompt": _Param(prompt),
        "prompt_base": _Param(prompt_base),
        "record_dir": _Param(str(tmp_path / "records")),
        "objects": _Param(objects),
        "objects_rate_hz": _Param(5.0),
        "score": _Param(score),
        "score_rate_hz": _Param(10.0),
        "score_scope": _Param(score_scope),
    })()
    tm._base_cache = _Base([], list(population))
    tm._pending_regions = []
    tm.aborted: list[str] = []

    driver = _Driver()
    tm._ensure_waypoint_driver = lambda: driver
    tm.driver = driver

    tm._ctx = SimpleNamespace(
        robots=robots if robots is not None else {"jackal": _Robot((0.0, 0.0), (20.0, 0.0))},
        world_manager=SimpleNamespace(
            map=world_map, loaded_world="pytest_world",
            world_compacted=lambda: SimpleNamespace(zones=[_Zone(n, c) for n, c in zones]),
        ),
        environment_manager=_Env(),
        abort_episode=lambda reason: tm.aborted.append(reason),
    )
    tm.node = SimpleNamespace(
        conf=SimpleNamespace(Obstacles=SimpleNamespace(SAFE_DIST=_Param(safe_dist))),
        rosparam=_RosParams({"task.scenario.file": scenario}),
        _episodes=SimpleNamespace(run_seed="run", current=SimpleNamespace(episode_id=1, seed=7)),
        has_parameter=lambda name: False,
        executor=None, event_loop=None,
    )
    return tm
