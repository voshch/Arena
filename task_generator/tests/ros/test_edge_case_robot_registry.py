from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


def test_edge_case_is_in_the_robot_enum():
    from task_generator.constants import Constants

    assert Constants.TaskMode.TM_Robots("edge_case") is Constants.TaskMode.TM_Robots.EDGE_CASE


def test_registered_without_a_humansim_backend():
    """Unlike the obstacle-side edge_case mode, this one has no HumanSim dependency, so it
    registers at import and is available under any `human:=` backend."""
    from task_generator.constants import Constants
    from task_generator.tasks.registry import ROBOTS_MODES

    assert Constants.TaskMode.TM_Robots.EDGE_CASE in ROBOTS_MODES


def test_loader_returns_the_class():
    from task_generator.constants import Constants
    from task_generator.tasks.registry import ROBOTS_MODES
    from task_generator.tasks.robots.edge_case.impl import TM_EdgeCase

    assert ROBOTS_MODES.get(Constants.TaskMode.TM_Robots.EDGE_CASE) is TM_EdgeCase


def test_namespace_is_separate_from_the_obstacle_mode():
    """Sharing `task.edge_case` with the obstacle mode would only fail at node start, on a
    leaf-name collision. Keep them apart."""
    from task_generator.constants import Constants
    from task_generator.tasks.registry import OBSTACLES_MODES, ROBOTS_MODES

    robots_ns = str(ROBOTS_MODES.meta(Constants.TaskMode.TM_Robots.EDGE_CASE).namespace)
    assert "edge_case_robot" in robots_ns

    try:
        obstacles_ns = str(OBSTACLES_MODES.meta(Constants.TaskMode.TM_Obstacles.EDGE_CASE).namespace)
    except KeyError:
        return  # obstacle mode registers from the HumanSim adapter; not loaded here
    assert robots_ns != obstacles_ns


def test_schema_declares_every_param():
    from arena_rclpy_mixins.shared import Namespace
    from task_generator.tasks.robots.edge_case import declare_schema

    declared: list[str] = []

    class _FakeRosParam:
        def declare_forward(self, name, default, *args, **kwargs):
            declared.append(str(name))

    class _FakeNode:
        rosparam = _FakeRosParam()

    declare_schema(_FakeNode(), Namespace("task")("edge_case_robot"))
    leaves = {n.rsplit(".", 1)[-1].rsplit("/", 1)[-1] for n in declared}
    assert {"traversals", "blocked_distance", "blocked_timeout", "on_blocked"} <= leaves


def test_reads_the_shared_scenario_param():
    """The route must come from the same scenario the obstacle mode is populating, so the
    file is bound to `task.scenario.file`, not to this mode's own namespace."""
    import inspect

    from task_generator.tasks.robots.edge_case import impl

    src = inspect.getsource(impl.TM_EdgeCase.__init__)
    assert '_REGISTRY_NAMESPACE("scenario")' in src
    assert "TM_Robots.__init__" in src, "must skip TM_Scenario.__init__, which binds the wrong namespace"


class _FakeRealizer:
    """Applies a fixed env grid offset, like the real realizer does for env_N."""

    OFFSET = (16.0, 13.0)

    def realize(self, target):
        from task_generator.shared import Pose, Position

        return Pose(
            Position(
                x=target.position.x + self.OFFSET[0],
                y=target.position.y + self.OFFSET[1],
                z=target.position.z,
            ),
            target.orientation,
        )


def _submit_target():
    """A RobotManager carrying only what submit_task's frame handling touches.

    `node` and `_logger` are read-only NodeInterface properties, so they are demoted to plain
    attributes on a subclass — the same trick test_edge_case_apply uses.
    """
    from task_generator.manager.robot_manager.robot_manager import RobotManager
    from task_generator.shared import Pose

    cls = type("_TestableRM", (RobotManager,), {"node": None, "_logger": None, "name": "jackal"})
    rm = object.__new__(cls)
    rm._start_pos = Pose()
    rm._goal_pos = Pose()
    rm._route_goal = Pose()
    rm._environment_manager = _FakeRealizer()
    rm._adapters = {}
    rm._unsupported_kinds_logged = set()
    rm._current_request = None
    rm._phase_index = 0
    return rm


async def _submit(rm, *poses):
    from task_generator.tasks.robots.request import GoToPhase, TaskRequest

    class _WM:
        @staticmethod
        def level_of_point(_x, _y):
            return ""

    class _Node:
        _world_manager = _WM()

    class _Robot:
        record_data_dir = ""

    class _Logger:
        def warning(self, *_a, **_k):
            pass

    rm.node = _Node()
    rm._robot = _Robot()
    rm._logger = _Logger()
    await rm.submit_task(TaskRequest(phases=[GoToPhase(pose=p) for p in poses]))


def test_route_goal_is_abstract_and_goal_pos_is_map_frame():
    """The two exist because they are in different frames and have different owners.

    `goal_pos` is published on `<ns>/goal_pose` with frame_id "map", so it must be realized.
    `route_goal` is compared against `start_pos` and against obstacle poses, all of which are
    abstract. Conflating them made the robot's "route" leave the world entirely.
    """
    import asyncio

    from task_generator.shared import Pose, Position

    rm = _submit_target()
    goal = Pose(Position(x=3.0, y=-2.0))
    asyncio.run(_submit(rm, goal))

    assert (rm.route_goal.position.x, rm.route_goal.position.y) == (3.0, -2.0)
    assert (rm.goal_pos.position.x, rm.goal_pos.position.y) == (19.0, 11.0)


def test_resubmitting_route_goal_does_not_realize_it_twice():
    """`tm_robots:=edge_case` reads the goal back to build its traversal legs. Feeding
    `goal_pos` in would realize an already-realized pose, putting the robot two env offsets
    outside the map — which is exactly what happened."""
    import asyncio

    from task_generator.shared import Pose, Position

    rm = _submit_target()
    asyncio.run(_submit(rm, Pose(Position(x=3.0, y=-2.0))))

    # Round-trip the abstract goal back through submit_task, as the robot mode does.
    asyncio.run(_submit(rm, rm.route_goal, rm.route_goal))

    assert (rm.route_goal.position.x, rm.route_goal.position.y) == (3.0, -2.0)
    assert (rm.goal_pos.position.x, rm.goal_pos.position.y) == (19.0, 11.0)


def test_goal_is_populated_at_all():
    """Regression: both were assigned only at construction, so every robot reported the map
    origin as its destination for its whole life."""
    import asyncio

    from task_generator.shared import Pose, Position

    rm = _submit_target()
    asyncio.run(_submit(rm, Pose(Position(x=5.0, y=5.0))))
    assert (rm.route_goal.position.x, rm.route_goal.position.y) != (0.0, 0.0)


def test_robot_spawn_clearance_is_robot_sized_not_pedestrian_sized():
    """A robot placed to pedestrian tolerances is routinely immobile.

    nav2 stops on anything inside its collision-monitor stop polygon (+/-0.3 m for the jackal)
    and inflates obstacles by a further 0.55 m in the costmap. `Obstacles.SAFE_DIST` is the
    clearance a *pedestrian* needs and can legitimately be as low as 0.15 m, which produces
    starts the robot cannot drive out of - observed live as a silent `cmd_vel` with the monitor
    latched on `StopPolygon`.
    """
    from task_generator.tasks.robots.scenario.impl import TM_Scenario

    # `node` is a read-only NodeInterface property; demote it so a bare instance can carry a fake.
    tm = object.__new__(type("_TestableScenario", (TM_Scenario,), {"node": None}))

    class _Robot:
        safe_distance = 0.52

    tm._ctx = type("_Ctx", (), {"robots": {"jackal": _Robot()}})()
    assert tm._spawn_clearance() == pytest.approx(0.52)

    # Before any robot exists, fall back to a nominal robot rather than the pedestrian value.
    tm._ctx = type("_Ctx", (), {"robots": {}})()
    tm.node = type(
        "_Node", (), {"conf": type("_C", (), {"Robot": type("_R", (), {"SPAWN_ROBOT_SAFE_DIST": type("_P", (), {"value": 0.25})()})()})()}
    )()
    assert tm._spawn_clearance() == pytest.approx(0.55)
    assert tm._spawn_clearance() > 0.15, "must not collapse to the pedestrian clearance"


def test_crowd_clearance_adds_the_pedestrian_body_radius():
    """The forbidden disc must keep pedestrian BODIES out, not merely their centres.

    Placement only requires an agent's *centre* to clear the disc, and the sampler erodes
    occupancy by `Obstacles.SAFE_DIST` (0.15 m), so an agent legally lands at
    `safe_distance + 0.15` and then puts a ~0.25 m body radius back toward the robot.
    Measured on arena_arena_002: agents at 0.66 m presenting surfaces ~0.41 m from the robot
    centre, inside nav2's 0.35 m inflation - `cmd_vel` silent, "Failed to make progress", and
    every recovery aborting on "Collision Ahead". No traversal ever completed.

    Regression guard for that fix. It is the one thing standing between a run and a robot
    that never moves, and nothing else pins it.
    """
    from task_generator.tasks.robots.scenario.impl import TM_Scenario

    tm = object.__new__(type("_TestableScenario", (TM_Scenario,), {"node": None}))
    tm.node = type(
        "_Node",
        (),
        {"conf": type("_C", (), {"Obstacles": type("_O", (), {"PEDESTRIAN_BODY_RADIUS": type("_P", (), {"value": 0.25})()})()})()},
    )()

    assert tm._crowd_clearance(0.52) == pytest.approx(0.77)
    # The measured failure: 0.52 alone admitted an agent whose body reached inside nav2's
    # 0.35 m inflation. Anything at or below the robot's own safe distance reproduces it.
    assert tm._crowd_clearance(0.52) > 0.52


def test_crowd_clearance_is_wider_than_spawn_clearance():
    """Stated as a relation rather than two numbers: whatever the robot needs around its own
    centre, the crowd must be held further out than that, by the pedestrian's own extent."""
    from task_generator.tasks.robots.scenario.impl import TM_Scenario

    tm = object.__new__(type("_TestableScenario", (TM_Scenario,), {"node": None}))
    tm.node = type(
        "_Node",
        (),
        {
            "conf": type(
                "_C",
                (),
                {
                    "Obstacles": type("_O", (), {"PEDESTRIAN_BODY_RADIUS": type("_P", (), {"value": 0.25})()})(),
                    "Robot": type("_R", (), {"SPAWN_ROBOT_SAFE_DIST": type("_P", (), {"value": 0.25})()})(),
                },
            )()
        },
    )()
    tm._ctx = type("_Ctx", (), {"robots": {}})()

    spawn = tm._spawn_clearance()
    assert tm._crowd_clearance(spawn) > spawn
