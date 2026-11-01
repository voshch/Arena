from __future__ import annotations

from types import SimpleNamespace


def test_task_context_robots_property_returns_managers_dict():
    from task_generator.tasks.context import TaskContext

    manager_a = SimpleNamespace(name="robot_a")
    manager_b = SimpleNamespace(name="robot_b")
    managers_dict = {"robot_a": manager_a, "robot_b": manager_b}

    robots_manager = SimpleNamespace(managers=managers_dict)
    env_manager = SimpleNamespace()
    world_manager = SimpleNamespace()

    ctx = TaskContext(
        environment_manager=env_manager,
        robots_manager=robots_manager,
        world_manager=world_manager,
        abort_episode=lambda _: None,
    )

    assert ctx.robots is managers_dict
    assert ctx.robots["robot_a"] is manager_a
    assert ctx.robots["robot_b"] is manager_b


def test_task_context_robots_empty():
    from task_generator.tasks.context import TaskContext

    robots_manager = SimpleNamespace(managers={})
    env_manager = SimpleNamespace()
    world_manager = SimpleNamespace()

    ctx = TaskContext(
        environment_manager=env_manager,
        robots_manager=robots_manager,
        world_manager=world_manager,
        abort_episode=lambda _: None,
    )

    assert ctx.robots == {}


def test_task_context_attributes_accessible():
    from task_generator.tasks.context import TaskContext

    env = SimpleNamespace(test_attr="env")
    rm = SimpleNamespace(managers={"r": SimpleNamespace()})
    wm = SimpleNamespace(test_attr="world")

    ctx = TaskContext(environment_manager=env, robots_manager=rm, world_manager=wm, abort_episode=lambda _: None)

    assert ctx.environment_manager is env
    assert ctx.robots_manager is rm
    assert ctx.world_manager is wm
