from arena_viz import env_namespace


def test_env_namespace_is_the_task_generator_node_without_its_name() -> None:
    assert env_namespace("/arena/env_3/arena_env_node") == "/arena/env_3"
    assert env_namespace("/arena/env_3/arena_env_node/") == "/arena/env_3"


def test_env_namespace_of_a_root_task_generator_is_empty() -> None:
    assert env_namespace("/arena_env_node") == ""
