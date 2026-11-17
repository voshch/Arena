from arena_runtime.sim.isaac_simulator import world_roots


def test_world_roots_cover_walls_floors_and_ceilings_of_the_env():
    assert world_roots(3) == ['env_3/Walls', 'env_3/Floors', 'env_3/Ceilings']


def test_world_roots_of_different_envs_are_disjoint():
    assert not set(world_roots(0)) & set(world_roots(1))
