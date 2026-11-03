from enum import Enum


class SimSimulator(Enum):
    DUMMY = "dummy"
    FLATSIM = "flatsim"
    FLATLAND = "flatland"
    GAZEBO = "gazebo"
    UNITY = "unity"
    ISAAC = "isaac"
    MUJOCO = "mujoco"
