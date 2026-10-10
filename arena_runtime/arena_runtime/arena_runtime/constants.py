from enum import Enum


class SimSimulator(Enum):
    DUMMY = "dummy"
    FLATLAND = "flatland"
    GAZEBO = "gazebo"
    UNITY = "unity"
    ISAAC = "isaac"
    MUJOCO = "mujoco"
