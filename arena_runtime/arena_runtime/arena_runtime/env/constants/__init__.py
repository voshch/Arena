from enum import Enum


class Constants:
    ENV_NODE_NAMESPACE = "arena_env_node"
    DEFAULT_PEDESTRIAN_MODEL = "arenian"

    class HumanSimulator(Enum):
        DUMMY = "dummy"
        NONE = "none"
        HUNAV = "hunav"
        ARENA = "arena"
        REPLAY = "replay"

    class AcousticsSimulator(Enum):
        NONE = "none"
        ARENA = "arena"

    HUMAN_PARAM_NAMESPACES = {HumanSimulator.ARENA: "humansim"}
    HUMAN_PARAM_RESERVED = "human"

    class MapGenerator:
        NODE_NAME = "map_generator"
        MAP_FOLDER_NAME = "dynamic_map"


class UnityConstants:
    WALL_HEIGHT = 4.0
    ATTACH_SAFE_DIST_SENSOR_TOPIC = "attach_safe_dist_sensor"
