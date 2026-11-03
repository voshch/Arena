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

    PLUGIN_FULL_RANGE_LASER = {
        "type": "Laser",
        "name": "full_static_laser",
        "frame": "full_laser",
        "topic": "full_scan",
        "body": "base_link",
        "broadcast_tf": "true",
        "origin": [0, 0, 0],
        "range": 2.0,
        "angle": {"min": -3.14, "max": 3.14, "increment": 0.01745},
        "noise_std_dev": 0.0,
        "update_rate": 10,
    }


class UnityConstants:
    WALL_HEIGHT = 4.0
    ATTACH_SAFE_DIST_SENSOR_TOPIC = "attach_safe_dist_sensor"
