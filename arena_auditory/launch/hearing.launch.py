"""Bring up the hearing belief layer beside a running Arena env.

Starts, under ``env_namespace``:

  * hearing_belief: the belief node, publishing the belief grid
  * hearing_policy: listen-then-yield, the speed-filter mask writer
  * costmap_filter_info_server: publishes hearing/costmap_filter_info
  * lifecycle_manager: transitions the filter-info server
  * seld_frontend: only with source:=seld, the live SELDnet front-end

``arena launch ... robot.hearing:=bus|seld`` includes this per env and merges
config/hearing/nav2_overlay.yaml (a SpeedFilter on the local costmap and the
controller's speed_limit_topic) into the robot's Nav2 parameters. The nodes bind
to the robot announced on <tg_node>/state/robots, so ``robot`` only needs to be
set with more than one robot per env.

source:=bus   consumes <tg_node>/<robot>/heard_sound off the simulator bus (map-frame bearings, 2 Hz)
source:=seld  runs the front-end on <tg_node>/<robot>/audio/raw_array and consumes
              <tg_node>/<robot>/heard_sound_seld (robot-frame bearings, 10 Hz)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg = get_package_share_directory("arena_auditory")
    default_params = os.path.join(pkg, "config", "hearing", "speed_filter.yaml")

    args = [
        DeclareLaunchArgument("env_namespace", default_value="/arena/env_0", description="Arena env namespace"),
        DeclareLaunchArgument("robot", default_value="", description="robot to bind; empty = the first robot on the fleet topic"),
        DeclareLaunchArgument("tg_node", default_value="task_generator_node", description="task generator node name; the bus topics live below it"),
        DeclareLaunchArgument("source", default_value="bus", choices=["bus", "seld"]),
        DeclareLaunchArgument("params_file", default_value=default_params),
        DeclareLaunchArgument("policy_params_file", default_value=os.path.join(pkg, "config", "hearing", "policy.yaml")),
        DeclareLaunchArgument("policy", default_value="full", choices=["belief", "listen", "full"], description="mask layers: belief only, plus the corner listen cap, plus yield"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("autostart", default_value="true"),
        DeclareLaunchArgument("device", default_value="cuda", description="torch device for the front-end"),
        DeclareLaunchArgument("lookahead_frames", default_value="5", description="front-end label frames of future context (100 ms each)"),
        DeclareLaunchArgument("bearing_source", default_value="gcc", choices=["gcc", "seld"], description="front-end bearing: GCC-PHAT fit over the array, or the model's azimuth"),
    ]

    ns = LaunchConfiguration("env_namespace")
    robot = LaunchConfiguration("robot")
    source = LaunchConfiguration("source")
    params_file = LaunchConfiguration("params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")
    is_seld = PythonExpression(["'", source, "' == 'seld'"])

    tg = LaunchConfiguration("tg_node")
    fleet_topic = [tg, "/state/robots"]
    heard_suffix = PythonExpression(["'heard_sound_seld' if ", is_seld, " else 'heard_sound'"])
    bearing_frame = PythonExpression(["'robot' if ", is_seld, " else 'map'"])
    event_rate = PythonExpression(["10.0 if ", is_seld, " else 2.0"])
    mask_topic = [ns, "/hearing/speed_filter_mask"]
    policy = LaunchConfiguration("policy")
    filter_info_topic = [ns, "/hearing/costmap_filter_info"]

    belief = Node(
        package="arena_auditory",
        executable="hearing_belief_node",
        name="hearing_belief",
        namespace=ns,
        output="screen",
        parameters=[
            params_file,
            {
                "use_sim_time": use_sim_time,
                "robot_fleet_topic": fleet_topic,
                "robot": robot,
                "heard_sound_suffix": heard_suffix,
                "map_topic": [tg, "/map"],
                "reset_topic": [tg, "/state/resetting"],
                "belief_topic": "hearing/belief_grid",
                "marker_topic": "hearing/belief_wedges",
                "bearing_frame": bearing_frame,
                "nominal_event_rate_hz": event_rate,
            },
        ],
    )

    policy_node = Node(
        package="arena_auditory",
        executable="hearing_policy",
        name="hearing_policy",
        namespace=ns,
        output="screen",
        parameters=[
            LaunchConfiguration("policy_params_file"),
            {
                "use_sim_time": use_sim_time,
                "robot_fleet_topic": fleet_topic,
                "robot": robot,
                "heard_sound_suffix": heard_suffix,
                "map_topic": [tg, "/map"],
                "reset_topic": [tg, "/state/resetting"],
                "belief_topic": "hearing/belief_grid",
                "speed_mask_topic": mask_topic,
                "listen_enabled": PythonExpression(["'", policy, "' != 'belief'"]),
                "yield_enabled": PythonExpression(["'", policy, "' == 'full'"]),
            },
        ],
    )

    filter_info = Node(
        package="nav2_map_server",
        executable="costmap_filter_info_server",
        name="costmap_filter_info_server",
        namespace=ns,
        output="screen",
        parameters=[
            params_file,
            {
                "use_sim_time": use_sim_time,
                "mask_topic": mask_topic,
                "filter_info_topic": filter_info_topic,
            },
        ],
    )

    lifecycle = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_hearing_filter",
        namespace=ns,
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "autostart": LaunchConfiguration("autostart"),
                "node_names": ["costmap_filter_info_server"],
            }
        ],
    )

    frontend = Node(
        package="arena_auditory",
        executable="hearing_seld_frontend",
        name="seld_frontend",
        namespace=ns,
        output="screen",
        condition=IfCondition(is_seld),
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "robot_fleet_topic": fleet_topic,
                "robot": robot,
                "device": LaunchConfiguration("device"),
                "lookahead_frames": LaunchConfiguration("lookahead_frames"),
                "bearing_source": LaunchConfiguration("bearing_source"),
            }
        ],
    )

    return LaunchDescription(args + [belief, policy_node, filter_info, lifecycle, frontend])
