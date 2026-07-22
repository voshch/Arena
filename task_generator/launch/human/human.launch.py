import launch
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from arena_bringup.substitutions import LaunchArgument, SelectAction

from task_generator.constants import Constants


def generate_launch_description():

    ld = []

    LaunchArgument.auto_append(ld)

    namespace = LaunchArgument(
        name='namespace',
    )

    enable_auditory = LaunchArgument(
        name="enable_auditory",
        default_value="false",
    )

    enable_sound_visualization = LaunchArgument(
        name="enable_sound_visualization",
        default_value="false",
    )

    robot = LaunchArgument(
        name='robot',
        default_value='jackal',
    )

    propagation_backend = LaunchArgument(
        name="propagation_backend",
        choices=["level3", "pyroomacoustics"],
        default_value="level3",
    )

    launch_human_simulator = SelectAction(launch.substitutions.LaunchConfiguration('simulator'))

    launch_human_simulator.add(
        Constants.HumanSimulator.DUMMY.value,
        launch.actions.GroupAction([])
    )

    launch_human_simulator.add(
        Constants.HumanSimulator.NONE.value,
        launch.actions.GroupAction([])
    )

    launch_human_simulator.add(
        Constants.HumanSimulator.HUNAV.value,
        launch.actions.IncludeLaunchDescription(
            PathJoinSubstitution([
                FindPackageShare('task_generator'),
                'launch', 'human', 'hunav', 'hunav.launch.py',
            ]),
            launch_arguments={
                'use_sim_time': 'true',
                'world_file': '',
                **namespace.dict
            }.items(),
        )
    )

    launch_human_simulator.add(
        Constants.HumanSimulator.ARENA.value,
        launch.actions.GroupAction([
            launch.actions.IncludeLaunchDescription(
                PathJoinSubstitution([
                    FindPackageShare('task_generator'),
                    'launch', 'human', 'arena_humansim', 'arena_humansim.launch.py',
                ]),
                launch_arguments={
                    'use_sim_time': 'true',
                    **namespace.dict
                }.items(),
            ),
            # 1. Sound propagation node
            Node(
                package='task_generator',
                executable='sound_propagation_node',
                name='sound_propagation_node',
                namespace=namespace.substitution,
                output='screen',
                condition=launch.conditions.IfCondition(enable_auditory.substitution),
                parameters=[{
                    "use_sim_time": True,
                    "sound_events_topic": "human_sound_events",
                    "heard_sound_events_topic": "heard_sound_events",
                    "arena_peds_topic": "arena_peds",
                    "map_topic": "map",
                    "world_topic": "state/world",
                    "robot_fleet_topic": "state/robots",
                    "robots_hear_self": True,
                    "propagation_level": 3,
                    "default_hearing_threshold_db": 20.0,
                    # "occlusion_penalty_db": 20.0,
                    "max_first_order_reflections": 8,
                    "reflection_floor_db": -60.0,
                    "ceiling_height_m": 3.0,
                    "publish_inaudible": True,
                    "odom_topic_template": "{namespace}/{name}_velocity_controller/odom",
                    "pyroom_sample_rate_hz": 44100,
                    "pyroom_max_order": 3,
                    "pyroom_temperature_c": 20.0,
                    "pyroom_relative_humidity_percent": 50.0,
                    "pyroom_ceiling_height_m": 3.0,
                    "propagation_backend": propagation_backend.substitution,
                    "portal_adjacency_tolerance_m": 0.08,
                    "portal_inset_m": 0.03,
                    "portal_loss_db": 3.0,
                    "portal_source_early_window_sec": 0.08,
                    "portal_max_rir_duration_sec": 2.0,
                    "portal_position_quantization_m": 0.10,
                    "portal_rir_cache_size": 256,
                    "validate_zone_coverage": True,
                    "zone_coverage_stride_cells": 10,
                }],
            ),

            Node(
                package='task_generator',
                executable='sound_propagation_visualizer',
                name='sound_propagation_visualizer',
                namespace=namespace.substitution,
                output='screen',
                condition=launch.conditions.IfCondition(
                    launch.substitutions.PythonExpression([
                        "'", enable_auditory.substitution,
                        "' == 'true' and '",
                        enable_sound_visualization.substitution,
                        "' == 'true'",
                    ])
                ),
                parameters=[{
                    "use_sim_time": True,
                    "heard_sound_events_topic": "heard_sound_events",
                    "marker_topic": "sound_propagation_markers",
                    "marker_lifetime_sec": 2.0,
                    "path_z_m": 1.0,
                }],
            ),

            # 2. Robot motor sound producer
            Node(
                package='task_generator',
                executable='robot_sound_node',
                name='robot_sound_node',
                namespace=namespace.substitution,
                output='screen',
                condition=launch.conditions.IfCondition(enable_auditory.substitution),
                parameters=[{
                    "use_sim_time": True,
                    "robot_fleet_topic": "state/robots",
                    "sound_events_topic": "human_sound_events",
                    "odom_topic_template": "{namespace}/{name}_velocity_controller/odom",
                    "sound_type": "motor",
                    "motor_start_asset_id": "motor_start",
                    "motor_stop_asset_id": "motor_stop",
                    "source_volume_db": 55.0,
                    "publish_period_sec": 0.5,
                    "only_when_moving": True,
                    "min_speed_mps": 0.05,
                    "stop_speed_mps": 0.03,
                    "publish_motor_markers": True,
                    "motor_marker_topic": "motor_sound_markers",
                    "motor_marker_lifetime_sec": 0.8,
                    "motor_marker_z_m": 0.16,
                    "motor_marker_line_width_m": 0.055,
                    "motor_marker_arc_degrees": 300.0,
                    "motor_marker_radii_m": [0.45, 0.75, 1.05],
                }],
            ),

             # 3. Robot hearing node
            Node(
                package='task_generator',
                executable='robot_hearing_node',
                name='robot_hearing_node',
                namespace=namespace.substitution,
                output='screen',
                condition=launch.conditions.IfCondition(enable_auditory.substitution),
                parameters=[{
                    "use_sim_time": True,
                    "robot_fleet_topic": "state/robots",
                    "heard_sound_events_topic": "heard_sound_events",
                    "heard_sound_topic_suffix": "heard_sound",
                    "marker_topic_suffix": "heard_sound_marker",
                    "ignore_self": True,
                    "min_snr_db": -5.0,
                    "honor_propagation_delay": True,
                    "marker_lifetime_sec": 1.5,
                    "marker_z_offset": 1.2,
                    "marker_text_height_m": 0.35,
                    "marker_sound_types": ["greeting", "footstep", "motor"],
                }],
            ),

            # 4. Human sound playback node
            Node(
                package='task_generator',
                executable='human_sound_playback',
                name='human_sound_playback',
                namespace=namespace.substitution,
                output='screen',
                condition=launch.conditions.IfCondition(enable_auditory.substitution),
                parameters=[{
                    "sound_events_topic": "human_sound_events",
                    "heard_sound_events_topic": "heard_sound_events",
                    "use_rir": True,
                    "listener_robot_name": "jackal",
                    "world_topic": "state/world",
                    "rir_sample_rate_hz": 44100,
                    "rir_max_order": 1,
                    "rir_temperature_c": 20.0,
                    "rir_relative_humidity_percent": 50.0,
                    "rir_ceiling_height_m": 3.0,
                    "rir_dry_fallback": True,
                    "portal_adjacency_tolerance_m": 0.08,
                    "portal_inset_m": 0.03,
                    "portal_loss_db": 3.0,
                    "portal_source_early_window_sec": 0.08,
                    "portal_max_rir_duration_sec": 2.0,
                    "portal_position_quantization_m": 0.10,
                    "portal_rir_cache_size": 256,
                    "play_inaudible_events": True,
                    "minimum_playback_gain_db": -60.0,
                    "episode_topic": "state/episode",
                    "output_sample_rate": 44100,
                    "output_channels": 2,
                    "block_size":  8192,
                    "audio_device": "hw:0,0",
                    "master_gain_db": 0.0,
                }],
            ),
        ])
    )

    simulator = LaunchArgument(
        name='simulator',
        choices=launch_human_simulator.keys,
    )

    ld = launch.LaunchDescription([
        *ld,
        launch_human_simulator,
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
