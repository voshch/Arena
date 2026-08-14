import launch
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from arena_bringup.substitutions import LaunchArgument, SelectAction

from task_generator.constants import Constants


def generate_launch_description():

    ld = []

    LaunchArgument.auto_append(ld)

    namespace = LaunchArgument(
        name='namespace',
    )

    environment_namespace = LaunchArgument(
        name='environment_namespace',
    )

    enable_auditory = LaunchArgument(
        name="enable_auditory",
        default_value="true",
    )

    enable_sound_visualization = LaunchArgument(
        name="enable_sound_visualization",
        default_value="true",
    )

    enable_robot_sound = LaunchArgument(
        name="enable_robot_sound",
        default_value="true",
    )

    enable_motor_playback = LaunchArgument(
        name="enable_motor_playback",
        default_value="true",
    )

    enable_environment_playback = LaunchArgument(
        name="enable_environment_playback",
        default_value="true",
    )

    audio_block_size = LaunchArgument(
        name="audio_block_size",
        default_value="2048",
    )

    audio_device = LaunchArgument(
        name="audio_device",
        default_value="pulse",
    )

    audio_asset_catalog = LaunchArgument(
        name="audio_asset_catalog",
        default_value=PathJoinSubstitution([
            FindPackageShare("task_generator"),
            "config", "auditory", "acoustic_assets.yaml",
        ]),
    )

    audio_sound_dir = LaunchArgument(
        name="audio_sound_dir",
        default_value=PathJoinSubstitution([
            FindPackageShare("task_generator"),
            "sounds",
        ]),
    )

    robot = LaunchArgument(
        name='robot',
        default_value='jackal',
    )

    propagation_backend = LaunchArgument(
        name="propagation_backend",
        choices=["level3", "pyroomacoustics"],
        default_value="pyroomacoustics",
    )
    enable_multi_portal_rir = LaunchArgument(
        name="enable_multi_portal_rir",
        default_value="true",
    )
    compute_rir_in_propagation = LaunchArgument(
        name="compute_rir_in_propagation",
        default_value="true",
    )

    motor_audio_mode = LaunchArgument(
        name="motor_audio_mode",
        choices=["procedural", "wav"],
        default_value="procedural",
    )

    audio_listener_robot = LaunchArgument(
        name="audio_listener_robot",
        default_value="jackal",
    )

    audio_listener_frame = LaunchArgument(
        name="audio_listener_frame",
        default_value="",
    )

    audio_listener_id = LaunchArgument(
        name="audio_listener_id",
        default_value="",
    )

    audio_listener_mode = LaunchArgument(
        name="audio_listener_mode",
        choices=["selected", "list", "all"],
        default_value="selected",
    )

    audio_listener_ids = LaunchArgument(
        name="audio_listener_ids",
        default_value="[]",
    )

    audio_robot_microphones = LaunchArgument(
        name="audio_robot_microphones",
        default_value="[]",
    )

    motor_playback_mode = LaunchArgument(
        name="motor_playback_mode",
        choices=["sequence", "single_loop"],
        default_value="sequence",
    )

    playback_parameters = {
        "sound_events_topic": "human_sound_events",
        "heard_sound_events_topic": "heard_sound_events",
        "use_rir": True,
        "listener_robot_name": audio_listener_robot.substitution,
        "listener_id": audio_listener_id.substitution,
        "listener_mode": audio_listener_mode.substitution,
        "listener_ids": audio_listener_ids.param_value(str),
        "microphone_listeners_topic": "microphone_listeners",
        "world_topic": "state/world",
        "rir_sample_rate_hz": 44100,
        "rir_max_order": 3,
        "rir_temperature_c": 20.0,
        "rir_relative_humidity_percent": 50.0,
        "rir_ceiling_height_m": 3.0,
        "rir_cache_position_quantization_m": 0.1,
        "rir_cache_size": 512,
        "rir_dry_fallback": True,
        "portal_adjacency_tolerance_m": 0.2,
        "portal_inset_m": 0.03,
        "portal_loss_db": 3.0,
        "opening_portal_loss_db": 0.5,
        "derive_opening_portals": True,
        "minimum_opening_width_m": 0.2,
        **enable_multi_portal_rir.param(bool),
        "max_portal_hops": 4,
        "route_distance_loss_db_per_m": 0.05,
        "portal_source_early_window_sec": 0.08,
        "portal_max_rir_duration_sec": 2.0,
        "portal_position_quantization_m": 0.10,
        "portal_rir_cache_size": 256,
        "rir_event_buffer_size": 128,
        "play_inaudible_events": True,
        "minimum_playback_gain_db": -60.0,
        "episode_topic": "state/episode",
        "output_sample_rate": 44100,
        "output_channels": 1,
        "block_size": audio_block_size.param_value(int),
        "audio_device": audio_device.substitution,
        "asset_catalog": audio_asset_catalog.substitution,
        "sound_dir": audio_sound_dir.substitution,
        "master_gain_db": 0.0,
    }

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
            Node(
                package='task_generator',
                executable='human_sound_node',
                name='human_sound_node',
                namespace=namespace.substitution,
                output='screen',
                condition=launch.conditions.IfCondition(
                    enable_auditory.substitution
                ),
                parameters=[{
                    "use_sim_time": True,
                    "arena_peds_topic": PathJoinSubstitution([
                        environment_namespace.substitution,
                        "arena_peds",
                    ]),
                    "world_topic": "state/world",
                    "sound_events_topic": "human_sound_events",
                    "sound_markers_topic": PathJoinSubstitution([
                        environment_namespace.substitution,
                        "pedestrian_markers",
                        "extra",
                    ]),
                }],
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
                    "continuous_audio_sources_topic":
                        "continuous_audio_sources",
                    "continuous_heard_sounds_topic":
                        "continuous_heard_sounds",
                    "robot_microphones":
                        audio_robot_microphones.param_value(str),
                    "microphone_listeners_topic":
                        "microphone_listeners",
                    "robot_listener_frame":
                        audio_listener_frame.substitution,
                    "arena_peds_topic": PathJoinSubstitution([
                        environment_namespace.substitution,
                        "arena_peds",
                    ]),
                    "map_topic": "map",
                    "world_topic": "state/world",
                    "episode_topic": "state/episode",
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
                    "pyroom_max_order": 1,
                    "pyroom_temperature_c": 20.0,
                    "pyroom_relative_humidity_percent": 50.0,
                    "pyroom_ceiling_height_m": 3.0,
                    "pyroom_cache_position_quantization_m": 0.10,
                    "pyroom_cache_size": 512,
                    "pyroom_robot_listeners_only": True,
                    **compute_rir_in_propagation.param(bool),
                    "propagation_backend": propagation_backend.substitution,
                    "portal_adjacency_tolerance_m": 0.2,
                    "portal_inset_m": 0.03,
                    "portal_loss_db": 3.0,
                    "opening_portal_loss_db": 0.5,
                    "derive_opening_portals": True,
                    "minimum_opening_width_m": 0.2,
                    # Each authored zone is one ordinary pyroomacoustics
                    # room. Cross-zone rendering may follow a portal route.
                    **enable_multi_portal_rir.param(bool),
                    "max_portal_hops": 4,
                    "route_distance_loss_db_per_m": 0.05,
                    "portal_source_early_window_sec": 0.08,
                    "portal_max_rir_duration_sec": 2.0,
                    "portal_position_quantization_m": 0.10,
                    "portal_rir_cache_size": 256,
                    "validate_zone_coverage": False,
                    "zone_coverage_stride_cells": 10,
                    "zone_coverage_tolerance_m": 0.25,
                    "buffer_events_until_scene_loaded": True,
                    "scene_event_buffer_size": 128,
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
                    "continuous_audio_sources_topic":
                        "continuous_audio_sources",
                    "continuous_heard_sounds_topic":
                        "continuous_heard_sounds",
                    "environmental_source_marker_topic":
                        "environmental_audio_source_markers",
                    "pedestrian_marker_topic":
                        "pedestrian_sound_propagation_markers",
                    "robot_marker_topic":
                        "robot_sound_propagation_markers",
                    "robot_fleet_topic": "state/robots",
                    "sync_robot_listener_to_tf": True,
                    "marker_lifetime_sec": 5.0,
                    "path_z_m": 1.0,
                }],
            ),

            # 2. Robot motor sound producer and playback
            Node(
                package='task_generator',
                executable='robot_sound_node',
                name='robot_sound_node',
                namespace=namespace.substitution,
                output='screen',
                condition=launch.conditions.IfCondition(enable_auditory.substitution),
                parameters=[{
                    "use_sim_time": True,
                    **playback_parameters,
                    **enable_robot_sound.param(bool),
                    **enable_motor_playback.param(bool),
                    "robot_fleet_topic": "state/robots",
                    "continuous_audio_sources_topic":
                        "continuous_audio_sources",
                    "continuous_heard_sounds_topic":
                        "continuous_heard_sounds",
                    "motor_audio_mode": motor_audio_mode.substitution,
                    "motor_rir_crossfade_sec": 0.10,
                    "motor_playback_mode": motor_playback_mode.substitution,
                    "motor_single_asset_id": "motor",
                    "odom_topic_template": "{namespace}/{name}_velocity_controller/odom",
                    "sound_type": "motor",
                    "motor_start_asset_id": "motor_start",
                    "motor_stop_asset_id": "motor_stop",
                    "source_volume_db": 45.0,
                    "publish_period_sec": 0.05,
                    "only_when_moving": True,
                    "min_speed_mps": 0.05,
                    "stop_speed_mps": 0.03,
                    "angular_speed_scale_m": 0.25,
                    "publish_motor_markers": True,
                    "motor_marker_topic": "",
                    "motor_marker_topic_suffix": "motor_sound_markers",
                    "motor_marker_lifetime_sec": 0.8,
                    "motor_marker_z_m": 0.16,
                    "motor_marker_line_width_m": 0.055,
                    "motor_marker_cone_degrees": 70.0,
                    "motor_marker_range_m": 1.25,
                }],
            ),

            Node(
                package='task_generator',
                executable='environmental_sound_playback',
                name='environmental_sound_playback',
                namespace=namespace.substitution,
                output='screen',
                condition=launch.conditions.IfCondition(
                    enable_auditory.substitution
                ),
                parameters=[{
                    **playback_parameters,
                    "continuous_heard_sounds_topic":
                        "continuous_heard_sounds",
                    **enable_environment_playback.param(bool),
                    "environment_rir_crossfade_sec": 0.10,
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
                    **playback_parameters,
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
