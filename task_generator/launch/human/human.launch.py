import launch
from launch.substitutions import PathJoinSubstitution
import launch_ros.parameter_descriptions
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

    auditory = LaunchArgument(
        name="auditory",
        choices=["none", "arena"],
        default_value="none",
    )
    auditory_viz = LaunchArgument(
        name="auditory.viz",
        default_value="false",
    )
    auditory_playback = LaunchArgument(
        name="auditory.playback",
        default_value="auto",
        description="PortAudio output device for local playback; auto = system default, none = no playback nodes",
    )
    auditory_block_size = LaunchArgument(
        name="auditory.block_size",
        default_value="2048",
    )
    auditory_assets = LaunchArgument(
        name="auditory.assets",
        default_value=PathJoinSubstitution([
            FindPackageShare("task_generator"),
            "config", "auditory", "acoustic_assets.yaml",
        ]),
    )
    auditory_sound_dir = LaunchArgument(
        name="auditory.sound_dir",
        default_value=PathJoinSubstitution([
            FindPackageShare("task_generator"),
            "sounds",
        ]),
    )
    auditory_propagation = LaunchArgument(
        name="auditory.propagation",
        choices=["level3", "pyroomacoustics"],
        default_value="pyroomacoustics",
    )
    auditory_multi_portal = LaunchArgument(
        name="auditory.multi_portal",
        default_value="true",
    )
    auditory_rir_in_propagation = LaunchArgument(
        name="auditory.rir_in_propagation",
        default_value="true",
    )
    auditory_robot_sound = LaunchArgument(
        name="auditory.robot_sound",
        default_value="true",
    )
    auditory_motor = LaunchArgument(
        name="auditory.motor",
        choices=["off", "wav", "procedural"],
        default_value="procedural",
    )
    auditory_motor_playback = LaunchArgument(
        name="auditory.motor.playback",
        choices=["sequence", "single_loop"],
        default_value="sequence",
    )
    auditory_environment_playback = LaunchArgument(
        name="auditory.environment_playback",
        default_value="true",
    )
    auditory_listener = LaunchArgument(
        name="auditory.listener",
        default_value="",
    )
    auditory_listener_frame = LaunchArgument(
        name="auditory.listener_frame",
        default_value="",
    )
    auditory_microphones = LaunchArgument(
        name="auditory.microphones",
        default_value="[]",
    )
    auditory_viewport_height = LaunchArgument(
        name="auditory.viewport_height",
        default_value="1.6",
    )

    robot = LaunchArgument(
        name='robot',
        default_value='jackal',
    )

    auditory_on = launch.conditions.IfCondition(
        launch.substitutions.PythonExpression([
            "'", auditory.substitution, "' != 'none'",
        ])
    )
    auditory_playback_on = launch.conditions.IfCondition(
        launch.substitutions.PythonExpression([
            "'", auditory.substitution, "' != 'none' and '",
            auditory_playback.substitution, "' != 'none'",
        ])
    )
    auditory_viz_on = launch.conditions.IfCondition(
        launch.substitutions.PythonExpression([
            "'", auditory.substitution, "' != 'none' and '",
            auditory_viz.substitution, "' == 'true'",
        ])
    )
    motor_enabled = launch.substitutions.PythonExpression([
        "'", auditory_motor.substitution, "' != 'off'",
    ])
    motor_audio_mode = launch.substitutions.PythonExpression([
        "'procedural' if '", auditory_motor.substitution,
        "' == 'off' else '", auditory_motor.substitution, "'",
    ])

    playback_parameters = {
        "sound_events_topic": "human_sound_events",
        "heard_sound_events_topic": "heard_sound_events",
        "use_rir": True,
        "listener_id": auditory_listener.substitution,
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
        "enable_multi_portal_rir": auditory_multi_portal.param_value(bool),
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
        "block_size": auditory_block_size.param_value(int),
        "audio_device": auditory_playback.substitution,
        "asset_catalog": auditory_assets.substitution,
        "sound_dir": auditory_sound_dir.substitution,
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
                condition=auditory_on,
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
                condition=auditory_on,
                parameters=[{
                    "use_sim_time": True,
                    "sound_events_topic": "human_sound_events",
                    "heard_sound_events_topic": "heard_sound_events",
                    "continuous_audio_sources_topic":
                        "continuous_audio_sources",
                    "continuous_heard_sounds_topic":
                        "continuous_heard_sounds",
                    "robot_microphones":
                        auditory_microphones.param_value(str),
                    "viewport_down_projection_height_m":
                        auditory_viewport_height.param_value(float),
                    "active_microphone_id":
                        auditory_listener.substitution,
                    "microphone_listeners_topic":
                        "microphone_listeners",
                    "robot_listener_frame":
                        auditory_listener_frame.substitution,
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
                    "compute_rir_in_propagation": auditory_rir_in_propagation.param_value(bool),
                    "propagation_backend": auditory_propagation.substitution,
                    "portal_adjacency_tolerance_m": 0.2,
                    "portal_inset_m": 0.03,
                    "portal_loss_db": 3.0,
                    "opening_portal_loss_db": 0.5,
                    "derive_opening_portals": True,
                    "minimum_opening_width_m": 0.2,
                    # Each authored zone is one ordinary pyroomacoustics
                    # room. Cross-zone rendering may follow a portal route.
                    "enable_multi_portal_rir": auditory_multi_portal.param_value(bool),
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
                condition=auditory_viz_on,
                parameters=[{
                    "use_sim_time": True,
                    "heard_sound_events_topic": "heard_sound_events",
                    "continuous_audio_sources_topic":
                        "continuous_audio_sources",
                    "continuous_heard_sounds_topic":
                        "continuous_heard_sounds",
                    "environment_source_marker_topic":
                        "environment_audio_source_markers",
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
                condition=auditory_on,
                parameters=[{
                    "use_sim_time": True,
                    **playback_parameters,
                    "enable_robot_sound": auditory_robot_sound.param_value(bool),
                    "enable_motor_playback": launch_ros.parameter_descriptions.ParameterValue(motor_enabled, value_type=bool),
                    "robot_fleet_topic": "state/robots",
                    "continuous_audio_sources_topic":
                        "continuous_audio_sources",
                    "continuous_heard_sounds_topic":
                        "continuous_heard_sounds",
                    "motor_audio_mode": motor_audio_mode,
                    "motor_rir_crossfade_sec": 0.10,
                    "motor_playback_mode": auditory_motor_playback.substitution,
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
                executable='environment_sound_playback',
                name='environment_sound_playback',
                namespace=namespace.substitution,
                output='screen',
                condition=auditory_playback_on,
                parameters=[{
                    **playback_parameters,
                    "continuous_heard_sounds_topic":
                        "continuous_heard_sounds",
                    "enable_environment_playback": auditory_environment_playback.param_value(bool),
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
                condition=auditory_on,
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
                condition=auditory_playback_on,
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
