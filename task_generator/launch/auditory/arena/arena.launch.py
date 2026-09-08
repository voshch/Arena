import launch
import launch_ros.parameter_descriptions
from arena_bringup.substitutions import LaunchArgument
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> launch.LaunchDescription:

    ld = []

    LaunchArgument.auto_append(ld)

    namespace = LaunchArgument(
        name='namespace',
    )

    environment_namespace = LaunchArgument(
        name='environment_namespace',
    )

    auditory_viz = LaunchArgument(
        name="auditory.viz",
        default_value="false",
    )
    auditory_playback = LaunchArgument(
        name="auditory.playback",
        default_value="auto",
        description="PortAudio output device for local playback; auto = pulse, pipewire, default, then the PortAudio default, none = no playback nodes",
    )
    auditory_block_size = LaunchArgument(
        name="auditory.block_size",
        default_value="2048",
    )
    auditory_assets = LaunchArgument(
        name="auditory.assets",
        default_value=PathJoinSubstitution(
            [
                FindPackageShare("arena_auditory"),
                "config",
                "acoustic_assets.yaml",
            ]
        ),
    )
    auditory_sound_dir = LaunchArgument(
        name="auditory.sound_dir",
        default_value=PathJoinSubstitution(
            [
                FindPackageShare("arena_auditory"),
                "sounds",
            ]
        ),
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
    auditory_ped_hearing = LaunchArgument(
        name="auditory.ped_hearing",
        default_value="true",
    )
    auditory_robot_sound = LaunchArgument(
        name="auditory.robot_sound",
        default_value="true",
    )
    auditory_source_volume_db = LaunchArgument(
        name="auditory.source_volume_db",
        default_value="45.0",
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
    auditory_motor_volume = LaunchArgument(
        name="auditory.motor.volume_db",
        default_value="-15.020599913279624",
        description="Four-microphone procedural motor drivetrain level in dB.",
    )
    auditory_motor_mems_calibration = LaunchArgument(
        name="auditory.motor.mems_calibration_db",
        default_value="-40.0",
        description="Four-microphone procedural motor calibration in dB.",
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
    microphone_mode = LaunchArgument(
        name="microphone_mode",
        choices=["stereo", "four_mic"],
        default_value="stereo",
    )
    auditory_viewport_height = LaunchArgument(
        name="auditory.viewport_height",
        default_value="1.6",
    )

    legacy_playback_on = launch.conditions.IfCondition(
        launch.substitutions.PythonExpression(
            [
                "'",
                auditory_playback.substitution,
                "' != 'none' and '",
                microphone_mode.substitution,
                "' != 'four_mic'",
            ]
        )
    )
    auditory_viz_on = launch.conditions.IfCondition(
        launch.substitutions.PythonExpression(
            [
                "'",
                auditory_viz.substitution,
                "' == 'true'",
            ]
        )
    )
    four_mic_on = launch.conditions.IfCondition(
        launch.substitutions.PythonExpression(
            [
                "'",
                microphone_mode.substitution,
                "' == 'four_mic'",
            ]
        )
    )
    legacy_audio_device = launch.substitutions.PythonExpression(
        [
            "'none' if '",
            microphone_mode.substitution,
            "' == 'four_mic' else '",
            auditory_playback.substitution,
            "'",
        ]
    )
    motor_enabled = launch.substitutions.PythonExpression(
        [
            "'",
            auditory_motor.substitution,
            "' != 'off'",
        ]
    )
    motor_audio_mode = launch.substitutions.PythonExpression(
        [
            "'procedural' if '",
            auditory_motor.substitution,
            "' == 'off' else '",
            auditory_motor.substitution,
            "'",
        ]
    )
    robot_hearing_events_topic = launch.substitutions.PythonExpression(
        [
            "'four_mic_heard_sound_events' if '",
            microphone_mode.substitution,
            "' == 'four_mic' else 'heard_sound_events'",
        ]
    )

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
        "audio_device": legacy_audio_device,
        "asset_catalog": auditory_assets.substitution,
        "sound_dir": auditory_sound_dir.substitution,
        "master_gain_db": 0.0,
    }

    auditory_stack = launch.actions.GroupAction(
        [
            # 0. Human sound producer, fed by the human simulator's arena_peds topic
            Node(
                package='arena_auditory',
                executable='human_sound_node',
                name='human_sound_node',
                namespace=namespace.substitution,
                output='screen',
                parameters=[
                    {
                        "use_sim_time": True,
                        "arena_peds_topic": PathJoinSubstitution(
                            [
                                environment_namespace.substitution,
                                "arena_peds",
                            ]
                        ),
                        "world_topic": "state/world",
                        "sound_events_topic": "human_sound_events",
                        "sound_markers_topic": PathJoinSubstitution(
                            [
                                environment_namespace.substitution,
                                "pedestrian_markers",
                                "extra",
                            ]
                        ),
                    }
                ],
            ),
            Node(
                package='arena_auditory',
                executable='sound_propagation_node',
                name='sound_propagation_node',
                namespace=namespace.substitution,
                output='screen',
                parameters=[
                    PathJoinSubstitution(
                        [
                            FindPackageShare("arena_auditory"),
                            "config",
                            "jackal_four_mic.yaml",
                        ]
                    ),
                    {
                        "use_sim_time": True,
                        "sound_events_topic": "human_sound_events",
                        "heard_sound_events_topic": "heard_sound_events",
                        "continuous_audio_sources_topic": "continuous_audio_sources",
                        "continuous_heard_sounds_topic": "continuous_heard_sounds",
                        "robot_microphones": auditory_microphones.param_value(str),
                        "microphone_mode": microphone_mode.substitution,
                        "viewport_down_projection_height_m": auditory_viewport_height.param_value(float),
                        "active_microphone_id": auditory_listener.substitution,
                        "microphone_listeners_topic": "microphone_listeners",
                        "robot_listener_frame": auditory_listener_frame.substitution,
                        "arena_peds_topic": PathJoinSubstitution(
                            [
                                environment_namespace.substitution,
                                "arena_peds",
                            ]
                        ),
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
                        "ped_hearing": auditory_ped_hearing.param_value(bool),
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
                        # Validation reports authoring gaps but is deliberately
                        # non-fatal, so an acoustic diagnostic cannot abort a run.
                        "validate_zone_coverage": True,
                        "zone_coverage_stride_cells": 10,
                        "zone_coverage_tolerance_m": 0.35,
                        "buffer_events_until_scene_loaded": True,
                        "scene_event_buffer_size": 128,
                    },
                ],
            ),
            Node(
                package='arena_auditory',
                executable='microphone_array_node',
                name='microphone_array_node',
                namespace=namespace.substitution,
                output='screen',
                condition=four_mic_on,
                parameters=[
                    PathJoinSubstitution(
                        [
                            FindPackageShare("arena_auditory"),
                            "config",
                            "jackal_four_mic.yaml",
                        ]
                    ),
                    {
                        "use_sim_time": True,
                        "asset_catalog": auditory_assets.substitution,
                        "sound_dir": auditory_sound_dir.substitution,
                        "heard_sound_events_topic": "heard_sound_events",
                        "fused_heard_sound_events_topic": "four_mic_heard_sound_events",
                        "continuous_heard_sounds_topic": "continuous_heard_sounds",
                        "robot_fleet_topic": "state/robots",
                        "microphone_marker_topic": "microphone_markers",
                        "visualization_enabled": auditory_viz.param_value(bool),
                        "audio_device": auditory_playback.substitution,
                        "motor_volume_db": auditory_motor_volume.param_value(float),
                        "motor_mems_calibration_db": auditory_motor_mems_calibration.param_value(float),
                    },
                ],
                # Single-threaded BLAS pins the block reduction order, so an
                # offline re-render of a recording matches bit for bit.
                additional_env={"OMP_NUM_THREADS": "1"},
            ),
            Node(
                package='arena_auditory',
                executable='sound_propagation_visualizer',
                name='sound_propagation_visualizer',
                namespace=namespace.substitution,
                output='screen',
                condition=auditory_viz_on,
                parameters=[
                    {
                        "use_sim_time": True,
                        "heard_sound_events_topic": "heard_sound_events",
                        "continuous_audio_sources_topic": "continuous_audio_sources",
                        "continuous_heard_sounds_topic": "continuous_heard_sounds",
                        "environment_source_marker_topic": "environment_audio_source_markers",
                        "pedestrian_marker_topic": "pedestrian_sound_propagation_markers",
                        "robot_marker_topic": "robot_sound_propagation_markers",
                        "robot_fleet_topic": "state/robots",
                        "sync_robot_listener_to_tf": True,
                        "marker_lifetime_sec": 5.0,
                        "path_z_m": 1.0,
                    }
                ],
            ),
            # 2. Robot motor sound producer and playback
            Node(
                package='arena_auditory',
                executable='robot_sound_node',
                name='robot_sound_node',
                namespace=namespace.substitution,
                output='screen',
                parameters=[
                    {
                        "use_sim_time": True,
                        **playback_parameters,
                        "enable_robot_sound": auditory_robot_sound.param_value(bool),
                        "enable_motor_playback": launch_ros.parameter_descriptions.ParameterValue(motor_enabled, value_type=bool),
                        "robot_fleet_topic": "state/robots",
                        "continuous_audio_sources_topic": "continuous_audio_sources",
                        "continuous_heard_sounds_topic": "continuous_heard_sounds",
                        "motor_audio_mode": motor_audio_mode,
                        "motor_rir_crossfade_sec": 0.10,
                        "motor_playback_mode": auditory_motor_playback.substitution,
                        "motor_single_asset_id": "motor",
                        "odom_topic_template": "{namespace}/{name}_velocity_controller/odom",
                        "sound_type": "motor",
                        "motor_start_asset_id": "motor_start",
                        "motor_stop_asset_id": "motor_stop",
                        "source_volume_db": auditory_source_volume_db.param_value(float),
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
                    }
                ],
            ),
            Node(
                package='arena_auditory',
                executable='environment_sound_playback',
                name='environment_sound_playback',
                namespace=namespace.substitution,
                output='screen',
                condition=legacy_playback_on,
                parameters=[
                    {
                        **playback_parameters,
                        "continuous_heard_sounds_topic": "continuous_heard_sounds",
                        "enable_environment_playback": auditory_environment_playback.param_value(bool),
                        "environment_rir_crossfade_sec": 0.10,
                    }
                ],
            ),
            # 3. Robot hearing node
            Node(
                package='arena_auditory',
                executable='robot_hearing_node',
                name='robot_hearing_node',
                namespace=namespace.substitution,
                output='screen',
                parameters=[
                    {
                        "use_sim_time": True,
                        "robot_fleet_topic": "state/robots",
                        "heard_sound_events_topic": robot_hearing_events_topic,
                        "heard_sound_topic_suffix": "heard_sound",
                        "marker_topic_suffix": "heard_sound_marker",
                        "ignore_self": True,
                        "min_snr_db": -5.0,
                        "honor_propagation_delay": True,
                        "marker_lifetime_sec": 1.5,
                        "marker_z_offset": 1.2,
                        "marker_text_height_m": 0.35,
                        "marker_sound_types": ["greeting", "footstep", "motor"],
                    }
                ],
            ),
            # 4. Human sound playback node
            Node(
                package='arena_auditory',
                executable='human_sound_playback',
                name='human_sound_playback',
                namespace=namespace.substitution,
                output='screen',
                condition=legacy_playback_on,
                parameters=[
                    {
                        **playback_parameters,
                    }
                ],
            ),
        ]
    )

    ld = launch.LaunchDescription(
        [
            *ld,
            auditory_stack,
        ]
    )
    return ld


if __name__ == '__main__':
    generate_launch_description()
