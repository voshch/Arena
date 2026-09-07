import launch
from arena_bringup.substitutions import LaunchArgument, SelectAction
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from task_generator.constants import Constants


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

    launch_auditory_simulator = SelectAction(launch.substitutions.LaunchConfiguration('simulator'))

    launch_auditory_simulator.add(Constants.AuditorySimulator.NONE.value, launch.actions.GroupAction([]))

    launch_auditory_simulator.add(
        Constants.AuditorySimulator.ARENA.value,
        launch.actions.IncludeLaunchDescription(
            PathJoinSubstitution(
                [
                    FindPackageShare('task_generator'),
                    'launch',
                    'auditory',
                    'arena',
                    'arena.launch.py',
                ]
            ),
            launch_arguments={
                'use_sim_time': 'true',
                **namespace.dict,
                **environment_namespace.dict,
                **auditory_viz.dict,
                **auditory_playback.dict,
                **auditory_block_size.dict,
                **auditory_assets.dict,
                **auditory_sound_dir.dict,
                **auditory_propagation.dict,
                **auditory_multi_portal.dict,
                **auditory_rir_in_propagation.dict,
                **auditory_ped_hearing.dict,
                **auditory_robot_sound.dict,
                **auditory_source_volume_db.dict,
                **auditory_motor.dict,
                **auditory_motor_playback.dict,
                **auditory_motor_mems_calibration.dict,
                **auditory_environment_playback.dict,
                **auditory_listener.dict,
                **auditory_listener_frame.dict,
                **auditory_microphones.dict,
                **microphone_mode.dict,
                **auditory_viewport_height.dict,
            }.items(),
        ),
    )

    simulator = LaunchArgument(
        name='simulator',
        choices=launch_auditory_simulator.keys,
    )

    ld = launch.LaunchDescription(
        [
            *ld,
            launch_auditory_simulator,
        ]
    )
    return ld


if __name__ == '__main__':
    generate_launch_description()
