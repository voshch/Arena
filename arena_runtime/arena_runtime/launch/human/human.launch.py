import launch
from arena_rclpy_mixins.launch_argument import LaunchArgument
from arena_bringup.substitutions import SelectAction
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from arena_runtime.env.constants import Constants


def generate_launch_description() -> launch.LaunchDescription:

    ld = []

    LaunchArgument.auto_append(ld)

    namespace = LaunchArgument(
        name='namespace',
        description='Namespace of the human simulator nodes.',
    )

    launch_human_simulator = SelectAction(launch.substitutions.LaunchConfiguration('simulator'))

    launch_human_simulator.add(Constants.HumanSimulator.DUMMY.value, launch.actions.GroupAction([]))

    launch_human_simulator.add(Constants.HumanSimulator.NONE.value, launch.actions.GroupAction([]))

    launch_human_simulator.add(Constants.HumanSimulator.REPLAY.value, launch.actions.GroupAction([]))

    launch_human_simulator.add(
        Constants.HumanSimulator.HUNAV.value,
        launch.actions.IncludeLaunchDescription(
            PathJoinSubstitution(
                [
                    FindPackageShare('arena_runtime'),
                    'launch',
                    'human',
                    'hunav',
                    'hunav.launch.py',
                ]
            ),
            launch_arguments={'use_sim_time': 'true', **namespace.dict}.items(),
        ),
    )

    launch_human_simulator.add(
        Constants.HumanSimulator.ARENA.value,
        launch.actions.IncludeLaunchDescription(
            PathJoinSubstitution(
                [
                    FindPackageShare('arena_runtime'),
                    'launch',
                    'human',
                    'arena_humansim',
                    'arena_humansim.launch.py',
                ]
            ),
            launch_arguments={'use_sim_time': 'true', **namespace.dict}.items(),
        ),
    )

    simulator = LaunchArgument(
        name='simulator',
        choices=launch_human_simulator.keys,
        description='Human simulator backend to launch.',
    )

    ld = launch.LaunchDescription(
        [
            *ld,
            launch_human_simulator,
        ]
    )
    return ld


if __name__ == '__main__':
    generate_launch_description()
