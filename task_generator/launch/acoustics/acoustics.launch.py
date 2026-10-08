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

    launch_acoustics_simulator = SelectAction(launch.substitutions.LaunchConfiguration('simulator'))

    launch_acoustics_simulator.add(Constants.AcousticsSimulator.NONE.value, launch.actions.GroupAction([]))

    launch_acoustics_simulator.add(
        Constants.AcousticsSimulator.ARENA.value,
        launch.actions.IncludeLaunchDescription(
            PathJoinSubstitution(
                [
                    FindPackageShare('task_generator'),
                    'launch',
                    'acoustics',
                    'arena',
                    'arena.launch.py',
                ]
            ),
            launch_arguments={
                **namespace.dict,
                **environment_namespace.dict,
            }.items(),
        ),
    )

    simulator = LaunchArgument(
        name='simulator',
        choices=launch_acoustics_simulator.keys,
    )

    ld = launch.LaunchDescription(
        [
            *ld,
            launch_acoustics_simulator,
        ]
    )
    return ld


if __name__ == '__main__':
    generate_launch_description()
