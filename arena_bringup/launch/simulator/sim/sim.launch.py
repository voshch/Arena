import launch.actions
import launch.substitutions

import launch
import launch_ros.actions
from arena_bringup.substitutions import LaunchArgument, SelectAction

from arena_runtime.constants import SimSimulator


def generate_launch_description():

    ld = []
    LaunchArgument.auto_append(ld)

    use_sim_time = LaunchArgument(
        name='use_sim_time',
    )

    headless = LaunchArgument(
        name='headless',
        default_value='False',
    )

    physics = LaunchArgument(
        name='sim.isaac.physics',
        default_value='physx',
        choices=['physx', 'newton'],
    )

    world = LaunchArgument(
        name='world'
    )

    launch_simulator = SelectAction(launch.substitutions.LaunchConfiguration('sim'))

    launch_simulator.add(
        SimSimulator.DUMMY.value,
        launch.actions.GroupAction([
            launch_ros.actions.Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                arguments=['--frame-id', 'map', '--child-frame-id', 'dummy'],
            ),
        ])
    )

    launch_simulator.add(
        SimSimulator.GAZEBO.value,
        launch.actions.ExecuteProcess(
            cmd=['bash', '-c', [
                launch.substitutions.TextSubstitution(text='arena feature gazebo launch use_sim_time:='),
                use_sim_time.substitution,
                launch.substitutions.TextSubstitution(text=' headless:='),
                headless.substitution,
                launch.substitutions.TextSubstitution(text=' world:='),
                world.substitution,
            ]],
            sigterm_timeout=launch.substitutions.LaunchConfiguration('sigterm_timeout', default='5'),
            sigkill_timeout=launch.substitutions.LaunchConfiguration('sigkill_timeout', default='5'),
            on_exit=[launch.actions.Shutdown()],
            output='log',
        )
    )

    launch_simulator.add(
        SimSimulator.ISAAC.value,
        launch.actions.ExecuteProcess(
            cmd=['bash', '-c', [
                launch.substitutions.TextSubstitution(text='arena feature isaac launch headless:='),
                headless.substitution,
                launch.substitutions.TextSubstitution(text=' log_level:='),
                launch.substitutions.LaunchConfiguration('log_level', default='debug'),
                launch.substitutions.TextSubstitution(text=' physics:='),
                physics.substitution,
            ]],
            sigterm_timeout=launch.substitutions.LaunchConfiguration('sigterm_timeout', default='5'),
            sigkill_timeout=launch.substitutions.LaunchConfiguration('sigkill_timeout', default='5'),
            on_exit=[launch.actions.Shutdown()],
            output='log',
        )
    )

    sim = LaunchArgument(
        name='sim',
        choices=launch_simulator.keys,
    )

    ld = launch.LaunchDescription([
        *ld,
        launch_simulator,
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
