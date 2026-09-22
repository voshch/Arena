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

    viewport = {
        key: LaunchArgument(
            name=f'sim.isaac.viewport.{key}',
            default_value=default,
            description=description,
        )
        for key, default, description in (
            ('preset', 'photoreal', 'Base render preset: photoreal | boring. GUI only.'),
            ('resolution', '', 'Viewport render size: WxH | dynamic. Empty keeps the preset value.'),
            ('scale', '', 'Viewport resolution scale factor. Empty keeps the preset value.'),
            ('dlss', '', 'DLSS mode: auto | quality | balanced | performance. Empty keeps the preset value.'),
            ('lighting', '', 'Lighting rig: lights_off | camera_light | stage_lights | colored_lights | default | grey_studio. Empty keeps the preset value.'),
            ('overlays', '', 'Viewport overlays to show, comma list of axis,grid,bbox or none. Empty keeps the preset value.'),
        )
    }

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
                launch.substitutions.TextSubstitution(text='exec env PYTHONPATH="${ARENA_DIR:?run source arena first}/_meta${PYTHONPATH:+:$PYTHONPATH}" python3 -m arena_cli feature gazebo launch use_sim_time:='),
                use_sim_time.substitution,
                launch.substitutions.TextSubstitution(text=' headless:='),
                headless.substitution,
                launch.substitutions.TextSubstitution(text=' world:='),
                world.substitution,
            ]],
            sigterm_timeout=launch.substitutions.LaunchConfiguration('sigterm_timeout', default='20'),
            sigkill_timeout=launch.substitutions.LaunchConfiguration('sigkill_timeout', default='5'),
            on_exit=[launch.actions.Shutdown()],
            output='log',
        )
    )

    launch_simulator.add(
        SimSimulator.ISAAC.value,
        launch.actions.ExecuteProcess(
            cmd=['bash', '-c', [
                launch.substitutions.TextSubstitution(text='exec env PYTHONPATH="${ARENA_DIR:?run source arena first}/_meta${PYTHONPATH:+:$PYTHONPATH}" python3 -m arena_cli feature isaac launch headless:='),
                headless.substitution,
                launch.substitutions.TextSubstitution(text=' log_level:='),
                launch.substitutions.LaunchConfiguration('log_level', default='debug'),
                launch.substitutions.TextSubstitution(text=' physics:='),
                physics.substitution,
                *(
                    sub
                    for key, arg in viewport.items()
                    for sub in (launch.substitutions.TextSubstitution(text=f' viewport.{key}:='), arg.substitution)
                ),
            ]],
            sigterm_timeout=launch.substitutions.LaunchConfiguration('sigterm_timeout', default='20'),
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
