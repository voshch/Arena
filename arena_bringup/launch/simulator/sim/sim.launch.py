import launch.actions
import launch.substitutions

import launch
import launch_ros.actions
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from arena_rclpy_mixins.launch_argument import LaunchArgument
from arena_bringup.substitutions import SelectAction

from arena_bringup.constants import SimSimulator


def _require_flatsim(context: launch.LaunchContext) -> None:
    try:
        get_package_share_directory('flatsim')
    except PackageNotFoundError as exc:
        raise RuntimeError('sim:=flatsim needs the flatsim package, install it with `arena feature flatsim install`') from exc


def generate_launch_description():

    ld = []
    LaunchArgument.auto_append(ld)

    use_sim_time = LaunchArgument(
        name='use_sim_time',
        description='Use the simulation clock instead of wall time.',
    )

    headless = LaunchArgument(
        name='headless',
        default_value='False',
        description='true = hide the simulator GUI.',
    )

    physics = LaunchArgument(
        name='sim.isaac.physics',
        default_value='physx',
        choices=['physx', 'newton'],
        description='Physics engine of Isaac Sim.',
    )

    world = LaunchArgument(
        name='world',
        description='World for the simulator to load.',
    )

    dynamics = LaunchArgument(
        name='sim.flatsim.dynamics',
        default_value='rigid',
        choices=['rigid', 'kinematic'],
        description='How walls and obstacles act on a robot in flatsim.',
    )

    flatsim_physics = LaunchArgument(
        name='sim.flatsim.physics',
        default_value='auto',
        choices=['auto', 'box2d', 'warp'],
        description='Physics engine of flatsim.',
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
            ('ceilings', '', 'Room ceilings: auto | on | off. Empty is auto, hidden so the GUI looks into the rooms from above unless the world declares lights.'),
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

    launch_simulator.add(SimSimulator.FLATSIM.value, launch.actions.OpaqueFunction(function=_require_flatsim))
    launch_simulator.add(
        SimSimulator.FLATSIM.value,
        launch.actions.ExecuteProcess(
            cmd=['bash', '-c', [
                launch.substitutions.TextSubstitution(text='exec env PYTHONPATH="${ARENA_DIR:?run source arena first}/_meta${PYTHONPATH:+:$PYTHONPATH}" python3 -m arena_cli feature flatsim launch dynamics:='),
                dynamics.substitution,
                launch.substitutions.TextSubstitution(text=' physics:='),
                flatsim_physics.substitution,
            ]],
            sigterm_timeout=launch.substitutions.LaunchConfiguration('sigterm_timeout', default='20'),
            sigkill_timeout=launch.substitutions.LaunchConfiguration('sigkill_timeout', default='5'),
            on_exit=[launch.actions.Shutdown()],
            output='screen',
        )
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

    launch_simulator.add(
        SimSimulator.MUJOCO.value,
        launch.actions.ExecuteProcess(
            cmd=['bash', '-c', [
                launch.substitutions.TextSubstitution(text='exec env PYTHONPATH="${ARENA_DIR:?run source arena first}/_meta${PYTHONPATH:+:$PYTHONPATH}" python3 -m arena_cli feature mujoco launch headless:='),
                headless.substitution,
                launch.substitutions.TextSubstitution(text=' log_level:='),
                launch.substitutions.LaunchConfiguration('log_level', default='info'),
            ]],
            sigterm_timeout=launch.substitutions.LaunchConfiguration('sigterm_timeout', default='5'),
            sigkill_timeout=launch.substitutions.LaunchConfiguration('sigkill_timeout', default='5'),
            on_exit=[launch.actions.Shutdown()],
            output='screen',
        )
    )

    sim = LaunchArgument(
        name='sim',
        choices=launch_simulator.keys,
        description='Simulator to start.',
    )

    ld = launch.LaunchDescription([
        *ld,
        launch_simulator,
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
