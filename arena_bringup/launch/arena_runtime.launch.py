import os

import launch
import launch.event_handlers
import launch_ros.actions
from arena_bringup.actions import IsolatedGroupAction
from arena_bringup.defaults import SIM_ORDER, default_sim, gz_discovery_env
from arena_bringup.extensions.NodeLogLevelExtension import SetGlobalLogLevelAction
from arena_bringup.substitutions import deprecated_launch_args
from arena_rclpy_mixins.launch_argument import LaunchArgument
from arena_simulation_setup import params as world_params
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

_RUNTIME_OWNED = frozenset(
    {
        'log_level',
        'sim',
        'sim.flatsim.dynamics',
        'sim.flatsim.physics',
        'sim.isaac.physics',
        *(f'sim.isaac.viewport.{k}' for k in ('preset', 'resolution', 'scale', 'dlss', 'lighting', 'overlays', 'ceilings')),
        'use_sim_time',
        'world',
        'headless',
        'lockstep',
        'lockstep.channels',
        'lockstep.rtf',
        'lockstep.paused',
        'env.bootstrap_timeout',
        'env.heartbeat_timeout',
    }
)


def generate_launch_description():
    ld_items = []
    LaunchArgument.auto_append(ld_items)

    log_level = LaunchArgument(
        name='log_level',
        default_value='warn',
        description='Per-node log level.',
    )

    sim = LaunchArgument(
        name='sim',
        default_value=default_sim(),
        description=f'simulator, default: arena settings sim, else the first installed of {", ".join(s.value for s in SIM_ORDER)}, else dummy',
    )

    use_sim_time = LaunchArgument(
        name='use_sim_time',
        default_value='true',
        description='Use the simulation clock instead of wall time (false only for real robots).',
    )

    world = LaunchArgument(
        name='world',
        default_value='map_empty',
        description='World for the simulator to load.',
    )

    headless = LaunchArgument(
        name='headless',
        default_value='False',
        description='true = hide the simulator GUI.',
    )

    for param in (world_params.WORLD_LIGHTING, world_params.WORLD_LEVELS, world_params.KG_DERIVE):
        LaunchArgument.from_param(param, keep_default=True)

    LaunchArgument(
        name='record.dir',
        default_value='',
        description='Episode recording directory (empty = no recording).',
    )

    LaunchArgument(
        name='record.auto',
        default_value='true',
        description='Spawn the data recorder per env when record.dir is set.',
    )

    LaunchArgument(
        name='lockstep',
        default_value='false',
        description='Start the lockstep scheduler at bringup.',
    )

    LaunchArgument(
        name='lockstep.channels',
        default_value='',
        description="Extra lockstep channels, name|topic|type|period_s|role entries separated by semicolons, {env} expanded per env.",
    )

    LaunchArgument(
        name='lockstep.rtf',
        default_value='',
        description='Target real-time factor of the lockstep scheduler (empty = as fast as it can).',
    )

    LaunchArgument(
        name='lockstep.paused',
        default_value='true',
        description='true = the lockstep scheduler comes up paused.',
    )

    LaunchArgument(
        name='env.bootstrap_timeout',
        default_value='',
        description='Seconds an env may take to become ready before eviction. Empty = node default.',
    )

    LaunchArgument(
        name='env.heartbeat_timeout',
        default_value='',
        description='Seconds of heartbeat silence before a ready env is evicted. Empty = node default.',
    )

    launch_sim = launch.actions.IncludeLaunchDescription(
        PathJoinSubstitution(
            [
                FindPackageShare('arena_bringup'),
                'launch',
                'simulator',
                'sim',
                'sim.launch.py',
            ]
        ),
        launch_arguments={
            **use_sim_time.dict,
            **sim.dict,
            **world.dict,
            'headless': headless.substitution,
        }.items(),
    )

    world_generator_node = launch_ros.actions.Node(
        package='arena_simulation_setup',
        executable='world_generator',
        name='world_generator',
        output='screen',
    )

    given_args: dict[str, str] = {}

    def _aliases(context: launch.LaunchContext) -> None:
        deprecated_launch_args(context)
        given_args.update(context.launch_configurations)

    def _arena_runtime_node(context: launch.LaunchContext) -> list[launch.Action]:
        env_args = [f'{k}:={v}' for k, v in given_args.items() if k not in _RUNTIME_OWNED]
        params: dict = {'sim': sim.substitution}
        if env_args:
            params['env_args'] = env_args
        if context.launch_configurations.get('lockstep', 'false').lower() in ('true', '1'):
            params['lockstep.autostart'] = True
        channels = [c for c in context.launch_configurations.get('lockstep.channels', '').split(';') if c]
        if channels:
            params['lockstep.channels'] = channels
        if rtf := context.launch_configurations.get('lockstep.rtf', ''):
            params['lockstep.target_rtf'] = float(rtf)
        params['lockstep.paused'] = context.launch_configurations.get('lockstep.paused', 'true').lower() in ('true', '1')
        if bootstrap_timeout := context.launch_configurations.get('env.bootstrap_timeout', ''):
            params['bootstrap_timeout_sec'] = float(bootstrap_timeout)
        if heartbeat_timeout := context.launch_configurations.get('env.heartbeat_timeout', ''):
            params['heartbeat_timeout_sec'] = float(heartbeat_timeout)
        arena_runtime_node = launch_ros.actions.LifecycleNode(
            package='arena_runtime',
            executable='arena_runtime_node',
            name='arena',
            namespace='',
            output='screen',
            parameters=[params],
        )
        return [
            arena_runtime_node,
            launch.actions.RegisterEventHandler(
                launch.event_handlers.OnProcessExit(
                    target_action=arena_runtime_node,
                    on_exit=[launch.actions.Shutdown(reason='arena_runtime_node exited')],
                )
            ),
        ]

    return launch.LaunchDescription(
        [
            launch.actions.OpaqueFunction(function=_aliases),
            *(launch.actions.SetEnvironmentVariable(k, v) for k, v in gz_discovery_env(int(os.environ.get('ROS_DOMAIN_ID', '0'))).items()),
            *ld_items,
            SetGlobalLogLevelAction(log_level.substitution),
            IsolatedGroupAction([launch_sim]),
            world_generator_node,
            launch.actions.OpaqueFunction(function=_arena_runtime_node),
        ]
    )


if __name__ == '__main__':
    generate_launch_description()
