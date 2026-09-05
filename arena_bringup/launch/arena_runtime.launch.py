import launch
import launch.event_handlers
import launch_ros.actions
from arena_bringup.actions import IsolatedGroupAction
from arena_bringup.extensions.NodeLogLevelExtension import SetGlobalLogLevelAction
from arena_bringup.substitutions import LaunchArgument, deprecated_launch_args
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

_RUNTIME_OWNED = frozenset({'log_level', 'sim', 'sim.isaac.physics', 'use_sim_time', 'world', 'headless', 'lockstep', 'lockstep.channels', 'lockstep.rtf', 'lockstep.paused', 'env.bootstrap_timeout'})


def generate_launch_description():
    ld_items = []
    LaunchArgument.auto_append(ld_items)

    log_level = LaunchArgument(
        name='log_level',
        default_value='warn',
        description='Per-node log level. See launch README.',
    )

    sim = LaunchArgument(
        name='sim',
        default_value='gazebo',
    )

    use_sim_time = LaunchArgument(
        name='use_sim_time',
        default_value='true',
    )

    world = LaunchArgument(
        name='world',
        default_value='map_empty',
    )

    headless = LaunchArgument(
        name='headless',
        default_value='False',
    )

    LaunchArgument(
        name='record.dir',
        default_value='',
        description='Directory for episode recording; empty disables.',
    )

    LaunchArgument(
        name='record.auto',
        default_value='true',
        description='Spawn the data recorder per env when record.dir is set.',
    )

    LaunchArgument(
        name='lockstep',
        default_value='false',
        description='Start the lockstep scheduler at bringup (control later via sim_lifecycle/lockstep/start|stop).',
    )

    LaunchArgument(
        name='lockstep.channels',
        default_value='',
        description="Semicolon-separated 'name|topic|type|period_s|role' entries; {env} expands per env.",
    )

    LaunchArgument(
        name='lockstep.rtf',
        default_value='',
        description='Target real-time factor for the lockstep scheduler; empty = flat out.',
    )

    LaunchArgument(
        name='lockstep.paused',
        default_value='true',
        description='Bring the lockstep scheduler up paused (resume via arena lockstep resume); false = start stepping immediately.',
    )

    LaunchArgument(
        name='env.bootstrap_timeout',
        default_value='',
        description='Seconds an env may spend never-ACTIVE before eviction, measured from reservation, empty = node default (never).',
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

    def _arena_node(context: launch.LaunchContext) -> list[launch.Action]:
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
        arena_node = launch_ros.actions.LifecycleNode(
            package='arena_runtime',
            executable='arena_node',
            name='arena',
            namespace='',
            output='screen',
            parameters=[params],
        )
        return [
            arena_node,
            launch.actions.RegisterEventHandler(
                launch.event_handlers.OnProcessExit(
                    target_action=arena_node,
                    on_exit=[launch.actions.Shutdown(reason='arena_node exited')],
                )
            ),
        ]

    return launch.LaunchDescription(
        [
            launch.actions.OpaqueFunction(function=_aliases),
            *ld_items,
            SetGlobalLogLevelAction(log_level.substitution),
            IsolatedGroupAction([launch_sim]),
            world_generator_node,
            launch.actions.OpaqueFunction(function=_arena_node),
        ]
    )


if __name__ == '__main__':
    generate_launch_description()
