import os

import launch
from arena_bringup.actions import IsolatedIncludeLaunchDescription
from task_generator.simulators.hearing import arena_hearing_share

_PREFIX = 'robot.hearing.'
_POLICY_KEY = f'{_PREFIX}policy'


def _include(context: launch.LaunchContext) -> list[launch.LaunchDescriptionEntity]:
    configs = context.launch_configurations
    frontend = configs['frontend']
    share = arena_hearing_share(frontend)
    forwarded = {key: value for key, value in configs.items() if key.startswith(_PREFIX) and key != _POLICY_KEY and value}
    return [
        IsolatedIncludeLaunchDescription(
            os.path.join(share, 'launch', 'hearing.launch.py'),
            args={
                **forwarded,
                'env.ns': configs['environment_namespace'],
                'tg_node': os.path.basename(configs['namespace']),
                'frontend': frontend,
                'policy': configs['policy'],
            },
        ),
    ]


def generate_launch_description():
    return launch.LaunchDescription([launch.actions.OpaqueFunction(function=_include)])
