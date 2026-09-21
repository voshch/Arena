import os

import launch
from ament_index_python.packages import get_package_share_directory
from arena_bringup.actions import IsolatedIncludeLaunchDescription

_PREFIX = 'humansim.'


def _include(context: launch.LaunchContext) -> list[launch.LaunchDescriptionEntity]:
    forwarded = {key.removeprefix(_PREFIX): value for key, value in context.launch_configurations.items() if key.startswith(_PREFIX)}
    return [
        IsolatedIncludeLaunchDescription(
            os.path.join(
                get_package_share_directory('arena_humansim'),
                'launch/arena_humansim.launch.py',
            ),
            args={
                'markers': '2',
                **forwarded,
                'mode': 'subsystem',
                'use_sim_time': 'true',
                'rviz': 'false',
                'namespace': context.launch_configurations.get('namespace', ''),
            },
        ),
    ]


def generate_launch_description():
    return launch.LaunchDescription([launch.actions.OpaqueFunction(function=_include)])
