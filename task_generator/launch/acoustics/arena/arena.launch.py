import os

import launch
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from arena_bringup.actions import IsolatedIncludeLaunchDescription

_PREFIX = 'auditory.'
_PASSTHROUGH = ('debug.map_source',)


def _include(context: launch.LaunchContext) -> list[launch.LaunchDescriptionEntity]:
    try:
        share = get_package_share_directory('arena_auditory')
    except PackageNotFoundError as exc:
        raise RuntimeError('acoustics:=arena needs the arena_auditory package, install it with `arena feature auditory install`') from exc
    configs = context.launch_configurations
    forwarded = {key.removeprefix(_PREFIX): value for key, value in configs.items() if key.startswith(_PREFIX)}
    passthrough = {key: configs[key] for key in _PASSTHROUGH if key in configs}
    return [
        IsolatedIncludeLaunchDescription(
            os.path.join(share, 'launch/arena_auditory.launch.py'),
            args={
                **forwarded,
                **passthrough,
                'namespace': configs.get('namespace', ''),
                'env.ns': configs.get('environment_namespace', ''),
                'hearing': configs.get('robot.hearing', 'none'),
            },
        ),
    ]


def generate_launch_description():
    return launch.LaunchDescription([launch.actions.OpaqueFunction(function=_include)])
