"""Launch the MuJoCo simulation server."""

import os
import re
import subprocess

import launch
from arena_bringup.substitutions import LaunchArgument
from launch import LaunchDescription
from launch.actions import ExecuteProcess, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import ExecutableInPackage

_VGLRUN = '/opt/VirtualGL/bin/vglrun'
_GLXINFO = '/opt/VirtualGL/bin/glxinfo'
_VGL = [_VGLRUN, '-d', 'egl']
_SOFTWARE_RENDERERS = ('llvmpipe', 'softpipe', 'swrast')


def _renderer(prefix: list[str]) -> str:
    """GL renderer string the display gives a window, empty when GL fails."""
    try:
        out = subprocess.run([*prefix, _GLXINFO, '-B'], capture_output=True, text=True, timeout=20, check=False).stdout
    except subprocess.TimeoutExpired:
        return ''
    match = re.search(r'OpenGL renderer string: (.*)', out)
    return match.group(1) if match else ''


def _software(renderer: str) -> bool:
    return any(name in renderer for name in _SOFTWARE_RENDERERS)


def _gpu_renderer(headless: bool) -> str:
    """GPU renderer VirtualGL gives the viewer when the display renders GL in software, else empty."""
    if headless or not os.environ.get('DISPLAY') or not os.path.exists(_VGLRUN):
        return ''
    if not _software(_renderer([])):
        return ''
    renderer = _renderer(_VGL)
    return '' if _software(renderer) else renderer


def _server(context: launch.LaunchContext) -> list[launch.Action]:
    headless = LaunchConfiguration('headless').perform(context)
    gpu = _gpu_renderer(headless.lower() == 'true')
    return [
        *([LogInfo(msg=f'display renders GL in software, the viewer goes through VirtualGL on {gpu}')] if gpu else []),
        ExecuteProcess(
            cmd=[
                *(_VGL if gpu else []),
                ExecutableInPackage(executable='run_mujoco', package='arena_mujoco'),
                '--log-level',
                LaunchConfiguration('log_level'),
                '--headless',
                headless,
            ],
            output='screen',
        ),
    ]


def generate_launch_description():
    """Return launch description for run_mujoco."""
    ld = []
    LaunchArgument.auto_append(ld)

    LaunchArgument(
        name='log_level',
        default_value='info',
        description='Logging level',
    )

    LaunchArgument(
        name='headless',
        default_value='False',
    )

    return LaunchDescription(
        [
            *ld,
            launch.actions.DeclareLaunchArgument(
                'log_level',
                default_value=['info'],
                description='Logging level',
            ),
            OpaqueFunction(function=_server),
        ]
    )
