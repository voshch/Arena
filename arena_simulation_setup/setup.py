import os
from collections.abc import Iterator

from setuptools import find_namespace_packages, setup

package_name = 'arena_simulation_setup'
python_root = 'src'


def _walk_data_files(*roots: str) -> Iterator[tuple[str, list[str]]]:
    # os.path.isfile filters out dangling symlinks. colcon --symlink-install
    # populates the build dir with per-file symlinks into source and does not
    # prune them when source files are deleted, so os.walk would otherwise
    # hand setuptools broken symlinks and the copy step would abort.
    for root in roots:
        for base, _dirs, files in os.walk(root):
            kept = [os.path.join(base, f) for f in files if os.path.isfile(os.path.join(base, f))]
            if kept:
                yield (os.path.join('share', package_name, base), kept)


def _walk_data_files_into(
    root: str,
    destination: str,
    *,
    skip_if_present_in: str | None = None,
) -> Iterator[tuple[str, list[str]]]:
    """Install a source tree below a different package-share directory.

    When two source trees intentionally overlay the same installed tree,
    ``skip_if_present_in`` gives the overlay precedence without asking
    setuptools to create two symlinks at one destination.
    """
    for base, _dirs, files in os.walk(root):
        relative = os.path.relpath(base, root)
        kept = [
            os.path.join(base, name)
            for name in files
            if os.path.isfile(os.path.join(base, name))
            and name != '.DS_Store'
            and not (
                skip_if_present_in is not None
                and os.path.isfile(os.path.join(skip_if_present_in, relative, name))
            )
            and '__pycache__' not in base.split(os.sep)
        ]
        if kept:
            target = destination if relative == '.' else os.path.join(destination, relative)
            yield (os.path.join('share', package_name, target), kept)


setup(
    name=package_name,
    version='1.0.0',
    packages=find_namespace_packages(
        where=python_root,
    ),
    package_dir={'': python_root},
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        *_walk_data_files('configs', 'launch', 'assets'),
        *_walk_data_files_into(
            'worlds',
            'worlds',
            skip_if_present_in='acoustics/worlds',
        ),
        (
            os.path.join('share', package_name, 'acoustics'),
            ['acoustics/README.md', 'acoustics/record_acoustics_dataset.sh'],
        ),
        *_walk_data_files_into('acoustics/worlds', 'worlds'),
    ],
    install_requires=[
        'setuptools',
        'requests',
        'attrs',
        'shapely',
        'pillow',
        'numpy',
        'pyarrow',
        'mcap',
        'mcap-ros2-support',
    ],
    extras_require={
        'test': ['pytest>=7', 'hypothesis>=6'],
    },
    zip_safe=True,
    maintainer='voshch',
    maintainer_email='dev@voshch.dev',
    description='arena_simulation_setup.',
    license='MIT',
    scripts=[
        'scripts/model_staging',
        'scripts/preload_world',
        'scripts/touch_world',
    ],
    entry_points={
        'console_scripts': [
            f'generate_world = {package_name}.utils.generative.world_generator:main',
            f'world_generator = {package_name}.utils.generative.world_generator_ros:main',
            f'export_acoustics_recording = {package_name}.acoustics.export_recording:main',
            f'wait_acoustics_capture = {package_name}.acoustics.wait_capture:main',
            f'normalize_acoustics_scenarios = {package_name}.acoustics.scenario_layout:main',
        ],
    },
)
