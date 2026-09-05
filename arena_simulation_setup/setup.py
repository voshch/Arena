import os
from setuptools import setup, find_namespace_packages

package_name = 'arena_simulation_setup'
python_root = 'src'


def _walk_data_files(*roots):
    # os.path.isfile filters out dangling symlinks. colcon --symlink-install
    # populates the build dir with per-file symlinks into source and does not
    # prune them when source files are deleted, so os.walk would otherwise
    # hand setuptools broken symlinks and the copy step would abort.
    for root in roots:
        for base, _dirs, files in os.walk(root):
            kept = [
                os.path.join(base, f)
                for f in files
                if os.path.isfile(os.path.join(base, f))
            ]
            if kept:
                yield (os.path.join('share', package_name, base), kept)


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
        *_walk_data_files('configs', 'launch', 'worlds', 'assets'),
    ],
    install_requires=[
        'setuptools',
        'requests',
        'attrs',
        'shapely',
        'pillow',
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
        ],
    },
)
