import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'arena_runtime'


def existing(*patterns):
    return [p for pat in patterns for p in glob(pat) if os.path.exists(p)]


setup(
    name=package_name,
    packages=find_packages(
        where='.',
        include=[f'{package_name}*']
    ),
    package_dir={'': '.'},
    package_data={package_name: ['humans/profiles/*.yaml']},
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), existing('launch/*.launch.py')),
        (os.path.join('share', package_name, 'launch', 'human'),
         existing('launch/human/*.launch.py', 'launch/human/*.md')),
        (os.path.join('share', package_name, 'launch', 'human', 'hunav'),
         existing('launch/human/hunav/*.launch.py')),
        (os.path.join('share', package_name, 'launch', 'human', 'arena_humansim'),
         existing('launch/human/arena_humansim/*.launch.py')),
        (os.path.join('share', package_name, 'launch', 'acoustics'),
         existing('launch/acoustics/*.launch.py', 'launch/acoustics/*.md')),
        (os.path.join('share', package_name, 'launch', 'acoustics', 'arena'),
         existing('launch/acoustics/arena/*.launch.py')),
        (os.path.join('share', package_name, 'launch', 'hearing'),
         existing('launch/hearing/*.launch.py', 'launch/hearing/*.md')),
        (os.path.join('share', package_name, 'launch', 'hearing', 'arena'),
         existing('launch/hearing/arena/*.launch.py')),
    ],
    zip_safe=True,
    maintainer='voshch',
    maintainer_email='dev@voshch.dev',
    description='Simulator engine, world driver and per-env episode loop for Arena-Rosnav',
    license='TODO',
    entry_points={
        'console_scripts': [
            'arena_runtime_node = arena_runtime.arena_runtime_node:main',
            'urdf_publisher = arena_runtime.urdf_publisher:main',
            'clock_relay = arena_runtime.clock_relay:main',
            'arena_env_node = arena_runtime.arena_env_node:main',
            'generate_map = arena_runtime.env.map_generator:main',
        ],
        'arena_mcp.domains': [
            'runtime = arena_runtime.mcp:domain',
            'entities = arena_runtime.env.entities_mcp:domain',
            'episode = arena_runtime.env.episode_mcp:domain',
        ],
    }
)
