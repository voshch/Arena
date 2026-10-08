import os
from collections import defaultdict
from glob import glob

from setuptools import find_packages, setup

package_name = 'task_generator'


def existing(*patterns):
    return [p for pat in patterns for p in glob(pat) if os.path.exists(p)]

setup(
    name=package_name,
    packages=find_packages(
        where='.',
        include=[f'{package_name}*']
    ),
    package_dir={'': '.'},
    package_data={package_name: ['simulators/human/profiles/*.yaml']},
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
        (os.path.join('share', package_name, 'simulators', 'human', 'animations'),
         existing('task_generator/simulators/human/animations/*.npy', 'task_generator/simulators/human/animations/*.npz', 'task_generator/simulators/human/animations/*.yaml')),
    ],
    zip_safe=True,
    maintainer='Name',
    maintainer_email='your@email.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'task_generator_node = task_generator.task_generator_node:main',
            'generate_map = task_generator.utils.map_generator:main',
            # 'server = task_generator.server:main',
            # 'filewatcher = task_generator.filewatcher:main'
        ]
    }
)
