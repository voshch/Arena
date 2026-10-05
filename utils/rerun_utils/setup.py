import os
from glob import glob

from setuptools import setup

package_name = 'rerun_utils'

setup(
    name=package_name,
    packages=[
        package_name,
        f'{package_name}.renderers',
    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*.launch.py'))),
    ],
    zip_safe=True,
    maintainer='voshch',
    maintainer_email='dev@voshch.dev',
    description='Rerun.io web-viewer bridge for the arena_viz contract.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'rerun_bridge = rerun_utils.scripts.rerun_bridge:main',
        ],
    },
)
