import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'arena_fleet'

setup(
    name=package_name,
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'configs'), glob('configs/*.yaml')),
        (os.path.join('share', package_name, 'examples', 'doorway'), glob('examples/doorway/*.yaml')),
        (os.path.join('share', package_name, 'examples', 'doorway', 'scenario'), glob('examples/doorway/scenario/*.yaml')),
    ],
    zip_safe=True,
    author='voshch',
    author_email='voshch@arena-rosnav.org',
    maintainer='voshch',
    maintainer_email='voshch@arena-rosnav.org',
    description='Fleet dispatcher for shared doors',
    license='MIT',
    entry_points={
        'console_scripts': [
            'dispatcher = arena_fleet.__main__:main',
        ],
    },
)
