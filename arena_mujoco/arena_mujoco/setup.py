import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'arena_mujoco'

setup(
    name=package_name,
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    zip_safe=True,
    maintainer='Vova Shcherbyna',
    maintainer_email='voshch@arena-rosnav.org',
    description='MuJoCo simulation backend for Arena-Rosnav',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'run_mujoco = arena_mujoco.run_mujoco:main',
        ],
    },
)
