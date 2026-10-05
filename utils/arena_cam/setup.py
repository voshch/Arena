import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'arena_cam'

setup(
    name=package_name,
    packages=find_packages(
        where='.',
        include=[f'{package_name}*'],
    ),
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'plugin.xml']),
        ('share/' + package_name + '/resource', ['resource/arena_cam_icon.png']),
        (os.path.join('share', package_name, 'configs', 'cam', 'shots'),
         glob('configs/cam/shots/*.yaml')),
    ],
    zip_safe=True,
    maintainer='voshch',
    maintainer_email='dev@voshch.dev',
    description='Viewport-camera client for Arena: scripted shots and keyboard flying.',
    license='TODO',
    entry_points={
        'console_scripts': [
            'cam = arena_cam.cli:main',
        ]
    }
)
