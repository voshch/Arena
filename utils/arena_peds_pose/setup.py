import os

from setuptools import setup

package_name = 'arena_peds_pose'

setup(
    name=package_name,
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
    ],
    package_data={
        package_name: ['bone_map.json'],
    },
    zip_safe=True,
    maintainer='voshch',
    maintainer_email='dev@voshch.dev',
    description='Pedestrian skeleton pose data shared by Arena simulators',
    license='MIT',
)
