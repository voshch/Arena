import os

from setuptools import find_packages, setup

package_name = 'arena_bringup'


def recursive_walk(base_dir, *, destination=None, relative_to=None):
    if destination is None:
        destination = os.path.join('share', package_name)
    if relative_to is None:
        relative_to = ''

    def process(base, files):
        adjusted_base = os.path.relpath(base, relative_to)
        kept = [
            os.path.join(base, file)
            for file in files
            if os.path.isfile(os.path.join(base, file))
        ]
        return (
            os.path.normpath(os.path.join(destination, adjusted_base)),
            kept,
        )

    return [
        entry
        for entry in (
            process(base, files)
            for base, _, files in os.walk(base_dir)
        )
        if entry[1]
    ]


setup(
    name=package_name,
    packages=find_packages(where='.', include=[f'{package_name}*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        *recursive_walk('scripts', destination=os.path.join('lib', package_name), relative_to='scripts'),
        *recursive_walk('launch'),
        *recursive_walk('configs'),
    ],
    zip_safe=True,
    maintainer='voshch',
    maintainer_email='dev@voshch.dev',
    description='Arena bringup package',
    license='MIT',
    entry_points={
        'launch_ros.node_action': [
            'NodeLogLevelExtension = arena_bringup.extensions.NodeLogLevelExtension:NodeLogLevelExtension',
        ],
    },
)
