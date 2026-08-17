from setuptools import find_packages, setup

package_name = 'arena_runtime'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(
        where='.',
        include=[f'{package_name}*']
    ),
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'pyyaml'],
    extras_require={
        'test': ['pytest>=7'],
    },
    zip_safe=True,
    maintainer='voshch',
    maintainer_email='dev@voshch.dev',
    description='Simulator engine and world driver for Arena-Rosnav',
    license='TODO',
    entry_points={
        'console_scripts': [
            'arena_node = arena_runtime.arena_node:main',
            'urdf_publisher = arena_runtime.urdf_publisher:main',
            'clock_relay = arena_runtime.clock_relay:main',
        ]
    }
)
