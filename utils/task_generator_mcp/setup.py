import os
from glob import glob

from setuptools import setup

package_name = 'task_generator_mcp'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools', 'mcp>=2.0'],
    zip_safe=True,
    author='voshch',
    author_email='voshch@arena-rosnav.org',
    maintainer='voshch',
    maintainer_email='voshch@arena-rosnav.org',
    description='MCP server wrapping task_generator services as Model Context Protocol tools and resources',
    license='MIT',
    entry_points={
        'console_scripts': [
            'task_generator_mcp = task_generator_mcp.server:main',
        ],
    },
)
