from setuptools import find_packages, setup

package_name = 'task_generator'


setup(
    name=package_name,
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
    zip_safe=True,
    maintainer='Name',
    maintainer_email='your@email.com',
    description='Task modes and their registry for Arena-Rosnav envs',
    license='TODO: License declaration',
    entry_points={
        'arena_runtime.task_layer': [
            'task_generator = task_generator.tasks.layer',
        ],
    }
)
