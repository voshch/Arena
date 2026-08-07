from setuptools import find_packages, setup

package_name = 'human_steering'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(
        where='.',
        include=[f'{package_name}*'],
    ),
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'plugin.xml']),
        ('share/' + package_name + '/resource', ['resource/human_steering_icon.png']),
    ],
    install_requires=['setuptools'],
    extras_require={
        'test': ['pytest>=7'],
    },
    zip_safe=True,
    maintainer='voshch',
    maintainer_email='dev@voshch.dev',
    description='rqt GUI motion engine for the human:=manual backend.',
    license='TODO',
)
