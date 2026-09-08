import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'arena_auditory'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(where='.', include=[f'{package_name}*']),
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'weights.yaml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config', 'hearing'), glob('config/hearing/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'sounds'), glob('sounds/*.wav')),
    ],
    install_requires=['setuptools'],
    extras_require={
        'test': ['pytest>=7'],
    },
    zip_safe=True,
    maintainer='voshch',
    maintainer_email='dev@voshch.dev',
    description='Arena auditory simulator',
    license='TODO',
    entry_points={
        'console_scripts': [
            'sound_propagation_node = arena_auditory.sound_propagation_node:main',
            'sound_propagation_visualizer = arena_auditory.sound_propagation_visualizer:main',
            'microphone_array_node = arena_auditory.microphone_array_node:main',
            'microphone_diagnostic = arena_auditory.microphone_diagnostic:main',
            'robot_hearing_node = arena_auditory.robot_hearing_node:main',
            'robot_sound_node = arena_auditory.robot_sound_node:main',
            'human_sound_node = arena_auditory.human_sound_node:main',
            'human_sound_playback = arena_auditory.human_sound_playback_node:main',
            'environment_sound_playback = arena_auditory.environment_sound_playback_node:main',
            'auditory_benchmark = arena_auditory.benchmark:main',
            'auditory_offline_render = arena_auditory.offline_render:main',
            'acoustic_world_audit = arena_auditory.acoustic_audit:main',
            'hearing_belief_node = arena_auditory.hearing.belief_node:main',
            'hearing_policy = arena_auditory.hearing.policy_node:main',
            'hearing_seld_frontend = arena_auditory.hearing.seld_frontend_node:main',
            'hearing_audio_replay = arena_auditory.hearing.audio_replay:main',
            'hearing_setup = arena_auditory.hearing.weights:main',
        ]
    },
)
