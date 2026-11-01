"""Generated stand-ins for bucket assets, so test suites resolve clips, sounds and materials offline."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import wave
import zlib
from pathlib import Path

import numpy as np
import yaml

from arena_simulation_setup.tree.assets.Animation import write_clip

FPS = 20.0
JOINTS = (
    'chest', 'l_ankle', 'l_elbow', 'l_knee', 'l_p_collar', 'l_p_hip', 'l_p_shoulder', 'l_r_hip', 'l_r_shoulder',
    'l_y_ankle', 'l_y_collar', 'l_y_hip', 'l_y_shoulder', 'p_head', 'r_ankle', 'r_chest', 'r_elbow', 'r_head',
    'r_knee', 'r_p_collar', 'r_p_hip', 'r_p_shoulder', 'r_r_hip', 'r_r_shoulder', 'r_spine', 'r_waist', 'r_y_ankle',
    'r_y_collar', 'r_y_hip', 'r_y_shoulder', 'spine', 'waist', 'y_chest', 'y_head', 'y_spine', 'y_waist',
)  # fmt: skip
REST = (
    0.048, 0.362, 0.376, -0.137, 0.026, 0.131, -0.13, -0.115, -1.467, -0.366, -0.159, 0.054, -1.497, -0.228, 0.371,
    0.077, 0.603, 0.0, -0.187, 0.159, 0.199, -0.18, -0.016, 0.209, -0.053, -0.004, 0.139, -0.124, 0.093, 0.216, 0.149,
    -0.152, -0.005, -0.003, -0.012, -0.012,
)  # fmt: skip
POINT = (
    0.008, 0.342, 0.46, -0.166, 0.094, 0.099, -0.244, 0.132, -2.115, 0.699, -0.07, -0.907, -1.899, -0.153, 0.388,
    -0.154, 0.237, 0.0, -0.154, 0.589, 0.192, -1.433, 0.097, 3.142, 0.001, 0.008, -0.344, 0.076, 1.0, 2.473, 0.177,
    -0.17, 0.116, -0.133, 0.124, 0.124,
)  # fmt: skip
ARM = frozenset({'r_p_collar', 'r_y_collar', 'r_p_shoulder', 'r_r_shoulder', 'r_y_shoulder', 'r_elbow'})
CLIPS: dict[str, dict] = {
    'point_at_template': {'seconds': 2.15, 'meta': {'loop': True}, 'joints': frozenset(JOINTS)},
    'wave': {'seconds': 3.95, 'meta': {'loop': True, 'joints': 'upper_body'}},
    'beckon': {'seconds': 3.0, 'meta': {'loop': True, 'joints': 'upper_body'}},
    'hug': {'seconds': 3.15, 'meta': {'loop': True, 'joints': 'upper_body'}},
    'shake_hand': {'seconds': 3.95, 'meta': {'loop': True, 'joints': 'upper_body'}},
    'talk_with_arm_gesture': {'seconds': 4.0, 'meta': {'loop': True, 'reverse': True, 'joints': 'upper_body'}},
    'talk_with_arm_gesture_v1': {'seconds': 4.0, 'meta': {'loop': True, 'reverse': True, 'joints': 'upper_body'}},
    'jump': {'seconds': 2.0, 'meta': {'loop': False}},
}
MATERIALS = {
    'Marble': 'kind::stone',
    'Porcelain_Tile_4': 'kind::ceramic',
    'Aluminum_Anodized': 'kind::metal',
    'Concrete_Smooth': 'kind::concrete',
}
FOOTSTEPS = {
    'walnut_planks': ('walnut', 48000, 1, 0.58),
    'oak_planks': ('oak', 44100, 2, 1.0),
    'marble_tile': ('marble', 48000, 2, 1.0),
    'smooth_concrete': ('concrete', 44100, 2, 1.0),
    'ceramic_tile': ('ceramic', 48000, 1, 1.0),
    'default': (None, 44100, 2, 1.0),
}
LOOPS = {
    'alarm_loop': ('alarm', 'Looping alarm for environment alarm sources.', 88.0, -17.5, ['alarm', 'loop', 'environment'], ['alarm']),
    'radio_loop': ('music', 'Looping radio music.', 62.0, -12.5, ['music', 'radio', 'loop', 'environment'], ['radio', 'music']),
}


def _synth(path: Path, rate: int, channels: int, seconds: float, *, decay: float = 0.0, tone: float = 0.0) -> None:
    """Write a seeded 16-bit WAV: noise under an optional decay, or a tone with a slow beat."""
    rng = np.random.default_rng(zlib.crc32(path.name.encode()))
    n = int(seconds * rate)
    t = np.arange(n) / rate
    signal = rng.normal(0.0, 0.25, n)
    if tone:
        signal = 0.4 * np.sin(2 * np.pi * tone * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t)) + 0.1 * signal
    if decay:
        signal = signal * np.exp(-decay * t)
    pcm = np.clip(signal, -1.0, 1.0)
    frames = (pcm[:, None].repeat(channels, axis=1) * 32767).astype('<i2')
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(frames.tobytes())


def _sound(root: Path, name: str, manifest: dict) -> Path:
    directory = root / 'Common' / 'Sound' / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'{name}.yaml').write_text(yaml.safe_dump({'version': 2, **manifest}, sort_keys=False))
    return directory


def write_sounds(root: Path) -> None:
    """The default asset of every sound kind, under the ids and manifests the catalog expects."""
    variants = []
    directory = _sound(root, 'footstep', {})
    for surface, (floor, rate, channels, seconds) in FOOTSTEPS.items():
        file = f'footstep_{surface}.wav'
        _synth(directory / file, rate, channels, seconds, decay=12.0)
        match = {'match': {'floor': ['ceramic', 'porcelain'] if floor == 'ceramic' else [floor]}} if floor else {'default': True}
        variants.append({'id': f'footstep_{surface}_01', 'file': file, **match, 'tags': ['walk', surface]})
    _sound(root, 'footstep', {
        'kind': 'footstep', 'desc': 'One generated footstep per floor.', 'tags': ['footstep', 'pedestrian', 'walk', 'floor'],
        'level_db': 45.0, 'reference_distance_m': 1.0, 'normalize_dbfs': -26.5, 'loop': False, 'model': 'wav', 'surface': 'floor',
        'variants': variants,
    })  # fmt: skip
    directory = _sound(root, 'greeting', {
        'kind': 'speech', 'desc': 'Generated greeting.', 'tags': ['speech', 'voice', 'greeting', 'pedestrian'],
        'level_db': 60.0, 'reference_distance_m': 1.0, 'normalize_dbfs': -19.0, 'loop': False, 'model': 'wav',
        'variants': [{'id': 'greeting_01', 'file': 'greeting.wav', 'tags': ['neutral', 'adult']}],
    })  # fmt: skip
    _synth(directory / 'greeting.wav', 44100, 2, 1.36, tone=180.0)
    directory = _sound(root, 'motor', {
        'kind': 'motor', 'desc': 'Procedural drivetrain on Jackals, a generated loop otherwise.', 'tags': ['motor', 'robot', 'drivetrain', 'ego_noise', 'loop'],
        'level_db': 45.0, 'reference_distance_m': 1.0, 'normalize_dbfs': -15.0, 'loop': True,
        'variants': [
            {'id': 'jackal_drivetrain', 'model': 'drivetrain', 'params': {'spec': 'jackal'}, 'match': {'robot_model': ['jackal']}},
            {'id': 'motor_loop_01', 'model': 'wav_loop', 'file': 'long_motor_sound_middle.wav', 'default': True, 'tags': ['robot', 'motor', 'steady']},
        ],
    })  # fmt: skip
    _synth(directory / 'long_motor_sound_middle.wav', 44100, 2, 0.78)
    for name, (kind, desc, level, normalize, tags, variant_tags) in LOOPS.items():
        directory = _sound(root, name, {
            'kind': kind, 'desc': desc, 'tags': tags, 'level_db': level, 'reference_distance_m': 1.0, 'normalize_dbfs': normalize,
            'loop': True, 'model': 'wav_loop', 'variants': [{'id': f'{name}_01', 'file': f'{name}.wav', 'tags': variant_tags}],
        })  # fmt: skip
        _synth(directory / f'{name}.wav', 44100, 2, 3.0, tone=440.0)


def write_clips(root: Path) -> None:
    """Rest to point and back at the humanoid rig's joints, all joints on the template, the arm on the rest."""
    rest, point = np.array(REST), np.array(POINT)
    for name, spec in CLIPS.items():
        moving = spec.get('joints', ARM)
        delta = np.array([p - r if j in moving else 0.0 for j, r, p in zip(JOINTS, rest, point, strict=True)])
        n = int(spec['seconds'] * FPS) + 1
        frames = []
        for k in range(n):
            phase = k / (n - 1)
            weight = np.sin(np.pi * min(phase / 1.4, 1.0)) ** 2
            frames.append({'t': k / FPS, 'angles': dict(zip(JOINTS, (rest + weight * delta).tolist(), strict=True))})
        write_clip(root / 'Common' / 'Animation' / name, frames, {'desc': name.replace('_', ' '), 'tags': ['domain::common'], 'fps': FPS, **spec['meta']})


def write_materials(root: Path) -> None:
    """The default materials as descriptors without textures."""
    for name, kind in MATERIALS.items():
        directory = root / 'Common' / 'Material' / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'annotation.yaml').write_text(yaml.safe_dump({'asa': 1, 'desc': '', 'name': name, 'path': f'Material/{name}', 'tags': ['domain::common', kind]}))
        (directory / f'{name}.mdl').write_text('mdl 1.6;\n')


def write_fake_assets(root: Path) -> None:
    write_sounds(root)
    write_clips(root)
    write_materials(root)


def install_fake_assets() -> Path:
    """Generate the assets into a temporary local root that resolves ahead of every bucket, for this process and its children."""
    root = Path(tempfile.mkdtemp(prefix='arena-fake-assets-'))
    write_fake_assets(root)
    os.environ['ARENA_ASSETS_DIR_LOCAL'] = str(root)
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    return root
