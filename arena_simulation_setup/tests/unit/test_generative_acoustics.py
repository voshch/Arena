"""The shipped acoustics worlds as an exact-geometry corpus for the sketch generator."""

import random
from pathlib import Path

import pytest
import shapely
import shapely.affinity
import yaml

from arena_simulation_setup.utils.generative.sketch import WorldGeneratorSketch

BUNDLE = Path(__file__).resolve().parents[3] / 'arena_evaluation/arena_evaluation/configs/benchmark/suites/acoustics/worlds'

SHAPES = {
    'bend': ' ┃\n━┛',
    'tee': '━┳━\n ┃',
    'cross': ' ┃\n━╋━\n ┃',
}

VARIANTS = {
    '': None,
    '_narrow': str.maketrans('┃━┳╋┛', '│─┬┼┘'),
    '_wide': str.maketrans('┃━┳╋┛', '║═╦╬╝'),
}

pytestmark = pytest.mark.skipif(not BUNDLE.is_dir(), reason='acoustics bundle is not installed')


def shipped(world: str) -> shapely.Polygon | shapely.MultiPolygon:
    data = yaml.safe_load((BUNDLE / world / '0' / 'world.yaml').read_text())
    return shapely.union_all([shapely.Polygon([(c['x'], c['y']) for c in zone['corners']]) for zone in data['zones']])


def generated(sketch: str) -> shapely.Polygon | shapely.MultiPolygon:
    generator = WorldGeneratorSketch({'sketch': sketch, 'cell': 8.0, 'light': 1.5, 'heavy': 3.0, 'double': 6.0}, random.Random(0))
    level = generator.compute()
    return shapely.union_all([shapely.Polygon([(c.x, c.y) for c in zone.corners]) for zone in level.zones])


@pytest.mark.parametrize('shape', sorted(SHAPES))
@pytest.mark.parametrize('variant', sorted(VARIANTS))
def test_sketch_reproduces_the_shipped_world(shape, variant):
    table = VARIANTS[variant]
    sketch = SHAPES[shape] if table is None else SHAPES[shape].translate(table)
    want = shipped(f'acoustics_{shape}{variant}')
    got = generated(sketch)
    origin = (want.bounds[0] - got.bounds[0], want.bounds[1] - got.bounds[1])
    assert shapely.affinity.translate(got, *origin).symmetric_difference(want).area == pytest.approx(0.0, abs=1e-9)
