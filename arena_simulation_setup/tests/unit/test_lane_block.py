import copy

from arena_simulation_setup.shared.lane import shift_lane_points


def _config() -> dict:
    return {
        'rate_profile': [[0.0, 1.0]],
        'lane': {
            'side': 'right',
            'offset': 0.6,
            'segments': [
                {'centerline': [[0.0, 0.0], [5.0, 0.0]]},
                {'centerline': [[5.0, 0.0], [5.0, 4.0]]},
            ],
        },
    }


def test_shift_moves_every_centerline_point_by_the_offset() -> None:
    shifted = shift_lane_points(_config(), 10.0, -2.0)
    segments = shifted['lane']['segments']
    assert segments[0]['centerline'] == [[10.0, -2.0], [15.0, -2.0]]
    assert segments[1]['centerline'] == [[15.0, -2.0], [15.0, 2.0]]


def test_shift_keeps_every_other_key() -> None:
    shifted = shift_lane_points(_config(), 10.0, -2.0)
    assert shifted['rate_profile'] == [[0.0, 1.0]]
    assert shifted['lane']['side'] == 'right'
    assert shifted['lane']['offset'] == 0.6


def test_shift_leaves_the_input_untouched() -> None:
    config = _config()
    original = copy.deepcopy(config)
    shift_lane_points(config, 10.0, -2.0)
    assert config == original


def test_config_without_a_lane_comes_back_equal() -> None:
    config = {'rate_profile': [[0.0, 1.0]]}
    assert shift_lane_points(config, 10.0, -2.0) == config
