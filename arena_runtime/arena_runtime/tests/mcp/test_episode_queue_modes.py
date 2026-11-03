from __future__ import annotations

import pytest

pytest.importorskip("arena_mcp")
pytest.importorskip("rclpy")

from arena_mcp.core import ToolError

from arena_runtime.env import episode_mcp as episode
from arena_runtime.env.task_layer import task_layer
from arena_runtime.testing.env_peer import ENV


@pytest.fixture(scope="module")
def layer() -> object:
    try:
        return task_layer()
    except LookupError:
        pytest.skip("no task layer is installed")


@pytest.mark.parametrize(
    "modes",
    [
        {"terrain": "flat"},
        {"robots": "__bad__"},
        {"obstacles": ""},
        {"robots": []},
    ],
)
def test_queue_refuses_what_the_layer_validator_refuses_with_its_message_and_sends_nothing(run, peers, layer, modes):
    with pytest.raises(ValueError) as raised:
        layer.validate_modes({axis: [name for name in ([value] if isinstance(value, str) else value) if name] for axis, value in modes.items()})
    sent = len(peers.humansim.queued)
    with pytest.raises(ToolError) as refused:
        run(episode.episode_queue(env_id=ENV, modes=modes))
    assert str(raised.value) in str(refused.value)
    assert len(peers.humansim.queued) == sent


def test_queue_sends_modes_only_and_the_record_carries_modes_and_params_only(run, peers):
    queued = run(episode.episode_queue(env_id=ENV, modes={"modules": []}))["queued"]
    request = peers.humansim.queued[-1]
    assert {pair.key: pair.value for pair in request.modes} == {"modules": ""}
    assert not request.params
    assert queued["modes"] == {"robots": ["random"], "obstacles": ["random"], "modules": []}
    assert queued["params"] == {}
    assert not {"tm_robots", "tm_obstacles", "tm_modules", "robots_params", "obstacles_params"} & set(queued)


def test_the_active_task_modes_answer_carries_modes_only(run):
    assert set(run(episode.episode_task_modes_active(env_id=ENV))) == {"env", "modes"}


def test_queue_refuses_a_param_of_the_mode_it_deselects_in_the_same_request(run, peers):
    sent = len(peers.humansim.queued)
    with pytest.raises(ToolError) as refused:
        run(episode.episode_queue(env_id=ENV, modes={"modules": []}, params={"task.fake_module.level": 5}))
    assert "task.fake_module.level" in str(refused.value)
    assert len(peers.humansim.queued) == sent
