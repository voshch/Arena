from __future__ import annotations

import pytest

pytest.importorskip("arena_mcp")
pytest.importorskip("rclpy")

from arena_mcp.core import ROS, current_episode
from arena_runtime_msgs.msg import EpisodeRecord

from arena_runtime.env import episode_mcp as episode

from arena_runtime.testing.env_peer import ENV, fqn


@pytest.fixture()
def replayed(run, peers):
    first = peers.humansim.episode_id
    peers.humansim.burst([EpisodeRecord.SUCCESS, EpisodeRecord.FAILED, EpisodeRecord.SKIPPED])
    run(ROS.shutdown())
    run(ROS.node())
    return first


def test_current_on_a_cold_cache_is_the_newest_record_of_the_replay(run, replayed):
    current = run(episode.episode_current(env_id=ENV))
    assert current["current"]["episode_id"] == replayed + 3
    assert current["current"]["outcome"] == "running"
    assert run(current_episode(fqn(ENV))).episode_id == replayed + 3


def test_history_lists_every_replayed_episode_newest_first_with_its_final_outcome(run, replayed):
    history = run(episode.episode_history(env_id=ENV, limit=4))
    assert [(entry["episode_id"], entry["outcome"]) for entry in history] == [(replayed + 3, "running"), (replayed + 2, "skipped"), (replayed + 1, "failed"), (replayed, "success")]


def test_reset_after_a_replay_waits_for_the_episode_after_the_newest(run, peers, replayed):
    reset = run(episode.episode_reset(env_id=ENV))
    assert reset["episode_id"] == replayed + 4
    assert peers.humansim.episode_id == replayed + 4
