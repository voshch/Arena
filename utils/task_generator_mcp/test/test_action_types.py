"""Test: RunEpisode action goal/result/feedback types instantiate and carry expected fields."""

from task_generator_msgs.action import RunEpisode


def test_goal_has_world_field():
    goal = RunEpisode.Goal()
    goal.world = "empty_map"
    assert goal.world == "empty_map"


def test_goal_default_world_is_empty():
    goal = RunEpisode.Goal()
    assert goal.world == ""


def test_result_state_constants():
    assert RunEpisode.Result.QUEUED == 0
    assert RunEpisode.Result.RUNNING == 1
    assert RunEpisode.Result.SUCCESS == 2
    assert RunEpisode.Result.FAILED == 3
    assert RunEpisode.Result.SKIPPED == 4
    assert RunEpisode.Result.FATAL == 5


def test_result_fields():
    result = RunEpisode.Result()
    result.state = RunEpisode.Result.SUCCESS
    result.info = "finished"
    result.episode_id = 42
    assert result.state == 2
    assert result.info == "finished"
    assert result.episode_id == 42


def test_feedback_state_constant():
    assert RunEpisode.Feedback.STARTED == 1


def test_feedback_field():
    fb = RunEpisode.Feedback()
    fb.state = RunEpisode.Feedback.STARTED
    assert fb.state == 1


def test_state_name_mapping():
    from task_generator_mcp.tools import _RUN_EPISODE_STATE_NAMES

    assert _RUN_EPISODE_STATE_NAMES[RunEpisode.Result.QUEUED] == "QUEUED"
    assert _RUN_EPISODE_STATE_NAMES[RunEpisode.Result.RUNNING] == "RUNNING"
    assert _RUN_EPISODE_STATE_NAMES[RunEpisode.Result.SUCCESS] == "SUCCESS"
    assert _RUN_EPISODE_STATE_NAMES[RunEpisode.Result.FAILED] == "FAILED"
    assert _RUN_EPISODE_STATE_NAMES[RunEpisode.Result.SKIPPED] == "SKIPPED"
    assert _RUN_EPISODE_STATE_NAMES[RunEpisode.Result.FATAL] == "FATAL"
