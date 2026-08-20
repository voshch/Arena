import pytest

from task_generator.auditory.microphone_config import parse_robot_microphones


def test_robot_microphones_require_robot_and_keep_stable_indices():
    specs = parse_robot_microphones(
        """
        - owner: robot
          robot: jackal_1
          placement: front
          frame: microphone_link
          index: 1
        - owner: robot
          robot: jackal_1
          placement: front
          frame: microphone_link_2
          index: 2
        """
    )

    assert [spec.listener_id for spec in specs] == [
        "microphone:robot:jackal_1:front:1",
        "microphone:robot:jackal_1:front:2",
    ]
    assert specs[0].resolve_frame("env_0/jackal_1") == (
        "env_0/jackal_1/microphone_link"
    )


def test_robot_microphones_reject_missing_robot():
    with pytest.raises(ValueError, match="robot must be a string"):
        parse_robot_microphones(
            "[{owner: robot, placement: front, frame: microphone_link}]"
        )


def test_robot_microphones_reject_duplicate_indexed_id():
    with pytest.raises(ValueError, match="duplicate microphone"):
        parse_robot_microphones(
            """
            - {robot: jackal_1, placement: front, frame: mic_a, index: 1}
            - {robot: jackal_1, placement: front, frame: mic_b, index: 1}
            """
        )
