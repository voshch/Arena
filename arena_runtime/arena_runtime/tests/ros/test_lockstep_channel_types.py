from __future__ import annotations

import pytest

from arena_runtime.lockstep import _parse_channel_spec


def test_channel_with_stamped_type_is_accepted() -> None:
    spec = _parse_channel_spec("beat|/arena/{env}/beat|arena_runtime_msgs/msg/LockstepHeartbeat|0.1|hard")
    assert spec.type_name == "arena_runtime_msgs/msg/LockstepHeartbeat"


@pytest.mark.parametrize("type_name", ["std_msgs/msg/String", "std_msgs/msg/Header"])
def test_channel_without_header_field_is_refused(type_name: str) -> None:
    with pytest.raises(ValueError, match="has no std_msgs/Header header"):
        _parse_channel_spec(f"search|/arena/{{env}}/search_state|{type_name}|0.1|soft")


def test_channel_with_unknown_type_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot resolve type"):
        _parse_channel_spec("x|/t|no_such_msgs/msg/Nope|0.1|hard")
