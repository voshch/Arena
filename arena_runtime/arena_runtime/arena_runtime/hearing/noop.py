"""Backend for robot.hearing:=none, no hearing layer runs."""

from arena_runtime.hearing import BaseHearing


class NoopHearing(BaseHearing):
    pass
