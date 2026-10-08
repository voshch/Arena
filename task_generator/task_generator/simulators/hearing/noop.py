"""Backend for robot.hearing:=none, no hearing layer runs."""

from task_generator.simulators.hearing import BaseHearing


class NoopHearing(BaseHearing):
    pass
