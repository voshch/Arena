import typing

from task_generator.tasks.mode import TaskMode

from . import audio_systems, clear_forbidden_zones, rviz_ui, staged

if typing.TYPE_CHECKING:
    from task_generator.tasks.task import Task


class TM_Module(TaskMode):
    _task: "Task"

    def __init__(self, *args: object, task: "Task", **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._task = task

    def before_reset(self): ...

    def after_reset(self): ...


__all__ = [
    "TM_Module",
    "audio_systems",
    "clear_forbidden_zones",
    "rviz_ui",
    "staged",
]
