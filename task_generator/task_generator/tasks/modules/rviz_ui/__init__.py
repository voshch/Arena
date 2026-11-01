import typing

from task_generator.modes import Modes
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, MODULE_MODES

if typing.TYPE_CHECKING:
    from task_generator.tasks.modules import TM_Module

_NS = _REGISTRY_NAMESPACE("rviz_ui")


@MODULE_MODES.register(Modes.TM_Module.RVIZ_UI, namespace=_NS)
def _load_rviz_ui() -> type["TM_Module"]:
    from .impl import Mod_OverrideRobot

    return Mod_OverrideRobot
