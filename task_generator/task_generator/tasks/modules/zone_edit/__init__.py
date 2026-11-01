import typing

from task_generator.modes import Modes
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, MODULE_MODES

if typing.TYPE_CHECKING:
    from task_generator.tasks.modules import TM_Module

_NS = _REGISTRY_NAMESPACE("zone_edit")


@MODULE_MODES.register(Modes.TM_Module.ZONE_EDIT, namespace=_NS)
def _load_zone_edit() -> type["TM_Module"]:
    from .impl import Mod_ZoneEdit

    return Mod_ZoneEdit
