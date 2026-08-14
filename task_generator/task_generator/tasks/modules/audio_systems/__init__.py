import typing

from task_generator.constants import Constants
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, MODULE_MODES

if typing.TYPE_CHECKING:
    from task_generator.tasks.modules import TM_Module

_NS = _REGISTRY_NAMESPACE("audio_systems")


@MODULE_MODES.register(
    Constants.TaskMode.TM_Module.AUDIO_SYSTEMS,
    namespace=_NS,
)
def _load_audio_systems() -> type["TM_Module"]:
    from .impl import Mod_AudioSystems

    return Mod_AudioSystems
