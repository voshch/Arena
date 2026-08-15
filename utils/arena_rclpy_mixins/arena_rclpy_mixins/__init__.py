"""Lazy package facade, importing one submodule must not drag in the others
(yaml_replace stays importable where `launch`/`rclpy` are unavailable)."""

import importlib
import typing

_EXPORTS = {
    'ActionClientWrapper': '.Async',
    'AsyncLaunchManager': '.Async',
    'AsyncNode': '.Async',
    'AsyncUtil': '.Async',
    'ClientWrapper': '.Async',
    'AsyncLifecycleClient': '.LifecycleClient',
    'LifecycleClient': '.LifecycleClient',
    'ArenaMixinNode': '.node',
    'launch_str_to_value': '.launch_params',
    'param_value_to_launch_str': '.launch_params',
    'AsyncFactoryRegistry': '.registry',
    'ClassRegistry': '.registry',
    'FactoryRegistry': '.registry',
    'ROSParamServer': '.ROSParamServer',
    'ROSParamT': '.ROSParamServer',
    'ServiceNamespace': '.ServiceNamespace',
    'DefaultParameter': '.shared',
    'FrameNamespace': '.shared',
    'Namespace': '.shared',
    'ParamNamespace': '.shared',
    'async_main': '.spin',
    'run_main': '.spin',
    'spin_context': '.spin',
    'spin_node': '.spin',
    'Time': '.Time',
    'TimeNode': '.Time',
    'YAMLReplacer': '.yaml_replace',
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> object:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return __all__


if typing.TYPE_CHECKING:
    from .Async import (
        ActionClientWrapper as ActionClientWrapper,
    )
    from .Async import (
        AsyncLaunchManager as AsyncLaunchManager,
    )
    from .Async import (
        AsyncNode as AsyncNode,
    )
    from .Async import (
        AsyncUtil as AsyncUtil,
    )
    from .Async import (
        ClientWrapper as ClientWrapper,
    )
    from .launch_params import launch_str_to_value as launch_str_to_value
    from .launch_params import param_value_to_launch_str as param_value_to_launch_str
    from .LifecycleClient import AsyncLifecycleClient as AsyncLifecycleClient
    from .LifecycleClient import LifecycleClient as LifecycleClient
    from .node import ArenaMixinNode as ArenaMixinNode
    from .registry import AsyncFactoryRegistry as AsyncFactoryRegistry
    from .registry import ClassRegistry as ClassRegistry
    from .registry import FactoryRegistry as FactoryRegistry
    from .ROSParamServer import ROSParamServer as ROSParamServer
    from .ROSParamServer import ROSParamT as ROSParamT
    from .ServiceNamespace import ServiceNamespace as ServiceNamespace
    from .shared import (
        DefaultParameter as DefaultParameter,
    )
    from .shared import (
        FrameNamespace as FrameNamespace,
    )
    from .shared import (
        Namespace as Namespace,
    )
    from .shared import (
        ParamNamespace as ParamNamespace,
    )
    from .spin import async_main as async_main
    from .spin import run_main as run_main
    from .spin import spin_context as spin_context
    from .spin import spin_node as spin_node
    from .Time import Time as Time
    from .Time import TimeNode as TimeNode
    from .yaml_replace import YAMLReplacer as YAMLReplacer
