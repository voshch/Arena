import typing

from arena_rclpy_mixins.shared import Namespace
from arena_runtime.env.task_api import NodeInterface
from arena_runtime.env.task_layer import TaskContext

if typing.TYPE_CHECKING:
    from arena_rclpy_mixins import ArenaMixinNode


class Namespaced:
    _namespace: Namespace = Namespace('').ParamNamespace()

    def namespace(self, *path: str) -> Namespace:
        return self._namespace(*path)


class TaskModeKwargs(typing.TypedDict):
    ctx: TaskContext
    namespace: Namespace
    node: "ArenaMixinNode"


class TaskMode(NodeInterface["ArenaMixinNode"], Namespaced):
    _ctx: TaskContext

    def __init__(self, *, ctx: TaskContext, namespace: Namespace, node: "ArenaMixinNode") -> None:
        super().__init__(node=node)
        self._ctx = ctx
        self._namespace = namespace
