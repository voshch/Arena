import abc
import traceback
import typing

import rcl_interfaces.msg
import rclpy.exceptions
import rclpy.node

from .shared import DefaultParameter

T = typing.TypeVar('T')
U = typing.TypeVar('U')


class ROSParamT(abc.ABC, typing.Generic[T]):
    @abc.abstractmethod
    def __init__(
        self,
        /,
        name: str,
        value: object,
        *,
        type_: rclpy.Parameter.Type | None = None,
        parse: typing.Callable[[object], T] | None = None,
        **kwargs: object,
    ) -> None: ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Get name.
        """

    @property
    @abc.abstractmethod
    def value(self) -> T:
        """
        Get cached value.
        """

    @value.setter
    @abc.abstractmethod
    def value(self, value: object) -> None:
        """
        Set value and publish.
        """

    @abc.abstractmethod
    def callback(self, value: object) -> bool:
        """
        Callback function for setting value.
        """

    @abc.abstractmethod
    def destroy(self) -> None:
        """
        Undeclare the parameter from the node. Idempotent.
        """


class _ROSParam(ROSParamT[T]):
    """
    Wrapper that handles callbacks.
    """

    _node: typing.ClassVar["ROSParamServer"]
    _name: str

    _type: typing.TypeVar
    _from_param: typing.Callable[[typing.Any], T]

    _value: T
    _parameter_value: typing.Any

    @staticmethod
    def identity(x: object, *args: object) -> object:
        """
        lambda x: x
        """
        del args
        return x

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> T:
        return self._value

    @value.setter
    def value(self, value: object) -> None:
        self._node.set_parameters([rclpy.Parameter(name=self._name, value=value)])

    @property
    def param(self) -> object:
        return self._parameter_value

    @param.setter
    def param(self, value: object) -> None:
        self._parameter_value = value
        self._value = self._from_param(value)

    def callback(self, value: object) -> bool:
        self.param = value
        return True

    def destroy(self) -> None:
        try:
            self._node.undeclare_parameter(self._name)
        except rclpy.exceptions.ParameterNotDeclaredException:
            pass

    def __init__(
        self,
        /,
        name: str,
        value: object | None = None,
        *,
        type_: rclpy.Parameter.Type | None = None,
        parse: typing.Callable[[object], T] | None = None,
        **kwargs: object,
    ) -> None:
        self._name = name

        if parse is None:
            parse = self.identity
        self._from_param = parse

        if type_ is not None:
            self._node.rosparam.declare_safe(
                self.name,
                type_,
            )

        self._node.register_param(self, value, **kwargs)


counter = 0


class _rosparam(typing.Generic[T]):
    """
    Light-weight stateless interface for singular typed rosparam actions (short-lived).
    Runtime checks are not performed.
    Use ROSParam instead for most use cases.
    """

    _node: typing.ClassVar["ROSParamServer"]

    class _UNSET: ...

    @classmethod
    def declare_safe(cls, param_name: str, value: object = None, *, descriptor: rcl_interfaces.msg.ParameterDescriptor | None = None, **kwargs: object) -> None:
        if cls._node.has_parameter(param_name):
            if descriptor is not None:
                existing = cls._node.get_parameter(param_name)
                if descriptor.type == rclpy.Parameter.Type.NOT_SET.value:
                    descriptor.type = existing.type_.value
                elif descriptor.type == rclpy.Parameter.Type.DOUBLE.value and existing.type_ == rclpy.Parameter.Type.INTEGER:
                    cls._node.undeclare_parameter(param_name)
                    cls._node.declare_parameter(param_name, float(existing.value), descriptor=descriptor, ignore_override=True)
                    return
                cls._node.set_descriptor(param_name, descriptor)
            return

        try:
            if descriptor is not None and descriptor.type != rclpy.Parameter.Type.NOT_SET.value and not isinstance(value, rclpy.Parameter.Type):
                type_enum = rclpy.Parameter.Type(descriptor.type)
                cls._node.declare_parameter(param_name, type_enum, descriptor=descriptor, **kwargs)
                if value is not None:
                    try:
                        already_set = cls._node.get_parameter(param_name).type_ != rclpy.Parameter.Type.NOT_SET
                    except rclpy.exceptions.ParameterUninitializedException:
                        already_set = False
                    if not already_set:
                        cls._node.set_parameters([rclpy.Parameter(name=param_name, type_=type_enum, value=value)])
            else:
                cls._node.declare_parameter(param_name, value, **({"descriptor": descriptor} if descriptor is not None else {}), **kwargs)
        except rclpy.exceptions.ParameterAlreadyDeclaredException:
            pass

    @classmethod
    def declare_forward(cls, name: str, value: object, *, descriptor: rcl_interfaces.msg.ParameterDescriptor) -> None:
        cls.declare_safe(name, value, descriptor=descriptor)

    @classmethod
    def get_unsafe(cls, param_name: str, default: T | type[_UNSET] = _UNSET) -> T:
        """
        Get value of parameter.
        """

        _default = default

        default_value = None
        if default is not cls._UNSET:
            default_value = DefaultParameter(default)

        result = cls._node.get_parameter_or(
            param_name,
            default_value,
        )
        if result.type_ is rclpy.Parameter.Type.NOT_SET:
            if _default is not cls._UNSET:
                return _default  # type: ignore
            raise ValueError(f'parameter {param_name} is unset and no default passed')

        return typing.cast(T, result.value)

    @classmethod
    def get(cls, param_name: str, default: T) -> T:
        """
        Get value of parameter. Declare if undeclared.
        """
        if default is not None:
            cls.declare_safe(param_name, default)
        return cls.get_unsafe(param_name, default)

    @classmethod
    def set_unsafe(cls, param_name: str, value: T) -> bool:
        """
        Set value of parameter.
        """

        return cls._node.set_parameters([rclpy.Parameter(param_name, value=value)])[0].successful

    @classmethod
    def set(cls, param_name: str, value: T) -> bool:
        """
        Set value of parameter. Declare if undeclared.
        """

        try:
            return cls.set_unsafe(param_name, value)
        except rclpy.exceptions.ParameterNotDeclaredException:
            cls._node.declare_parameter(param_name, value)
            return True

    @classmethod
    def callback(cls, param_name: str, callback: typing.Callable[[object], bool]) -> None:
        try:
            value = cls.get_unsafe(param_name)
        except ValueError:
            value = None

        cls._node.add_param_callback(param_name, callback)

        if value is not None:
            if cls._node.executor is not None:
                cls._node.executor.create_task(lambda: callback(value))
            else:
                callback(value)


class ROSParamServer(rclpy.node.Node):
    """
    Interface for interacting with this node's ros2 parameters.
    """

    # this confuses my type checker
    # ROSParam: type[_ROSParam[typing.Any]]
    # rosparam: type[_rosparam[typing.Any]]

    __callbacks: dict[str, set[typing.Callable[[object], bool]]]

    def add_param_callback(self, param_name: str, callback: typing.Callable[[object], bool]) -> None:
        """
        Add callback for parameter changes.
        """

        self.__callbacks.setdefault(param_name, set()).add(callback)

    def register_param(self, param: ROSParamT[T], value: object, **kwargs: object) -> None:
        del kwargs  # unused

        current_value = self.rosparam[T].get(param.name, value)

        self.add_param_callback(
            param.name,
            param.callback,
        )

        result = self._callback([rclpy.Parameter(name=param.name, value=current_value)])

        if not result.successful:
            raise RuntimeError(f'initial configuration of parameter {param.name} failed with {result.reason}')

    def _callback(self, params: list[rclpy.Parameter]) -> rcl_interfaces.msg.SetParametersResult:
        successful = True
        reason: list[str] = []
        for param in params:
            for callback in self.__callbacks.get(param.name, set()):
                self.get_logger().debug(f"setting param {param.name} with value {param.value} (callback {callback})")
                try:
                    successful &= callback(param.value)
                except BaseException as e:
                    self.get_logger().warn(f'setting parameter {param.name} with value {param.value} failed: {e}')
                    reason.append(''.join(traceback.TracebackException.from_exception(e).format()))
                    successful = False

        if not successful:
            # this function can cause inconsistent states when some callbacks
            # succeed, some fail. revert back to old value here.
            ...

        return rcl_interfaces.msg.SetParametersResult(successful=successful, reason="\n".join(reason))

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.__callbacks = {}
        self.add_on_set_parameters_callback(self._callback)
        self._setup_rosparam()

    def _setup_rosparam(self):
        class ROSParam_impl(_ROSParam[T], typing.Generic[T]):
            _node = self

        self.ROSParam = ROSParam_impl

        class rosparam_impl(_rosparam[T], typing.Generic[T]):
            _node = self

        self.rosparam = rosparam_impl
