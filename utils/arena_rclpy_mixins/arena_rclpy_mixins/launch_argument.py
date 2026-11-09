"""Launch arguments: declared by hand or from the Param they forward to, passed on to the node, and read back from any launch file."""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Iterable

import launch
import launch.actions
import launch.some_substitutions_type
import launch.substitutions
import launch_ros.parameter_descriptions

from .launch_params import value_to_launch_str
from .params import Choices, Param


class LaunchArgument(launch.actions.DeclareLaunchArgument):
    _auto_append: typing.ClassVar[list | None] = None

    @classmethod
    def auto_append(cls, target: list | None = None) -> None:
        """Append every LaunchArgument created from now on to target."""
        cls._auto_append = target

    def __init__(
        self,
        name: str,
        *,
        default_value: launch.some_substitutions_type.SomeSubstitutionsType | None = None,
        description: str,
        choices: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(name, default_value=default_value, description=description, choices=choices, **kwargs)
        if self._auto_append is not None:
            self._auto_append.append(self)

    @classmethod
    def from_param(cls, param: Param[typing.Any], *, prefix: str = "", keep_default: bool = False) -> LaunchArgument:
        """The launch argument prefix + name of param with its description and choices, defaulting to empty (the node default) or, with keep_default, to the param default."""
        values = list(param.constraint.values) if isinstance(param.constraint, Choices) and not param.constraint.many else None
        if keep_default:
            return cls(prefix + param.name, default_value=value_to_launch_str(param.ros_default()), description=param.description, choices=values)
        description = param.description if param.default == "" else f"{param.description} Empty = node default."
        return cls(prefix + param.name, default_value="", description=description, choices=None if values is None else list(dict.fromkeys(["", *values])))

    @property
    def substitution(self) -> launch.substitutions.LaunchConfiguration:
        return launch.substitutions.LaunchConfiguration(self.name)

    @property
    def dict(self) -> dict[str, launch.substitutions.LaunchConfiguration]:
        return {self.name: self.substitution}

    def param_value(self, type_: type) -> launch_ros.parameter_descriptions.ParameterValue:
        return launch_ros.parameter_descriptions.ParameterValue(self.substitution, value_type=type_)

    def param(self, type_: type) -> dict[str, launch_ros.parameter_descriptions.ParameterValue]:
        return {self.name: self.param_value(type_)}

    @property
    def str_param(self) -> dict[str, launch_ros.parameter_descriptions.ParameterValue]:
        return self.param(str)


def forwarded(context: launch.LaunchContext, params: Iterable[Param[typing.Any]], *, prefix: str = "") -> dict[str, object]:
    """Node parameters of params whose launch argument is set, by parameter name, as ROS values."""
    configs = context.launch_configurations
    return {param.name: param.coerce(configs[prefix + param.name]) for param in params if configs.get(prefix + param.name)}


class UnreadableLaunchFile(Exception):
    """The launch file is missing, ambiguous in its package or fails to load."""


@dataclasses.dataclass(frozen=True)
class DeclaredArgument:
    name: str
    description: str
    default: str | None
    choices: tuple[str, ...] | None


def declared_arguments(package: str, file: str) -> list[DeclaredArgument]:
    """Every argument the launch file declares, in declaration order."""
    import launch.launch_description_sources
    from ament_index_python.packages import PackageNotFoundError
    from ros2launch.api import InvalidLaunchFileError, MultipleLaunchFilesError, get_share_file_path_from_package

    try:
        path = get_share_file_path_from_package(package_name=package, file_name=file)
        description = launch.launch_description_sources.get_launch_description_from_any_launch_file(path)
    except (PackageNotFoundError, FileNotFoundError, MultipleLaunchFilesError, InvalidLaunchFileError) as exc:
        raise UnreadableLaunchFile(f"{package}/{file}: {exc}") from exc
    return [DeclaredArgument(name=argument.name, description=_description(argument), default=_default(argument), choices=None if argument.choices is None else tuple(argument.choices)) for argument in description.get_launch_arguments()]


def _description(argument: launch.actions.DeclareLaunchArgument) -> str:
    if argument.choices is None:
        return argument.description
    return argument.description.removesuffix(f" Valid choices are: {argument.choices}")


def _default(argument: launch.actions.DeclareLaunchArgument) -> str | None:
    import launch.substitutions.substitution_failure
    import launch.utilities

    if argument.default_value is None:
        return None
    try:
        return launch.utilities.perform_substitutions(launch.LaunchContext(), argument.default_value)
    except launch.substitutions.substitution_failure.SubstitutionFailure:
        return " + ".join(token.describe() for token in argument.default_value)
