"""The arguments a launch file declares, read from its launch description."""

from __future__ import annotations

import dataclasses
import typing

if typing.TYPE_CHECKING:
    import launch.actions


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
    import launch
    import launch.substitutions.substitution_failure
    import launch.utilities

    if argument.default_value is None:
        return None
    try:
        return launch.utilities.perform_substitutions(launch.LaunchContext(), argument.default_value)
    except launch.substitutions.substitution_failure.SubstitutionFailure:
        return " + ".join(token.describe() for token in argument.default_value)
