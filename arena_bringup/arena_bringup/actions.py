import typing
from collections.abc import Mapping

import launch
import launch.launch_description_source


class IsolatedGroupAction(launch.actions.GroupAction):
    def __init__(self, actions: typing.Iterable[launch.Action], *args: object, **kwargs: object) -> None:
        return super().__init__(
            (
                launch.actions.PushEnvironment(),
                launch.actions.PushLaunchConfigurations(),
                *actions,
                launch.actions.PopLaunchConfigurations(),
                launch.actions.PopEnvironment(),
            ),
            *args,
            **kwargs,
        )


class _ClearLaunchConfigurations(launch.Action):
    def execute(self, context: launch.LaunchContext) -> None:
        context.launch_configurations.clear()
        return None


class IsolatedIncludeLaunchDescription(launch.Action):
    """IncludeLaunchDescription with symmetric LaunchConfiguration scope.

    Substitutions in `args` are resolved against the parent context first, then the live
    launch_configurations dict is cleared so the child sees only the resolved `args` plus
    its own DeclareLaunchArgument defaults. Parent state is restored on exit. Use when
    the included launch reuses generic arg names (e.g. ``global_planner``) that would
    otherwise leak in from the parent context.
    """

    def __init__(
        self,
        source: launch.launch_description_source.LaunchDescriptionSource,
        *,
        args: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        self._source = source
        self._args = dict(args) if args else {}

    def execute(self, context: launch.LaunchContext) -> list[launch.LaunchDescriptionEntity]:
        from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions

        resolved: dict[str, str] = {}
        for k, v in self._args.items():
            resolved[k] = v if isinstance(v, str) else perform_substitutions(context, normalize_to_list_of_substitutions(v))
        return [
            launch.actions.PushLaunchConfigurations(),
            _ClearLaunchConfigurations(),
            launch.actions.IncludeLaunchDescription(self._source, launch_arguments=list(resolved.items())),
            launch.actions.PopLaunchConfigurations(),
        ]


class IncludeLaunchDescriptionForward(launch.Action):
    """IncludeLaunchDescription that forwards every parent launch configuration to the child.

    `overrides` pin specific args (e.g. {'env.n': '0'}); they win over forwarded values.
    """

    def __init__(
        self,
        source: launch.launch_description_source.LaunchDescriptionSource,
        *,
        overrides: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._source = source
        self._overrides = dict(overrides) if overrides else {}

    def execute(self, context: launch.LaunchContext) -> list[launch.LaunchDescriptionEntity]:
        args = dict(context.launch_configurations)
        args.update(self._overrides)
        return [launch.actions.IncludeLaunchDescription(self._source, launch_arguments=args.items())]
