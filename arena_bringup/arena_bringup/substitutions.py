from __future__ import annotations

import copy
import os
import tempfile
import typing
from collections.abc import Callable, Mapping, MutableMapping

import launch
import launch.actions
import launch.logging
import launch.substitutions
import launch.utilities
import launch_ros.parameter_descriptions
import yaml
from arena_rclpy_mixins.yaml_replace import YAMLReplacer

_YAMLReplacer = YAMLReplacer


class NoAliasDumper(yaml.Dumper):
    def ignore_aliases(self, data: object) -> bool:
        return True


def _yaml_iter(obj: list | dict) -> range | typing.KeysView:
    if isinstance(obj, dict):
        return obj.keys()
    elif isinstance(obj, list):
        return range(len(obj))
    else:
        raise RuntimeError()


class LaunchArgument(launch.actions.DeclareLaunchArgument):
    _auto_append: typing.ClassVar[list | None] = None

    @classmethod
    def auto_append(cls, target: list | None = None):
        """
        auto append self to list after creation
        """
        cls._auto_append = target

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        if self._auto_append is not None:
            self._auto_append.append(self)

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


DEPRECATED_LAUNCH_ARGS: Mapping[str, str] = {
    'isaac.physics': 'sim.isaac.physics',
    'env_id': 'env.id',
    'ns': 'env.ns',
    'managed': 'env.managed',
    'tm_robots': 'task.robots',
    'tm_obstacles': 'task.obstacles',
    'tm_modules': 'task.modules',
    'task_config': 'task.config',
    'scenario_file': 'task.scenario',
    'parameter_file': 'task.params',
    'episodes': 'task.episodes',
    'auto_reset': 'task.auto_reset',
    'fail_on_collision': 'task.fail_on_collision',
    'mobile': 'robot.mobile',
    'mobile.': 'robot.mobile.',
    'arm': 'robot.arm',
    'arm.': 'robot.arm.',
    'planner': 'robot.planner',
    'train_mode': 'robot.train',
    'record_data_dir': 'record.dir',
    'disable_auto_recorder': 'record.auto',
}
INVERTED_LAUNCH_ARGS: frozenset[str] = frozenset({'disable_auto_recorder'})


def resolve_deprecated_args(
    configs: MutableMapping[str, str],
    warn: Callable[[str, str], None],
    aliases: Mapping[str, str] = DEPRECATED_LAUNCH_ARGS,
    inverted: frozenset[str] = INVERTED_LAUNCH_ARGS,
) -> None:
    """Move deprecated keys onto their successors in place; a trailing '.' aliases a whole prefix, `inverted` flips booleans."""
    for old, new in aliases.items():
        if old.endswith('.'):
            hits = [(k, new + k[len(old) :]) for k in list(configs) if k.startswith(old)]
        elif old in configs:
            hits = [(old, new)]
        else:
            continue
        for old_key, new_key in hits:
            warn(old_key, new_key)
            value = configs.pop(old_key)
            if new_key in configs:
                continue
            if old in inverted:
                value = 'false' if value.strip().lower() in ('true', '1') else 'true'
            configs[new_key] = value


def deprecated_launch_args(context: launch.LaunchContext) -> None:
    logger = launch.logging.get_logger('arena')
    resolve_deprecated_args(
        context.launch_configurations,
        lambda old, new: logger.warning(f"\033[33mlaunch arg '{old}' is deprecated, use '{new}'\033[0m"),
    )


class OptionalLaunchArgument:
    """undeclared launch configuration that defers default to downstream consumer"""

    def __init__(self, name: str) -> None:
        self.name = name

    def value_if_given(self, context: launch.LaunchContext) -> str | None:
        return context.launch_configurations.get(self.name) or None

    def dict_if_given(self, context: launch.LaunchContext) -> dict[str, str]:
        v = self.value_if_given(context)
        return {self.name: v} if v is not None else {}

    def cli_if_given(self, context: launch.LaunchContext, flag: str) -> list[str]:
        v = self.value_if_given(context)
        return [flag, v] if v is not None else []


class SelectAction(launch.Action):
    _actions: dict[str, list[launch.Action]]
    _selector: list[launch.Substitution]

    def __init__(self, selector: launch.SomeSubstitutionsType) -> None:
        launch.Action.__init__(self)
        self._actions = {}
        self._selector = launch.utilities.normalize_to_list_of_substitutions(selector)

    def add(self, value: str, action: launch.Action):
        self._actions.setdefault(value, []).append(action)

    @property
    def keys(self) -> list[str]:
        return list(self._actions.keys())

    def execute(self, context: launch.LaunchContext) -> list[launch.Action]:  # type: ignore
        key = launch.utilities.perform_substitutions(context, self._selector)
        return self._actions.get(key, [])


class YAMLFileSubstitution(launch.Substitution):
    _path: list[launch.Substitution]
    _default: dict | YAMLFileSubstitution | None
    _substitute: bool

    def __init__(
        self,
        path: launch.SomeSubstitutionsType,
        *,
        default: dict | YAMLFileSubstitution | None = None,
        substitute: bool = False,
    ):
        launch.Substitution.__init__(self)
        self._path = launch.utilities.normalize_to_list_of_substitutions(path)
        self._default = default
        self._substitute = substitute

    def perform(
        self,
        context: launch.LaunchContext,
    ) -> str:
        yaml_path = launch.utilities.perform_substitutions(context, self._path)
        try:
            with open(yaml_path) as f:
                contents = yaml.safe_load(f)
        except BaseException as e:
            if self._default is not None:
                if isinstance(self._default, type(self)):
                    contents = self._default.perform_load(context)
                else:
                    contents = self._default
            else:
                raise e

        if self._substitute:

            def substitute(obj: dict | list, k: int | str) -> None:
                if isinstance(obj[k], launch.Substitution):
                    obj[k] = obj[k].perform(context)

            def substitute_recursive(obj: dict | list) -> dict | list:
                for k in _yaml_iter(obj):
                    substitute(obj, k)
                    v = obj[k]
                    if isinstance(v, (list, dict)):
                        obj[k] = substitute_recursive(v)

                return obj

            assert isinstance(contents, (dict, list))
            contents = substitute_recursive(contents)

        yaml_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        yaml.dump(contents, yaml_file, Dumper=NoAliasDumper)
        yaml_file.close()

        return yaml_file.name

    def perform_load(self, context: launch.LaunchContext) -> dict:
        with open(yaml_path := self.perform(context)) as f:
            content = yaml.safe_load(f)
        if not isinstance(content, dict):
            raise yaml.YAMLError(f"{yaml_path} does not contain a top-level dictionary")
        return content

    @classmethod
    def from_dict(cls, obj: dict, /, *, substitute: bool = True) -> YAMLFileSubstitution:
        return cls(path=[], default=obj, substitute=substitute)


class YAMLRetrieveSubstitution(launch.Substitution):
    _obj: YAMLFileSubstitution
    _key: list[launch.Substitution]

    def __init__(
        self,
        obj: YAMLFileSubstitution,
        key: launch.SomeSubstitutionsType,
    ):
        launch.Substitution.__init__(self)
        self._obj = obj
        self._key = launch.utilities.normalize_to_list_of_substitutions(key)

    def perform(self, context: launch.LaunchContext) -> str:
        obj = self._obj.perform_load(context)
        key = launch.utilities.perform_substitutions(context, self._key)

        remainder: str = key

        while len(remainder):
            if remainder in _yaml_iter(obj):
                return obj[remainder]

            newkey, *remainders = remainder.split(os.sep, 1)
            remainder = ''.join(remainders)

            try:
                newkey = int(newkey)
            except ValueError:
                pass

            obj = obj[newkey]

        return str(obj)


class YAMLMergeSubstitution(launch.Substitution):
    _base: YAMLFileSubstitution
    _yamls: typing.Iterable[YAMLFileSubstitution]

    def __init__(
        self,
        base: YAMLFileSubstitution,
        *yamls: YAMLFileSubstitution,
    ):
        launch.Substitution.__init__(self)
        self._base = base
        self._yamls = yamls

    @classmethod
    def _recursive_merge(cls, obj1: dict, obj2: dict) -> dict:
        for k, v in obj2.items():
            if k in obj1 and isinstance(v, dict):
                v = cls._recursive_merge(obj1[k], v)
            obj1[k] = v
        return obj1

    @classmethod
    def _append_yaml(cls, base: dict, obj: dict) -> dict:
        base = cls._recursive_merge(base, obj)
        return base

    def perform(self, context: launch.LaunchContext) -> str:

        combined = copy.deepcopy(self._base.perform_load(context))

        for yaml_path in self._yamls:
            combined = self._append_yaml(combined, yaml_path.perform_load(context))

        return YAMLFileSubstitution.from_dict(combined).perform(context)


class YAMLReplaceSubstitution(launch.Substitution):
    _substitutions: YAMLFileSubstitution
    _obj: YAMLFileSubstitution

    def __init__(
        self,
        *,
        substitutions: YAMLFileSubstitution,
        obj: YAMLFileSubstitution,
    ):
        launch.Substitution.__init__(self)
        self._substitutions = substitutions
        self._obj = obj

    def perform(self, context: launch.LaunchContext) -> str:

        substitutions = self._substitutions.perform_load(context)

        replacer = _YAMLReplacer(substitutions)

        result = replacer.replace(self._obj.perform_load(context))

        return YAMLFileSubstitution.from_dict(
            result,
            substitute=True,
        ).perform(context)


class CurrentNamespaceSubstitution(launch.Substitution):
    def perform(self, context: launch.LaunchContext) -> str:
        """Perform the substitution."""
        return context.launch_configurations.get('ros_namespace', '/')
