from __future__ import annotations

import typing

import attrs

from arena_simulation_setup.utils.cattrs import Parseable, Serializable, converter


@attrs.define
class SemanticCfg(Parseable, Serializable):
    """One state or predicate annotation on a door, elevator, or zone."""

    role: typing.Literal['state', 'predicate']
    name: str
    value: float | bool | str | None = None
    binding: str = 'kinematic'
    trigger: str = 'proximity'
    distance: float = -1.0
    params: dict = attrs.field(factory=dict)

    @classmethod
    def parse(cls, value: dict) -> 'SemanticCfg':
        """Parse a single state/predicate primitive, not a preset."""
        value = dict(value)
        if 'state' in value:
            role: typing.Literal['state', 'predicate'] = 'state'
            name = value.pop('state')
        elif 'predicate' in value:
            role = 'predicate'
            name = value.pop('predicate')
        else:
            raise ValueError(f'semantics primitive must have a state or predicate key: {value}')

        binding = value.pop('binding', 'kinematic')
        if binding == 'joint':
            raise ValueError('binding: joint is reserved and not supported in v1')
        trigger = value.pop('trigger', 'proximity')
        if trigger == 'contact':
            raise ValueError('trigger: contact is reserved and not supported in v1')
        cfg_value = value.pop('value', None)
        distance = float(value.pop('distance', -1.0))
        params = dict(value.pop('params', {}))

        if value:
            raise ValueError(f'unknown semantics keys: {sorted(value)}')

        return cls(role=role, name=name, value=cfg_value, binding=binding, trigger=trigger, distance=distance, params=params)

    def serialize(self) -> dict:
        result: dict = {self.role: self.name}
        if self.value is not None:
            result['value'] = self.value
        if self.binding != 'kinematic':
            result['binding'] = self.binding
        if self.trigger != 'proximity':
            result['trigger'] = self.trigger
        if self.distance != -1.0:
            result['distance'] = self.distance
        if self.params:
            result['params'] = self.params
        return result


_PRESETS: dict[str, list[dict]] = {
    'signal': [
        {'state': 'state'},
        {'state': 'phase_remaining'},
        {'predicate': 'stop'},
    ],
    'schedule': [
        {'state': 'state'},
        {'predicate': 'active'},
        {'state': 'window_remaining'},
    ],
    'gate': [
        {'predicate': 'locked'},
        {'predicate': 'blocked'},
    ],
    'pressure_plate': [
        {'predicate': 'pressed'},
    ],
    'occupancy_cap': [
        {'state': 'occupancy'},
        {'state': 'cap'},
        {'predicate': 'over_cap'},
    ],
}


def parse_semantics(value: list) -> list[SemanticCfg]:
    """attrs converter: expand presets, build SemanticCfgs, reject reserved binding/trigger."""
    result: list[SemanticCfg] = []
    for item in value:
        if isinstance(item, SemanticCfg):
            result.append(item)
            continue
        if 'preset' in item:
            item = dict(item)
            preset_name = item.pop('preset')
            if preset_name not in _PRESETS:
                raise ValueError(f'unknown semantics preset: {preset_name}')
            item_params = item.pop('params', {})
            for primitive in _PRESETS[preset_name]:
                primitive = dict(primitive)
                primitive_params = primitive.pop('params', {})
                merged = {**primitive, **item}
                merged_params = {**primitive_params, **item_params}
                if merged_params:
                    merged['params'] = merged_params
                result.append(SemanticCfg.parse(merged))
        else:
            result.append(SemanticCfg.parse(item))
    return result


converter.register_structure_hook_func(
    lambda t: t == list[SemanticCfg],
    lambda value, _: parse_semantics(value),
)
