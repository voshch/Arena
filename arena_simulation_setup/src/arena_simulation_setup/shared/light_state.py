from __future__ import annotations

import attrs
import cattrs

from arena_simulation_setup.utils.cattrs import Parseable, Serializable, converter


def check_light_state(owner: str, cct_K: float, level: float, dead_fraction: float, light_on: str, lit: bool | None) -> None:
    if not cct_K > 0.0:
        raise ValueError(f'{owner} cct_K must be positive')
    if not 0.0 <= level <= 1.0:
        raise ValueError(f'{owner} level must be within 0..1')
    if not 0.0 <= dead_fraction <= 1.0:
        raise ValueError(f'{owner} dead_fraction must be within 0..1')
    if light_on == '!':
        raise ValueError(f"{owner} light_on needs a regime name after '!'")
    if lit is not None and light_on:
        raise ValueError(f'{owner} takes either lit or light_on, not both')


@attrs.define
class ObjectLightSettings(Parseable, Serializable):
    """Per-placement state of the lights an object carries in its annotation."""

    light_on: str = attrs.field(converter=lambda value: str(value).strip(), default='')
    lit: bool | None = None
    level: float = attrs.field(converter=float, default=1.0)

    def __attrs_post_init__(self) -> None:
        check_light_state('object light', 6500.0, self.level, 0.0, self.light_on, self.lit)

    def serialize(self) -> dict:
        return cattrs.gen.make_dict_unstructure_fn(type(self), converter, _cattrs_omit_if_default=True)(self)
