from __future__ import annotations

import re
import typing

import attrs

from arena_simulation_setup.utils.cattrs import Parseable, Serializable

_ENV_PREFIX = re.compile(r'^env_\d+/')
_ENTITY_ATOM = re.compile(r'^(\S+)\.(\S+) == (\S+)$')
_MEMBERSHIP_ATOM = re.compile(r'^(\S+) in (\S+)$')

_OPS: frozenset[str] = frozenset({'always', 'never', 'eventually', 'before', 'never_during'})
_BINARY_OPS: frozenset[str] = frozenset({'before', 'never_during'})


@attrs.frozen
class EntityAtom:
    """Entity field test, `<entity>.<field> == <value>`."""

    entity: str
    field: str
    value: str


@attrs.frozen
class MembershipAtom:
    """Zone membership test, `<subject> in <zone>`."""

    subject: str
    zone: str


def _reject_env_prefix(component: str, label: str) -> None:
    if _ENV_PREFIX.match(component):
        raise ValueError(f'atom {label} must be bare (env-stripped), got {component!r}')


def parse_atom(value: str) -> EntityAtom | MembershipAtom:
    """Total parser over the two surface forms, raises ValueError on any other shape."""
    text = value.strip()

    match = _ENTITY_ATOM.match(text)
    if match:
        entity, field, atom_value = match.groups()
        _reject_env_prefix(entity, 'entity')
        return EntityAtom(entity=entity, field=field, value=atom_value)

    match = _MEMBERSHIP_ATOM.match(text)
    if match:
        subject, zone = match.groups()
        _reject_env_prefix(subject, 'subject')
        _reject_env_prefix(zone, 'zone')
        return MembershipAtom(subject=subject, zone=zone)

    raise ValueError(f'atom does not match either surface form: {value!r}')


@attrs.define
class EpisodeCondition(Parseable, Serializable):
    """One clause: one operator over one or two atoms in surface form."""

    op: typing.Literal['always', 'never', 'eventually', 'before', 'never_during']
    p: str
    q: str | None = None
    text: str = ''

    def __attrs_post_init__(self) -> None:
        binary = self.op in _BINARY_OPS
        if binary and self.q is None:
            raise ValueError(f"clause op {self.op!r} requires a 'q' atom")
        if not binary and self.q is not None:
            raise ValueError(f"clause op {self.op!r} does not take a 'q' atom")
        parse_atom(self.p)
        if self.q is not None:
            parse_atom(self.q)

    @classmethod
    def parse(cls, value: dict) -> 'EpisodeCondition':
        value = dict(value)
        op = value.pop('op', None)
        if op not in _OPS:
            raise ValueError(f'unknown clause op: {op!r}')

        p = value.pop('p', None)
        if not isinstance(p, str):
            raise ValueError(f"clause 'p' must be a string atom: {p!r}")
        parse_atom(p)

        q = value.pop('q', None)
        if op in _BINARY_OPS:
            if not isinstance(q, str):
                raise ValueError(f"clause op {op!r} requires a 'q' atom")
            parse_atom(q)
        elif q is not None:
            raise ValueError(f"clause op {op!r} does not take a 'q' atom")

        text = str(value.pop('text', ''))

        if value:
            raise ValueError(f'unknown clause keys: {sorted(value)}')

        return cls(op=op, p=p, q=q, text=text)

    def serialize(self) -> dict:
        result: dict = {'op': self.op, 'p': self.p}
        if self.q is not None:
            result['q'] = self.q
        if self.text:
            result['text'] = self.text
        return result
