from __future__ import annotations

import re
import typing

import attrs

from arena_simulation_setup.utils.cattrs import Parseable, Serializable

_ENV_PREFIX = re.compile(r'^env_\d+/')
_ENTITY_ATOM = re.compile(r'^(\S+)\.(\S+) == (\S+)$')
_MEMBERSHIP_ATOM = re.compile(r'^(\S+) in (\S+)$')
_WITHIN_ATOM = re.compile(r'^(\S+) within (\S+) of (\S+)$')
_NOT_PREFIX = 'not '

_OPS: frozenset[str] = frozenset({'always', 'never', 'eventually', 'before', 'never_during'})
_BINARY_OPS: frozenset[str] = frozenset({'before', 'never_during'})
_ON_FAILURE: frozenset[str] = frozenset({'continue', 'stop_task', 'abort_episode'})

OnFailure = typing.Literal['continue', 'stop_task', 'abort_episode']


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


@attrs.frozen
class WithinAtom:
    """Proximity test, `<subject> within <radius> of <other>`, both subjects robots or peds."""

    subject: str
    radius: float
    other: str


@attrs.frozen
class NotAtom:
    """Negation, `not <atom>`."""

    atom: EntityAtom | MembershipAtom | WithinAtom


Atom = EntityAtom | MembershipAtom | WithinAtom | NotAtom


def _reject_env_prefix(component: str, label: str) -> None:
    if _ENV_PREFIX.match(component):
        raise ValueError(f'atom {label} must be bare (env-stripped), got {component!r}')


def parse_atom(value: str) -> Atom:
    """Total parser over the atom surface forms, raises ValueError on any other shape."""
    text = value.strip()

    if text.startswith(_NOT_PREFIX):
        inner = parse_atom(text[len(_NOT_PREFIX) :])
        if isinstance(inner, NotAtom):
            raise ValueError(f'atom negates a negation: {value!r}')
        return NotAtom(atom=inner)

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

    match = _WITHIN_ATOM.match(text)
    if match:
        subject, radius, other = match.groups()
        _reject_env_prefix(subject, 'subject')
        _reject_env_prefix(other, 'subject')
        try:
            radius_f = float(radius)
        except ValueError:
            raise ValueError(f'atom radius must be a number: {value!r}') from None
        if radius_f < 0:
            raise ValueError(f'atom radius must not be negative: {value!r}')
        return WithinAtom(subject=subject, radius=radius_f, other=other)

    raise ValueError(f'atom does not match any surface form: {value!r}')


def render_atom(atom: Atom) -> str:
    """Surface form of a parsed atom, the inverse of parse_atom."""
    if isinstance(atom, NotAtom):
        return f'not {render_atom(atom.atom)}'
    if isinstance(atom, EntityAtom):
        return f'{atom.entity}.{atom.field} == {atom.value}'
    if isinstance(atom, MembershipAtom):
        return f'{atom.subject} in {atom.zone}'
    return f'{atom.subject} within {atom.radius:g} of {atom.other}'


def atom_subjects(atom: Atom) -> frozenset[str]:
    """Robot or ped names the atom refers to, `robot` included."""
    if isinstance(atom, NotAtom):
        return atom_subjects(atom.atom)
    if isinstance(atom, EntityAtom):
        return frozenset()
    if isinstance(atom, MembershipAtom):
        return frozenset({atom.subject})
    return frozenset({atom.subject, atom.other})


@attrs.define
class EpisodeCondition(Parseable, Serializable):
    """One clause: one operator over one or two atoms in surface form."""

    op: typing.Literal['always', 'never', 'eventually', 'before', 'never_during']
    p: str
    q: str | None = None
    text: str = ''
    on_failure: OnFailure = 'continue'

    def __attrs_post_init__(self) -> None:
        binary = self.op in _BINARY_OPS
        if binary and self.q is None:
            raise ValueError(f"clause op {self.op!r} requires a 'q' atom")
        if not binary and self.q is not None:
            raise ValueError(f"clause op {self.op!r} does not take a 'q' atom")
        if self.on_failure not in _ON_FAILURE:
            raise ValueError(f'clause on_failure must be one of {sorted(_ON_FAILURE)}, got {self.on_failure!r}')
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
        on_failure = str(value.pop('on_failure', 'continue'))

        if value:
            raise ValueError(f'unknown clause keys: {sorted(value)}')

        return cls(op=op, p=p, q=q, text=text, on_failure=on_failure)

    def serialize(self) -> dict:
        result: dict = {'op': self.op, 'p': self.p}
        if self.q is not None:
            result['q'] = self.q
        if self.text:
            result['text'] = self.text
        if self.on_failure != 'continue':
            result['on_failure'] = self.on_failure
        return result


__all__ = [
    'Atom',
    'EntityAtom',
    'EpisodeCondition',
    'MembershipAtom',
    'NotAtom',
    'OnFailure',
    'WithinAtom',
    'atom_subjects',
    'parse_atom',
    'render_atom',
]
