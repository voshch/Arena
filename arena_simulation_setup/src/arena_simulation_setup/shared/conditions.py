from __future__ import annotations

import copy
import re
import typing
from collections.abc import Callable, Mapping

import attrs

from arena_simulation_setup.utils.cattrs import Parseable, Serializable

_ENV_PREFIX = re.compile(r'^env_\d+/')
_ENTITY_ATOM = re.compile(r'^([^\s.]+)\.(\S+) == (\S+)$')
_MEMBERSHIP_ATOM = re.compile(r'^(\S+) in (\S+)$')
_WITHIN_ATOM = re.compile(r'^(\S+) within (\S+) of (\S+)$')
_SEES_ATOM = re.compile(r'^(\S+) sees (\S+?)(?: within (\S+))?$')
_PLAYS_ATOM = re.compile(r'^(\S+) plays (\S+)$')
_HOLDS_ATOM = re.compile(r'^(\S+) holds (\S+)$')
_NUMBER = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'
_POINT = re.compile(rf'^({_NUMBER}),({_NUMBER})$')
_FOR_SUFFIX = re.compile(r'^(.*\S) for (\S+)$')
_NAME = re.compile(r'^\S+$')
_AND = ' and '
_OR = ' or '
_NOT_PREFIX = 'not '

SEES_RANGE_M = 12.0

_OPS: frozenset[str] = frozenset({'always', 'never', 'eventually', 'before', 'never_during'})
_BINARY_OPS: frozenset[str] = frozenset({'before', 'never_during'})
_ON_FAILURE: frozenset[str] = frozenset({'continue', 'stop_task', 'abort_episode'})

OnFailure = typing.Literal['continue', 'stop_task', 'abort_episode']
CompareOp = typing.Literal['eq', 'ne', 'lt', 'le', 'gt', 'ge', 'within']
Quantifier = typing.Literal['some', 'each']
CompareValue = str | float | tuple[float, float]
ConditionSource = str | dict[str, typing.Any]

COMPARE_OPS: tuple[str, ...] = ('eq', 'ne', 'lt', 'le', 'gt', 'ge', 'within')
_ORDER_OPS: frozenset[str] = frozenset({'lt', 'le', 'gt', 'ge'})
_COMPARE_SYMBOLS: dict[str, str] = {'eq': '==', 'ne': '!=', 'lt': '<', 'le': '<=', 'gt': '>', 'ge': '>='}
_SUBJECT_FORMS: tuple[str, ...] = ('in', 'near', 'sees', 'plays', 'holds')
_HEADS: tuple[str, ...] = ('field', 'some', 'each', 'all', 'any', 'not', 'subject')
_HEAD_KEYS: dict[str, frozenset[str]] = {
    'field': frozenset({'field', *COMPARE_OPS}),
    'some': frozenset({'some', *COMPARE_OPS}),
    'each': frozenset({'each', *COMPARE_OPS}),
    'all': frozenset({'all'}),
    'any': frozenset({'any'}),
    'not': frozenset({'not'}),
    'subject': frozenset({'subject', 'in', 'near', 'radius', 'sees', 'range', 'plays', 'holds'}),
}
_KNOWN_KEYS: frozenset[str] = frozenset().union(*_HEAD_KEYS.values(), {'for'})


@attrs.frozen
class EntityAtom:
    """Entity field test, `<entity>.<field> == <value>`, split at the first dot."""

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
    """Proximity test, `<subject> within <radius> of <other>`, each subject a robot, a ped or an `x,y` point."""

    subject: str
    radius: float
    other: str


@attrs.frozen
class SeesAtom:
    """Perception test, `<robot> sees <ped> [within <range>]`: within range and no wall cell on the segment between them."""

    subject: str
    other: str
    range: float = SEES_RANGE_M


@attrs.frozen
class PlaysAtom:
    """Gesture test, `<ped> plays <clip>`, the clip is among the gestures the pedestrian publishes."""

    subject: str
    clip: str


@attrs.frozen
class HoldsAtom:
    """Grasp test, `<robot> holds <object>`, the robot's gripper holds the entity."""

    subject: str
    object: str


@attrs.frozen
class NotAtom:
    """Negation, `not <atom>`."""

    atom: EntityAtom | MembershipAtom | WithinAtom | SeesAtom | PlaysAtom | HoldsAtom


Atom = EntityAtom | MembershipAtom | WithinAtom | SeesAtom | PlaysAtom | HoldsAtom | NotAtom
ATOM_TYPES: tuple[type, ...] = (EntityAtom, MembershipAtom, WithinAtom, SeesAtom, PlaysAtom, HoldsAtom, NotAtom)


def _check_compare(op: str, value: object) -> None:
    if op not in COMPARE_OPS:
        raise ValueError(f'unknown comparison {op!r}, expected one of {list(COMPARE_OPS)}')
    if op == 'within':
        if not (isinstance(value, tuple) and len(value) == 2 and all(isinstance(bound, float) for bound in value)):
            raise ValueError(f"'within' takes a [lo, hi] pair of numbers, got {value!r}")
        if value[0] > value[1]:
            raise ValueError(f"'within' needs lo <= hi, got {list(value)!r}")
        return
    if op in _ORDER_OPS:
        if not isinstance(value, float):
            raise ValueError(f'{op!r} takes a number, got {value!r}')
        return
    if not isinstance(value, (str, float)):
        raise ValueError(f'{op!r} takes a string or a number, got {value!r}')


@attrs.frozen
class FieldTest:
    """Comparison of one entity field, `{field: <entity>.<field>, <op>: <value>}`."""

    entity: str
    field: str
    op: CompareOp
    value: CompareValue

    def __attrs_post_init__(self) -> None:
        _check_compare(self.op, self.value)


@attrs.frozen
class FamilyTest:
    """Comparison quantified over the `<family>.<member>` fields of one entity, `{some|each: <entity>.<family>, <op>: <value>}`."""

    entity: str
    family: str
    quantifier: Quantifier
    op: CompareOp
    value: CompareValue

    def __attrs_post_init__(self) -> None:
        if self.quantifier not in ('some', 'each'):
            raise ValueError(f"family quantifier must be 'some' or 'each', got {self.quantifier!r}")
        _check_compare(self.op, self.value)


@attrs.frozen
class AllNode:
    """Conjunction of its nodes."""

    nodes: tuple[Node, ...]


@attrs.frozen
class AnyNode:
    """Disjunction of its nodes."""

    nodes: tuple[Node, ...]


@attrs.frozen
class NotNode:
    """Negation of a node."""

    node: Node


@attrs.frozen
class HoldNode:
    """A node that has held without a gap for `seconds` of sample time."""

    node: Node
    seconds: float


Node = Atom | FieldTest | FamilyTest | AllNode | AnyNode | NotNode | HoldNode


def _reject_env_prefix(component: str, label: str) -> None:
    if _ENV_PREFIX.match(component):
        raise ValueError(f'atom {label} must be bare (env-stripped), got {component!r}')


def _reject_point(component: str, value: object) -> None:
    if parse_point(component) is not None:
        raise ValueError(f'atom takes an x,y point only as a within subject: {value!r}')


def parse_point(subject: str) -> tuple[float, float] | None:
    """The point an `x,y` subject literal names, None for a robot or ped name."""
    match = _POINT.match(subject)
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


def _membership_atom(subject: str, zone: str, value: object) -> MembershipAtom:
    _reject_env_prefix(subject, 'subject')
    _reject_env_prefix(zone, 'zone')
    _reject_point(subject, value)
    return MembershipAtom(subject=subject, zone=zone)


def _within_atom(subject: str, radius: object, other: str, value: object) -> WithinAtom:
    _reject_env_prefix(subject, 'subject')
    _reject_env_prefix(other, 'subject')
    return WithinAtom(subject=subject, radius=_non_negative(radius, 'radius', value), other=other)


def _sees_atom(subject: str, other: str, range_value: object, value: object) -> SeesAtom:
    _reject_env_prefix(subject, 'subject')
    _reject_env_prefix(other, 'subject')
    _reject_point(subject, value)
    _reject_point(other, value)
    return SeesAtom(subject=subject, other=other, range=_non_negative(range_value, 'range', value) if range_value is not None else SEES_RANGE_M)


def _plays_atom(subject: str, clip: str, value: object) -> PlaysAtom:
    _reject_env_prefix(subject, 'subject')
    _reject_point(subject, value)
    return PlaysAtom(subject=subject, clip=clip)


def _holds_atom(subject: str, obj: str, value: object) -> HoldsAtom:
    _reject_env_prefix(subject, 'subject')
    _reject_env_prefix(obj, 'object')
    _reject_point(subject, value)
    _reject_point(obj, value)
    return HoldsAtom(subject=subject, object=obj)


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
        return _membership_atom(subject, zone, value)

    match = _WITHIN_ATOM.match(text)
    if match:
        subject, radius, other = match.groups()
        return _within_atom(subject, radius, other, value)

    match = _SEES_ATOM.match(text)
    if match:
        subject, other, range_text = match.groups()
        return _sees_atom(subject, other, range_text, value)

    match = _PLAYS_ATOM.match(text)
    if match:
        subject, clip = match.groups()
        return _plays_atom(subject, clip, value)

    match = _HOLDS_ATOM.match(text)
    if match:
        subject, obj = match.groups()
        return _holds_atom(subject, obj, value)

    raise ValueError(f'atom does not match any surface form: {value!r}')


def _non_negative(raw: object, label: str, value: object) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
        raise ValueError(f'atom {label} must be a number: {value!r}')
    try:
        number = float(raw)
    except ValueError:
        raise ValueError(f'atom {label} must be a number: {value!r}') from None
    if number < 0:
        raise ValueError(f'atom {label} must not be negative: {value!r}')
    return number


def parse_until(value: str) -> Node:
    """Node of the text form: clauses joined by ` or `, each atoms joined by ` and ` with an optional trailing ` for <seconds>` hold."""
    clauses = tuple(_parse_clause(part, value) for part in value.strip().split(_OR))
    return clauses[0] if len(clauses) == 1 else AnyNode(nodes=clauses)


def _parse_clause(text: str, value: str) -> Node:
    text = text.strip()
    if not text:
        raise ValueError(f'until has an empty clause: {value!r}')
    dwell = 0.0
    match = _FOR_SUFFIX.match(text)
    if match:
        text, dwell_text = match.groups()
        dwell = _non_negative(dwell_text, 'dwell', value)
    atoms = tuple(parse_atom(part) for part in text.split(_AND))
    node: Node = atoms[0] if len(atoms) == 1 else AllNode(nodes=atoms)
    return HoldNode(node=node, seconds=dwell) if dwell > 0 else node


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _compare_value(raw: object) -> object:
    if isinstance(raw, bool):
        return 'true' if raw else 'false'
    if _is_number(raw):
        return float(typing.cast('float', raw))
    if isinstance(raw, (list, tuple)) and all(_is_number(bound) for bound in raw):
        return tuple(float(bound) for bound in raw)
    return raw


def _dotted(raw: object, key: str) -> tuple[str, str]:
    if not isinstance(raw, str) or not _NAME.match(raw):
        raise ValueError(f'{key!r} must be an <entity>.<name> string, got {raw!r}')
    entity, dot, name = raw.partition('.')
    if not dot or not entity or not name:
        raise ValueError(f'{key!r} must be an <entity>.<name> string, got {raw!r}')
    _reject_env_prefix(entity, 'entity')
    return entity, name


def _name(raw: object, key: str) -> str:
    if not isinstance(raw, str) or not _NAME.match(raw):
        raise ValueError(f'{key!r} must be a name without spaces, got {raw!r}')
    return raw


def _comparison(data: dict[str, object], head: str) -> tuple[CompareOp, CompareValue]:
    ops = [op for op in COMPARE_OPS if op in data]
    if len(ops) != 1:
        raise ValueError(f'{head!r} condition takes exactly one operator of {list(COMPARE_OPS)}, got {ops}')
    op = ops[0]
    value = _compare_value(data[op])
    _check_compare(op, value)
    return typing.cast('CompareOp', op), typing.cast('CompareValue', value)


def _children(data: dict[str, object], head: str) -> tuple[Node, ...]:
    items = data[head]
    if not isinstance(items, (list, tuple)) or not items:
        raise ValueError(f'{head!r} takes a non-empty list of conditions, got {items!r}')
    return tuple(parse_node(item) for item in items)


def _subject_node(data: dict[str, object], value: object) -> Node:
    forms = [form for form in _SUBJECT_FORMS if form in data]
    if len(forms) != 1:
        raise ValueError(f"'subject' condition takes exactly one of {list(_SUBJECT_FORMS)}, got {forms}")
    form = forms[0]
    stray = [key for key in ('radius', 'range') if key in data and not (key == 'radius' and form == 'near') and not (key == 'range' and form == 'sees')]
    if stray:
        raise ValueError(f"'subject' condition with {form!r} takes no keys {stray}")
    subject = _name(data['subject'], 'subject')
    if form == 'in':
        return _membership_atom(subject, _name(data['in'], 'in'), value)
    if form == 'near':
        if 'radius' not in data:
            raise ValueError(f"'near' needs a 'radius': {value!r}")
        return _within_atom(subject, _non_negative_number(data['radius'], 'radius'), _name(data['near'], 'near'), value)
    if form == 'sees':
        range_value = data.get('range')
        return _sees_atom(subject, _name(data['sees'], 'sees'), None if range_value is None else _non_negative_number(range_value, 'range'), value)
    if form == 'holds':
        return _holds_atom(subject, _name(data['holds'], 'holds'), value)
    return _plays_atom(subject, _name(data['plays'], 'plays'), value)


def _non_negative_number(raw: object, key: str) -> float:
    if not _is_number(raw) or float(typing.cast('float', raw)) < 0:
        raise ValueError(f'{key!r} must be a non-negative number, got {raw!r}')
    return float(typing.cast('float', raw))


def parse_node(value: object) -> Node:
    """Condition node of a text form (see parse_until) or a structured mapping, raises ValueError naming the offending key or value."""
    if isinstance(value, str):
        return parse_until(value)
    if not isinstance(value, Mapping):
        raise ValueError(f'a condition is a string or a mapping, got {value!r}')
    data = {str(key): item for key, item in value.items()}
    unknown = sorted(set(data) - _KNOWN_KEYS)
    if unknown:
        raise ValueError(f'condition has unknown keys {unknown}, expected some of {sorted(_KNOWN_KEYS)}')
    seconds: float | None = None
    if 'for' in data:
        seconds = _non_negative_number(data.pop('for'), 'for')
    heads = [head for head in _HEADS if head in data]
    if len(heads) != 1:
        raise ValueError(f'condition takes exactly one of {list(_HEADS)}, got {heads}')
    head = heads[0]
    stray = sorted(set(data) - _HEAD_KEYS[head])
    if stray:
        raise ValueError(f'{head!r} condition takes no keys {stray}')
    node: Node
    if head == 'field':
        entity, field = _dotted(data['field'], 'field')
        op, compare = _comparison(data, head)
        node = FieldTest(entity=entity, field=field, op=op, value=compare)
    elif head in ('some', 'each'):
        entity, family = _dotted(data[head], head)
        if '.' in family:
            raise ValueError(f'{head!r} names a family without dots, got {data[head]!r}')
        op, compare = _comparison(data, head)
        node = FamilyTest(entity=entity, family=family, quantifier=typing.cast('Quantifier', head), op=op, value=compare)
    elif head == 'all':
        node = AllNode(nodes=_children(data, head))
    elif head == 'any':
        node = AnyNode(nodes=_children(data, head))
    elif head == 'not':
        node = NotNode(node=parse_node(data['not']))
    else:
        node = _subject_node(data, value)
    return node if seconds is None else HoldNode(node=node, seconds=seconds)


def render_atom(atom: Atom) -> str:
    """Surface form of a parsed atom, the inverse of parse_atom."""
    if isinstance(atom, NotAtom):
        return f'not {render_atom(atom.atom)}'
    if isinstance(atom, EntityAtom):
        return f'{atom.entity}.{atom.field} == {atom.value}'
    if isinstance(atom, MembershipAtom):
        return f'{atom.subject} in {atom.zone}'
    if isinstance(atom, SeesAtom):
        return f'{atom.subject} sees {atom.other} within {atom.range:g}'
    if isinstance(atom, PlaysAtom):
        return f'{atom.subject} plays {atom.clip}'
    if isinstance(atom, HoldsAtom):
        return f'{atom.subject} holds {atom.object}'
    return f'{atom.subject} within {atom.radius:g} of {atom.other}'


def render_until(node: Node) -> str:
    """Text form of a node the until grammar expresses, the inverse of parse_until, else render_node."""
    clauses = node.nodes if isinstance(node, AnyNode) else (node,)
    parts = []
    for clause in clauses:
        text = _render_clause(clause)
        if text is None:
            return render_node(node)
        parts.append(text)
    return _OR.join(parts)


def _render_clause(node: Node) -> str | None:
    dwell = 0.0
    if isinstance(node, HoldNode):
        dwell, node = node.seconds, node.node
    atoms = node.nodes if isinstance(node, AllNode) else (node,)
    if not all(isinstance(atom, ATOM_TYPES) for atom in atoms):
        return None
    text = _AND.join(render_atom(typing.cast('Atom', atom)) for atom in atoms)
    return f'{text} for {dwell:g}' if dwell > 0 else text


def _render_value(value: CompareValue) -> str:
    if isinstance(value, tuple):
        return f'[{value[0]:g}, {value[1]:g}]'
    if isinstance(value, str):
        return value
    return f'{value:g}'


def _render_compare(op: str, value: CompareValue) -> str:
    if op == 'within':
        return f'within {_render_value(value)}'
    return f'{_COMPARE_SYMBOLS[op]} {_render_value(value)}'


def needs_parentheses(parent: Node, child: Node) -> bool:
    """Whether a child renders in parentheses under its parent, `and` binding tighter than `or` and a hold taking its whole clause."""
    if isinstance(parent, NotNode):
        return isinstance(child, (AllNode, AnyNode, HoldNode, NotNode, NotAtom))
    if isinstance(parent, AllNode):
        return isinstance(child, (AnyNode, HoldNode))
    if isinstance(parent, AnyNode):
        return isinstance(child, AnyNode)
    return isinstance(child, (AnyNode, HoldNode))


def render_node(node: Node) -> str:
    """Readable one-line form of any node."""
    if isinstance(node, FieldTest):
        return f'{node.entity} {node.field} {_render_compare(node.op, node.value)}'
    if isinstance(node, FamilyTest):
        return f'{node.quantifier} {node.entity} {node.family} {_render_compare(node.op, node.value)}'
    if isinstance(node, AllNode):
        return _AND.join(_render_child(node, child) for child in node.nodes)
    if isinstance(node, AnyNode):
        return _OR.join(_render_child(node, child) for child in node.nodes)
    if isinstance(node, NotNode):
        return f'not {_render_child(node, node.node)}'
    if isinstance(node, HoldNode):
        return f'{_render_child(node, node.node)} for {node.seconds:g} s'
    return render_atom(node)


def _render_child(parent: Node, child: Node) -> str:
    text = render_node(child)
    return f'({text})' if needs_parentheses(parent, child) else text


def node_children(node: Node) -> tuple[Node, ...]:
    """Direct child nodes, empty for a leaf."""
    if isinstance(node, (AllNode, AnyNode)):
        return node.nodes
    if isinstance(node, (NotNode, HoldNode)):
        return (node.node,)
    return ()


def atom_subjects(atom: Atom) -> frozenset[str]:
    """Robot or ped names the atom refers to, `robot` included, `x,y` points left out."""
    if isinstance(atom, NotAtom):
        return atom_subjects(atom.atom)
    if isinstance(atom, EntityAtom):
        return frozenset()
    if isinstance(atom, (MembershipAtom, PlaysAtom, HoldsAtom)):
        return frozenset({atom.subject})
    return frozenset(subject for subject in (atom.subject, atom.other) if parse_point(subject) is None)


def node_subjects(node: Node) -> frozenset[str]:
    """Robot or ped names a node refers to, field and family tests contributing none."""
    if isinstance(node, ATOM_TYPES):
        return atom_subjects(typing.cast('Atom', node))
    return frozenset().union(*(node_subjects(child) for child in node_children(node)))


def node_fields(node: Node) -> set[tuple[str, str]]:
    """(entity, field) of every field test and entity atom in a node."""
    if isinstance(node, FieldTest):
        return {(node.entity, node.field)}
    if isinstance(node, NotAtom):
        return node_fields(node.atom)
    if isinstance(node, EntityAtom):
        return {(node.entity, node.field)}
    return set().union(*(node_fields(child) for child in node_children(node)))


def node_families(node: Node) -> set[tuple[str, str]]:
    """(entity, family) of every family test in a node."""
    if isinstance(node, FamilyTest):
        return {(node.entity, node.family)}
    return set().union(*(node_families(child) for child in node_children(node)))


def check_fields(node: Node, fields_of: Callable[[str], frozenset[str] | None]) -> list[str]:
    """Problems with the fields and families a node reads, skipping entities whose field set is None."""
    problems: list[str] = []
    for entity, field in sorted(node_fields(node)):
        names = fields_of(entity)
        if names is not None and field not in names:
            problems.append(f'{entity} has no field {field!r}, its fields: {", ".join(sorted(names))}')
    for entity, family in sorted(node_families(node)):
        names = fields_of(entity)
        if names is not None and not any(name.startswith(f'{family}.') for name in names):
            problems.append(f'{entity} has no {family!r} fields, its fields: {", ".join(sorted(names))}')
    return problems


def _source(raw: object, label: str) -> ConditionSource:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, Mapping):
        return copy.deepcopy(dict(raw))
    raise ValueError(f'{label} must be a string or a mapping, got {raw!r}')


@attrs.define
class EpisodeCondition(Parseable, Serializable):
    """One clause: one operator over one or two conditions, each a text form or a structured mapping."""

    op: typing.Literal['always', 'never', 'eventually', 'before', 'never_during']
    p: ConditionSource
    q: ConditionSource | None = None
    text: str = ''
    on_failure: OnFailure = 'continue'
    _p_node: Node = attrs.field(init=False, eq=False, repr=False)
    _q_node: Node | None = attrs.field(init=False, eq=False, repr=False)

    def __attrs_post_init__(self) -> None:
        binary = self.op in _BINARY_OPS
        if binary and self.q is None:
            raise ValueError(f"clause op {self.op!r} requires a 'q' atom")
        if not binary and self.q is not None:
            raise ValueError(f"clause op {self.op!r} does not take a 'q' atom")
        if self.on_failure not in _ON_FAILURE:
            raise ValueError(f'clause on_failure must be one of {sorted(_ON_FAILURE)}, got {self.on_failure!r}')
        self._p_node = parse_node(self.p)
        self._q_node = parse_node(self.q) if self.q is not None else None

    @property
    def p_node(self) -> Node:
        return self._p_node

    @property
    def q_node(self) -> Node | None:
        return self._q_node

    @classmethod
    def parse(cls, value: dict) -> EpisodeCondition:
        value = dict(value)
        op = value.pop('op', None)
        if op not in _OPS:
            raise ValueError(f'unknown clause op: {op!r}')

        p = _source(value.pop('p', None), "clause 'p'")

        q = value.pop('q', None)
        if op in _BINARY_OPS:
            if q is None:
                raise ValueError(f"clause op {op!r} requires a 'q' atom")
            q = _source(q, "clause 'q'")
        elif q is not None:
            raise ValueError(f"clause op {op!r} does not take a 'q' atom")

        text = str(value.pop('text', ''))
        on_failure = typing.cast('OnFailure', str(value.pop('on_failure', 'continue')))

        if value:
            raise ValueError(f'unknown clause keys: {sorted(value)}')

        return cls(op=op, p=p, q=q, text=text, on_failure=on_failure)

    def serialize(self) -> dict:
        result: dict = {'op': self.op, 'p': copy.deepcopy(self.p)}
        if self.q is not None:
            result['q'] = copy.deepcopy(self.q)
        if self.text:
            result['text'] = self.text
        if self.on_failure != 'continue':
            result['on_failure'] = self.on_failure
        return result


__all__ = [
    'ATOM_TYPES',
    'COMPARE_OPS',
    'SEES_RANGE_M',
    'AllNode',
    'AnyNode',
    'Atom',
    'CompareOp',
    'CompareValue',
    'ConditionSource',
    'EntityAtom',
    'EpisodeCondition',
    'FamilyTest',
    'FieldTest',
    'HoldNode',
    'HoldsAtom',
    'MembershipAtom',
    'Node',
    'NotAtom',
    'NotNode',
    'OnFailure',
    'PlaysAtom',
    'Quantifier',
    'SeesAtom',
    'WithinAtom',
    'atom_subjects',
    'check_fields',
    'needs_parentheses',
    'node_children',
    'node_families',
    'node_fields',
    'node_subjects',
    'parse_atom',
    'parse_node',
    'parse_point',
    'parse_until',
    'render_atom',
    'render_node',
    'render_until',
]
