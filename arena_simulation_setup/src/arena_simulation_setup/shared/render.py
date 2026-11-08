"""Natural-language rendering of task phases and conditions, one fixed template per form."""

from __future__ import annotations

import typing
from collections.abc import Mapping, Sequence

from arena_simulation_setup.shared.conditions import (
    ATOM_TYPES,
    AllNode,
    AnyNode,
    Atom,
    CompareValue,
    EntityAtom,
    EpisodeCondition,
    FamilyTest,
    FieldTest,
    HoldNode,
    MembershipAtom,
    Node,
    NotAtom,
    NotNode,
    PlaysAtom,
    SeesAtom,
    WithinAtom,
    needs_parentheses,
    parse_point,
)
from arena_simulation_setup.shared.task import GoToPhase, LoopPhase, PlayGesturePhase, ReachPhase, TaskPhase

Nouns = Mapping[str, str]


def noun(name: str, nouns: Nouns | None = None) -> str:
    """Display name of a zone, robot, ped or entity: its authored description, else the name with underscores as spaces."""
    if nouns and name in nouns and nouns[name]:
        return nouns[name]
    return name.replace("_", " ")


def render_atom_text(atom: Atom, nouns: Nouns | None = None, robot: str = "you") -> str:
    if isinstance(atom, NotAtom):
        inner = atom.atom
        if isinstance(inner, MembershipAtom):
            return f"{_subject(inner.subject, nouns, robot)} {_be(inner.subject)} not in {noun(inner.zone, nouns)}"
        if isinstance(inner, WithinAtom):
            return f"{_subject(inner.subject, nouns, robot)} {_be(inner.subject)} more than {inner.radius:g} m from {_subject(inner.other, nouns, robot)}"
        if isinstance(inner, SeesAtom):
            return f"{_subject(inner.subject, nouns, robot)} cannot see {_subject(inner.other, nouns, robot)} within {inner.range:g} m"
        if isinstance(inner, PlaysAtom):
            return f"{_subject(inner.subject, nouns, robot)} {_be(inner.subject)} not {_playing(inner.clip, nouns)}"
        return f"{noun(inner.entity, nouns)} {inner.field.replace('_', ' ')} is not {inner.value}"
    if isinstance(atom, MembershipAtom):
        return f"{_subject(atom.subject, nouns, robot)} {_be(atom.subject)} in {noun(atom.zone, nouns)}"
    if isinstance(atom, WithinAtom):
        return f"{_subject(atom.subject, nouns, robot)} {_be(atom.subject)} within {atom.radius:g} m of {_subject(atom.other, nouns, robot)}"
    if isinstance(atom, SeesAtom):
        return f"{_subject(atom.subject, nouns, robot)} can see {_subject(atom.other, nouns, robot)} within {atom.range:g} m"
    if isinstance(atom, PlaysAtom):
        return f"{_subject(atom.subject, nouns, robot)} {_be(atom.subject)} {_playing(atom.clip, nouns)}"
    return f"{noun(atom.entity, nouns)} {atom.field.replace('_', ' ')} is {atom.value}"


def render_until_text(node: Node, nouns: Nouns | None = None, robot: str = "you") -> str:
    """A condition node as a sentence part: `and`, `or` and `not` as words, a hold as a trailing `for <n> seconds`."""
    if isinstance(node, ATOM_TYPES):
        return render_atom_text(typing.cast("Atom", node), nouns, robot)
    if isinstance(node, FieldTest):
        return f"{noun(node.entity, nouns)} {_field_words(node.field)} {_compare_text(node.op, node.value)}"
    if isinstance(node, FamilyTest):
        quantifier = "some" if node.quantifier == "some" else "every"
        return f"the {_field_words(node.family)} of {quantifier} joint of {noun(node.entity, nouns)} {_compare_text(node.op, node.value)}"
    if isinstance(node, AllNode):
        return " and ".join(_child_text(node, child, nouns, robot) for child in node.nodes)
    if isinstance(node, AnyNode):
        return " or ".join(_child_text(node, child, nouns, robot) for child in node.nodes)
    if isinstance(node, NotNode):
        if isinstance(node.node, ATOM_TYPES) and not isinstance(node.node, NotAtom):
            return render_atom_text(NotAtom(atom=typing.cast("EntityAtom | MembershipAtom | WithinAtom | SeesAtom | PlaysAtom", node.node)), nouns, robot)
        return f"it is not the case that {_child_text(node, node.node, nouns, robot)}"
    assert isinstance(node, HoldNode)
    text = _child_text(node, node.node, nouns, robot)
    return f"{text} for {node.seconds:g} seconds" if node.seconds > 0 else text


def _child_text(parent: Node, child: Node, nouns: Nouns | None, robot: str) -> str:
    text = render_until_text(child, nouns, robot)
    return f"({text})" if needs_parentheses(parent, child) else text


def _field_words(field: str) -> str:
    return field.replace(".", " ").replace("_", " ")


def _compare_text(op: str, value: CompareValue) -> str:
    if isinstance(value, tuple):
        return f"is between {value[0]:g} and {value[1]:g}"
    shown = f"{value:g}" if isinstance(value, float) else value
    phrase = {"eq": "is", "ne": "is not", "lt": "is below", "le": "is at most", "gt": "is above", "ge": "is at least"}[op]
    return f"{phrase} {shown}"


def _playing(clip: str, nouns: Nouns | None) -> str:
    return f"making the {noun(clip, nouns)} gesture"


def _subject(subject: str, nouns: Nouns | None, robot: str) -> str:
    if subject == "robot":
        return robot
    point = parse_point(subject)
    if point is not None:
        return f"the point ({point[0]:g}, {point[1]:g})"
    return noun(subject, nouns)


def _be(subject: str) -> str:
    return "are" if subject == "robot" else "is"


def render_condition(condition: EpisodeCondition, nouns: Nouns | None = None) -> str:
    """One sentence per operator, the authored `text` winning when set."""
    if condition.text:
        return condition.text
    p = condition.p_node
    op = condition.op
    if op == "always":
        return f"Make sure that {render_until_text(p, nouns)} the whole time."
    if op == "never":
        return f"Never let it happen that {render_until_text(p, nouns)}."
    if op == "eventually":
        return f"At some point, {render_until_text(p, nouns)}."
    q = condition.q_node
    assert q is not None
    if op == "before":
        return f"Make sure that {render_until_text(p, nouns)} before {render_until_text(q, nouns)}."
    return f"Never let it happen that {render_until_text(p, nouns)} while {render_until_text(q, nouns)}."


def render_phase(phase: TaskPhase, nouns: Nouns | None = None) -> str:
    """One sentence per phase kind, the authored `text` winning when set."""
    if phase.text:
        return phase.text
    if isinstance(phase, LoopPhase):
        return _render_loop(phase, nouns)
    if isinstance(phase, GoToPhase):
        sentence = _render_goto(phase, nouns)
    elif isinstance(phase, PlayGesturePhase):
        sentence = f"Perform the {noun(phase.gesture, nouns)} gesture" if phase.gesture else "Perform a gesture"
    elif isinstance(phase, ReachPhase):
        if phase.random:
            sentence = "Reach to a point in your workspace"
        elif phase.named_target is not None:
            sentence = f"Move your arm to the {noun(phase.named_target, nouns)} position"
        else:
            sentence = "Move your hand in a straight line to the given pose" if phase.motion == "linear" else "Reach to the given pose"
        stop = phase.stop_node
        if stop is not None:
            sentence += f" and stop once {render_until_text(stop, nouns)}"
    else:
        sentence = "Carry out the step"
    until = phase.until_node
    if until is not None:
        sentence += f" and wait until {render_until_text(until, nouns)}"
    for condition in phase.conditions:
        sentence += ", " + _condition_clause(condition, nouns)
    return sentence + "."


def _render_loop(phase: LoopPhase, nouns: Nouns | None) -> str:
    steps = render_phases(phase.phases, nouns)
    until = phase.until_node
    if until is None:
        return f"Keep repeating these steps: {steps}"
    return f"Repeat these steps until {render_until_text(until, nouns)}: {steps}"


def _render_goto(phase: GoToPhase, nouns: Nouns | None) -> str:
    if phase.target is not None:
        sentence = f"Go to {noun(phase.target, nouns)}"
    elif phase.pose is not None and not phase.hold:
        sentence = f"Go to ({phase.pose.position.x:.1f}, {phase.pose.position.y:.1f})"
    else:
        sentence = "Stay where you are"
    if phase.hold_time:
        sentence += f" and stay there for {phase.hold_time:g} seconds"
    return sentence


def _condition_clause(condition: EpisodeCondition, nouns: Nouns | None) -> str:
    if condition.text:
        return condition.text
    p = condition.p_node
    op = condition.op
    if op == "never":
        return f"never letting it happen that {render_until_text(p, nouns)}"
    if op == "always":
        return f"making sure that {render_until_text(p, nouns)} the whole time"
    if op == "eventually":
        return f"making sure that at some point {render_until_text(p, nouns)}"
    q = condition.q_node
    assert q is not None
    if op == "before":
        return f"making sure that {render_until_text(p, nouns)} before {render_until_text(q, nouns)}"
    return f"never letting it happen that {render_until_text(p, nouns)} while {render_until_text(q, nouns)}"


def render_phases(phases: Sequence[TaskPhase], nouns: Nouns | None = None, conditions: Sequence[EpisodeCondition] = ()) -> str:
    """The whole instruction: phases in order, then the request conditions."""
    sentences = []
    for i, phase in enumerate(phases):
        sentence = render_phase(phase, nouns)
        if i > 0 and not phase.text:
            sentence = "Then " + sentence[0].lower() + sentence[1:]
        sentences.append(sentence)
    sentences.extend(render_condition(c, nouns) for c in conditions)
    return " ".join(sentences)


__all__ = ["Nouns", "noun", "render_atom_text", "render_condition", "render_phase", "render_phases", "render_until_text"]
