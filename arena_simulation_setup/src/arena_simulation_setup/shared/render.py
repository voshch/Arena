"""Natural-language rendering of task phases and conditions, one fixed template per form."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from arena_simulation_setup.shared.conditions import (
    Atom,
    EpisodeCondition,
    MembershipAtom,
    NotAtom,
    WithinAtom,
    parse_atom,
)
from arena_simulation_setup.shared.task import GoToPhase, PlayGesturePhase, ReachPhase, TaskPhase

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
        return f"{noun(inner.entity, nouns)} {inner.field.replace('_', ' ')} is not {inner.value}"
    if isinstance(atom, MembershipAtom):
        return f"{_subject(atom.subject, nouns, robot)} {_be(atom.subject)} in {noun(atom.zone, nouns)}"
    if isinstance(atom, WithinAtom):
        return f"{_subject(atom.subject, nouns, robot)} {_be(atom.subject)} within {atom.radius:g} m of {_subject(atom.other, nouns, robot)}"
    return f"{noun(atom.entity, nouns)} {atom.field.replace('_', ' ')} is {atom.value}"


def _subject(subject: str, nouns: Nouns | None, robot: str) -> str:
    return robot if subject == "robot" else noun(subject, nouns)


def _be(subject: str) -> str:
    return "are" if subject == "robot" else "is"


def render_condition(condition: EpisodeCondition, nouns: Nouns | None = None) -> str:
    """One sentence per operator, the authored `text` winning when set."""
    if condition.text:
        return condition.text
    p = parse_atom(condition.p)
    op = condition.op
    if op == "always":
        return f"Make sure that {render_atom_text(p, nouns)} the whole time."
    if op == "never":
        return f"Never let it happen that {render_atom_text(p, nouns)}."
    if op == "eventually":
        return f"At some point, {render_atom_text(p, nouns)}."
    q = parse_atom(condition.q or "")
    if op == "before":
        return f"Make sure that {render_atom_text(p, nouns)} before {render_atom_text(q, nouns)}."
    return f"Never let it happen that {render_atom_text(p, nouns)} while {render_atom_text(q, nouns)}."


def render_phase(phase: TaskPhase, nouns: Nouns | None = None) -> str:
    """One sentence per phase kind, the authored `text` winning when set."""
    if phase.text:
        return phase.text
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
            sentence = "Reach to the given pose"
    else:
        sentence = "Carry out the step"
    if phase.until is not None:
        sentence += f" and wait until {render_atom_text(parse_atom(phase.until), nouns)}"
    for condition in phase.conditions:
        sentence += ", " + _condition_clause(condition, nouns)
    return sentence + "."


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
    p = parse_atom(condition.p)
    op = condition.op
    if op == "never":
        return f"never letting it happen that {render_atom_text(p, nouns)}"
    if op == "always":
        return f"making sure that {render_atom_text(p, nouns)} the whole time"
    if op == "eventually":
        return f"making sure that at some point {render_atom_text(p, nouns)}"
    q = parse_atom(condition.q or "")
    if op == "before":
        return f"making sure that {render_atom_text(p, nouns)} before {render_atom_text(q, nouns)}"
    return f"never letting it happen that {render_atom_text(p, nouns)} while {render_atom_text(q, nouns)}"


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


__all__ = ["Nouns", "noun", "render_atom_text", "render_condition", "render_phase", "render_phases"]
