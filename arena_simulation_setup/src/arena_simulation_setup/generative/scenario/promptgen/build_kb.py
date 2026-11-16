"""Render the committed `system_{a,bc,d}.md` from the `.tmpl.md` notes, the parser constants and `kb_ranges.yaml`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import attrs
import yaml

from arena_simulation_setup.generative.core.profiles import ALIASES, PROFILES, TIERS, tier_value

from ..populate import MAX_VELOCITY_DEFAULT
from ..route import ROBOT_SPEED_MPS
from . import prompts as P
from . import step_d as D
from .lexicon import load as lex_load
from .spec import C_AFTER, CONFLICT_M, ERRATIC_AROUND, FORMATION_TRIGGER_M, PHRASES, BDecision, CDecision, DDecision

_DIR = Path(__file__).parent

#: Semantics that live nowhere in code constants: one line per case / variant / mode.
B_CASE_NOTES = {
    "erratic_route": "their routes change every few seconds",
    "erratic_behavior": "they keep switching what they are doing; falls back to erratic_route on plain walkers",
    "intrinsic": "one to three parameters pinned at a tier",
}
C_AFTER_NOTES = {
    "continue": "keeps going",
    "stop_facing": "halts in front of the robot for `duration` seconds",
    "follow": "tracks the robot for `duration` seconds",
    "veer": "turns off to the `veer_to` zone",
}
D_VARIANT_NOTES = {
    "stubborn": "one pedestrian walks against the robot's direction along its route, wants to keep walking, and does not yield to the robot or to anyone. Fields: `profile`, `speed` (tier), `route_fraction` (where it meets the robot, 0..1)",
    "meet": "members walk from their rooms to a point and start talking when they arrive",
    "standing": "members are already talking there",
    "blocking": "members stand in a line across the passage the robot's route runs through, shoulder to shoulder, and do not break",
    "circle": "a circle of talkers",
    "line": "a queue",
}
ANGLES = ((180, "head-on"), (90, "crossing from the right"), (270, "crossing from the left"), (0, "overtaking from behind"))
A_CASE_NOTES = {
    "blackout": "the lights or power go out; everyone slows down and feels their way",
    "evacuation": "an alarm or evacuation; everybody heads for the exits",
    "stampede": "a surge - shift change, break over, crowd pushes back to work all at once, nobody yields",
    "busy": "more people about than usual; `k` is the crowd multiplier",
}


def _default(cls: type, field: str) -> object:
    return next(a.default for a in attrs.fields(cls) if a.name == field)


def _ranges() -> dict[str, Any]:
    """The observed answer ranges committed in `kb_ranges.yaml`."""
    return yaml.safe_load((_DIR / "kb_ranges.yaml").read_text())


def _span(vals: list[float | str], unit: str = "") -> str:
    if not vals:
        return "none yet"
    if len(vals) <= 6:
        return "{" + ", ".join(f"{v:g}" if isinstance(v, (int, float)) else str(v) for v in vals) + "}" + unit
    return f"{min(vals):g}..{max(vals):g}{unit}"


def _bc_observed(r: dict[str, Any]) -> str:
    b, c = r["b"], r["c"]
    return "\n".join(
        (
            f"## Sample ranges ({r['scenarios']} previous runs)",
            "",
            f"- B `when.at` {_span(b['when_at'], ' s')}; selects used: {', '.join(b['selects'])}; params seen: {', '.join(b['params'])}.",
            f"- C `route_fraction` {_span(c['route_fraction'])} (anchors 0.3 / 0.5 / 0.7; off-anchor values come from the {CONFLICT_M:g} m conflict shift), `duration` {_span(c['duration'], ' s')}, `angle` {_span(c['angle'])}, `offset` {_span(c['offset'], ' s')}, `speed` {_span(c['speed'])}.",
            "- These are what previous answers used, not limits - the contract above is the limit.",
        )
    )


def _d_observed(r: dict[str, Any]) -> str:
    d = r["d"]
    return "\n".join(
        (
            f"## Sample ranges ({r['scenarios']} previous runs)",
            "",
            f"- `members` {_span(d['members'])}, place `route_fraction` {_span(d['route_fraction'])}, `when.at` {_span(d['when_at'], ' s')}, `robot_within` {_span(d['robot_within'], ' m')}; zone placements so far: {d['zone_places']} (the installed core is universal-scope).",
            "- These are what previous answers used, not limits - the contract above is the limit.",
        )
    )


def _time_table() -> str:
    groups: dict[str, list[str]] = {}
    for phrase, when in PHRASES:
        groups.setdefault(json.dumps(when.as_dict()), []).append(phrase)
    lines = ["Time phrases (`when`):"]
    lines += [f"- `{key}`: " + ", ".join(f'"{p}"' for p in ps) for key, ps in groups.items()]
    return "\n".join(lines)


def _place_table() -> str:
    lines = ["Places on the route (`route_fraction`):"]
    lines += [f'- "{phrase}" -> {fraction:g}' for phrase, fraction in P.PLACE_ROUTE]
    return "\n".join(lines)


def _duration_table() -> str:
    return "C duration phrases (seconds): " + "; ".join(f'"{p}" {s}' for p, s in P.DUR_C) + "."


def _pace_line() -> str:
    return f"The robot plans at {ROBOT_SPEED_MPS:g} m/s: \"twenty seconds in\" is about {20 * ROBOT_SPEED_MPS:g} m along the route - convert times to route fractions with the route length given in the context."


def _profile_table() -> str:
    words: dict[str, list[str]] = {}
    for word, prof in ALIASES.items():
        if prof in PROFILES:
            words.setdefault(prof, []).append(word)
    lines = ["## Profiles", "", "| profile | slow | walks | brisk | is |", "|---|---|---|---|---|"]
    for prof in PROFILES.values():
        lo, mid, hi = (tier_value("speed", t, prof=prof) for t in ("low", "mid", "high"))
        alias = "; words: " + ", ".join(words[prof.name]) if prof.name in words else ""
        lines.append(f"| `{prof.name}` | {lo:g} | {mid:g} | {hi:g} m/s | {prof.note}{alias} |")
    return "\n".join(lines)


def _tier_table(params: list[str]) -> str:
    lines = ["Tier values (what low / mid / high pin):", "", "| param | low | mid | high |", "|---|---|---|---|"]
    lines += [f"| {p} | {TIERS[p][0]:g} | {TIERS[p][1]:g} | {TIERS[p][2]:g} |" for p in params]
    lines.append("")
    lines.append(f"A `speed` above the engine cap ({MAX_VELOCITY_DEFAULT:g} m/s by default) lifts the cap automatically - `speed: high` really runs.")
    return "\n".join(lines)


def _bc_contract() -> str:
    params = [p for p in TIERS if p != "max_speed"]
    b_cases = "; ".join(f"`{c}` ({B_CASE_NOTES[c]})" for c in B_CASE_NOTES)
    after = "; ".join(f"`{a}` ({C_AFTER_NOTES[a]})" for a in C_AFTER)
    angles = ", ".join(f"{a} {note}" for a, note in ANGLES)
    return "\n".join(
        (
            "## B - tune some pedestrians",
            "",
            f"- `case`: {b_cases}.",
            '- `select` (who, from the unowned agents listed): `{"profile": "<name>"}` | `{"amount": "half" | "all_but_one" | "all"}` | `{"n": <count>}` | `{"zone": "<name>"}` | `{"names": [...]}`.',
            f"- `params` (intrinsic, one to three, tier per key): {', '.join(params)}.",
            f"- `period` (seconds between reroutes, default 4), `radius` (meters, default 3) and `around` for the erratic cases: {' or '.join(f'`{a}`' for a in ERRATIC_AROUND)} (points either side of the robot's route, or around each walker's own start; default `{_default(BDecision, 'around')}`).",
            "- `when`: see the time table; without a time phrase, omit it (the runtime defaults to 10 s).",
            "",
            _tier_table(params),
            "",
            "## C - one new pedestrian moving relative to the robot",
            "",
            "- `case` `blocking`: `mode` `stand` (stops on the robot's route at `route_fraction` for `duration` seconds) or `lead` (walks at the profile's slow speed ahead of the robot in its direction).",
            f"- `case` `approach`: `angle` {angles}; `after`: {after}.",
            f"- Fields and defaults: `profile`, `speed` tier (default `{_default(CDecision, 'speed')}`), `route_fraction` (strictly between 0 and 1, default {_default(CDecision, 'route_fraction'):g}), "
            f"`duration` seconds (default {_default(CDecision, 'duration'):g}), `offset` seconds after the robot (negative = before, default {_default(CDecision, 'offset'):g}).",
            f"- Keep C at least {CONFLICT_M:g} m along the route from any Type D placement.",
            "",
            _profile_table(),
            "",
            _bc_observed(_ranges()),
        )
    )


def _d_contract() -> str:
    lo, mid, _hi = TIERS["duration"]
    return "\n".join(
        (
            "## The three Type D cases",
            "",
            f"- `stubborn` - {D_VARIANT_NOTES['stubborn']}.",
            "- `interaction_unbroken` - a placed interaction that does not end on its own. `variant`: "
            + "; ".join(f"`{v}` ({D_VARIANT_NOTES[v]})" for v in ("meet", "standing", "blocking"))
            + "."
            + f" Fields: `members` (2-4 profiles; `blocking` allows up to 8 for a wide passage), `place`, `when`, `duration` (tier: low ends after {lo:g} s, mid after {mid:g} s, high never ends).",
            "- `formation_break` - a formation the step builds (`variant`: "
            + " or ".join(f"`{v}` ({D_VARIANT_NOTES[v]})" for v in ("circle", "line"))
            + f") that breaks when the robot comes near or after a while. Fields: `members`, `place`, `when` (default: robot within {FORMATION_TRIGGER_M:g} m), `break`: `scatter` (members step apart and stand) or `disperse` (members walk off to other rooms), `radius` (meters for scatter, default {_default(DDecision, 'radius'):g}).",
            "",
            '`place` is `{"route_fraction": f}` with 0 < f < 1, or `{"zone": "<name>"}` from the world context.',
            "",
            "## Formation numbers (how the realizer builds what you decide)",
            "",
            f"- Members walk their own routes first and form at about {D.FORM_AT_S:g} s unless `when` says later.",
            f"- A blocking wall stands shoulder to shoulder at {D.LINE_SPACING:g} m spacing; the realizer sizes it to the passage (an n-member wall spans n x {D.LINE_SPACING:g} m, up to 8).",
            f"- A queue strings out along the route at {D.QUEUE_SPACING:g} m spacing.",
            f"- A talking circle stands {D.CIRCLE_SPACING_M:g} m between neighbors (2 members {2 * D.circle_radius(2):.2f} m apart, 4 members on a {D.circle_radius(4):.2f} m radius), too tight for the robot to pass between; in a narrow passage it huddles to a {D.HUDDLE_RADIUS_M:g} m radius rather than leave the route.",
            f"- A formation counts as broken once members drift more than {D.FORMATION_RADIUS_M:g} m from its center.",
            f"- A route shorter than {D.MIN_ROUTE_FOR_FORMATION_M:g} m cannot host an on-route formation clear of the robot's endpoints.",
            "",
            _profile_table(),
            "",
            _d_observed(_ranges()),
        )
    )


def _profile_words() -> str:
    words: dict[str, list[str]] = {}
    for word, prof in ALIASES.items():
        if prof in PROFILES:
            words.setdefault(prof, []).append(word)
    lines = ["Profiles for `who` (name - words a prompt may use):"]
    lines += [f"- `{p.name}` - {p.note}" + ("; " + ", ".join(words[p.name]) if p.name in words else "") for p in PROFILES.values()]
    return "\n".join(lines)


def _a_contract() -> str:
    k_words = "; ".join(f'"{w}" -> {v:g}' for w, v in (lex_load()["busy"].get("k") or {}).items())
    return "\n".join(
        (
            "## The four crowd-wide cases",
            "",
            *(f"- `{c}` - {note}." for c, note in A_CASE_NOTES.items()),
            "",
            f"`k` words for busy: {k_words}; no quantity word -> 2.",
            "",
            _profile_words(),
        )
    )


def build() -> dict[str, str]:
    """filename -> content, deterministically."""
    out: dict[str, str] = {}
    mappings = "\n\n".join(("## Reading the sentences", _time_table(), _place_table(), _duration_table(), _pace_line()))
    a_mappings = "\n\n".join(("## Reading the sentences", _time_table()))
    for name, contract, maps in (("system_bc.md", _bc_contract(), mappings), ("system_d.md", _d_contract(), mappings), ("system_a.md", _a_contract(), a_mappings)):
        text = (_DIR / name.replace(".md", ".tmpl.md")).read_text()
        text = text.replace("<!-- CONTRACT -->", contract)
        text = text.replace("<!-- MAPPINGS -->", maps)
        out[name] = text
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify the committed files match the build")
    args = ap.parse_args()
    built = build()
    stale = [name for name, text in built.items() if not (_DIR / name).is_file() or (_DIR / name).read_text() != text]
    if args.check:
        for name in stale:
            print(f"stale: {name} (run python3 -m arena_simulation_setup.generative.scenario.promptgen.build_kb)")
        return 1 if stale else 0
    for name, text in built.items():
        (_DIR / name).write_text(text)
        print(f"wrote {name} ({len(text.encode())} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
