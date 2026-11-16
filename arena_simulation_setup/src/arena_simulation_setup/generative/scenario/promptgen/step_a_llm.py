"""Step 1b: Type A and the shared `who` cast read by a model from the prompt text."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from arena_simulation_setup.generative.core.profiles import ProfileError, profile

from .ask import ask, system_prompt
from .lexicon import ONSETS
from .llm import DEFAULT_MODEL
from .spec import A_CASES, ADecision, Record, SchemaError, When, parse_when

A_MODEL = os.environ.get("ARENA_PROMPT_LLM_A", DEFAULT_MODEL)


def user_prompt(prompt: str) -> str:
    return f"prompt: {prompt}"


def who_line(who: list[dict[str, Any]]) -> str:
    """The shared cast reading handed to Steps D and B+C."""
    if not who:
        return ""
    parts = [f"{w['count'] or 'some'} x {w['profile']} (\"{w['mention']}\")" for w in who]
    return "who (actors read from the prompt): " + "; ".join(parts)


def _parse(raw: dict[str, Any]) -> tuple[ADecision | None, list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    who: list[dict[str, Any]] = []
    for entry in raw.get("who") or []:
        if not isinstance(entry, dict) or not entry.get("mention"):
            warnings.append(f"who: dropped malformed entry {entry!r}")
            continue
        try:
            prof = profile(str(entry.get("profile") or "adult")).name
        except ProfileError:
            warnings.append(f"who: dropped {entry.get('mention')!r} (unknown profile {entry.get('profile')!r})")
            continue
        try:
            count = max(0, int(entry.get("count") or 0))
        except (TypeError, ValueError):
            count = 0
        who.append({"mention": str(entry["mention"])[:60], "profile": prof, "count": count})
    case = str(raw.get("case") or "none").strip().lower()
    if case in ("none", ""):
        return None, who, warnings
    if case not in A_CASES:
        raise SchemaError(f"A.case: {case!r} is not one of {list(A_CASES)} or 'none'")
    k_raw = raw.get("k")
    try:
        k = 2.0 if k_raw is None else float(k_raw)
    except (TypeError, ValueError):
        raise SchemaError(f"A.k: {raw.get('k')!r} is not a number") from None
    if not math.isfinite(k):
        raise SchemaError(f"A.k: {raw.get('k')!r} is not a finite number")
    k = min(3.0, max(1.0, k))
    when = parse_when(raw.get("when"), default=ONSETS[case], where="A")
    return ADecision(case=case, k=k, when=when if isinstance(when, When) else ONSETS[case]), who, warnings


def decide(prompt: str, record: Record, *, cache_dir: Path | None, model: str | None = None) -> tuple[ADecision | None, list[dict[str, Any]], list[str]]:
    """(decision-or-None, who block, warnings). Raises `StepError` on a twice-rejected answer."""
    return ask("A", system_prompt("a"), user_prompt(prompt), record, _parse, cache_dir=cache_dir, model=model or A_MODEL)
