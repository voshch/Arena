from __future__ import annotations

import json
from pathlib import Path

import pytest
from arena_simulation_setup.generative.scenario.promptgen import step_a_llm
from arena_simulation_setup.generative.scenario.promptgen.ask import system_prompt
from arena_simulation_setup.generative.scenario.promptgen.lexicon import match, score, when_in
from arena_simulation_setup.generative.scenario.promptgen.spec import Record, SchemaError, When, parse_b, parse_c, parse_d, parse_selection, parse_when
from arena_simulation_setup.generative.scenario.propose.client import prompt_sha256


@pytest.mark.parametrize(
    "text",
    [
        "A surgeon crosses the hall.",
        "The boxes are unpacked in the corridor.",
        "Someone mentions a crusher in the basement.",
    ],
)
def test_lexicon_phrases_do_not_match_inside_words(text) -> None:
    assert score(text) == {}
    assert match(text) is None


def test_busy_multiplier_words_match_whole_words_outside_time_phrases() -> None:
    assert match("The floor is busy and everyone is here.").k == 2.0
    assert match("The floor gets busy a little later.").k == 2.0
    assert match("The floor gets a little busy.").k == 1.5
    assert match("The floor is very busy.").k == 3.0


def test_a_word_containing_a_case_phrase_does_not_own_its_sentence() -> None:
    assert when_in("A surgeon walks by after ten seconds.", exclude_cases=("stampede",)).as_dict() == {"at": 10.0}


@pytest.mark.parametrize("raw, names", [("ped_1", ("ped_1",)), (["ped_1", "ped_2"], ("ped_1", "ped_2"))])
def test_names_and_reuse_take_a_list_or_one_string(raw, names) -> None:
    assert parse_selection({"names": raw}, where="B.select").names == names
    assert parse_d({"case": "stubborn", "reuse": raw}).reuse == names


@pytest.mark.parametrize("raw", [5, {"ped_1": 1}])
def test_names_and_reuse_reject_a_non_list(raw) -> None:
    with pytest.raises(SchemaError, match=r"B\.select\.names"):
        parse_selection({"names": raw}, where="B.select")
    with pytest.raises(SchemaError, match=r"D\.reuse"):
        parse_d({"case": "stubborn", "reuse": raw})


@pytest.mark.parametrize(
    "parse, base, keys",
    [
        (parse_b, {"case": "erratic_route", "select": {"amount": "all"}}, ("period", "radius", "around", "when", "params", "reason")),
        (parse_c, {"case": "approach"}, ("profile", "mode", "angle", "route_fraction", "speed", "after", "duration", "offset", "veer_to", "reason")),
        (parse_d, {"case": "formation_break"}, ("variant", "members", "place", "when", "duration", "break", "radius", "profile", "speed", "route_fraction", "reuse", "reason")),
    ],
)
def test_null_fields_read_as_absent(parse, base, keys) -> None:
    for key in keys:
        assert parse({**base, key: None}) == parse(base), key


def test_null_k_and_who_profile_read_as_absent_in_the_model_answer(tmp_path: Path) -> None:
    text = "A nurse walks by."
    answer = json.dumps({"case": "busy", "k": None, "who": [{"mention": "a nurse", "profile": None, "count": None}]})
    (tmp_path / f"{prompt_sha256(step_a_llm.A_MODEL, system_prompt('a'), step_a_llm.user_prompt(text))}.json").write_text(json.dumps({"answer": answer}))
    decision, who, warnings = step_a_llm.decide(text, Record(prompt_id="t", prompt=text, world="w", base="b"), cache_dir=tmp_path)
    assert decision is not None and decision.k == 2.0
    assert who == [{"mention": "a nurse", "profile": "adult", "count": 0}] and warnings == []


@pytest.mark.parametrize("raw", [{"at": 5, "robot_within": 2}, {"at": 5, "route_fraction": 0.5}, {"robot_within": 2, "route_fraction": 0.5}])
def test_when_takes_one_trigger(raw) -> None:
    with pytest.raises(SchemaError, match=r"B\.when: set exactly one"):
        parse_when(raw, default=When(at=0.0), where="B")


@pytest.mark.parametrize("raw", [{"at": 5}, {"robot_within": 2}, {"route_fraction": 0.5}, {"at": None, "robot_within": 2}])
def test_when_round_trips_through_its_effect(raw) -> None:
    when = parse_when(raw, default=When(at=0.0), where="B")
    assert when.as_effect() == {k: float(v) for k, v in raw.items() if v is not None}


@pytest.mark.parametrize("value", ["inf", "nan", float("inf"), -5])
def test_when_rejects_a_time_that_is_not_a_finite_non_negative_number(value) -> None:
    with pytest.raises(SchemaError, match=r"B\.when\.at"):
        parse_when({"at": value}, default=When(at=0.0), where="B")


@pytest.mark.parametrize(
    ("parse", "raw", "where"),
    [
        (parse_c, {"case": "approach", "offset": "nan"}, r"C\.offset"),
        (parse_c, {"case": "approach", "angle": "inf"}, r"C\.angle"),
        (parse_c, {"case": "approach", "duration": -3}, r"C\.duration"),
        (parse_c, {"case": "approach", "route_fraction": "nan"}, r"C\.route_fraction"),
        (parse_b, {"case": "erratic_route", "select": {"amount": "all"}, "period": "inf"}, r"B\.period"),
        (parse_b, {"case": "erratic_route", "select": {"amount": "all"}, "radius": -1}, r"B\.radius"),
        (parse_b, {"case": "erratic_route", "select": {"n": "nan"}}, r"B\.select\.n"),
        (parse_d, {"case": "formation_break", "radius": "nan"}, r"D\.radius"),
        (parse_d, {"case": "formation_break", "when": {"robot_within": "inf"}}, r"D\.when\.robot_within"),
    ],
)
def test_model_numbers_must_be_finite_and_in_range(parse, raw, where) -> None:
    with pytest.raises(SchemaError, match=where):
        parse(raw)


def test_a_non_finite_busy_multiplier_is_rejected(tmp_path: Path) -> None:
    from arena_simulation_setup.generative.scenario.promptgen.step_a import StepError

    text = "The place is packed."
    retry = f"{step_a_llm.user_prompt(text)}\n\nYour previous answer was rejected: A.k: 'inf' is not a finite number. Answer again."
    for user in (step_a_llm.user_prompt(text), retry):
        (tmp_path / f"{prompt_sha256(step_a_llm.A_MODEL, system_prompt('a'), user)}.json").write_text(json.dumps({"answer": json.dumps({"case": "busy", "k": "inf"})}))
    with pytest.raises(StepError, match=r"A\.k: 'inf' is not a finite number"):
        step_a_llm.decide(text, Record(prompt_id="t", prompt=text, world="w", base="b"), cache_dir=tmp_path)
