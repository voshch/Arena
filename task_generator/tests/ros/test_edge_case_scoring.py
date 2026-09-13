"""Tests for the scoring lifecycle — the join between a case and what the episode did.

`criticality.py` is already covered as pure maths. What is untested there, and is where the
mistakes actually live, is the *wiring*: which episode a score belongs to, what the panel was
computed over, and whether an instrumentation failure can take an episode down with it.

The last point is the load-bearing one. Scoring is optional. An episode that dies because its
own measurement raised is strictly worse than an episode with no measurement, so every failure
path here is asserted to be silent-and-degrading rather than fatal.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("arena_humansim")

from edge_case_fixtures import _mode  # noqa: E402


class _Scorer:
    """Stands in for EpisodeScorer: no ROS, but the same surface flush_score uses."""

    def __init__(self, *, injected=(), result=None, ambient=None):
        self._injected = list(injected)
        self._result = result
        self._ambient = ambient
        self.reset_calls = 0
        self.scored_with: list[tuple] = []

    def injected_seen(self):
        return list(self._injected)

    def result(self, *, encounter=None, only=None):
        self.scored_with.append((encounter, tuple(only) if only else None))
        return self._ambient if (only is None and self._ambient is not None) else self._result

    def reset(self):
        self.reset_calls += 1


def _criticality(**kwargs):
    """A panel that counts as *observed*.

    `Criticality.observed` needs both samples and a clearance - a row with neither is the
    "nothing was measurable" case, which must not be written. Tests that want that case ask
    for it explicitly by calling `Criticality()` with no arguments.
    """
    from task_generator.tasks.obstacles.edge_case.criticality import Criticality

    kwargs.setdefault("samples", 10)
    kwargs.setdefault("duration_s", 1.0)
    kwargs.setdefault("min_clearance_m", 0.5)
    return Criticality(**kwargs)


def _scores(tmp_path):
    path = tmp_path / "records" / "scores.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# Joining a score to its case
# ---------------------------


def test_score_is_written_against_the_case_that_was_running(tmp_path):
    """The row must carry the episode's identity, not the next one's.

    An episode is scored at the *start* of the following reset, so the case has to be held
    across that boundary. Getting this wrong shifts every score by one episode - which still
    produces a plausible-looking file, and is the reason this is asserted directly.
    """
    tm = _mode(tmp_path)
    tm._scorer = _Scorer(result=_criticality(min_ttc_s=0.8, samples=120, duration_s=12.0))
    tm._pending_case = {"run_seed": "abc", "episode_id": 7, "world": "arena_arena_002", "case_id": "vision_fov"}

    tm.flush_score()

    (row,) = _scores(tmp_path)
    assert row["episode_id"] == 7
    assert row["case_id"] == "vision_fov"
    assert row["min_ttc_s"] == 0.8


def test_pending_case_is_cleared_so_an_episode_is_never_scored_twice(tmp_path):
    tm = _mode(tmp_path)
    tm._scorer = _Scorer(result=_criticality(min_ttc_s=1.0))
    tm._pending_case = {"run_seed": "a", "episode_id": 1}

    tm.flush_score()
    tm.flush_score()

    assert len(_scores(tmp_path)) == 1


def test_nothing_is_written_when_no_case_ran(tmp_path):
    """First reset of a run: the scorer exists but no episode has been built yet."""
    tm = _mode(tmp_path)
    tm._scorer = _Scorer(result=_criticality(min_ttc_s=0.5))
    tm._pending_case = None

    tm.flush_score()

    assert _scores(tmp_path) == []


def test_an_episode_with_no_samples_writes_no_row(tmp_path):
    """`observed` is False for an empty panel. Writing it would put a row of nulls in the
    file that is indistinguishable from a genuinely uneventful episode."""
    from task_generator.tasks.obstacles.edge_case.criticality import Criticality

    tm = _mode(tmp_path)
    tm._scorer = _Scorer(result=Criticality())
    tm._pending_case = {"run_seed": "a", "episode_id": 1}

    tm.flush_score()

    assert _scores(tmp_path) == []


# Scope
# -----


def test_auto_scope_isolates_the_injected_agent(tmp_path):
    """In inject mode the injected agent *is* the case; scoring the whole crowd would bury
    its effect in the bystanders' ambient near-misses."""
    tm = _mode(tmp_path, score_scope="auto")
    tm._scorer = _Scorer(
        injected=["edge_0"],
        result=_criticality(min_ttc_s=0.4),
        ambient=_criticality(min_ttc_s=0.9),
    )
    tm._pending_case = {"run_seed": "a", "episode_id": 1, "injected": 1}

    tm.flush_score()

    (row,) = _scores(tmp_path)
    assert row["scope"] == "injected"
    assert row["scored_agents"] == ["edge_0"]
    assert row["min_ttc_s"] == 0.4
    # The crowd figure is kept alongside, so "was the crowd already this tight?" is
    # answerable without running a separate arm.
    assert row["min_ttc_s_ambient"] == 0.9


def test_auto_scope_scores_everyone_when_nothing_was_injected(tmp_path):
    """perturb / Level B / baseline arms have no single agent to isolate."""
    tm = _mode(tmp_path, score_scope="auto")
    tm._scorer = _Scorer(injected=[], result=_criticality(min_ttc_s=1.2))
    tm._pending_case = {"run_seed": "a", "episode_id": 1, "injected": 0}

    tm.flush_score()

    (row,) = _scores(tmp_path)
    assert row["scope"] == "all"
    assert row["scored_agents"] == []
    assert row["min_ttc_s_ambient"] is None


def test_scope_falls_back_loudly_when_the_injected_agent_never_appeared(tmp_path):
    """Silently widening to the whole crowd would answer a different question under the same
    column name. The row must say so."""
    tm = _mode(tmp_path, score_scope="injected")
    tm._scorer = _Scorer(injected=[], result=_criticality(min_ttc_s=1.1))
    tm._pending_case = {"run_seed": "a", "episode_id": 1, "injected": 2}

    tm.flush_score()

    (row,) = _scores(tmp_path)
    assert row["scope"] == "all"
    assert any("none appeared" in str(m) for m in tm._logger.messages)


def test_explicit_all_scope_never_narrows(tmp_path):
    tm = _mode(tmp_path, score_scope="all")
    tm._scorer = _Scorer(injected=["edge_0"], result=_criticality(min_ttc_s=1.3))
    tm._pending_case = {"run_seed": "a", "episode_id": 1, "injected": 1}

    tm.flush_score()

    (row,) = _scores(tmp_path)
    assert row["scope"] == "all"
    assert row["scored_agents"] == []


def test_encounter_point_is_passed_through_for_pet(tmp_path):
    """PET is measured at the designed encounter point. Dropping it silently yields a null
    PET column that looks like "the metric does not apply" rather than "we lost the input"."""
    tm = _mode(tmp_path, score_scope="all")
    scorer = _Scorer(result=_criticality(min_ttc_s=1.0))
    tm._scorer = scorer
    tm._pending_case = {"run_seed": "a", "episode_id": 1, "encounter_point": [3.5, -2.0]}

    tm.flush_score()

    assert scorer.scored_with[0][0] == (3.5, -2.0)


# Degradation
# -----------


def test_scoring_is_a_noop_when_disabled(tmp_path):
    """`score:=false` must create nothing and write nothing."""
    tm = _mode(tmp_path, score=False)
    tm._pending_case = {"run_seed": "a", "episode_id": 1}

    assert tm._ensure_scorer() is None
    tm.flush_score()
    assert _scores(tmp_path) == []


def test_a_scorer_that_raises_does_not_take_the_episode_down(tmp_path):
    """The whole point of keeping scoring optional: instrumentation failure degrades to "no
    score", never to a lost episode."""

    class _Exploding(_Scorer):
        def result(self, *, encounter=None, only=None):
            raise RuntimeError("peds stream went away")

    tm = _mode(tmp_path)
    tm._scorer = _Exploding()
    tm._pending_case = {"run_seed": "a", "episode_id": 1}

    tm.flush_score()  # must not raise

    assert _scores(tmp_path) == []
    assert any("scoring this episode failed" in str(m) for m in tm._logger.messages)


def test_arming_discards_the_reset_transient(tmp_path):
    """The respawn teleport looks like a metres-per-tick velocity spike. Sampling it would
    put a fictitious near-collision at the head of every episode."""
    tm = _mode(tmp_path, score=True)
    scorer = _Scorer()
    tm._scorer = scorer

    tm._arm_score()

    assert scorer.reset_calls == 1


def test_a_broken_scorer_is_not_retried_every_episode(tmp_path):
    """A per-episode retry storm on a permanently unavailable stream is its own failure mode."""
    tm = _mode(tmp_path)
    tm._score_broken = True

    assert tm._ensure_scorer() is None
    assert tm._scorer is None


# Teardown — the final episode
# ----------------------------
# An episode is scored at the start of the *next* reset, so the last one in a run has no
# next reset to be scored by. That is what `teardown` is for.


def test_teardown_scores_the_final_episode(tmp_path):
    """Without this, `scores.jsonl` is always one row short of `cases.jsonl` and the run's
    most recent arm - often the one being looked at - has no measurement at all."""
    import asyncio

    tm = _mode(tmp_path)
    tm._scorer = _Scorer(result=_criticality(min_ttc_s=0.42, samples=90, duration_s=9.0))
    tm._pending_case = {"run_seed": "abc", "episode_id": 11, "case_id": "vision_fov"}

    asyncio.run(tm.teardown())

    (row,) = _scores(tmp_path)
    assert row["episode_id"] == 11
    assert row["min_ttc_s"] == 0.42


def test_teardown_does_not_double_score_an_already_flushed_episode(tmp_path):
    """`reset` flushes too. Teardown after a normal reset must add nothing, or the last
    episode appears twice and any per-knob aggregate counts it twice."""
    import asyncio

    tm = _mode(tmp_path)
    tm._scorer = _Scorer(result=_criticality(min_ttc_s=1.0))
    tm._pending_case = {"run_seed": "a", "episode_id": 3}

    tm.flush_score()
    asyncio.run(tm.teardown())

    assert len(_scores(tmp_path)) == 1


def test_teardown_survives_a_scorer_that_cannot_be_released(tmp_path):
    """Same discipline as everywhere else in this file: instrumentation degrades, it does
    not take the process down. `run_main` gives the whole teardown chain 5 seconds, and a
    raise here would strand the rest of it."""
    import asyncio

    tm = _mode(tmp_path)
    tm._scorer = _Scorer(result=_criticality(min_ttc_s=0.9))
    tm._pending_case = {"run_seed": "a", "episode_id": 5}

    # The fake scorer has no `shutdown`, and the fake node has no `executor` - between them
    # that is exactly the AttributeError a half-built scorer would raise.
    asyncio.run(tm.teardown())

    assert len(_scores(tmp_path)) == 1, "the score must survive the release failing"
    assert tm._scorer is None


def test_teardown_with_nothing_pending_writes_nothing(tmp_path):
    import asyncio

    tm = _mode(tmp_path)
    tm._scorer = _Scorer(result=_criticality(min_ttc_s=0.5))
    tm._pending_case = None

    asyncio.run(tm.teardown())

    assert _scores(tmp_path) == []


def test_teardown_releases_the_delegate_mode(tmp_path):
    """`_base()` constructs and caches a whole second task mode. Leaving it behind when the
    mode is replaced leaks whatever that mode holds - `prompt` keeps an inference client and
    a scenario-local temp dir."""
    import asyncio

    tm = _mode(tmp_path)
    tm._scorer = None
    tm._pending_case = None

    released: list[str] = []

    class _Base:
        async def teardown(self):
            released.append("base")

    tm._base_cache = _Base()

    asyncio.run(tm.teardown())

    assert released == ["base"]
    assert tm._base_cache is None
