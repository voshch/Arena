from __future__ import annotations

import pytest

from task_generator.simulators.human.gestures import qa


@pytest.mark.parametrize("case", qa.default_cases(), ids=lambda c: c.name)
def test_default_cases_are_clean(case: qa.Case) -> None:
    res = qa.run(case)
    assert not res.warnings, res.warnings
    assert res.max_link_m <= qa.MAX_LINK_STEP_M, (res.max_link, res.max_link_t, res.max_link_m)
    for rep in res.reports:
        assert rep.get("ok", True), rep
        assert not rep.get("relaxed"), rep
        assert not rep.get("near_target_fallback"), rep
