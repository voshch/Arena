from __future__ import annotations

import pytest

from task_generator.simulators.human.gestures import qa


@pytest.mark.parametrize("case", qa.default_cases(), ids=lambda c: c.name)
def test_default_cases_are_clean(case: qa.Case) -> None:
    res = qa.run(case)
    assert not res.warnings, res.warnings
    assert res.max_link_m <= qa.MAX_LINK_STEP_M, (res.max_link, res.max_link_t, res.max_link_m)
    assert res.max_collar_rad <= qa.MAX_COLLAR_RAD, res.max_collar_rad
    assert res.max_hold_aim_deg <= qa.MAX_HOLD_AIM_DEG, res.max_hold_aim_deg
    assert res.arm_clips >= case.min_clips, (res.arm_clips, case.min_clips)
    assert not res.missing_rest, (res.missing_rest, res.rest_times)
    for rep in res.reports:
        assert rep.get("ok", True), rep
        assert not rep.get("relaxed"), rep
        assert not rep.get("near_target_fallback"), rep
