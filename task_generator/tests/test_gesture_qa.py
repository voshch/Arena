from __future__ import annotations

import pytest

from task_generator.simulators.human.gestures import qa

pytestmark = pytest.mark.clips("point_at_template", "talk_with_arm_gesture", "wave_high")


KNOWN_BAD: dict[str, str] = {}  # case name -> why it fails, run as a strict xfail


@pytest.mark.parametrize("case", qa.default_cases(loops=["talk_with_arm_gesture", "wave_high"]), ids=lambda c: c.name)
def test_default_cases_are_clean(case: qa.Case, request: pytest.FixtureRequest) -> None:
    if case.name in KNOWN_BAD:
        request.node.add_marker(pytest.mark.xfail(reason=KNOWN_BAD[case.name], strict=True))
    res = qa.run(case)
    assert not res.warnings, res.warnings
    assert res.max_link_m <= qa.MAX_LINK_STEP_M, (res.max_link, res.max_link_t, res.max_link_m)
    assert res.max_collar_rad <= case.collar_limit, res.max_collar_rad
    assert res.max_hold_aim_deg <= case.aim_limit, res.max_hold_aim_deg
    assert res.arm_clips >= case.min_clips, (res.arm_clips, case.min_clips)
    assert not res.missing_rest, (res.missing_rest, res.rest_times)
    for rep in res.reports:
        assert rep.get("ok", True), rep
        assert not rep.get("relaxed"), rep
        assert not rep.get("near_target_fallback"), rep
