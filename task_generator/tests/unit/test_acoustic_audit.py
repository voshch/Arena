from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from task_generator.auditory.acoustic_audit import audit_world


def test_audit_reports_coverage_and_portal_connectivity() -> None:
    root = (
        Path(get_package_share_directory("arena_simulation_setup"))
        / "worlds"
    )
    airport = audit_world("airport", worlds_root=root, stride_cells=10)
    hospital = audit_world(
        "hospital_1",
        worlds_root=root,
        stride_cells=10,
    )

    assert airport.complete
    assert airport.sampled_traversable_cells > 0
    assert hospital.rooms == 13
    assert hospital.door_portals == 12
    assert hospital.opening_portals == 1
    assert hospital.connected_components == 1
