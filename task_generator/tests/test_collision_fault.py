"""Who caused the contact: the scorer attributes the first collision by closing speeds."""

from __future__ import annotations

from task_generator.tasks.obstacles.edge_case.criticality import Sample, score


def _run(robot_vel, ped_vel):
    # two samples; contact happens on the second (centres 0.5 m apart, radii 0.3 + 0.25)
    return score(
        [
            Sample(t=0.0, robot=(0.0, 0.0), robot_vel=robot_vel, robot_radius=0.3,
                   peds={"p": ((3.0, 0.0), ped_vel, 0.25)}),
            Sample(t=1.0, robot=(0.0, 0.0), robot_vel=robot_vel, robot_radius=0.3,
                   peds={"p": ((0.5, 0.0), ped_vel, 0.25)}),
        ],
        settle_s=0.0,
    )


def test_robot_driving_into_a_standing_pedestrian_is_robot_fault():
    r = _run(robot_vel=(0.8, 0.0), ped_vel=(0.0, 0.0))
    assert r.collided and r.collision_fault == "robot"
    assert r.collision_with == "p"
    assert r.collision_robot_closing > 0.5 and abs(r.collision_ped_closing) < 0.05


def test_pedestrian_walking_into_a_standing_robot_is_pedestrian_fault():
    r = _run(robot_vel=(0.0, 0.0), ped_vel=(-0.9, 0.0))
    assert r.collided and r.collision_fault == "pedestrian"
    assert r.collision_ped_closing > 0.5


def test_both_closing_is_mutual_and_neither_is_static():
    assert _run((0.6, 0.0), (-0.6, 0.0)).collision_fault == "mutual"
    assert _run((0.0, 0.0), (0.0, 0.0)).collision_fault == "static"


def test_receding_pedestrian_contact_is_not_their_fault():
    # the pedestrian moves away from the robot at contact; the robot is closing
    r = _run(robot_vel=(0.7, 0.0), ped_vel=(0.7, 0.0))
    assert r.collision_fault == "robot"


def test_no_contact_leaves_the_fault_empty():
    r = score([Sample(t=0.0, robot=(0.0, 0.0), peds={"p": ((5.0, 0.0), (0.0, 0.0), 0.25)})])
    assert not r.collided and r.collision_fault == "" and r.collision_at is None
