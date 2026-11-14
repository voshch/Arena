import json
import math
import pathlib

import numpy as np
import pytest

from arena_fleet import features
from arena_fleet.config import CostConfig, RulesConfig, load_cost, load_rules
from arena_fleet.dispatcher import FleetDispatcher, Info, PublishPlan, SendGoal, Stop, robot_names
from arena_fleet.geometry import Point, hungarian, seg_cross
from arena_fleet.site import Site
from arena_fleet.view import MapOfDynamics

PKG = pathlib.Path(__file__).resolve().parents[1]
WORLD = PKG.parent / "arena_simulation_setup" / "worlds" / "mechanisms_doorway" / "world.yaml"
EXAMPLE = PKG / "examples" / "doorway"
TICK = 0.25

Peds = dict[str, tuple[float, float, float, float]]


def site() -> Site:
    return Site.load(WORLD, EXAMPLE / "resources.yaml", EXAMPLE / "scenario" / "scenario.yaml")


def cost(name: str = "cost.yaml") -> CostConfig:
    return load_cost(PKG / "configs" / name)


def rules() -> RulesConfig:
    return load_rules(PKG / "configs" / "rules.yaml")


class Episode:
    """The dispatcher on the doorway world; robots walk their latest plan at 0.4 m/s and stand when stopped."""

    def __init__(self, out: pathlib.Path, n: int = 4, profile: str = "base", info: Info = "none", peds: Peds | None = None, seed: int = 6, mod: MapOfDynamics | None = None, cost_file: str = "cost.yaml") -> None:
        s = site()
        if n > len(s.starts):
            s = Site(walls=s.walls, leaves=s.leaves, doors=s.doors, starts=tuple((2.6 + 1.2 * (i // 4), 4.5 + (i % 4)) for i in range(n)))
        self.out = out
        self.d = FleetDispatcher(s, cost(cost_file), rules(), names=robot_names("jackal", n), info=info, profile=profile, seed=seed, out=out, duration=1e9, mod=mod)
        self.t = 0.0
        self.d.set_time(0.0)
        assert self.d.try_setup({m: (*p, 0.0) for m, p in zip(self.d.names, s.starts, strict=False)})
        self.peds: Peds = dict(peds or {})
        self.plans: dict[str, list[Point]] = {}
        self.stopped: set[str] = set()
        self.goals_sent = dict.fromkeys(self.d.names, 0)
        self.cross_max = 0

    def tick(self) -> None:
        self.d.set_time(self.t)
        self.d.set_peds(self.peds)
        self.d.tick()
        for cmd in self.d.drain():
            if isinstance(cmd, PublishPlan):
                self.plans[cmd.robot] = list(cmd.path)
            elif isinstance(cmd, SendGoal):
                self.goals_sent[cmd.robot] += 1
                self.stopped.discard(cmd.robot)
            elif isinstance(cmd, Stop):
                self.stopped.add(cmd.robot)

    def run(self, until: float = 200.0, v: float = 0.4) -> None:
        while self.t < until and not self.d.finished:
            self.tick()
            self.peds = {k: (x + vx * TICK, y + vy * TICK, vx, vy) for k, (x, y, vx, vy) in self.peds.items()}
            for door in self.d.doors:
                self.cross_max = max(self.cross_max, sum(1 for r in self.d.rob.values() if r.door == door and r.phase == "cross"))
            for n in self.d.names:
                self.walk(n, v)
            self.t += TICK

    def walk(self, n: str, v: float) -> None:
        p = self.plans.get(n)
        if n in self.stopped or not p or self.d.rob[n].phase == "done":
            return
        x, y, _ = self.d.pose[n]
        left, q = v * TICK, (x, y)
        near = min(range(len(p)), key=lambda i: math.dist((x, y), p[i]))
        for w in p[near + 1 :]:
            dd = math.dist(q, w)
            if dd <= 1e-6:
                continue
            if dd >= left:
                q = (q[0] + (w[0] - q[0]) / dd * left, q[1] + (w[1] - q[1]) / dd * left)
                break
            q, left = w, left - dd
        self.d.set_pose(n, q[0], q[1], 0.0)

    def events(self, kind: str | None = None) -> list[dict[str, object]]:
        rows = [json.loads(line) for line in (self.out / "events.jsonl").read_text().splitlines()]
        return [e for e in rows if kind is None or e["ev"] == kind]


def walker(e: Episode, ahead: float = 6.0, speed: float = 1.0) -> Peds:
    """A person on robot 0's plan, ahead metres away, walking towards it."""
    n = e.d.names[0]
    p = e.plans[n]
    s, i = 0.0, 0
    while s < ahead and i + 1 < len(p):
        s += math.dist(p[i], p[i + 1])
        i += 1
    x0, y0, _ = e.d.pose[n]
    length = math.dist((x0, y0), p[i])
    return {"w": (p[i][0], p[i][1], (x0 - p[i][0]) / length * speed, (y0 - p[i][1]) / length * speed)}


def map_of_dynamics(path: pathlib.Path, box: tuple[slice, slice], t0: int = 0) -> MapOfDynamics:
    rho = np.zeros((18, 52, 104))
    rho[t0:, box[0], box[1]] = 3.0
    np.savez(path, rho=rho, x_edges=np.arange(105) * 0.25, y_edges=np.arange(53) * 0.25, t_edges=np.arange(19) * 10.0)
    return MapOfDynamics.load(path)


def test_hungarian_finds_the_optimum_of_a_known_matrix() -> None:
    assert hungarian([[4, 1, 3], [2, 0, 5], [3, 2, 2]]) == [1, 0, 2]


def test_goals_follow_the_seed_and_smaller_fleets_take_the_first_goals(tmp_path: pathlib.Path) -> None:
    two, eight = Episode(tmp_path / "2", 2), Episode(tmp_path / "8", 8)
    assert two.d.goals == eight.d.goals[:2]
    assert Episode(tmp_path / "7", 2, seed=7).d.goals != two.d.goals
    goals = eight.d.goals
    assert all(g[0] > 15 for g in goals) and min(math.dist(a, b) for i, a in enumerate(goals) for b in goals[i + 1 :]) >= 1.5


def test_plan_through_the_side_door_passes_its_middle(tmp_path: pathlib.Path) -> None:
    d = Episode(tmp_path, 2).d
    at_door = [q for q in d.grid.descend(d.fields[(d.goals[0], "side_door")], 2.6, 4.5) if 12.0 <= q[0] <= 14.0]
    assert at_door and all(10.2 <= q[1] <= 11.0 for q in at_door)


def test_base_profile_brings_every_robot_to_its_goal_with_one_goal_each(tmp_path: pathlib.Path) -> None:
    e = Episode(tmp_path)
    e.run()
    assign = e.events("d1_assign")[0]["assign"]
    assert isinstance(assign, dict) and len({tuple(v) for v in assign.values()}) == 4
    assert all(r.phase == "done" for r in e.d.rob.values())
    assert all(v == 1 for v in e.goals_sent.values())
    assert not e.events("hold") and not e.events("reroute") and not e.events("goal_swap")
    for row in e.events("assign"):
        options = row["options"]
        assert isinstance(options, dict) and all({"J", "T", "H", "F", "W"} <= set(v) for v in options.values())


def test_queue_lets_one_robot_cross_a_door_at_a_time_in_order_of_arrival(tmp_path: pathlib.Path) -> None:
    e = Episode(tmp_path, profile="queue")
    e.run()
    assert e.cross_max <= 1
    assert all(r.phase == "done" for r in e.d.rob.values())
    assert e.events("to_slot") and e.events("at_slot") and not e.events("hold")
    for door in e.d.doors:
        assert [x["robot"] for x in e.events("join") if x["door"] == door] == [x["robot"] for x in e.events("enter") if x["door"] == door]
    moves: dict[object, set[tuple[float, ...]]] = {}
    for x in e.events("to_slot"):
        slot = x["slot"]
        assert isinstance(slot, list)
        moves.setdefault(x["robot"], set()).add(tuple(slot))
    assert all(len(v) == 1 for v in moves.values())


def test_queue_slots_stand_beside_the_door_in_a_zigzag(tmp_path: pathlib.Path) -> None:
    e = Episode(tmp_path, profile="queue")
    e.run()
    rows = e.events("slots")
    assert rows
    for row in rows:
        slots, door = row["slots"], row["door"]
        assert isinstance(slots, list) and isinstance(door, str)
        cy = e.d.doors[door].c[1]
        for p, q in zip(slots, slots[1:], strict=False):
            assert abs(p[1] - cy) >= 1.5 and abs(q[0] - p[0]) >= 1.1 and math.dist(p, q) >= 1.9


def test_a_door_is_held_until_its_robot_is_through_unless_a_timeout_is_set(tmp_path: pathlib.Path) -> None:
    e = Episode(tmp_path, 2, profile="queue")
    a, b = e.d.names
    r = e.d.rob[a]
    r.phase, r.door, r.t_cross, r.goal_xy = "cross", "main_door", -100.0, (20.0, 6.0)
    assert e.d.occupants("main_door", b) == [a]
    e.d.rcfg["capacity"]["occupy_timeout_s"] = 40.0
    assert e.d.occupants("main_door", b) == []
    assert e.events("door_timeout")


def test_people_change_the_door_only_with_human_input(tmp_path: pathlib.Path) -> None:
    block = {f"p{i}": (13.0 + 0.5 * (i - 2), 6.0 + 0.3 * (i % 3 - 1), 0.0, 0.0) for i in range(6)}
    blind = Episode(tmp_path / "none", 2, peds=block)
    blind.run(1)
    rows = blind.events("assign")
    for row in rows:
        options = row["options"]
        assert isinstance(options, dict) and all(v["H"] == 0 and v["people_at_door"] == 0 for v in options.values())
    seeing = Episode(tmp_path / "oracle", 2, info="oracle", peds=block)
    seeing.run(1)
    assert all(row["door"] == "side_door" for row in seeing.events("assign"))


def test_a_robot_waits_for_a_person_in_the_door_and_goes_after_max_wait(tmp_path: pathlib.Path) -> None:
    e = Episode(tmp_path, 1, profile="queue_yield", info="oracle", peds={"p": (13.0, 10.6, 0.0, 0.0), "q": (13.0, 6.0, 0.0, 0.0)})
    e.run(120)
    assert any(x.get("why") == "peds_in_door" for x in e.events())
    assert any(x["forced"] for x in e.events("enter"))


def test_conflict_monitor_yields_before_a_person_on_the_plan_and_goes_on_when_clear(tmp_path: pathlib.Path) -> None:
    e = Episode(tmp_path, 1, profile="queue_yield", info="oracle")
    e.tick()
    e.peds = walker(e)
    e.run(150)
    yields = e.events("yield")
    first = yields[0]["ttc"] if yields else 0.0
    assert isinstance(first, float) and first >= 1.0
    assert any(not x["forced"] for x in e.events("yield_end"))
    assert e.d.rob[e.d.names[0]].phase == "done"


def test_no_conflict_monitor_without_the_rule_or_human_input(tmp_path: pathlib.Path) -> None:
    e = Episode(tmp_path, 1, profile="queue")
    e.tick()
    e.peds = walker(e)
    e.run(10)
    assert not e.events("yield")


def test_conflict_monitor_never_stops_a_robot_inside_the_door_zone(tmp_path: pathlib.Path) -> None:
    e = Episode(tmp_path, 1, profile="queue_yield", info="oracle")
    e.tick()
    n = e.d.names[0]
    r = e.d.rob[n]
    r.phase, r.door, r.goal_xy, r.t_cross = "cross", "main_door", e.d.goals[0], e.t
    e.d.set_pose(n, 12.6, 6.0, 0.0)
    r.plan = [(12.6 + 0.2 * i, 6.0) for i in range(20)]
    e.peds = {"w": (14.0, 6.0, -1.0, 0.0)}
    e.t += 0.5
    e.tick()
    assert r.phase == "cross"
    e.d.set_pose(n, 10.0, 6.0, 0.0)
    r.plan = [(10.0 + 0.2 * i, 6.0) for i in range(20)]
    e.peds = {"w": (12.0, 6.0, -0.5, 0.0)}
    e.t += 0.5
    e.tick()
    assert r.phase == "yield" and r.resume == "cross"


def test_perceived_people_need_range_field_of_view_and_line_of_sight(tmp_path: pathlib.Path) -> None:
    e = Episode(tmp_path, 1, profile="queue_yield", info="perceived")
    d, n = e.d, e.d.names[0]
    d.set_pose(n, 6.0, 4.0, 0.0)
    d.set_peds({"front": (9.0, 4.0, 0.2, 0.0), "back": (3.5, 4.0, 0.0, 0.0), "far": (10.5, 4.0, 0.0, 0.0), "wall": (14.0, 4.0, 0.0, 0.0)})
    d.update_known()
    assert set(d.seen) == {"front"}
    d.set_peds({"front": (9.5, 4.0, 0.2, 0.0)})
    d.set_pose(n, 6.0, 4.0, math.pi)
    d.set_time(1.0)
    d.update_known()
    assert d.seen["front"][0] == 9.0
    d.set_time(2.5)
    d.update_known()
    assert "front" not in d.seen


def test_plan_around_bends_around_cells_the_map_marks_occupied(tmp_path: pathlib.Path) -> None:
    e = Episode(tmp_path, 1, profile="full", info="oracle")
    d, n = e.d, e.d.names[0]
    g = d.goals[0]
    x, y, _ = d.pose[n]
    static = d.grid.descend(d.fields[(g, "main_door")], x, y)
    mid = static[len(static) // 4]
    d.set_peds({"p": (mid[0], mid[1], 0.0, 0.0)})
    d.update_known()
    around = d.path(g, "main_door", x, y)

    def gap(plan: list[Point]) -> float:
        return min(math.dist(q, mid) for q in plan)

    assert gap(static) < 0.2 and gap(around) > 0.65
    assert any(12.5 <= q[0] <= 13.5 and 5.4 <= q[1] <= 6.6 for q in around)


def test_queue_stands_on_the_side_of_the_door_away_from_people(tmp_path: pathlib.Path) -> None:
    e = Episode(tmp_path, 3, profile="full", info="oracle")
    d = e.d
    before = d.choose_slots("main_door", -1.0)
    axis = before[0][1] - d.doors["main_door"].c[1]
    d.set_peds({f"p{i}": (before[0][0] + 0.8 * (i - 2) + 0.4, before[0][1] + 0.3 * (i % 2), 0.0, 0.0) for i in range(5)})
    d.update_known()
    after = d.choose_slots("main_door", -1.0)
    assert (after[0][1] - d.doors["main_door"].c[1]) * axis < 0


def test_flow_level_sees_no_person_only_the_map(tmp_path: pathlib.Path) -> None:
    mod = map_of_dynamics(tmp_path / "mod.npz", (slice(20, 24), slice(40, 44)), t0=3)
    e = Episode(tmp_path / "run", 1, profile="full", info="flow", mod=mod)
    d = e.d
    d.set_peds({"p": (6.0, 3.0, 0.5, 0.0)})
    d.update_known()
    v = d.view
    assert v is not None and d.seen == {} and d.known == {}
    assert v.occupied(0, 6.0, 3.0, inflate=False) and not v.occupied(0, 8.0, 3.0, inflate=False)
    assert v.occupied(4, 8.0, 3.0, inflate=False) and not v.occupied(4, 6.0, 3.0, inflate=False)
    assert v.occupied(40, 10.5, 5.5, inflate=False) and not v.occupied(0, 10.5, 5.5, inflate=False)


def test_flow_level_full_profile_brings_robots_to_their_goals(tmp_path: pathlib.Path) -> None:
    mod = map_of_dynamics(tmp_path / "mod.npz", (slice(20, 24), slice(40, 44)), t0=3)
    e = Episode(tmp_path / "run", 2, profile="full", info="flow", mod=mod)
    e.run(200)
    assert all(r.phase == "done" for r in e.d.rob.values())


def test_hata_bids_avoid_the_door_the_map_shows_busy_and_apply_no_door_rules(tmp_path: pathlib.Path) -> None:
    mod = map_of_dynamics(tmp_path / "mod.npz", (slice(18, 30), slice(40, 60)))
    e = Episode(tmp_path / "run", 4, profile="hata", info="mod", mod=mod, cost_file="hata_cost.yaml", peds={"p": (13.0, 10.6, 0.0, 0.0)})
    e.run(200)
    v = e.d.view
    assert v is not None and e.d.seen == {} and v.prob(0, 13.0, 10.6) < 0.05 and v.prob(0, 13.0, 6.0) > 0.65
    rows = e.events("assign")
    assert rows
    for row in rows:
        options = row["options"]
        assert isinstance(options, dict) and row["door"] == "side_door"
        assert options["main_door"]["sum_p"] > options["side_door"]["sum_p"]
        assert all(abs(o["J"] - (1.15 * o["len_m"] + 0.95 * o["sum_p"])) < 0.2 for o in options.values())
    assert not any(e.events(k) for k in ("to_slot", "hold", "yield", "join"))
    assert any(x["mode"] == "hungarian" for x in e.events("d1_assign"))


def test_samples_pair_a_fixed_length_input_with_the_decision(tmp_path: pathlib.Path) -> None:
    e = Episode(tmp_path, profile="queue")
    e.run()
    rows = [json.loads(line) for line in (tmp_path / "samples.jsonl").read_text().splitlines()]
    width = len(features.names(list(e.d.doors)))
    assert len(rows) > 50 and all(len(z["x"]) == width for z in rows)
    assert all(z["y"]["door"] in (0, 1) and z["y"]["action"] in (0, 1, 2) for z in rows)
    assert any(z["y"]["action"] == 1 for z in rows)


def test_rules_file_with_an_unknown_key_is_refused(tmp_path: pathlib.Path) -> None:
    text = (PKG / "configs" / "rules.yaml").read_text().replace("queue:                   #", "queu:                   #", 1)
    bad = tmp_path / "rules.yaml"
    bad.write_text(text)
    with pytest.raises(ValueError, match="queu"):
        load_rules(bad)


def test_flow_level_without_a_map_is_refused(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="map of dynamics"):
        FleetDispatcher(site(), cost(), rules(), names=["jackal"], info="flow", out=tmp_path)


def test_a_single_robot_is_named_like_arena_names_it() -> None:
    assert robot_names("jackal", 1) == ["jackal"]
    assert robot_names("jackal", 3) == ["jackal_0", "jackal_1", "jackal_2"]


def yielding(tmp_path: pathlib.Path, at: Point, people: Peds) -> Episode:
    """Two robots; robot 0 stands at `at` among `people` with a fresh people map, robot 1 parked out of the way."""
    e = Episode(tmp_path, 2, profile="queue_yield", info="oracle")
    e.tick()
    a, b = e.d.names
    e.d.set_pose(a, *at, 0.0)
    e.d.set_pose(b, 3.0, 1.5, 0.0)
    e.d.rob[b].phase, e.d.rob[b].plan = "done", []
    e.d.set_time(10.0)
    e.d.set_peds(people)
    e.d.update_known()
    return e


def test_step_aside_point_keeps_clear_of_another_robot_and_its_plan(tmp_path: pathlib.Path) -> None:
    e = yielding(tmp_path, (7.0, 6.0), {"p": (7.0, 6.0, 0.0, 0.0)})
    d = e.d
    a, b = d.names
    first = d.aside_point(a)
    assert first is not None
    d.set_pose(b, *first, 0.0)
    second = d.aside_point(a)
    assert second is not None and math.dist(second, first) >= 1.2
    r = d.rob[b]
    r.phase, r.plan = "approach", [(first[0] + (second[0] - first[0]) * i / 10, first[1] + (second[1] - first[1]) * i / 10) for i in range(11)]
    third = d.aside_point(a)
    assert third is None or min(math.dist(third, q) for q in r.plan) >= 1.2


def test_two_yielding_robots_do_not_get_the_same_step_aside_point(tmp_path: pathlib.Path) -> None:
    e = yielding(tmp_path, (7.15, 5.92), {"p": (7.6, 5.9, -0.5, 0.0)})
    d = e.d
    a, b = d.names
    d.set_pose(b, 8.39, 6.39, 0.0)
    d.rob[b].phase = "approach"
    d.rob[a].aside = d.aside_point(a)
    other = d.aside_point(b)
    assert d.rob[a].aside is not None
    assert other is None or math.dist(other, d.rob[a].aside) >= 1.2


def test_way_to_the_step_aside_point_never_brings_a_person_closer_than_standing_still(tmp_path: pathlib.Path) -> None:
    crowd = {"between": (7.0, 6.9, 0.0, 0.0), "coming": (8.2, 6.0, -0.5, 0.0)}
    e = yielding(tmp_path, (7.0, 6.0), crowd)
    d = e.d
    p = d.aside_point(d.names[0])
    assert p is not None and p[1] < 6.0
    length = math.dist((7.0, 6.0), p)
    for i in range(1, 21):
        s = length * i / 20
        t = s / 0.4
        q = (7.0 + (p[0] - 7.0) * s / length, 6.0 + (p[1] - 6.0) * s / length)
        for x, y, vx, vy in crowd.values():
            there = (x + vx * t, y + vy * t)
            assert math.dist(q, there) >= min(math.dist((7.0, 6.0), there), 1.0) - 0.06


def test_queue_slots_leave_the_side_where_people_were_seen(tmp_path: pathlib.Path) -> None:
    e = Episode(tmp_path, 3, profile="full", info="oracle")
    d = e.d
    view = d.view
    assert view is not None
    before = d.choose_slots("main_door", -1.0)
    stream = [view.idx(x / 4, y / 4) for x in range(24, 52) for y in range(int(before[0][1] * 4) - 8, int(before[0][1] * 4) + 8)]
    d.history.update({c: 30.0 for c in stream if c is not None})
    after = d.choose_slots("main_door", -1.0)
    cy = d.doors["main_door"].c[1]
    assert (after[0][1] - cy) * (before[0][1] - cy) < 0


def test_door_gate_never_lets_a_robot_into_a_door_with_people(tmp_path: pathlib.Path) -> None:
    """v13: a person standing in both doors keeps the robot out for good, no forced entry."""
    e = Episode(tmp_path, 1, profile="v13", info="oracle", peds={"p": (13.0, 10.6, 0.0, 0.0), "q": (13.0, 6.0, 0.0, 0.0)})
    e.run(120)
    assert any(x.get("why") == "peds_in_door" for x in e.events())
    assert not e.events("enter") and e.d.rob[e.d.names[0]].phase in ("to_slot", "queued")


def test_door_gate_lets_the_robot_in_once_the_person_has_left(tmp_path: pathlib.Path) -> None:
    e = Episode(tmp_path, 1, profile="v13", info="oracle", peds={"p": (13.0, 10.6, 0.0, 0.0), "q": (13.0, 6.0, 0.0, 0.0)})
    e.run(40)
    e.peds = {"p": (13.0, 10.6, 0.0, 0.0)}
    e.run(200)
    enter = e.events("enter")
    assert enter and enter[0]["door"] == "main_door" and not enter[0]["forced"] and e.events("trip_done")


def test_door_gate_sends_a_granted_robot_back_to_the_queue_when_people_come(tmp_path: pathlib.Path) -> None:
    e2 = Episode(tmp_path, 1, profile="v13", info="oracle")
    while not e2.events("enter") and e2.t < 100:
        e2.run(e2.t + TICK)
    n = e2.d.names[0]
    d = e2.d.rob[n].route_door
    assert e2.d.rob[n].phase == "cross" and e2.d.pose[n][0] < 12.0
    e2.peds = {"w": (17.0, e2.d.doors[d].c[1], -1.2, 0.0)}
    e2.run(e2.t + 1.0)
    assert e2.events("retreat") and e2.d.rob[n].phase in ("to_slot", "queued") and n not in e2.d.inside[d]


def test_door_gate_wait_term_is_large_while_a_person_stands_in_the_door(tmp_path: pathlib.Path) -> None:
    e = Episode(tmp_path, 1, profile="v13", info="oracle", peds={"q": (13.0, 6.0, 0.0, 0.0)})
    e.run(2)
    n = e.d.names[0]
    opts, _ = e.d.options(n, e.d.rob[n].target)
    assert opts["main_door"]["Wp"] > 30 and opts["side_door"]["Wp"] == 0


def watch_slots(e: Episode, until: float = 200.0) -> tuple[int, int, int]:
    """Run, counting ticks where two robots of one queue share a slot, ways to slots cross, or a robot is sent ahead of one ranked before it."""
    shared = crossed = jumped = 0
    while e.t < until and not e.d.finished:
        e.run(until=e.t + TICK)
        for d, Q in e.d.queue_of.items():
            on = [m for m in Q if e.d.rob[m].slot_xy is not None and e.d.rob[m].phase in ("to_slot", "queued")]
            pts = [e.d.rob[m].slot_xy for m in on]
            shared += len(pts) != len(set(pts))
            way = [m for m in on if e.d.rob[m].phase == "to_slot"]
            crossed += any(seg_cross(e.d.pose[a][:2], e.d.rob[a].slot_xy, e.d.pose[b][:2], e.d.rob[b].slot_xy) for i, a in enumerate(way) for b in way[i + 1 :])
            idx = [e.d.slot_set[d].index(e.d.rob[m].slot_xy) for m in on]
            jumped += idx != sorted(idx)
    return shared, crossed, jumped


def test_queue_of_four_at_one_door_gets_four_slots_without_crossing_ways(tmp_path: pathlib.Path) -> None:
    e = Episode(tmp_path, n=4, profile="queue")
    shared, crossed, jumped = watch_slots(e)
    assert all(r.phase == "done" for r in e.d.rob.values())
    assert (shared, crossed, jumped) == (0, 0, 0)
    assert all(len(x["slots"]) == 4 for x in e.events("slots"))  # one slot per robot; with N - 1 the fourth robot shared one


def test_a_late_robot_never_takes_the_head_slot_left_free_ahead_of_the_queue(tmp_path: pathlib.Path) -> None:
    e = Episode(tmp_path, n=4, profile="queue")
    e.d.rcfg["queue"]["swap_gain_m"] = 1e9  # only slot_index decides
    _, _, jumped = watch_slots(e)
    assert jumped == 0


def test_four_robots_arriving_side_by_side_at_one_door_queue_without_crossing_ways(tmp_path: pathlib.Path) -> None:
    """The v1.3 smoke run (seed 6, N = 4) at 12.75 s: four robots in a column across the way to side_door, all routed there."""
    e = Episode(tmp_path, n=4, profile="queue")
    e.tick()
    at = {"jackal_0": (6.57, 7.68), "jackal_1": (6.76, 8.95), "jackal_2": (6.40, 9.56), "jackal_3": (6.98, 10.23)}
    goal = {"jackal_0": (20.59, 5.38), "jackal_1": (18.66, 6.02), "jackal_2": (21.23, 8.29), "jackal_3": (19.08, 11.01)}
    for n, (x, y) in at.items():
        e.d.set_pose(n, x, y, 0.0)
        e.d.set_plan(n, goal[n], "side_door", e.d.path(goal[n], "side_door", x, y))
        e.plans[n] = e.d.rob[n].plan
    e.d.rules["reassign_door"] = False
    shared, crossed, jumped = watch_slots(e, until=60.0)
    assert e.events("to_slot")
    assert (shared, crossed, jumped) == (0, 0, 0)


def directional_map(path: pathlib.Path, bin_: int, box: tuple[slice, slice] = (slice(0, 52), slice(0, 104)), speed: float = 1.2) -> MapOfDynamics:
    """A stream of 0.5 ped/m^2 walking towards direction bin_ (of 8, 0 = +x) everywhere in box."""
    rho_dir = np.zeros((18, 52, 104, 8))
    rho_dir[:, box[0], box[1], bin_] = 0.5
    spd = np.where(rho_dir.sum(-1) > 0, speed, np.nan)
    np.savez(path, rho=rho_dir.sum(-1), rho_dir=rho_dir, spd=spd, x_edges=np.arange(105) * 0.25, y_edges=np.arange(53) * 0.25, t_edges=np.arange(19) * 10.0)
    return MapOfDynamics.load(path)


def test_walking_against_the_stream_meets_more_people_than_walking_with_it(tmp_path: pathlib.Path) -> None:
    mod = directional_map(tmp_path / "mod.npz", 4)  # stream walks west
    east, west, north = (mod.meet_rate(0.0, 10.0, 5.0, vx, vy, 2.0) for vx, vy in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0)))
    assert east > north > west >= 0.0
    assert east == pytest.approx(0.5 * 2.0 * 2.2) and west == pytest.approx(0.5 * 2.0 * 0.2)
    assert mod.heading(0.0, 10.0, 5.0) == pytest.approx((-1.0, 0.0), abs=1e-9)
    assert mod.heading(0.0, 30.0, 5.0) is None


def test_dmod_forecast_turns_a_person_onto_the_stream_of_the_map(tmp_path: pathlib.Path) -> None:
    mod = directional_map(tmp_path / "mod.npz", 2)  # stream walks north (+y)
    e = Episode(tmp_path / "run", 1, profile="full", info="dmod", mod=mod)
    v = e.d.view
    assert v is not None
    tr = v.track(6.0, 3.0, 1.0, 0.0, 0.0, 0.0)  # the person walks east now
    x4, y4 = tr[4]
    assert y4 - 3.0 > x4 - 6.0 > 0.0  # it went north more than east
    assert math.dist((6.0, 3.0), tr[1]) == pytest.approx(1.0, abs=0.05)  # at its own speed


def test_dmod_prices_a_plan_against_the_stream_above_one_with_it(tmp_path: pathlib.Path) -> None:
    mod = directional_map(tmp_path / "mod.npz", 4)
    e = Episode(tmp_path / "run", 1, profile="full", info="dmod", mod=mod)
    against = [(4.0 + 0.25 * i, 6.0) for i in range(30)]
    along = list(reversed(against))
    v = e.d.cost["time"]["v_nom"]
    ratio = e.d.flow_encounters(against) / e.d.flow_encounters(along)
    assert ratio == pytest.approx((1.2 + v) / abs(1.2 - v), rel=1e-6)  # relative speeds; people walking along still overtake a slow robot


def test_dmod_needs_a_directional_map_and_brings_robots_to_their_goals(tmp_path: pathlib.Path) -> None:
    flat = map_of_dynamics(tmp_path / "flat.npz", (slice(20, 24), slice(40, 44)))
    with pytest.raises(ValueError, match="directional"):
        Episode(tmp_path / "a", 1, profile="full", info="dmod", mod=flat)
    e = Episode(tmp_path / "b", 2, profile="full", info="dmod", mod=directional_map(tmp_path / "mod.npz", 4, (slice(20, 28), slice(40, 60))))
    e.run(200)
    assert all(r.phase == "done" for r in e.d.rob.values())


def test_dmod_layers_keep_the_map_for_the_door_gate_and_h_counts_known_people_only(tmp_path: pathlib.Path) -> None:
    mod = directional_map(tmp_path / "mod.npz", 4, (slice(20, 24), slice(40, 44)))
    e = Episode(tmp_path / "run", 1, profile="v13", info="dmod", mod=mod)
    d = e.d
    d.set_peds({})
    d.update_known()
    v = d.view
    assert v is not None and v.live_layers is not None
    assert v.occupied(40, 10.5, 5.5, inflate=False)  # the map's stream, seen by gate, queue and slots
    assert not v.occupied(40, 10.5, 5.5, inflate=False, live=True)  # but not by H: nobody known
