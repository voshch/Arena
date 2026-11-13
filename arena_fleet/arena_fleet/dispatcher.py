import json
import math
import pathlib
import random
import typing
from collections.abc import Sequence

import attrs

from arena_fleet import features
from arena_fleet.config import RULES, CostConfig, Profile, RulesConfig
from arena_fleet.geometry import Funnel, Grid, Point, grown, hungarian, plan_len, seg_cross
from arena_fleet.site import Site
from arena_fleet.view import MapOfDynamics, PeopleView

Info = typing.Literal["none", "oracle", "perceived", "flow", "mod"]
INFOS: tuple[Info, ...] = typing.get_args(Info)
Pose2 = tuple[float, float, float]
Terms = dict[str, float]

VERSION = "1.2"
WAITING = ("approach", "hold", "cross", "to_slot", "queued", "yield")
_FIELDS: dict[tuple[int, float, float, str], list[float]] = {}


@attrs.frozen
class PublishPlan:
    robot: str
    path: tuple[Point, ...]


@attrs.frozen
class SendGoal:
    robot: str
    goal: Point
    tolerance: float


@attrs.frozen
class Stop:
    robot: str


Command = PublishPlan | SendGoal | Stop


@attrs.define
class LeafState:
    c: Point
    span: float
    prog: float = 0.0


@attrs.define
class RobotState:
    """What the dispatcher tracks of one robot."""

    phase: str = "init"
    goal_xy: Point | None = None
    door: str | None = None
    door_used: str | None = None
    plan: list[Point] = attrs.field(factory=list)
    goal: Point | None = None
    t_at: float | None = None
    why: str | None = None
    t0: float | None = None
    t_done: float | None = None
    t_switch: float = -1e9
    t_goal_sw: float = -1e9
    t_eval: float = -1e9
    t_pub: float = -1e9
    mark: Pose2 | None = None
    track: Terms = attrs.field(factory=dict)
    diag: dict[str, object] | None = None
    slot_xy: Point | None = None
    rank: int | None = None
    t_head: float | None = None
    t_cross: float | None = None
    resume: str = "approach"
    y_t0: float = 0.0
    aside: Point | None = None
    t_clear: float | None = None
    no_yield: float = -1e9
    t_aside: float = -1e9
    ttc: float | None = None
    slot_i: int | None = None
    slot_door: str | None = None
    stuck: bool = False
    timed_out: bool = False

    @property
    def target(self) -> Point:
        if self.goal_xy is None:
            raise RuntimeError("robot has no goal yet")
        return self.goal_xy

    @property
    def route_door(self) -> str:
        d = self.door or self.door_used
        if d is None:
            raise RuntimeError("robot has no door")
        return d


def robot_names(robot: str, n: int) -> list[str]:
    """Arena names a fleet <robot>_0..<robot>_{n-1} and a single robot <robot>."""
    return [robot] if n == 1 else [f"{robot}_{i}" for i in range(n)]


class FleetDispatcher:
    """Assigns goals, one plan per robot through a door, and applies the profile's rules every tick; ROS-free."""

    def __init__(self, site: Site, cost: CostConfig, rules: RulesConfig, *, names: Sequence[str], info: Info, profile: str = "", seed: int = 0, out: pathlib.Path, duration: float = 180.0, mod: MapOfDynamics | None = None, samples: bool = True) -> None:
        name = profile or rules["default_profile"]
        if name not in rules["profiles"]:
            raise ValueError(f"profile {name!r} not in {sorted(rules['profiles'])}")
        if info in ("flow", "mod") and mod is None:
            raise ValueError(f"information level {info} needs a map of dynamics")
        if len(site.starts) < len(names):
            raise ValueError(f"{len(names)} robots but the scenario parks {len(site.starts)}")
        self.site, self.cost, self.rcfg, self.info, self.duration = site, cost, rules, info, duration
        self.profile, self.rules = name, rules["profiles"][name]
        check_profile(self.rules)
        self.w = cost["weights"]
        self.names = list(names)
        self.starts = site.starts[: len(names)]
        self.walls = site.walls
        self.doors = site.doors
        self.grid = Grid(self.walls)
        self.door_open = {leaf.name: LeafState(leaf.c, leaf.span) for leaf in site.leaves}
        self.hata = cost.get("mode") == "hata"
        self.view = None if info == "none" else PeopleView(cost, self.walls, mod)
        self.pose: dict[str, Pose2] = {}
        self.peds: dict[str, tuple[float, float, float, float]] = {}
        self.seen: dict[str, tuple[float, float, float, float, float]] = {}
        self.known: dict[str, tuple[float, float, float]] = {}
        self.off: Point | None = None
        self.t_start = 0.0
        self.t_view = -1e9
        self.inside: dict[str, dict[str, float]] = {d: {} for d in self.doors}
        self.queue_of: dict[str, list[str]] = {d: [] for d in self.doors}
        self.slot_set: dict[str, list[Point]] = {}
        self.slot_fields: dict[Point, list[float]] = {}
        self.history: dict[int, float] = {}
        self.t_hist = -1e9
        self.rob = {n: RobotState() for n in self.names}
        self.trips_done = dict.fromkeys(self.names, 0)
        self.commands: list[Command] = []
        self.started = self.finished = False
        self.t_gs, self.t_done_all, self.t_samp = -1e9, None, -1e9
        self.last_xy: dict[str, tuple[float, float, float]] = {}
        self._now = 0.0
        out.mkdir(parents=True, exist_ok=True)
        self.ev = (out / "events.jsonl").open("a", buffering=1)
        self.st = (out / "state.jsonl").open("a", buffering=1)
        self.heat_f = (out / "heat.jsonl").open("a", buffering=1) if self.view else None
        self.samp_f = (out / "samples.jsonl").open("a", buffering=1) if samples else None
        self.goals = self.draw_goals(seed, len(self.names))
        self.fields: dict[tuple[Point, str], list[float]] = {}
        key0 = hash(self.walls)
        pl = cost["plan"]
        for g in self.goals:
            for d, door in self.doors.items():
                key = (key0, round(g[0], 2), round(g[1], 2), d)
                if key not in _FIELDS:
                    closed = [grown(o.polygon, 0.0) for k, o in self.doors.items() if k != d]
                    _FIELDS[key] = self.grid.field(g, closed, (*door.c, pl["funnel_len_m"], pl["funnel_k"]))
                self.fields[(g, d)] = _FIELDS[key]
        (out / "dispatch.json").write_text(json.dumps({"profile": name, "rules": self.rules, "info": info, "cost": cost, "rules_params": rules, "goals": self.goals, "seed": seed, "robots": self.names, "version": VERSION}, indent=1))

    def close(self) -> None:
        for f in (self.ev, self.st, self.heat_f, self.samp_f):
            if f is not None:
                f.close()

    def now(self) -> float:
        return self._now

    def t_rel(self) -> float:
        return self._now - self.t_start

    def log(self, ev: str, **kw: object) -> None:
        self.ev.write(json.dumps({"t": round(self.t_rel(), 2), "ev": ev, **kw}) + "\n")

    def draw_goals(self, seed: int, n: int) -> list[Point]:
        """Eight goals from the seed in the goal zone, the first n used, so fleets of 2, 4 and 8 share goals."""
        t = self.cost["task"]
        (x0, x1), (y0, y1) = t["goal_zone"]["x"], t["goal_zone"]["y"]
        rng = random.Random(f"fleetbench-goals-{seed}")
        out: list[Point] = []
        for _ in range(20000):
            p = (round(rng.uniform(x0, x1), 2), round(rng.uniform(y0, y1), 2))
            if self.grid.clearance(p) >= t["clearance_m"] and all(math.dist(p, q) >= t["min_separation_m"] for q in out):
                out.append(p)
                if len(out) == 8:
                    break
        if len(out) < n:
            raise ValueError("cannot place the goals")
        return out[:n]

    # inputs

    def set_time(self, now: float) -> None:
        self._now = now

    def set_pose(self, n: str, x: float, y: float, yaw: float) -> None:
        """Robot pose in the map frame."""
        if self.off is not None:
            self.pose[n] = (x - self.off[0], y - self.off[1], yaw)

    def set_peds(self, peds: dict[str, tuple[float, float, float, float]]) -> None:
        """Every pedestrian (x, y, vx, vy) in the map frame."""
        if self.off is not None:
            self.peds = {k: (x - self.off[0], y - self.off[1], vx, vy) for k, (x, y, vx, vy) in peds.items()}

    def set_leaves(self, leaves: Sequence[tuple[str, float, float]]) -> None:
        """Door leaf positions (entity name, x, y) in the map frame."""
        if self.off is None:
            return
        for name, x, y in leaves:
            parts = name.split("/")
            d = parts[1] if len(parts) > 1 else name
            if d in self.door_open:
                g = self.door_open[d]
                g.prog = max(0.0, min(1.0, math.dist((x - self.off[0], y - self.off[1]), g.c) / g.span))

    def try_setup(self, raw: dict[str, Pose2]) -> bool:
        """Map offset from the parked robots at their scenario starts; False until all stand there."""
        if any(n not in raw for n in self.names):
            return False
        offs = [(raw[n][0] - s[0], raw[n][1] - s[1]) for n, s in zip(self.names, self.starts, strict=True)]
        mx, my = sum(o[0] for o in offs) / len(offs), sum(o[1] for o in offs) / len(offs)
        spread = max(math.hypot(o[0] - mx, o[1] - my) for o in offs)
        if spread > 0.3:
            return False
        self.off, self.t_start = (mx, my), self._now
        self.log("setup", offset=[round(mx, 3), round(my, 3)], spread=round(spread, 3), sim_t0=round(self.t_start, 3), profile=self.profile, info=self.info)
        for n in self.names:
            self.set_pose(n, *raw[n])
        return True

    # what the dispatcher knows about people

    def update_known(self) -> None:
        now = self._now
        if self.info == "oracle":
            self.seen = {k: (*p, now) for k, p in self.peds.items()}
        elif self.info == "perceived":
            pc = self.cost["perception"]
            for k, (x, y, vx, vy) in self.peds.items():
                for n in self.names:
                    if n not in self.pose:
                        continue
                    rx, ry, th = self.pose[n]
                    off = (math.atan2(y - ry, x - rx) - th + math.pi) % (2 * math.pi) - math.pi
                    if math.hypot(x - rx, y - ry) <= pc["range_m"] and abs(math.degrees(off)) <= pc["fov_deg"] / 2 and not any(seg_cross((rx, ry), (x, y), a, b) for a, b in self.walls):
                        self.seen[k] = (x, y, vx, vy, now)
                        break
            self.seen = {k: v for k, v in self.seen.items() if now - v[4] <= pc["memory_s"]}
        else:
            self.seen = {}
        self.known = {k: (v[0], v[1], v[4]) for k, v in self.seen.items()}
        if self.view is None or self.off is None:
            return
        if self.t_hist > -1e9:
            for c in self.view.layer(0):
                self.history[c] = self.history.get(c, 0.0) + now - self.t_hist
        self.t_hist = now
        period = self.cost["flow"]["period_s"] if self.info in ("flow", "mod") else self.cost["people_view"]["period_s"]
        if now - self.t_view < period:
            return
        self.t_view = now
        if self.info == "flow":
            self.view.update_flow([(*p, 0.0) for p in self.peds.values()], self.t_rel())
        elif self.info == "mod":
            self.view.update_flow([], self.t_rel(), live=False)
        else:
            self.view.update_people([(v[0], v[1], v[2], v[3], now - v[4]) for v in self.seen.values()])
        if self.heat_f:
            self.heat_f.write(json.dumps({"t": round(self.t_rel(), 2), "heat": {str(k): v for k, v in self.view.heat.items()}}) + "\n")

    def door_is_open(self, d: str) -> bool:
        return d not in self.door_open or self.door_open[d].prog >= 0.9

    # predictions

    def walk(self, path: Sequence[Point], horizon: float) -> list[tuple[float, float, float, float]]:
        """(x, y, seconds ahead, step length) along a plan at v_nom up to horizon."""
        v, out, t = self.cost["time"]["v_nom"], [], 0.0
        for i, p in enumerate(path):
            ds = math.dist(path[i - 1], p) if i else 0.0
            t += ds / v
            if t > horizon:
                break
            out.append((p[0], p[1], t, ds))
        return out

    def encounters(self, path: Sequence[Point]) -> float:
        """H: plan length through occupied cells over one person's diameter."""
        if self.view is None:
            return 0.0
        view = self.view
        hit = sum(ds for x, y, t, ds in self.walk(path, self.cost["encounter"]["horizon_s"]) if view.occupied(t, x, y))
        return hit / (2 * view.person_r)

    def people_at_door(self, d: str, t_arrive: float) -> float:
        return self.view.count(t_arrive, *self.doors[d].c, self.cost["facility"]["crowd_r_m"]) if self.view else 0.0

    def funnel(self, d: str) -> Funnel:
        pl = self.cost["plan"]
        return (*self.doors[d].c, pl["funnel_len_m"], pl["funnel_k"])

    def path(self, goal: Point, d: str, x: float, y: float, around: bool = True) -> list[Point]:
        """Plan to the goal through door d, around people when plan_around is on and people are known."""
        fld = self.fields[(goal, d)]
        if around and self.rules["plan_around"] and self.view is not None:
            pl = self.cost["plan"]
            return self.grid.route(fld, x, y, self.view, self.cost["time"]["v_nom"], pl["occupied_cost_per_m"], self.funnel(d), pl["around_horizon_s"])
        return self.grid.descend(fld, x, y)

    def arrival(self, path: Sequence[Point], d: str) -> float:
        c = self.doors[d].c
        s, best, bd = 0.0, 0.0, math.inf
        for i, p in enumerate(path):
            if i:
                s += math.dist(path[i - 1], p)
            dd = math.dist(p, c)
            if dd < bd:
                bd, best = dd, s
        return best / self.cost["time"]["v_nom"]

    def queue_wait(self, n: str, d: str, t_arrive: float) -> float:
        """W: seconds robot n would stand at door d behind the other robots routed through it."""
        cross = self.cost["wait"]["cross_m"] / self.cost["time"]["v_nom"]
        others = [self.arrival(r.plan, d) if r.phase == "approach" else 0.0 for m, r in self.rob.items() if m != n and r.door == d and r.phase in WAITING and r.plan]
        free = 0.0
        for t in sorted(others):
            if t > t_arrive:
                break
            free = max(free, t) + cross
        return max(0.0, free - t_arrive)

    def terms(self, n: str, goal: Point, d: str, ignore_robots: bool = False, around: bool = True, path: Sequence[Point] | None = None) -> tuple[Terms | None, list[Point]]:
        """Every cost term of robot n going to goal through door d, and the plan."""
        c, w = self.cost, self.w
        x, y, _ = self.pose[n]
        plan = list(path) if path else self.path(goal, d, x, y, around)
        if not plan:
            return None, []
        length = plan_len(plan)
        ta = self.arrival(plan, d)
        if self.hata:
            return self.hata_terms(plan, ta), plan
        T = length / c["time"]["v_nom"]
        H = self.encounters(plan)
        f = c["facility"]
        rs = 0 if ignore_robots else sum(1 for m, r in self.rob.items() if m != n and r.door == d and r.phase in WAITING)
        pd = self.people_at_door(d, ta)
        op = f["open_door_s"] if not self.door_is_open(d) else 0.0
        F = op + f["robot_same_door_s"] * rs + f["person_at_door_s"] * pd
        W = 0.0 if ignore_robots else self.queue_wait(n, d, ta)
        J = w["time"] * T + w["encounter"] * H + w["facility"] * F + w["wait"] * W
        terms = {"J": round(J, 2), "T": round(T, 1), "H": round(H, 2), "F": round(F, 1), "W": round(W, 1), "len_m": round(length, 1), "t_door": round(ta, 1), "robots_same_door": rs, "people_at_door": round(pd, 1), "door_closed_s": op}
        return terms, plan

    def hata_terms(self, path: Sequence[Point], ta: float) -> Terms:
        """HATA bid (arXiv 2508.19731, eq. 3): sum over cells of w0 * d_k + w1 * p_k, no door or robot terms."""
        h = self.cost.get("hata")
        if h is None:
            raise ValueError("cost mode hata needs a hata section")
        length = prob = 0.0
        for x, y, t, ds in self.walk(path, 1e9):
            length += ds
            if self.view is not None and ds > 0:
                prob += self.view.prob(t, x, y) * ds / self.view.cell
        J = h["w0"] * length + h["w1"] * prob
        return {"J": round(J, 2), "T": round(length / self.cost["time"]["v_nom"], 1), "H": round(prob, 2), "F": 0.0, "W": 0.0, "len_m": round(length, 1), "t_door": round(ta, 1), "sum_p": round(prob, 2), "robots_same_door": 0, "people_at_door": 0.0, "door_closed_s": 0.0}

    def options(self, n: str, goal: Point, ignore_robots: bool = False, around: bool = True) -> tuple[dict[str, Terms], dict[str, list[Point]]]:
        out: dict[str, Terms] = {}
        paths: dict[str, list[Point]] = {}
        for d in self.doors:
            t, p = self.terms(n, goal, d, ignore_robots, around)
            if t:
                out[d], paths[d] = t, p
        return out, paths

    # commands

    def publish_plan(self, n: str, path: Sequence[Point]) -> None:
        self.commands.append(PublishPlan(n, tuple(path)))

    def send(self, n: str, p: Point, tol: float) -> None:
        self.rob[n].goal = p
        self.commands.append(SendGoal(n, p, tol))

    def stop(self, n: str) -> None:
        self.rob[n].goal = None
        self.commands.append(Stop(n))

    def drain(self) -> list[Command]:
        out, self.commands = self.commands, []
        return out

    def go(self, n: str) -> None:
        r = self.rob[n]
        self.publish_plan(n, r.plan)
        self.send(n, r.target, self.rcfg["episode"]["goal_tol_m"])

    def set_plan(self, n: str, goal: Point, door: str, path: list[Point]) -> None:
        r = self.rob[n]
        r.goal_xy, r.door, r.plan, r.phase = goal, door, path, "approach"

    # goal assignment and first door

    def start_episode(self) -> None:
        """Goals by minimum total J (Hungarian), then doors one robot at a time, each seeing the choices before it."""
        for n in self.names:
            self.rob[n].t0 = self._now
        C: list[list[float]] = []
        for n in self.names:
            C.append([min(v["J"] for v in self.options(n, g, ignore_robots=True, around=False)[0].values()) for g in self.goals])
        mode = self.cost["task"]["assign"]
        assign = hungarian(C) if mode == "hungarian" else list(range(len(self.names)))
        self.log("d1_assign", mode=mode, goals=[list(g) for g in self.goals], cost=C, assign={n: list(self.goals[j]) for n, j in zip(self.names, assign, strict=True)}, total=round(sum(C[i][j] for i, j in enumerate(assign)), 1))
        for i in sorted(range(len(self.names)), key=lambda i: C[i][assign[i]]):
            n, g = self.names[i], self.goals[assign[i]]
            opts, paths = self.options(n, g)
            d = min(opts, key=lambda k: opts[k]["J"])
            self.set_plan(n, g, d, paths[d])
            self.log("assign", robot=n, goal=list(g), door=d, options=opts)
            self.go(n)
        self.started = True

    # door capacity

    def past_door(self, n: str) -> bool:
        r = self.rob[n]
        cx = self.doors[r.route_door].c[0]
        x = self.pose[n][0]
        return (x - cx) * (r.target[0] - cx) > 0 and abs(x - cx) >= self.rcfg["capacity"]["through_m"]

    def occupants(self, d: str, skip: str) -> list[str]:
        """Robots holding door d; with occupy_timeout_s, one holding it longer no longer counts."""
        out: list[str] = []
        lim = self.rcfg["capacity"]["occupy_timeout_s"]
        for m, q in self.rob.items():
            if m != skip and q.door == d and (q.phase == "cross" or (q.phase == "yield" and q.resume == "cross")):
                t_cross = q.t_cross if q.t_cross is not None else self._now
                if lim is None or self._now - t_cross <= lim:
                    out.append(m)
                elif not q.timed_out:
                    q.timed_out = True
                    self.log("door_timeout", robot=m, door=d)
        return out

    def people_in_door(self, d: str) -> bool:
        if self.view is None:
            return False
        view = self.view
        y = self.rcfg["yield_people"]
        x0, x1, y0, y1 = grown(self.doors[d].polygon, y["margin_m"])
        c = view.cell
        xs = [x0 + i * c for i in range(int((x1 - x0) / c) + 1)]
        ys = [y0 + j * c for j in range(int((y1 - y0) / c) + 1)]
        return any(view.occupied(t, xx, yy, inflate=False) for t in range(0, int(y["lookahead_s"]) + 1) for xx in xs for yy in ys)

    def entry_reason(self, n: str, d: str) -> str:
        """Why robot n may not enter door d now, empty when it may."""
        R, r = self.rules, self.rob[n]
        if R["capacity"] and len(self.occupants(d, n)) >= self.rcfg["capacity"]["k"]:
            return "door_busy"
        if R["yield_people"] and self.view is not None:
            t_ref = r.t_head if r.t_head is not None else r.t_at
            waited = self._now - t_ref if t_ref is not None else 0.0
            if waited <= self.rcfg["yield_people"]["max_wait_s"] and self.people_in_door(d):
                return "peds_in_door"
        return ""

    # queue slots beside the door

    def slot_score(self, slots: Sequence[Point]) -> float:
        """Occupied slot-seconds over the next 10 s, plus the seconds the slots were seen occupied so far."""
        if self.view is None:
            return 0.0
        view = self.view
        ahead = sum(view.occupied(t, x, y) for x, y in slots[:3] for t in range(0, 11, 2))
        seen = sum(self.history.get(c, 0.0) for c in (view.idx(x, y) for x, y in slots[:3]) if c is not None)
        return float(ahead) + self.rcfg["queue"]["history_weight"] * seen

    def choose_slots(self, d: str, side: float) -> list[Point]:
        """Zig-zag slot column beside the door axis, clear of walls and other queues; by people when slot_by_people is on."""
        q, (cx, cy) = self.rcfg["queue"], self.doors[d].c
        k = max(len(self.names) - 1, 1)
        best: list[Point] | None = None
        best_s = math.inf
        taken = [p for dd, sl in self.slot_set.items() if dd != d and self.queue_of.get(dd) for p in sl]
        sep = q["other_queue_m"]
        for lat in (-q["lateral_m"], q["lateral_m"]):
            for head in q["head_m"]:
                out = math.copysign(q["stagger_m"], lat)
                sl = [(cx + side * (head + i * q["spacing_m"]), cy + lat + (i % 2) * out) for i in range(k)]
                if all(self.grid.clearance(p) >= q["clearance_m"] for p in sl) and all(math.dist(p, o) >= sep for p in sl for o in taken):
                    score = self.slot_score(sl)
                    if not self.rules["slot_by_people"]:
                        self.log("slots", door=d, slots=sl, chosen_by="fixed")
                        return sl
                    if score < best_s:
                        best, best_s = sl, score
        if best is None:
            raise RuntimeError(f"no valid queue slots at {d}")
        self.log("slots", door=d, slots=best, chosen_by="people", score=best_s)
        return best

    def slot_index(self, n: str, d: str, rank: int) -> int:
        """keep_slot: the nearest slot free when first sent, kept until its turn; else its queue rank."""
        r, last = self.rob[n], len(self.slot_set[d]) - 1
        if not self.rcfg["queue"]["keep_slot"]:
            return min(rank, last)
        if r.slot_i is None or r.slot_door != d:
            used = {self.rob[m].slot_i for m in self.queue_of[d] if m != n and self.rob[m].slot_door == d}
            r.slot_i = next((i for i in range(last + 1) if i not in used), last)
            r.slot_door = d
        return r.slot_i

    def slot_path(self, p: Point, x: float, y: float) -> list[Point]:
        key = (round(p[0], 2), round(p[1], 2))
        if key not in self.slot_fields:
            self.slot_fields[key] = self.grid.field(p, [grown(o.polygon, 0.0) for o in self.doors.values()])
        fld = self.slot_fields[key]
        if self.rules["plan_around"] and self.view is not None:
            pl = self.cost["plan"]
            return self.grid.route(fld, x, y, self.view, self.cost["time"]["v_nom"], pl["occupied_cost_per_m"], None, pl["around_horizon_s"])
        return self.grid.descend(fld, x, y)

    def to_slot(self, n: str, d: str, rank: int, why: str) -> None:
        r, q = self.rob[n], self.rcfg["queue"]
        x, y, _ = self.pose[n]
        p = self.slot_set[d][self.slot_index(n, d, rank)]
        waited = round(self._now - (r.t_at if r.t_at is not None else self._now), 1)
        if r.phase in ("to_slot", "queued") and r.slot_xy == p:
            r.why, r.rank, r.diag = why, rank, {"why": why, "waited_s": waited}
            if r.phase == "to_slot" and math.dist((x, y), p) <= q["slot_tol_m"]:
                r.phase = "queued"
                self.stop(n)
                self.log("at_slot", robot=n, door=d, rank=rank, slot=list(p))
            return
        r.phase, r.slot_xy, r.rank, r.why, r.plan = "to_slot", p, rank, why, self.slot_path(p, x, y)
        r.diag = {"why": why, "waited_s": waited}
        self.publish_plan(n, r.plan)
        self.send(n, p, q["slot_tol_m"])
        self.log("to_slot", robot=n, door=d, rank=rank, slot=list(p), why=why)

    def resume_goal(self, n: str) -> None:
        """Back to the real goal with a plan through the door."""
        r = self.rob[n]
        x, y, _ = self.pose[n]
        r.plan = self.path(r.target, r.route_door, x, y) or r.plan
        r.slot_xy, r.rank, r.why, r.diag, r.aside, r.slot_i, r.slot_door = None, None, None, None, None, None, None
        self.go(n)

    def grant(self, n: str, d: str, from_slot: bool) -> None:
        r = self.rob[n]
        waited = round(self._now - (r.t_at if r.t_at is not None else self._now), 1)
        forced = bool(self.rules["yield_people"] and r.t_head is not None and self._now - r.t_head > self.rcfg["yield_people"]["max_wait_s"])
        if n in self.queue_of[d]:
            self.queue_of[d].remove(n)
        r.phase, r.t_cross, r.slot_i, r.slot_door = "cross", self._now, None, None
        self.inside[d][n] = self._now
        self.log("enter", robot=n, door=d, waited=waited, forced=forced)
        if from_slot:
            self.resume_goal(n)

    def queue_step(self, n: str, d: str, dd: float) -> None:
        """One tick of the queue at door d, robots in order of arrival at join_m."""
        r, rc = self.rob[n], self.rcfg
        Q = self.queue_of[d]
        if n not in Q:
            if dd > rc["queue"]["join_m"]:
                return
            if not Q:
                self.slot_set[d] = self.choose_slots(d, -1.0 if self.pose[n][0] < self.doors[d].c[0] else 1.0)
            Q.append(n)
            r.t_at, r.t_head = self._now, None
            self.log("join", robot=n, door=d, pos=len(Q))
        ready = [m for m in Q if self.rob[m].phase != "yield"]
        if n not in ready:
            return
        rank = ready.index(n)
        if rank == 0 and r.t_head is None:
            r.t_head = self._now
        why = self.entry_reason(n, d) if rank == 0 else "queue"
        if not why:
            at_slot = r.phase in ("to_slot", "queued")
            if dd <= rc["capacity"]["hold_m"] or r.phase == "queued":
                self.grant(n, d, at_slot)
            elif at_slot:
                r.phase = "approach"
                self.resume_goal(n)
            return
        self.to_slot(n, d, rank, why)

    # conflict monitor: stop before meeting a person

    def conflict(self, n: str, path: Sequence[Point] | None = None) -> float | None:
        """Seconds until a person is predicted on the plan, or at a standing robot; None when clear."""
        if self.view is None:
            return None
        view = self.view
        y, r = self.rcfg["yield_people"], self.rob[n]
        x, yy, _ = self.pose[n]
        hor = y["hold_ttc_s"]
        if r.phase == "queued" or (r.phase == "yield" and path is None):
            return next((float(t) for t in range(int(hor) + 1) if view.occupied(t, x, yy)), None)
        p = path if path is not None else r.plan
        if not p:
            return None
        near = min(range(len(p)), key=lambda i: math.dist((x, yy), p[i]))
        for px, py, t, _ds in self.walk(p[near:], hor):
            if view.occupied(t, px, py):
                return t
        return None

    def in_no_return(self, n: str) -> bool:
        r = self.rob[n]
        d = r.door or r.door_used
        return bool(d) and abs(self.pose[n][0] - self.doors[r.route_door].c[0]) <= self.rcfg["yield_people"]["no_return_m"]

    def claims(self, n: str) -> tuple[list[Point], list[Point]]:
        """Other robots' bodies, and the points they hold or will drive over soon (step-aside point, slot, plan ahead)."""
        ahead = self.rcfg["yield_people"]["aside_plan_ahead_s"]
        bodies: list[Point] = []
        held: list[Point] = []
        for m, r in self.rob.items():
            if m == n or m not in self.pose:
                continue
            x, y, _ = self.pose[m]
            bodies.append((x, y))
            held += [p for p in (r.aside, r.slot_xy) if p is not None]
            if r.plan and r.phase in ("approach", "cross", "to_goal", "to_slot"):
                near = min(range(len(r.plan)), key=lambda i: math.dist((x, y), r.plan[i]))
                held += [(px, py) for px, py, _t, _ds in self.walk(r.plan[near:], ahead)]
        return bodies, held

    def aside_ok(self, n: str, p: Point, bodies: Sequence[Point], held: Sequence[Point]) -> bool:
        """Step-aside point clear of walls and robots; on the way there no person comes closer than if the robot stood still."""
        y = self.rcfg["yield_people"]
        x, yy, _ = self.pose[n]
        gap = y["aside_robot_gap_m"]
        if self.grid.clearance(p) < y["aside_clearance_m"] or any(seg_cross((x, yy), p, u, v) for u, v in self.walls):
            return False
        if any(math.dist(p, o) < gap for o in (*bodies, *held)):
            return False
        length = math.dist((x, yy), p)
        steps = max(int(length / 0.2), 1)
        v, now = self.cost["time"]["v_nom"], self._now
        clear = y["aside_person_gap_m"]
        people = [(px, py, vx, vy, now - ts) for px, py, vx, vy, ts in self.seen.values()]
        for i in range(1, steps + 1):
            s = length * i / steps
            t = s / v
            q = (x + (p[0] - x) * s / length, yy + (p[1] - yy) * s / length)
            if any(math.dist(q, o) < 0.75 * gap for o in bodies):
                return False
            for px, py, vx, vy, age in people:
                there = (px + vx * (age + t), py + vy * (age + t))
                if math.dist(q, there) < min(math.dist((x, yy), there), clear) - 0.05:
                    return False
            if not people and s > clear and self.view is not None and self.view.occupied(t, *q):
                return False
        return True

    def aside_point(self, n: str) -> Point | None:
        """Reachable free point the forecast leaves free longest, then farthest from known people; None to stay."""
        if self.view is None:
            return None
        view = self.view
        y = self.rcfg["yield_people"]
        x, yy, _ = self.pose[n]
        bodies, held = self.claims(n)
        people = [(v[0], v[1]) for v in self.known.values()]

        def score(px: float, py: float) -> int:
            return sum(view.occupied(t, px, py) for t in range(int(y["aside_check_s"]) + 1))

        here = score(x, yy)
        best: Point | None = None
        best_key = (math.inf, math.inf)
        for rad in y["aside_radii_m"]:
            for k in range(12):
                a = k * math.pi / 6
                p = (round(x + rad * math.cos(a), 2), round(yy + rad * math.sin(a), 2))
                if not self.aside_ok(n, p, bodies, held):
                    continue
                key = (float(score(*p)), -min([math.dist(p, q) for q in people] or [5.0]))
                if key < best_key:
                    best, best_key = p, key
        return best if best is not None and best_key[0] < here else None

    def start_yield(self, n: str, ttc: float) -> None:
        r = self.rob[n]
        r.resume = "cross" if r.phase == "cross" else "to_goal" if r.phase == "to_goal" else "approach"
        r.phase, r.y_t0, r.t_clear, r.why, r.t_aside = "yield", self._now, None, "people_ahead", self._now
        r.diag = {"why": "people_ahead", "waited_s": 0.0}
        p = self.aside_point(n)
        r.aside = p
        x, y, _ = self.pose[n]
        if p:
            self.publish_plan(n, [(x, y), p])
            self.send(n, p, self.rcfg["yield_people"]["aside_tol_m"])
        else:
            self.stop(n)
        self.log("yield", robot=n, ttc=round(ttc, 1), aside=list(p) if p else None, x=round(x, 2), y=round(y, 2))

    def yield_step(self, n: str) -> None:
        y, r, now = self.rcfg["yield_people"], self.rob[n], self._now
        x, yy, _ = self.pose[n]
        d = r.door or r.door_used
        waited = now - r.y_t0
        r.diag = {"why": "people_ahead", "waited_s": round(waited, 1)}
        if r.aside and math.dist((x, yy), r.aside) <= y["aside_tol_m"]:
            r.aside = None
            self.stop(n)
        elif r.aside and not self.aside_ok(n, r.aside, *self.claims(n)):
            self.log("aside_dropped", robot=n, aside=list(r.aside))
            r.aside, r.t_aside = None, now - 1.0
            self.stop(n)
        elif r.aside is None and now - r.t_aside >= 1.0 and self.view is not None and self.view.occupied(0, x, yy):
            r.t_aside = now
            p = self.aside_point(n)
            if p:
                r.aside = p
                self.publish_plan(n, [(x, yy), p])
                self.send(n, p, y["aside_tol_m"])
        plan = self.path(r.target, d, x, yy) if d else r.plan
        clear = self.conflict(n, plan) is None
        r.t_clear = (r.t_clear if r.t_clear is not None else now) if clear else None
        forced = waited > y["max_wait_s"]
        if (r.t_clear is not None and now - r.t_clear >= y["clear_s"]) or forced:
            r.phase, r.no_yield = r.resume, now + (y["grace_s"] if forced else y["regrace_s"])
            self.log("yield_end", robot=n, waited=round(waited, 1), forced=forced)
            self.resume_goal(n)

    def monitor(self, n: str) -> bool:
        """True when the conflict monitor owns robot n this tick."""
        if not (self.rules["yield_people"] and self.view is not None):
            return False
        r = self.rob[n]
        if r.phase == "yield":
            self.yield_step(n)
            return True
        r.ttc = None
        if r.phase in ("approach", "to_slot", "queued", "cross", "to_goal") and self._now >= r.no_yield and not self.in_no_return(n):
            t = self.conflict(n)
            r.ttc = t
            if t is not None:
                self.start_yield(n, t)
                return True
        return False

    # re-assignment

    def dist_door(self, n: str, d: str) -> float:
        return math.dist(self.pose[n][:2], self.doors[d].c)

    def reassign_door(self, n: str) -> None:
        r, rc = self.rob[n], self.rcfg["reassign_door"]
        d = r.route_door
        behind = n in self.queue_of[d] and self.queue_of[d].index(n) > 0
        if self._now - r.t_switch < rc["min_dwell_s"] or (self.dist_door(n, d) <= rc["lock_m"] and not behind):
            return
        opts, paths = self.options(n, r.target)
        best = min(opts, key=lambda k: opts[k]["J"])
        if best != d and opts[d]["J"] - opts[best]["J"] >= rc["hysteresis_j"]:
            self.log("reroute", robot=n, old=d, new=best, options=opts)
            was_slot = r.phase in ("to_slot", "queued")
            if n in self.queue_of[d]:
                self.queue_of[d].remove(n)
            r.slot_i, r.slot_door = None, None
            r.door, r.plan, r.t_switch, r.t_at, r.t_head = best, paths[best], self._now, None, None
            if was_slot:
                r.phase = "approach"
                self.resume_goal(n)
            else:
                self.publish_plan(n, r.plan)

    def reassign_goal(self) -> None:
        rc = self.rcfg["reassign_goal"]
        if self._now - self.t_gs < rc["period_s"]:
            return
        self.t_gs = self._now
        ok = [n for n in self.names if self.rob[n].phase == "approach" and self.dist_door(n, self.rob[n].route_door) > rc["lock_m"] and self._now - self.rob[n].t_goal_sw >= rc["min_dwell_s"]]
        for i, a in enumerate(ok):
            for b in ok[i + 1 :]:
                ra, rb = self.rob[a], self.rob[b]
                cur = min(v["J"] for v in self.options(a, ra.target, around=False)[0].values()) + min(v["J"] for v in self.options(b, rb.target, around=False)[0].values())
                (oa, pa), (ob, pb) = self.options(a, rb.target, around=False), self.options(b, ra.target, around=False)
                da, db = min(oa, key=lambda k: oa[k]["J"]), min(ob, key=lambda k: ob[k]["J"])
                if cur - (oa[da]["J"] + ob[db]["J"]) >= rc["margin_j"]:
                    self.log("goal_swap", a=a, b=b, before=round(cur, 1), after=round(oa[da]["J"] + ob[db]["J"], 1))
                    ga, gb = ra.target, rb.target
                    for m in (a, b):
                        q = self.queue_of[self.rob[m].route_door]
                        if m in q:
                            q.remove(m)
                        self.rob[m].t_at, self.rob[m].t_head = None, None
                    self.set_plan(a, gb, da, pa[da])
                    self.set_plan(b, ga, db, pb[db])
                    self.rob[a].t_goal_sw = self.rob[b].t_goal_sw = self._now
                    self.go(a)
                    self.go(b)
                    return

    # main loop

    def tick(self) -> None:
        """One decision step at the time set by set_time; sets finished at the end of the episode."""
        if self.off is None or self.finished:
            return
        self.update_known()
        if not self.started:
            if any(n not in self.pose for n in self.names):
                return
            self.start_episode()
        now, ep, cp = self._now, self.rcfg["episode"], self.rcfg["capacity"]
        for n in self.names:
            r = self.rob[n]
            if r.phase == "done" or n not in self.pose:
                continue
            x, y, _ = self.pose[n]
            if math.dist((x, y), r.target) <= ep["goal_tol_m"]:
                r.phase, r.t_done = "done", now
                self.stop(n)
                self.trips_done[n] += 1
                if r.door:
                    self.inside[r.door].pop(n, None)
                self.log("trip_done", robot=n, t_trip=round(now - (r.t0 if r.t0 is not None else now), 1), door_used=r.door_used)
                continue
            mk = r.mark
            if mk is None or math.dist(mk[1:], (x, y)) >= 0.1:
                r.mark = (now, x, y)
            elif now - mk[0] >= ep["stuck_s"] and r.phase not in ("hold", "queued", "yield") and not r.stuck:
                r.stuck = True
                self.log("stuck", robot=n, phase=r.phase, door=r.door, x=round(x, 2), y=round(y, 2))
            if self.monitor(n):
                continue
            if r.phase == "cross" and self.past_door(n):
                d = r.route_door
                self.inside[d].pop(n, None)
                r.phase, r.door_used, r.door = "to_goal", d, None
                self.log("release", robot=n, door=d)
                r.plan = self.path(r.target, d, x, y) or r.plan
                self.publish_plan(n, r.plan)
                continue
            if r.phase in ("approach", "to_slot", "queued") and self.rules["queue"]:
                if r.phase == "approach" and self.past_door(n):
                    q = self.queue_of[r.route_door]
                    if n in q:
                        q.remove(n)
                    r.phase = "cross"
                    continue
                if self.rules["reassign_door"] and now - r.t_eval >= self.rcfg["reassign_door"]["period_s"]:
                    r.t_eval = now
                    self.reassign_door(n)
                d = r.route_door
                self.queue_step(n, d, self.dist_door(n, d))
            elif r.phase in ("approach", "hold"):
                if r.phase == "approach" and self.past_door(n):
                    r.phase = "cross"
                    continue
                if r.phase == "approach" and self.rules["reassign_door"] and now - r.t_eval >= self.rcfg["reassign_door"]["period_s"]:
                    r.t_eval = now
                    self.reassign_door(n)
                d = r.route_door
                if self.dist_door(n, d) <= cp["hold_m"]:
                    self.hold_or_enter(n, d, x, y)
            self.refresh_plan(n, x, y)
        if self.rules["reassign_goal"]:
            self.reassign_goal()
        self.write_state()
        if self.samp_f is not None and now - self.t_samp >= 0.5:
            self.t_samp = now
            self.write_samples()
        if all(self.rob[n].phase == "done" for n in self.names):
            self.t_done_all = self.t_done_all or now
        if self.t_rel() >= self.duration or (self.t_done_all and now - self.t_done_all >= ep["tail_s"]):
            for n in self.names:
                r = self.rob[n]
                if r.phase != "done" and n in self.pose:
                    self.log("not_finished", robot=n, phase=r.phase, dist=round(math.dist(self.pose[n][:2], r.target), 2))
            self.log("end", trips=self.trips_done, makespan=round(self.t_done_all - self.t_start, 1) if self.t_done_all else None)
            self.finished = True

    def hold_or_enter(self, n: str, d: str, x: float, y: float) -> None:
        """Without the queue rule: stop a robot at its door until it may enter."""
        r, now = self.rob[n], self._now
        if r.t_at is None:
            r.t_at = now
        why = self.entry_reason(n, d)
        if why:
            if r.phase != "hold":
                r.phase = "hold"
                self.stop(n)
                self.log("hold", robot=n, door=d, why=why)
            r.why, r.diag = why, {"why": why, "waited_s": round(now - r.t_at, 1)}
            return
        waited = round(now - r.t_at, 1)
        was_hold = r.phase == "hold"
        forced = bool(self.rules["yield_people"] and waited > self.rcfg["yield_people"]["max_wait_s"])
        r.phase, r.why, r.diag = "cross", None, None
        self.inside[d][n] = now
        self.log("enter", robot=n, door=d, waited=waited, forced=forced)
        if was_hold:
            r.plan = self.grid.descend(self.fields[(r.target, d)], x, y)
            self.go(n)

    def refresh_plan(self, n: str, x: float, y: float) -> None:
        """Re-plan a moving robot once a second."""
        r, now = self.rob[n], self._now
        if now - r.t_pub >= 1.0 and r.phase in ("approach", "cross", "to_goal"):
            r.t_pub = now
            r.plan = self.path(r.target, r.route_door, x, y) or r.plan
            self.publish_plan(n, r.plan)
            r.track = (self.terms(n, r.target, r.door, path=r.plan)[0] or {}) if r.door else {}

    def write_samples(self) -> None:
        if self.samp_f is None:
            return
        now = self._now
        for n in self.names:
            r = self.rob[n]
            if r.phase in ("done", "init") or n not in self.pose or r.goal_xy is None:
                continue
            x, y, _ = self.pose[n]
            last = self.last_xy.get(n)
            speed = math.dist((x, y), last[1:]) / max(now - last[0], 1e-3) if last else 0.0
            self.last_xy[n] = (now, x, y)
            self.samp_f.write(json.dumps({"t": round(self.t_rel(), 2), "robot": n, "x": features.features(self, n, speed), "y": features.labels(self, n)}) + "\n")

    def write_state(self) -> None:
        robots: dict[str, dict[str, object]] = {}
        for n in self.names:
            r = self.rob[n]
            goal = list(r.goal_xy) if r.goal_xy else None
            robots[n] = {
                "state": r.phase,
                "phase": r.phase,
                "door": r.door or r.door_used,
                "goal": goal,
                "cmd_kind": "goal",
                "final": goal,
                "why": r.why,
                "diag": r.diag,
                "track": r.track,
                "slot": list(r.slot_xy) if r.slot_xy else None,
                "rank": r.rank,
                "aside": list(r.aside) if r.aside else None,
                "ttc": r.ttc,
            }
        doors = {d: {"prog": round(g.prog, 2), "state": "open" if g.prog >= 0.9 else ("closed" if g.prog <= 0.05 else "moving")} for d, g in self.door_open.items()}
        self.st.write(json.dumps({"t": round(self.t_rel(), 2), "robots": robots, "inside": {d: list(m) for d, m in self.inside.items()}, "queue": self.queue_of, "doors": doors, "known": sorted(self.known), "n_peds": len(self.peds), "trips": dict(self.trips_done)}) + "\n")


def check_profile(profile: Profile) -> None:
    bad = set(profile) ^ set(RULES)
    if bad:
        raise ValueError(f"profile rule keys differ from {RULES}: {sorted(bad)}")
