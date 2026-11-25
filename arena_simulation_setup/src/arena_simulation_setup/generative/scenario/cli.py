"""`generate_scenario`: base populations, prompt-driven cases and benchmark suites."""

from __future__ import annotations

import argparse
import functools
import itertools
import math
import os
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from arena_simulation_setup.shared.pathing import PathError, plan_route

from .formats import first_level, is_world, levels, load_world, read_scenario, world_roles
from .materialize import WouldOverwrite, prune_generated, write_base
from .occupancy import UNKNOWN_FOOTPRINT_RADIUS, level_grid, world_statics
from .populate import MIN_ROUTE_CLEARANCE, PROFILES, basis_for, robot_zones, route_points, zone_centroids
from .populate import build as build_base
from .promptgen import prompts as P
from .promptgen.world import ROBOT_CLEAR, Grid, WorldError, endpoint
from .propose import DEFAULT_MODEL, Unavailable, build_prompt, classify_tail, merge, unknown_names
from .route import MIN_EPISODE_S, ROBOT_SPEED_MPS, robot_for, route_seconds
from .zone_screen import ref_safe_zones, screen_zones
from .zones import zones_by_role

if TYPE_CHECKING:
    from arena_simulation_setup.tree.World.model import World

PROFILE_NOTES = {
    "normal": "working hours: the building's dwellers walk between rooms, pairs talk in rooms",
    "break": "break time: the same crowd in the break rooms",
    "night": "night: half the crowd, slower",
    "empty": "nobody - the no-population control",
}


def _finite(value: str, parse: Callable[[str], float]) -> float:
    try:
        number = parse(value)
    except ValueError:
        number = math.nan
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError(f"{value!r} is not a number")
    return number


def _positive_float(value: str) -> float:
    """An argparse type: a finite number above 0."""
    if (number := _finite(value, float)) <= 0:
        raise argparse.ArgumentTypeError(f"{value!r}: expected a number above 0, e.g. 0.5")
    return number


def _non_negative_float(value: str) -> float:
    """An argparse type: a finite number of at least 0."""
    if (number := _finite(value, float)) < 0:
        raise argparse.ArgumentTypeError(f"{value!r}: expected a number of at least 0, e.g. 0 or 1.5")
    return number


def _positive_int(value: str) -> int:
    """An argparse type: a whole number of at least 1."""
    if (number := int(_finite(value, int))) < 1:
        raise argparse.ArgumentTypeError(f"{value!r}: expected a whole number of at least 1, e.g. 1 or 3")
    return number


def _arena_world_path() -> list[Path]:
    return [Path(p) for p in os.environ.get("ARENA_WORLD_PATH", "").split(":") if p]


def world_roots(worlds: Path | None) -> list[Path]:
    """The source world roots: `--worlds`, else the `ARENA_WORLD_PATH` roots, then `$ARENA_DIR/arena_simulation_setup/worlds`."""
    if worlds is not None:
        return [worlds]
    roots = _arena_world_path()
    if arena := os.environ.get("ARENA_DIR"):
        roots.append(Path(arena) / "arena_simulation_setup" / "worlds")
    return roots


def world_root(worlds: Path | None, world: str, *, write: bool = False) -> Path:
    """`<root>/<world>` under the first source root that has it, else the world `WorldIdentifier` resolves for a read."""
    from arena_simulation_setup.tree.World import WorldIdentifier

    roots = world_roots(worlds)
    for root in roots:
        if (root / world).is_dir():
            return root / world
    searched = ", ".join(map(str, roots)) or "no source root (ARENA_DIR and ARENA_WORLD_PATH are unset)"
    if worlds is not None:
        raise FileNotFoundError(f"no world {world!r} under {searched}. Worlds there: {_world_names(worlds)}")
    identifier = WorldIdentifier(world)
    try:
        path = identifier.resolve_source_sync().path if write else identifier.resolve_sync().path
    except FileNotFoundError:
        path = None
    if path is None:
        raise FileNotFoundError(f"no world {world!r} under {searched}, and no resolver has it. Worlds there: {_world_names(worlds)}. Set ARENA_DIR to the Arena checkout or ARENA_WORLD_PATH to world roots, or pass --worlds DIR") from None
    if write:
        raise ValueError(f"world {world!r} resolves to {path}, which is no source world root ({searched}), so it is not written. Copy it into one of them, or pass --worlds DIR")
    return Path(path)


def _all_worlds(worlds: Path | None) -> list[str]:
    names: set[str] = set()
    for root in world_roots(worlds):
        if root.is_dir():
            names.update(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".") and is_world(p))
    return sorted(names)


def _world_names(worlds: Path | None) -> str:
    return ", ".join(_all_worlds(worlds)) or "none"


def _writable(path: Path) -> Path:
    """`path`, refused under an install prefix or the asset cache."""
    from arena_simulation_setup import ARENA_ASSETS_DIR

    real = path.resolve()
    prefixes = [Path(p) for var in ("AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH") for p in os.environ.get(var, "").split(":") if p]
    for root, what in [*((p, "an install prefix") for p in prefixes), (ARENA_ASSETS_DIR, "the asset cache")]:
        if real.is_relative_to(root.resolve()):
            raise ValueError(f"refusing to write {path}: it is under {root}, {what} that a rebuild or a pull overwrites. Pass --worlds DIR or --out FILE in the source tree or an ARENA_WORLD_PATH root")
    return path


def _runtime_note(root: Path) -> None:
    if root.parent.resolve() not in {p.resolve() for p in _arena_world_path()}:
        print(f"rebuild arena_simulation_setup before launching, or add {root.parent} to ARENA_WORLD_PATH, for the runtime to find what was written")


def _bare(args: argparse.Namespace) -> int:
    """A verb run without a world: its help and the worlds it can take."""
    print(args.help_text())
    print(f"worlds{f' under {args.worlds}' if args.worlds is not None else ''}: {_world_names(args.worlds)}")
    return 2


def default_suites_dir() -> Path:
    """`configs/suites` of arena_evaluation under `$ARENA_DIR`, read by the runner after its share."""
    if not (arena := os.environ.get("ARENA_DIR")):
        raise FileNotFoundError("ARENA_DIR is unset: set it to the Arena checkout, or pass --out to name the suite file")
    out = Path(arena) / "arena_evaluation" / "arena_evaluation" / "configs" / "suites"
    if not out.is_dir():
        raise FileNotFoundError(f"{out} does not exist: pass --out to name the suite file")
    return out


def _shadowing_suite(out: Path, name: str) -> Path | None:
    """The installed arena_evaluation suite of the same name the runner reads before `out`, if any."""
    try:
        from ament_index_python.packages import get_package_share_path

        installed = get_package_share_path("arena_evaluation") / "configs" / "suites" / name / "suite.yaml"
    except (ImportError, LookupError):
        return None
    return installed if installed.is_file() and not (out.exists() and installed.samefile(out)) else None


def _grid_loader(root: Path, world: World, statics: Sequence[Mapping[str, Any]] | None = None) -> Callable[[], Grid | None]:
    """The planning grid of `world` at `root` with `statics` (default its furniture) stamped, built on the first call."""

    @functools.cache
    def grid() -> Grid | None:
        loaded = level_grid(root, world_statics(root, world) if statics is None else statics, world)
        if loaded is None:
            return None
        occupied, res, ox, oy = loaded
        return Grid(occupied=occupied, res=res, ox=ox, oy=oy)

    return grid


def _route_reachable(grid: Callable[[], Grid | None], zones: list[dict[str, Any]], robot: dict[str, Any], radius: float = ROBOT_CLEAR) -> bool:
    """Whether the planner finds a route clear by `radius` along the robot's legs from snapped endpoints."""
    pts = route_points(robot, zones)
    held = grid() if len(pts) >= 2 else None
    if held is None:
        return True
    try:
        for a, b in itertools.pairwise(pts):
            plan_route(held, endpoint(held, a), endpoint(held, b), is_valid=held.clear_of(radius))
    except PathError:
        return False
    return True


def _zones(root: Path, loaded: World) -> list[dict[str, Any]]:
    """The named zones of a world's first level."""
    doc = first_level(root, loaded)
    if doc is None:
        raise FileNotFoundError(f"{root.name}: no world.yaml")
    return doc.get("zones") or []


def _world_parts(root: Path, loaded: World, grid: Callable[[], Grid | None], radius: float = ROBOT_CLEAR) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """A world's zones and a robot copied from one of its scenarios, else the two most separated reachable zones."""
    world = root.name
    zones = _zones(root, loaded)

    for cand in sorted((root / "scenarios").glob("*/scenario.yaml")):
        robots = read_scenario(cand).get("robots") or []
        if robots and isinstance(robots[0], dict) and robots[0].get("start"):
            robot = dict(robots[0])
            if _route_reachable(grid, zones, robot, radius):
                return zones, robot
            print(f"  ! {world}: {cand.parent.name}'s robot route is not plannable on this map, ignoring it")

    centroids = zone_centroids(zones)
    if len(centroids) < 2:
        raise FileNotFoundError(f"{world}: no scenario with a robot to copy, and fewer than two named zones to fall back on. Author a scenario with a `robots:` block first.")

    names = list(centroids)
    pairs = sorted(((math.dist(centroids[a], centroids[b]), a, b) for i, a in enumerate(names) for b in names[i + 1 :]), reverse=True)
    _, start, goal = pairs[0]
    for _d, a_name, b_name in pairs:
        if _route_reachable(grid, zones, {"start": a_name, "goal": b_name}, radius):
            start, goal = a_name, b_name
            break
    print(f"  ! {world}: no scenario declares a robot, using the two most separated reachable zones as endpoints - start={start!r} goal={goal!r}")
    print("  ! override with --robot-start / --robot-goal if that is not the traverse you want")
    return zones, {"start": start, "goal": goal}


def _all_zones(root: Path, world: World) -> list[dict[str, Any]]:
    """Every named zone across all levels, the first of each name kept."""
    zones: list[dict[str, Any]] = []
    seen: set[str] = set()
    for doc in levels(root, world):
        for zone in doc.get("zones") or []:
            if zone.get("name") and str(zone["name"]) not in seen:
                seen.add(str(zone["name"]))
                zones.append(zone)
    return zones


def _cmd_zones(args: argparse.Namespace) -> int:
    if args.world is None:
        return _bare(args)
    root = world_root(args.worlds, args.world)
    world = load_world(root)
    zones = _zones(root, world)
    overrides = world_roles(root, world)
    by_role = zones_by_role(zones, overrides)
    for role, names in by_role.items():
        if names:
            print(f"{role.value:9s} " + ", ".join(f"{n}*" if n in overrides else n for n in names))
    if any(n in overrides for names in by_role.values() for n in names):
        print("\n* stated in the world's kg.yaml, not a substring rule")
    if unresolved := unknown_names(zones, overrides):
        print(f"\nunclassified: {', '.join(unresolved)}")
        print(f"  run `generate_scenario classify {args.world}` to have a model propose roles for these")
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    from arena_simulation_setup.tree.World.view import KG_FILE

    if args.world and args.all:
        raise ValueError("pass world names or --all, not both")
    if not args.world and not args.all:
        return _bare(args)
    names = sorted(args.world) if args.world else _all_worlds(args.worlds)
    pending: dict[str, tuple[Path, list[dict[str, Any]], list[str]]] = {}
    for world in dict.fromkeys(names):
        root = world_root(args.worlds, world, write=args.write)
        loaded = load_world(root)
        zones = _all_zones(root, loaded)
        if not zones:
            print(f"  ! {world}: no zones declared on any level")
            continue
        if todo := unknown_names(zones, world_roles(root, loaded)):
            pending[world] = (root, zones, todo)

    if not pending:
        print("every zone in every world is already classified by a rule or a fact in its kg.yaml")
        return 0
    print(f"{sum(len(todo) for _, _, todo in pending.values())} unclassified zone name(s) across {len(pending)} world(s)")
    if not args.write:
        for world, (_, zones, todo) in pending.items():
            print(f"\n### {world}  ({len(todo)})")
            print(build_prompt(zones, todo, world=world))
        print("(dry run - this is the prompt that would be sent, pass --write to ask and store)")
        return 0

    for root, _, _ in pending.values():
        _writable(root / KG_FILE)
    written = 0
    for world, (root, zones, todo) in pending.items():
        print(f"\n### {world}")
        try:
            proposals, complaints = classify_tail(zones, world=world, names=todo, model=args.model)
        except Unavailable as exc:
            print(f"  ! {exc}")
            return 1
        for proposal in proposals:
            print(f"  {' ' if proposal.confident else '?'} {proposal.zone:20s} {proposal.role.value:8s} {proposal.why}")
        for complaint in complaints:
            print(f"  ! {complaint}")
        landed, kept = merge(proposals, root / KG_FILE)
        for note in kept:
            print(f"  = {note}")
        if landed:
            print(f"  {landed} fact(s) written to {root / KG_FILE}")
        written += landed
    print(f"\n{written} fact(s) written")
    print("review them in each kg.yaml: drop `by: llm` from any fact you change or confirm and it will not be revisited")
    return 0


def _endpoint_arg(flag: str, value: str, zones: list[dict[str, Any]]) -> str | list[float]:
    """A `--robot-start`/`--robot-goal` value as a zone name or an `[x, y]`/`[x, y, yaw]` pose."""
    names = sorted(str(z["name"]) for z in zones if z.get("name"))
    forms = f"a zone name ({', '.join(names) or 'this level has none'}) or 'x,y[,yaw]' in meters and radians, e.g. '1.5,2,0'"
    if "," not in value:
        if value in names:
            return value
        raise ValueError(f"{flag} {value!r}: no zone of that name on the robot's level. Pass {forms}")
    try:
        pose = [float(v) for v in value.split(",")]
    except ValueError:
        pose = []
    if len(pose) not in (2, 3) or not all(math.isfinite(v) for v in pose):
        raise ValueError(f"{flag} {value!r}: not a pose. Pass {forms}")
    return pose


def _override_robot(robot: dict[str, Any], zones: list[dict[str, Any]], *, start: str | None, goal: str | None) -> dict[str, Any]:
    """`robot` with the given start and goal, a new goal replacing the route's legs with the single leg to it."""
    out = dict(robot)
    if start is not None:
        out["start"] = _endpoint_arg("--robot-start", start, zones)
    if goal is not None:
        out.pop("phases", None)
        out.pop("goal_level", None)
        out["goal"] = _endpoint_arg("--robot-goal", goal, zones)
    return out


def _cmd_populate(args: argparse.Namespace) -> int:
    if args.world is None:
        return _bare(args)
    if args.prune and not args.write:
        raise ValueError("--prune removes stale generated bases, which only a writing run does: add --write")
    root = world_root(args.worlds, args.world, write=args.write)
    if args.write:
        _writable(root)
    world = load_world(root)
    statics = world_statics(root, world)
    grid = _grid_loader(root, world, statics)
    zones, robot = _world_parts(root, world, grid, args.robot_radius)
    if args.robot_start is not None or args.robot_goal is not None:
        robot = _override_robot(robot, zones, start=args.robot_start, goal=args.robot_goal)
        print(f"  robot endpoints overridden: start={robot['start']!r} goal={robot.get('goal')!r}")

    overrides = world_roles(root, world)
    if unresolved := unknown_names(zones, overrides):
        print(f"  ! {len(unresolved)} zone(s) no rule classifies: {', '.join(unresolved)}")
        print(f"  !   they are populated as neither work nor break, `generate_scenario classify {args.world}` proposes roles")
    if (basis := basis_for(zones, overrides)) != "workplace":
        print(f"  ! zone mapping: {basis} - this world has no work zones, so the active/retreat")
        print("  !   distinction is drawn between living areas and private rooms instead")

    spec = robot_for(robot, zones)
    if unresolved := sorted({str(s["model"]) for s in statics if not s["resolved"]}):
        print(f"  ! {len(unresolved)} asset(s) with no annotated footprint, stamped as the runtime's {2 * UNKNOWN_FOOTPRINT_RADIUS:g} m box: {', '.join(unresolved)}")
    held = grid()
    clearance_map = None if held is None else (held.dist, held.res, held.ox, held.oy)
    verdicts = screen_zones(zones, world_root=root, route=route_points(spec, zones), statics=statics, min_route_clearance=MIN_ROUTE_CLEARANCE, robot_zones=robot_zones(robot), clearance_map=clearance_map, world=world)
    homes = ref_safe_zones(verdicts)
    print(f"  homes: {len(homes)}/{len(verdicts)} zone(s) entirely clear of the route" + ("" if homes else " - falling back to every target zone but the robot's own"))
    for name, verdict in sorted(verdicts.items()):
        if not verdict.ref_safe and verdict.reason:
            print(f"    - {name}: {verdict.reason}")
    print(f"  route: {route_seconds(spec, zones):.0f} s at {ROBOT_SPEED_MPS} m/s ({len(spec['phases'])} leg(s), floor {MIN_EPISODE_S:.0f} s)")

    written: list[str] = []
    for profile in PROFILES:
        doc = build_base(profile, zones=zones, robot=spec, seed=args.seed, homes=homes, overrides=overrides, world=args.world)
        agents = doc.get("humans") or []
        if not agents and PROFILES[profile]["count"] > 0:
            print(f"  ! {profile}: no zone to populate - this base is measurable as `empty`,")
            print("  !   and every case needing a crowd will refuse it")
        talking = sum(1 for a in agents if "talker" in str(a["agent"].get("agent_type")))
        line = f"  {profile:7s} {len(agents):3d} agent(s): {talking} talking, {len(agents) - talking} walking, from {len({str(a['pose']) for a in agents})} room(s)"
        if args.write:
            try:
                out = write_base(doc, world_root=root, profile=profile, force=args.force, note=PROFILE_NOTES[profile])
            except WouldOverwrite as exc:
                print(f"  {profile:7s} SKIPPED - {exc}")
                continue
            written.append(out.name)
        print(line)

    if args.write and args.prune:
        keep = [*written, *(d.name for d in (root / "scenarios").glob("*__*"))]
        for gone in prune_generated(root, keep):
            print(f"  - pruned stale generated base {gone}")
    if written:
        _runtime_note(root)
    if not args.write:
        print("(dry run - pass --write to write them)")
    return 0


def _cmd_promptgen(args: argparse.Namespace) -> int:
    from .promptgen import RobotPlan, generate, write_result
    from .promptgen.world import LoadedWorld, WorldContext

    if args.world is None:
        return _bare(args)
    root = world_root(args.worlds, args.world, write=args.write)
    if args.write:
        _writable(root)
    prompts = P.load_sets(args.world, extra=[Path(p) for p in args.prompts or []])
    if args.ids:
        wanted = [i.strip() for i in args.ids.split(",") if i.strip()]
        if unknown := [i for i in wanted if i not in {p.id for p in prompts}]:
            raise ValueError(f"--ids {', '.join(unknown)}: no such prompt in {P.shipped_dir()} or the --prompts files. Known ids: {', '.join(p.id for p in prompts) or 'none'}")
        prompts = [p for p in prompts if p.id in wanted]
    if not prompts:
        print(f"no prompts: {P.shipped_dir()} and the --prompts files hold none for {args.world}")
        return 1
    bases = list(PROFILES)
    if args.bases:
        bases = [b.strip() for b in args.bases.split(",") if b.strip()]
        have = sorted(d.parent.name for d in (root / "scenarios").glob("*/scenario.yaml") if "__" not in d.parent.name)
        if unknown := [b for b in bases if b not in have]:
            raise ValueError(f"--bases {', '.join(unknown)}: no such base scenario in {args.world}. Its bases: {', '.join(have) or 'none, run `populate --write` first'}")
    cache = Path(args.cache) if args.cache else None
    robot = RobotPlan(speed=args.robot_speed, traversals=args.traversals, hold=args.hold, tolerance=args.tolerance, radius=args.robot_radius)
    seen: dict[str, str] = {}
    counts: dict[str, int] = {}
    written = 0
    refused = 0
    loaded = 0
    parts: LoadedWorld | None = None
    for base_id in bases:
        base_path = root / "scenarios" / base_id / "scenario.yaml"
        if not base_path.is_file():
            print(f"  ! base {base_id}: not found - run `populate --write` first")
            continue
        base_doc = read_scenario(base_path)
        try:
            if parts is None:
                parts = LoadedWorld.read(root)
                if parts.unresolved:
                    print(f"  ! {len(parts.unresolved)} asset(s) with no annotated footprint, stamped as the runtime's {2 * UNKNOWN_FOOTPRINT_RADIUS:g} m box: {', '.join(parts.unresolved)}")
            world = WorldContext.load(root, base_doc, parts, robot_radius=robot.radius)
        except WorldError as exc:
            print(f"  ! {base_id}: {exc}")
            continue
        loaded += 1
        for prompt in prompts:
            try:
                result = generate(prompt, world=world, base_id=base_id, base_doc=base_doc, base_dir=base_path.parent, seed=args.seed, cache_dir=cache, model=args.model, oracle=args.oracle, robot=robot)
            except Exception as exc:
                print(f"  ! {base_id}/{prompt.id}: {exc!r}")
                if args.verbose:
                    traceback.print_exc()
                counts["crashed"] = counts.get("crashed", 0) + 1
                continue
            status = result.record.status
            if status not in ("infeasible", "denied"):
                key = result.record.dedup_key()
                if key in seen:
                    status = f"duplicate of {seen[key]}"
                else:
                    seen[key] = result.scenario_id
            if args.write and status in ("ok", "partial"):
                try:
                    if write_result(result, root, keep_partial=not args.strict) is not None:
                        written += 1
                except WouldOverwrite as exc:
                    print(f"  ! {base_id}/{prompt.id}: {exc}")
                    refused += 1
                    status = "refused"
                except ValueError as exc:
                    print(f"  ! {base_id}/{prompt.id}: the scenario model rejects the case: {exc}")
                    status = "rejected"
            counts[status.split(" ")[0]] = counts.get(status.split(" ")[0], 0) + 1
            fails = ", ".join(f"{f.step}:{f.kind}" for f in result.record.failures)
            print(f"  {result.scenario_id:40s} {status:10s} [{', '.join(result.record.steps) or '-'}]" + (f"  failures: {fails}" if fails else ""))
            if args.verbose:
                for note in result.record.warnings:
                    print(f"      - {note}")
                for failure in result.record.failures:
                    print(f"      ! {failure.step}: {failure.reason}")
    print(f"{sum(counts.values())} prompt run(s): " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) + (f", wrote {written}" if args.write else " (dry run - pass --write)"))
    if written:
        _runtime_note(root)
    crashed = counts.get("crashed", 0)
    if crashed:
        print(f"generate_scenario: {crashed} prompt(s) crashed" + ("" if args.verbose else ", rerun with -v for the tracebacks"), file=sys.stderr)
    if rejected := counts.get("rejected", 0):
        print(f"generate_scenario: {rejected} case(s) not written, the scenario model rejects them (see the ! lines above)", file=sys.stderr)
    if refused:
        print(f"generate_scenario: {refused} case(s) not written over a scenario this tool did not generate, see the ! lines above", file=sys.stderr)
    if not counts.get("ok", 0) + counts.get("partial", 0):
        why = "no prompt was realized as ok or partial on any base" if loaded else f"none of the bases {', '.join(bases)} could be loaded, see the ! lines above"
        print(f"generate_scenario: nothing generated: {why}", file=sys.stderr)
        return 1
    return 1 if crashed or rejected or refused else 0


def _cmd_suite(args: argparse.Namespace) -> int:
    from .suite import build, scenarios

    if not args.world:
        return _bare(args)
    out = _writable(Path(args.out) if args.out else default_suites_dir() / args.name / "suite.yaml")
    name = out.stem if args.out else args.name
    worlds = {world: scenarios(world_root(args.worlds, world), args.match) for world in args.world}
    for world, names in worlds.items():
        print(f"  {world:28s} {len(names):4d} scenario(s)")
    if not any(worlds.values()):
        print("no scenarios matched" + ("" if args.match else ": none of these worlds has generated scenarios (pass --match for hand-authored ones)"))
        return 1
    doc = build(worlds, robot=args.robot, episodes=args.episodes, seed=args.seed, timeout=args.timeout)
    text = yaml.safe_dump(doc, sort_keys=False)
    if args.write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"wrote {out}: {len(doc['stages'])} stage(s) x {args.episodes} episode(s). Run with `arena evaluation benchmark --suite {name} --contest <contest>`")
    else:
        print(text, end="")
        print(f"(dry run - pass --write to write these {len(doc['stages'])} stage(s) to {out})")
    if (shadow := _shadowing_suite(out, name)) is not None:
        print(f"  ! the runner reads the installed {shadow} first: rebuild arena_evaluation for it to read this one")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_scenario",
        description=(
            "Generate scenarios for an Arena world. Verbs: zones (print each zone's role), classify (ask a model for the roles no rule covers), "
            "populate (write the four base scenarios), promptgen (realize prompts on the bases as timeline scenarios), suite (write a benchmark suite over them). "
            "Writing verbs are dry runs without --write. `generate_scenario <verb> --help` lists a verb's options."
        ),
    )
    verbs = parser.add_subparsers(dest="verb", required=True, metavar="{zones,classify,populate,promptgen,suite}")

    def verb(name: str, fn: Callable[[argparse.Namespace], int], text: str) -> argparse.ArgumentParser:
        p = verbs.add_parser(name, help=text, description=text)
        p.add_argument("--worlds", type=Path, default=None, help="worlds root to read and write (default: the ARENA_WORLD_PATH roots, then $ARENA_DIR/arena_simulation_setup/worlds, and for reading also every world the world resolvers find)")
        p.set_defaults(fn=fn, help_text=p.format_help)
        return p

    p = verb("zones", _cmd_zones, "classify a world's zones into roles (work, break, transit, private, unknown)")
    p.add_argument("world", nargs="?", help="the world whose zones to print (bare, the verb lists the worlds)")

    p = verb("classify", _cmd_classify, "ask a model for the zone roles no rule or KG fact covers and write them as has_purpose facts into each world's kg.yaml")
    p.add_argument("world", nargs="*", help="worlds to classify (bare, the verb lists the worlds)")
    p.add_argument("--all", action="store_true", help="classify every world of the source roots")
    p.add_argument("--model", default=None, help=f"override the model (default {DEFAULT_MODEL})")
    p.add_argument("--write", action="store_true", help="call the model and write what it returns into the world's kg.yaml")

    p = verb("populate", _cmd_populate, "write the four base scenarios (normal, break, night, empty) for a world")
    p.add_argument("world", nargs="?", help="the world to populate (bare, the verb lists the worlds)")
    p.add_argument("--seed", type=int, default=0, help="seed of the crowd draws, the same world and seed write the same bases (default 0)")
    p.add_argument("--write", action="store_true", help="write the bases under the world's scenarios/ (default: a dry run that prints them)")
    p.add_argument("--robot-start", default=None, help="robot start: a zone name of its level or 'x,y[,yaw]' (default: the route of the world's first scenario with a robot)")
    p.add_argument("--robot-goal", default=None, help="robot goal: a zone name of its level or 'x,y[,yaw]', replacing the route's legs with the single leg to it")
    p.add_argument("--prune", action="store_true", help="with --write, remove generated bases this run did not write (never hand-authored ones)")
    p.add_argument("--force", action="store_true", help="replace scenarios this tool did not generate (they may be hand-authored)")
    p.add_argument("--robot-radius", type=_positive_float, default=ROBOT_CLEAR, help=f"meters the robot's route keeps clear of walls and furniture when the base route is picked, above 0 (default {ROBOT_CLEAR})")

    p = verb("promptgen", _cmd_promptgen, "realize prompt x base scenarios for a world as scenarios with a timeline")
    p.add_argument("world", nargs="?", help="the world whose bases to realize the prompts on (bare, the verb lists the worlds)")
    p.add_argument("--bases", default="", help="comma-separated base scenario ids of the world (default: normal, break, night, empty)")
    p.add_argument("--ids", default="", help="comma-separated prompt ids to restrict to (default: every shipped prompt for the world and every --prompts one)")
    p.add_argument("--prompts", nargs="*", help="extra prompt files, YAML lists in the shipped prompts/*.yaml format")
    p.add_argument("--seed", type=int, default=0, help="seed of every random draw, the same seed writes the same cases (default 0)")
    p.add_argument("--cache", default=None, help="model answer cache directory (default $ARENA_DATA_DIR/promptgen/llm_cache)")
    p.add_argument("--model", default=None, help=f"override the model of every step (default {DEFAULT_MODEL}, step A $ARENA_PROMPT_LLM_A when set)")
    p.add_argument("--oracle", action="store_true", help="use each prompt's own template slots instead of a model")
    p.add_argument("--robot-speed", type=_positive_float, default=ROBOT_SPEED_MPS, help=f"m/s the robot is designed to drive at, which times every interception release (default {ROBOT_SPEED_MPS})")
    p.add_argument("--traversals", type=_positive_int, default=1, help="round trips through the population: N traversals are 2N-1 legs (default 1, the single start->goal leg)")
    p.add_argument("--hold", type=_non_negative_float, default=0.0, help="seconds the robot holds at its start before the first leg (default 0, no hold phase)")
    p.add_argument("--tolerance", type=_non_negative_float, default=0.0, help="goal tolerance radius of every leg in meters (default 0: the runtime's goal tolerance)")
    p.add_argument("--robot-radius", type=_positive_float, default=ROBOT_CLEAR, help=f"robot radius in meters, the clearance its planned route keeps from walls and furniture (default {ROBOT_CLEAR})")
    p.add_argument("--strict", action="store_true", help="do not write partial scenarios")
    p.add_argument("--verbose", "-v", action="store_true", help="print each case's warnings and failures, and the traceback of a crashed prompt")
    p.add_argument("--write", action="store_true", help="write each ok or partial case under the world's scenarios/ (default: a dry run)")

    p = verb("suite", _cmd_suite, "write a benchmark suite with one scenario stage per (world, scenario)")
    p.add_argument("world", nargs="*", help="the worlds whose scenarios become stages (bare, the verb lists the worlds)")
    p.add_argument("--name", default="edge_case", help="suite name, the bundle folder under $ARENA_DIR/arena_evaluation/arena_evaluation/configs/suites (default edge_case)")
    p.add_argument("--out", default=None, help="write here instead of $ARENA_DIR/arena_evaluation/arena_evaluation/configs/suites/<name>/suite.yaml")
    p.add_argument("--match", nargs="*", default=[], help="scenario globs (default: every scenario the generator wrote)")
    p.add_argument("--robot", default="jackal", help="the chassis, as a contest compares planners on one drive family (default jackal)")
    p.add_argument("--episodes", type=_positive_int, default=1, help="episodes per stage, seeded seed..seed+episodes-1, at least 1 (default 1)")
    p.add_argument("--seed", type=int, default=1000, help="first episode seed, keep screening and dataset seed blocks disjoint (default 1000)")
    p.add_argument("--timeout", default="240s", help="sim-time budget per episode")
    p.add_argument("--write", action="store_true", help="write the suite file (default: a dry run that prints it)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point `generate_scenario`."""
    args = _parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        print(f"generate_scenario: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
