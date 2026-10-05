from __future__ import annotations

import argparse
from pathlib import Path

import attrs
import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory
from arena_simulation_setup.tree.World import WorldIdentifier
from PIL import Image
from shapely.geometry import Point, Polygon

from .acoustic_room_spec import AcousticRoomSpecBuilder
from .acoustic_scene import AcousticScene
from .acoustic_world_graph import AcousticWorldGraph


@attrs.frozen
class AcousticWorldAudit:
    world_name: str
    rooms: int
    door_portals: int
    opening_portals: int
    connected_components: int
    unpaired_doors: int
    overlapping_zone_pairs: tuple[tuple[str, str], ...]
    sampled_traversable_cells: int
    uncovered_traversable_cells: int
    uncovered_examples: tuple[tuple[float, float], ...]
    map_issues: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return self.uncovered_traversable_cells == 0 and not self.overlapping_zone_pairs and not self.map_issues


def audit_world(
    world_name: str,
    *,
    worlds_root: Path | None = None,
    stride_cells: int = 10,
) -> AcousticWorldAudit:
    if stride_cells <= 0:
        raise ValueError("stride_cells must be positive")
    world = WorldIdentifier(world_name).resolve_sync().load()
    scene = AcousticScene.from_world(world)
    rooms = AcousticRoomSpecBuilder().from_world(world)
    graph = AcousticWorldGraph.from_world(world, rooms)

    overlaps: list[tuple[str, str]] = []
    for level in world.levels.values():
        level_zones = [
            (
                str(zone.name),
                Polygon([(float(c.x), float(c.y)) for c in zone.corners]),
            )
            for zone in level.zones
        ]
        for index, (first_name, first) in enumerate(level_zones):
            for second_name, second in level_zones[index + 1 :]:
                if first.intersection(second).area > 1e-6:
                    overlaps.append((first_name, second_name))

    root = worlds_root or (Path(get_package_share_directory("arena_simulation_setup")) / "worlds")
    traversable = 0
    uncovered = 0
    examples: list[tuple[float, float]] = []
    map_issues: list[str] = []
    for level_id in sorted(world.levels):
        level_zone_polygons = tuple(Polygon([(float(corner.x), float(corner.y)) for corner in level_zone.corners]) for level_zone in world.levels[level_id].zones)
        level_root = root / world_name / str(level_id)
        map_yaml = level_root / "map.yaml"
        if not map_yaml.exists():
            map_issues.append(f"level {level_id}: missing map.yaml")
            continue
        config = yaml.safe_load(map_yaml.read_text(encoding="utf-8"))
        image_path = level_root / str(config["image"])
        if not image_path.exists():
            map_issues.append(f"level {level_id}: missing map image {image_path.name!r}")
            continue
        opened = Image.open(image_path)
        alpha: np.ndarray | None = None
        if "transparency" in opened.info or opened.mode == "RGBA":
            rgba = opened.convert("RGBA")
            alpha = np.asarray(rgba.getchannel("A"))
            opened = rgba
        image = np.asarray(opened.convert("L"))
        resolution = float(config["resolution"])
        origin_x, origin_y, _ = map(float, config.get("origin", (0, 0, 0)))
        free_threshold = float(config.get("free_thresh", 0.196))
        negate = bool(config.get("negate", 0))

        for row in range(0, image.shape[0], stride_cells):
            for column in range(0, image.shape[1], stride_cells):
                if alpha is not None and alpha[row, column] < 255:
                    continue
                intensity = float(image[row, column]) / 255.0
                occupancy = intensity if negate else 1.0 - intensity
                if occupancy >= free_threshold:
                    continue
                x = origin_x + (column + 0.5) * resolution
                y = origin_y + (image.shape[0] - row - 0.5) * resolution
                traversable += 1
                if not any(polygon.covers(Point(x, y)) for polygon in level_zone_polygons):
                    uncovered += 1
                    if len(examples) < 8:
                        examples.append((x, y))

    return AcousticWorldAudit(
        world_name=world_name,
        rooms=len(rooms),
        door_portals=sum(p.portal_kind == "door" for p in graph.portals),
        opening_portals=sum(p.portal_kind == "opening" for p in graph.portals),
        connected_components=len(graph.connected_components()),
        unpaired_doors=len(graph.unpaired_doors),
        overlapping_zone_pairs=tuple(overlaps),
        sampled_traversable_cells=traversable,
        uncovered_traversable_cells=uncovered,
        uncovered_examples=tuple(examples),
        map_issues=tuple(map_issues),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Arena worlds for acoustic coverage and connectivity")
    parser.add_argument("worlds", nargs="*", help="world names; default: all")
    parser.add_argument("--stride-cells", type=int, default=10)
    args = parser.parse_args()
    root = Path(get_package_share_directory("arena_simulation_setup")) / "worlds"
    names = args.worlds or sorted(path.name for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
    failed = False
    for name in names:
        report = audit_world(
            name,
            worlds_root=root,
            stride_cells=args.stride_cells,
        )
        status = "PASS" if report.complete else "INCOMPLETE"
        failed |= not report.complete
        print(f"{status:10} {name:28} rooms={report.rooms:3} doors={report.door_portals:3} openings={report.opening_portals:3} components={report.connected_components:2} unpaired={report.unpaired_doors:3} uncovered={report.uncovered_traversable_cells}/{report.sampled_traversable_cells}")
        if report.uncovered_examples:
            print(f"  uncovered examples: {report.uncovered_examples}")
        if report.overlapping_zone_pairs:
            print(f"  overlapping zones: {report.overlapping_zone_pairs}")
        if report.map_issues:
            print(f"  map issues: {report.map_issues}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
