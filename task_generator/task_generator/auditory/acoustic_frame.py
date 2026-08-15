from __future__ import annotations

import attrs
from nav_msgs.msg import OccupancyGrid
from shapely.affinity import translate

from .acoustic_room_spec import AcousticRoomSpec
from .acoustic_scene import AcousticScene, AcousticWall
from .acoustic_world_graph import (
    AcousticPortal,
    AcousticWorldGraph,
    UnpairedDoor,
)

Offset2D = tuple[float, float]


def runtime_acoustic_offset(
    occupancy_map: OccupancyGrid,
    authored_map_origin: Offset2D,
) -> Offset2D:
    """Return the authored-to-runtime translation already applied to the map."""
    return (
        float(occupancy_map.info.origin.position.x) - authored_map_origin[0],
        float(occupancy_map.info.origin.position.y) - authored_map_origin[1],
    )


def realize_acoustic_geometry(
    scene: AcousticScene,
    rooms: tuple[AcousticRoomSpec, ...],
    graph: AcousticWorldGraph,
    offset: Offset2D,
) -> tuple[AcousticScene, tuple[AcousticRoomSpec, ...], AcousticWorldGraph]:
    """Translate authored acoustic geometry into the runtime map frame."""
    dx, dy = offset
    realized_rooms = tuple(_translate_room(room, dx, dy) for room in rooms)
    realized_scene = attrs.evolve(
        scene,
        zones=tuple(
            attrs.evolve(
                zone,
                polygon=translate(zone.polygon, xoff=dx, yoff=dy),
            )
            for zone in scene.zones
        ),
        walls=tuple(_translate_wall(wall, dx, dy) for wall in scene.walls),
    )
    realized_graph = attrs.evolve(
        graph,
        rooms=realized_rooms,
        portals=tuple(
            _translate_portal(portal, dx, dy) for portal in graph.portals
        ),
        zone_polygons=tuple(
            (name, translate(polygon, xoff=dx, yoff=dy))
            for name, polygon in graph.zone_polygons
        ),
        unpaired_doors=tuple(
            _translate_unpaired_door(door, dx, dy)
            for door in graph.unpaired_doors
        ),
    )
    return realized_scene, realized_rooms, realized_graph


def realize_rooms_and_graph(
    rooms: tuple[AcousticRoomSpec, ...],
    graph: AcousticWorldGraph,
    offset: Offset2D,
) -> tuple[tuple[AcousticRoomSpec, ...], AcousticWorldGraph]:
    """Translate playback room and portal geometry into the runtime map frame."""
    dx, dy = offset
    realized_rooms = tuple(_translate_room(room, dx, dy) for room in rooms)
    return realized_rooms, attrs.evolve(
        graph,
        rooms=realized_rooms,
        portals=tuple(
            _translate_portal(portal, dx, dy) for portal in graph.portals
        ),
        zone_polygons=tuple(
            (name, translate(polygon, xoff=dx, yoff=dy))
            for name, polygon in graph.zone_polygons
        ),
        unpaired_doors=tuple(
            _translate_unpaired_door(door, dx, dy)
            for door in graph.unpaired_doors
        ),
    )


def _translate_room(
    room: AcousticRoomSpec,
    dx: float,
    dy: float,
) -> AcousticRoomSpec:
    return attrs.evolve(
        room,
        boundary=tuple(
            attrs.evolve(
                boundary,
                start=_translate_xy(boundary.start, dx, dy),
                end=_translate_xy(boundary.end, dx, dy),
            )
            for boundary in room.boundary
        ),
    )


def _translate_wall(
    wall: AcousticWall,
    dx: float,
    dy: float,
) -> AcousticWall:
    return attrs.evolve(
        wall,
        start=_translate_xy(wall.start, dx, dy),
        end=_translate_xy(wall.end, dx, dy),
    )


def _translate_portal(
    portal: AcousticPortal,
    dx: float,
    dy: float,
) -> AcousticPortal:
    return attrs.evolve(
        portal,
        start=_translate_xy(portal.start, dx, dy),
        end=_translate_xy(portal.end, dx, dy),
    )


def _translate_unpaired_door(
    door: UnpairedDoor,
    dx: float,
    dy: float,
) -> UnpairedDoor:
    return attrs.evolve(
        door,
        start=_translate_xy(door.start, dx, dy),
        end=_translate_xy(door.end, dx, dy),
    )


def _translate_xy(
    point: tuple[float, float],
    dx: float,
    dy: float,
) -> tuple[float, float]:
    return point[0] + dx, point[1] + dy
