from arena_simulation_setup.tree.World import WorldIdentifier

from task_generator.auditory.acoustic_room_spec import AcousticRoomSpecBuilder
from task_generator.auditory.acoustic_scene import AcousticScene
from task_generator.auditory.acoustic_world_graph import AcousticWorldGraph


def test_current_multilevel_world_schema_builds_rooms_and_portals():
    world = WorldIdentifier("hospital_1").resolve_sync().load()

    assert hasattr(world, "levels")
    scene = AcousticScene.from_world(world)
    rooms = AcousticRoomSpecBuilder().from_world(world)
    graph = AcousticWorldGraph.from_world(world, rooms)

    assert len(scene.zones) == len(rooms) > 1
    assert scene.zone_at_xy(-1e-10, 0.0).name == "reception"
    assert graph.portals
    assert any(
        portal.connects("central_hallway", "reception")
        for portal in graph.portals
    )
    opening = next(
        portal
        for portal in graph.portals
        if portal.connects("central_hallway", "sub_hallway")
    )
    assert opening.portal_kind == "opening"
    route = graph.find_portal_route(
        "operating_room",
        "waiting_area",
        source_xy=(13.69, 18.00),
        listener_xy=(23.95, 6.81),
        max_portals=4,
    )
    assert route is not None
    assert route.zones == (
        "operating_room",
        "central_hallway",
        "sub_hallway",
        "waiting_area",
    )
    assert route.hop_count == 3
    assert graph.find_portal_route(
        "operating_room",
        "waiting_area",
        source_xy=(13.69, 18.00),
        listener_xy=(23.95, 6.81),
        max_portals=2,
    ) is None

    doors_only = AcousticWorldGraph.from_world(
        world,
        rooms,
        derive_opening_portals=False,
    )
    assert not any(
        portal.connects("central_hallway", "sub_hallway")
        for portal in doors_only.portals
    )
    assert doors_only.find_portal_route(
        "operating_room",
        "waiting_area",
        source_xy=(13.69, 18.00),
        listener_xy=(23.95, 6.81),
        max_portals=12,
    ) is None
