"""Street blocks: road, curb, verge, sidewalk and frontage bands around street axes, with crosswalks, street furniture and a sidewalk robot mission."""

from __future__ import annotations

import dataclasses
import math
import random
import re
import typing

import pydantic
import shapely
import yaml

from arena_simulation_setup.generative.scenario.promptgen.profiles import PROFILES
from arena_simulation_setup.tree.World import World
from arena_simulation_setup.tree.World.Scenario import SCENARIO_FORMAT

from . import PedestrianConfiguration, WithPedestrians
from .layout import Area, Layout, Note, Region, compile_layout

SCENARIO = 'default'

LAYOUTS = ('straight', 'curve', 'corner', 'intersection', 'crosswalk', 'narrow_passage', 'plaza', 'block')
BEHAVIORS = ('obstructing', 'conversing', 'queueing', 'frontal', 'lateral', 'overtaking', 'crossing', 'gesturing')
SCENARIOS = ('empty', 'default', *BEHAVIORS)
KINDS = ('crosswalk', 'road', 'curb', 'verge', 'sidewalk', 'frontage')

GRID = 0.01
MIN_PART_AREA = 0.05
LEAD_IN = 6.0
LEAD_OUT = 4.0
CROSSWALK_WIDTH = 3.0
PASSAGE_WIDTH = 1.0
PASSAGE_LENGTH = 5.0
BLOCK_PITCH = 72.0
CURVE_STEP = 0.5
FILLET_RADIUS = 2.0
ARC_SEGMENTS = 16
DEFAULT_PEDESTRIANS = 4
FLOW_SPEED = (1.1, 1.65)
FLOW_START_S = 3.0
GROUP_SPEED = 1.2
CROSSING_LEAD = 0.5
CROSSING_LATERAL = 0.6
PED_KEEPOUT = 1.5
CROSSWALK_KEEPOUT = 1.0
FURNITURE_GAP = 0.3
SPOT_TRIES = 400

FACING = {'along': 0.0, 'back': math.pi, 'left': math.pi / 2, 'right': -math.pi / 2}

PLACEMENT: dict[str, list[tuple[float, float, str]]] = {
    'obstructing': [(12.0, 0.0, 'along')],
    'conversing': [(12.0, -0.6, 'left'), (12.0, 0.0, 'left'), (12.0, 0.6, 'right')],
    'queueing': [(12.0, -1.5 + 0.6 * index, 'left') for index in range(4)],
    'frontal': [(25.0, 0.0, 'back')],
    'lateral': [(8.0, -3.0, 'left')],
    'overtaking': [(-4.0, 0.8, 'along')],
    'gesturing': [(12.0, 0.0, 'back')],
}

FURNITURE: dict[str, tuple[str, str, float]] = {
    'bench': ('Hospital/Waiting_Room_Bench', 'frontage', 0.8),
    'bollard': ('Warehouse/Traffic_Cone', 'verge', 0.3),
    'trash_bin': ('Office/SM_TrashCan', 'frontage', 0.35),
    'lamp_post': ('Hospital/IV_Stand', 'verge', 0.3),
    'tree': ('Office/SM_Plant03', 'verge', 0.5),
    'sign': ('Warehouse/Wet_Floor_Sign', 'verge', 0.3),
}

_ZONE = re.compile(r'^(?P<kind>[a-z]+)_\d+(?:_\d+)?$')

Point = tuple[float, float]


@dataclasses.dataclass
class Plan:
    """Street axes with the sidewalk route along them and what is laid over or cut out of the bands."""

    axes: list[shapely.LineString]
    route: shapely.LineString
    crosswalks: list[shapely.Polygon] = dataclasses.field(default_factory=list)
    voids: list[shapely.Polygon] = dataclasses.field(default_factory=list)
    plazas: list[shapely.Polygon] = dataclasses.field(default_factory=list)


def _snap(geometry: shapely.geometry.base.BaseGeometry) -> shapely.geometry.base.BaseGeometry:
    return shapely.set_precision(geometry, GRID)


def _parts(geometry: shapely.geometry.base.BaseGeometry) -> list[shapely.Polygon]:
    if isinstance(geometry, shapely.Polygon):
        found = [geometry]
    elif isinstance(geometry, shapely.MultiPolygon | shapely.GeometryCollection):
        found = [part for item in geometry.geoms for part in _parts(item)]
    else:
        found = []
    return sorted((part for part in found if part.area >= MIN_PART_AREA), key=lambda part: (round(part.bounds[0], 3), round(part.bounds[1], 3)))


class WorldGeneratorSidewalk(WithPedestrians):
    """A street segment, corner, junction or city-block grid with its sidewalk as the robot's route, street furniture in the verge and frontage, and pedestrians by scenario."""

    class Configuration(PedestrianConfiguration):
        layout: str = pydantic.Field('straight', description=f'Street layout, one of {", ".join(LAYOUTS)}.')
        length: float = pydantic.Field(40.0, gt=0, description='Length of the street in meters.')
        blocks: int = pydantic.Field(2, ge=1, description='Number of city blocks per side in the block layout.')
        sidewalk_width: float = pydantic.Field(4.0, gt=0, description='Sidewalk band width in meters.')
        road_width: float = pydantic.Field(8.0, gt=0, description='Full road width in meters, half of it on each side of the axis.')
        curb_width: float = pydantic.Field(0.3, gt=0, description='Curb band width in meters between the road and the verge.')
        verge_width: float = pydantic.Field(1.5, gt=0, description='Verge band width in meters between the curb and the sidewalk.')
        frontage_width: float = pydantic.Field(2.0, gt=0, description='Frontage band width in meters in front of the buildings.')
        obstacles: int = pydantic.Field(6, ge=0, description='Number of street furniture pieces in the verge and frontage (0 = none).')
        scenario: str = pydantic.Field('empty', description=f'Pedestrian scenario, one of {", ".join(SCENARIOS)}.')

        @pydantic.field_validator('layout')
        @classmethod
        def _layout(cls, value: str) -> str:
            if value not in LAYOUTS:
                raise ValueError(f'layout {value!r} is not one of {", ".join(LAYOUTS)}')
            return value

        @pydantic.field_validator('scenario')
        @classmethod
        def _scenario(cls, value: str) -> str:
            if value not in SCENARIOS:
                raise ValueError(f'scenario {value!r} is not one of {", ".join(SCENARIOS)}')
            return value

    config: Configuration
    _plan: Plan
    _seed: int
    _layout: Layout | None
    _bands: dict[str, shapely.geometry.base.BaseGeometry] | None

    def configure(self, configuration: dict) -> None:
        self.config = self.Configuration.model_validate(configuration)
        self.warnings = []
        self.diagnostics = None
        self._layout = None
        self._bands = None
        self._seed = self.rng.getrandbits(32)
        self._plan = {
            'straight': self._straight,
            'curve': self._curve,
            'corner': self._corner,
            'intersection': self._intersection,
            'crosswalk': self._crosswalk,
            'narrow_passage': self._narrow_passage,
            'plaza': self._plaza,
            'block': self._block,
        }[self.config.layout]()
        self._group()

    @property
    def route(self) -> shapely.LineString:
        """The sidewalk centerline from the robot start to its goal."""
        return self._plan.route

    @property
    def edges(self) -> list[float]:
        """Distances from a street axis to the outer edge of road, curb, verge, sidewalk and frontage."""
        c = self.config
        edges = [c.road_width / 2]
        for width in (c.curb_width, c.verge_width, c.sidewalk_width, c.frontage_width):
            edges.append(edges[-1] + width)
        return edges

    @property
    def _offset(self) -> float:
        return self.edges[2] + self.config.sidewalk_width / 2

    def _arm(self) -> float:
        return self._offset + CROSSWALK_WIDTH / 2 + 4.0

    def _across(self, center: Point, direction: Point) -> shapely.Polygon:
        """Crosswalk over the arm leaving *center* along *direction*, at the sidewalk centerline of the cross street."""
        half = self.edges[2]
        px, py = center[0] + direction[0] * self._offset, center[1] + direction[1] * self._offset
        hx, hy = (half, CROSSWALK_WIDTH / 2) if direction[0] == 0 else (CROSSWALK_WIDTH / 2, half)
        return shapely.box(px - hx, py - hy, px + hx, py + hy)

    def _straight(self) -> Plan:
        length = self.config.length
        axis = shapely.LineString([(-LEAD_IN, 0.0), (length + LEAD_OUT, 0.0)])
        return Plan(axes=[axis], route=shapely.LineString([(0.0, self._offset), (length, self._offset)]))

    def _curve(self) -> Plan:
        length = self.config.length
        offset = self._offset
        radius = 3 * self.edges[4]
        sign = random.Random(self._seed).choice((1.0, -1.0))
        steps = max(2, round(length / CURVE_STEP))
        step = length / steps
        x = y = heading = 0.0
        axis: list[Point] = [(-LEAD_IN, 0.0), (0.0, 0.0)]
        route: list[Point] = [(0.0, offset)]
        for index in range(steps):
            turn = sign if index < steps / 2 else -sign
            middle = heading + turn * step / (2 * radius)
            x += step * math.cos(middle)
            y += step * math.sin(middle)
            heading += turn * step / radius
            axis.append((x, y))
            route.append((x - offset * math.sin(heading), y + offset * math.cos(heading)))
        axis.append((x + LEAD_OUT * math.cos(heading), y + LEAD_OUT * math.sin(heading)))
        return Plan(axes=[shapely.LineString(axis)], route=shapely.LineString(route))

    def _corner(self) -> Plan:
        offset = self._offset
        leg = max((self.config.length - math.pi / 2 * offset) / 2, 2.0)
        axis = shapely.LineString([(-LEAD_IN, 0.0), (leg, 0.0), (leg, -(leg + LEAD_OUT))])
        arc = [(leg + offset * math.cos(a), offset * math.sin(a)) for a in (math.pi / 2 * (1 - k / ARC_SEGMENTS) for k in range(ARC_SEGMENTS + 1))]
        return Plan(axes=[axis], route=shapely.LineString([(0.0, offset), *arc, (leg + offset, -leg)]))

    def _junction(self, through: bool) -> Plan:
        length = self.config.length
        center = length / 2
        reach = self._arm()
        main = shapely.LineString([(-LEAD_IN, 0.0), (length + LEAD_OUT, 0.0)])
        side = shapely.LineString([(center, -reach if through else 0.0), (center, reach)])
        arms = [(0.0, 1.0), (0.0, -1.0), (1.0, 0.0), (-1.0, 0.0)] if through else [(0.0, 1.0)]
        return Plan(axes=[main, side], route=shapely.LineString([(0.0, self._offset), (length, self._offset)]), crosswalks=[self._across((center, 0.0), arm) for arm in arms])

    def _intersection(self) -> Plan:
        return self._junction(True)

    def _crosswalk(self) -> Plan:
        return self._junction(False)

    def _narrow_passage(self) -> Plan:
        plan = self._straight()
        offset = self._offset
        edges = self.edges
        left, right = self.config.length / 2 - PASSAGE_LENGTH / 2, self.config.length / 2 + PASSAGE_LENGTH / 2
        plan.voids = [shapely.box(left, offset + PASSAGE_WIDTH / 2, right, edges[4] + 1.0), shapely.box(left, edges[0], right, offset - PASSAGE_WIDTH / 2)]
        return plan

    def _plaza(self) -> Plan:
        plan = self._straight()
        size = max(3 * self.config.sidewalk_width, self.config.length / 2)
        center = self.config.length / 2
        plan.plazas = [shapely.box(center - size / 2, self.edges[2], center + size / 2, self.edges[2] + size)]
        return plan

    def _block(self) -> Plan:
        n = self.config.blocks
        offset = self._offset
        span = n * BLOCK_PITCH
        arm = self._arm()
        axes = [shapely.LineString([(-arm, k * BLOCK_PITCH), (span + arm, k * BLOCK_PITCH)]) for k in range(n + 1)]
        axes += [shapely.LineString([(k * BLOCK_PITCH, -arm), (k * BLOCK_PITCH, span + arm)]) for k in range(n + 1)]
        crosswalks = [self._across((i * BLOCK_PITCH, j * BLOCK_PITCH), arm_direction) for i in range(n + 1) for j in range(n + 1) for arm_direction in ((0.0, 1.0), (0.0, -1.0), (1.0, 0.0), (-1.0, 0.0))]
        start = self.edges[4] + 0.2
        corner = (span - offset, offset)
        center = (corner[0] - FILLET_RADIUS, corner[1] + FILLET_RADIUS)
        arc = [(center[0] + FILLET_RADIUS * math.cos(a), center[1] + FILLET_RADIUS * math.sin(a)) for a in (-math.pi / 2 * (1 - k / ARC_SEGMENTS) for k in range(ARC_SEGMENTS + 1))]
        route = shapely.LineString([(start, offset), *arc, (corner[0], span - start)])
        return Plan(axes=axes, route=route, crosswalks=crosswalks)

    def _bands_of(self) -> dict[str, shapely.geometry.base.BaseGeometry]:
        if self._bands is None:
            plan = self._plan
            reach = []
            for index, edge in enumerate(self.edges):
                shape = shapely.union_all([axis.buffer(edge, cap_style='flat', join_style='round') for axis in plan.axes])
                if index >= 3 and plan.plazas:
                    shape = shapely.union_all([shape, *plan.plazas])
                if plan.voids:
                    shape = shape.difference(shapely.union_all(plan.voids))
                reach.append(_snap(shape))
            crossing = _snap(shapely.union_all(plan.crosswalks).intersection(reach[4])) if plan.crosswalks else shapely.Polygon()
            bands: dict[str, shapely.geometry.base.BaseGeometry] = {'crosswalk': crossing}
            inner: shapely.geometry.base.BaseGeometry = shapely.Polygon()
            for kind, outer in zip(KINDS[1:], reach, strict=True):
                bands[kind] = _snap(outer.difference(inner).difference(crossing))
                inner = outer
            bands['free'] = reach[4]
            self._bands = bands
        return self._bands

    def compute(self) -> World:
        bands = self._bands_of()
        regions = [Region(name=f'{kind}_{index}', polygon=part) for kind in KINDS for index, part in enumerate(_parts(bands[kind]))]
        self._layout = Layout(areas=[Area(polygon=shapely.MultiPolygon(_parts(bands['free'])))], regions=regions)
        world, self.diagnostics = compile_layout(self._layout, grid=GRID)
        grouped: dict[str, list] = {}
        for zone in world.all_levels[0].zones:
            match = _ZONE.match(zone.name)
            if match:
                grouped.setdefault(match['kind'], []).append(zone)
        for kind, zones in grouped.items():
            for index, zone in enumerate(zones):
                zone.name = kind if len(zones) == 1 else f'{kind}_{index}'
        return world

    def layout(self) -> Layout | None:
        return self._layout

    def params(self) -> dict[str, typing.Any]:
        return {'modes': {'robots': 'scenario', 'obstacles': 'scenario'}, 'params': {'robots': {'file': SCENARIO}, 'obstacles': {'file': SCENARIO}}}

    def files(self) -> dict[str, bytes]:
        humans = self._humans()
        document: dict[str, typing.Any] = {'format': SCENARIO_FORMAT}
        objects = self._objects([human['pose'] for human in humans])
        if objects:
            document['objects'] = objects
        if humans:
            document['humans'] = humans
        start, goal = self._pose(0.0, 0.0), self._pose(self._plan.route.length, 0.0)
        document['robots'] = [{'start': self._list(start), 'phases': [{'goto': self._list(goal)}]}]
        return {f'scenarios/{SCENARIO}/scenario.yaml': yaml.safe_dump(document, sort_keys=False, default_flow_style=None).encode('utf-8')}

    @staticmethod
    def _list(pose: tuple[float, float, float]) -> list[float]:
        return [round(pose[0], 2), round(pose[1], 2), round(pose[2], 3)]

    def _pose(self, s: float, d: float, facing: str = 'along') -> tuple[float, float, float]:
        """Position at arc *s* along the route and lateral *d* to its left, with the heading turned by *facing*."""
        route = self._plan.route
        if s < 0:
            ahead = route.interpolate(0.1)
            origin = route.interpolate(0.0)
            heading = math.atan2(ahead.y - origin.y, ahead.x - origin.x)
            x, y = origin.x + s * math.cos(heading), origin.y + s * math.sin(heading)
        else:
            low, high = max(s - 0.1, 0.0), min(s + 0.1, route.length)
            before, after = route.interpolate(low), route.interpolate(high)
            heading = math.atan2(after.y - before.y, after.x - before.x)
            here = route.interpolate(min(s, route.length))
            x, y = here.x, here.y
        return x - d * math.sin(heading), y + d * math.cos(heading), heading + FACING[facing]

    def _group(self) -> list[tuple[float, float, str]]:
        """The (s, d, facing) of each pedestrian of the scenario's behavior group."""
        behavior = self.config.scenario
        if behavior not in BEHAVIORS:
            return []
        if behavior == 'crossing':
            return self._crossing()
        placement = PLACEMENT[behavior]
        farthest = max(s for s, _, _ in placement)
        if farthest > self._plan.route.length - 1.0:
            raise ValueError(f'{behavior} places a pedestrian {farthest} m along a route of {self._plan.route.length:.1f} m')
        return placement

    def _crossing(self) -> list[tuple[float, float, str]]:
        route = self._plan.route
        spans = []
        for rect in self._plan.crosswalks:
            hit = rect.intersection(route)
            if not hit.is_empty:
                along = [route.project(shapely.Point(x, y)) for x, y in shapely.get_coordinates(hit)]
                spans.append((min(along), max(along)))
        if not spans:
            raise ValueError(f'the {self.config.layout} layout has no crosswalk on its route')
        near, far = min(spans)
        return [(near - CROSSING_LEAD, CROSSING_LATERAL, 'along'), (far + CROSSING_LEAD, -CROSSING_LATERAL, 'back')]

    @staticmethod
    def _human(name: str, model: str, pose: tuple[float, float, float], agent_type: str, speed: float, waypoints: list[list[float]], mode: str) -> dict[str, typing.Any]:
        return {'name': name, 'model': model, 'pose': WorldGeneratorSidewalk._list(pose), 'agent': {'agent_type': agent_type, 'desired_velocity': round(speed, 2)}, 'waypoints': waypoints, 'waypoint_mode': mode}

    def _humans(self) -> list[dict[str, typing.Any]]:
        c = self.config
        if c.scenario == 'empty':
            return []
        rng = random.Random(self._seed + 2)
        meshes = PROFILES['adult'].meshes
        humans = []
        for index, (s, d, facing) in enumerate(self._group(), start=1):
            pose = self._pose(s, d, facing)
            humans.append(self._human(f'{c.scenario}_{index}', rng.choice(meshes), pose, f'sidewalk_{c.scenario}', GROUP_SPEED, [self._list(pose)], 'once'))
        count = c.pedestrians if c.pedestrians >= 0 else (DEFAULT_PEDESTRIANS if c.scenario == 'default' else 0)
        length = self._plan.route.length
        half = max(c.sidewalk_width / 2 - 0.6, 0.0)
        for index in range(count):
            s = min(max(FLOW_START_S, (index + 1) / (count + 1) * length + rng.uniform(-1.0, 1.0)), length - 1.0)
            d = rng.uniform(-half, half)
            forward = index % 2 == 0
            pose = self._pose(s, d, 'along' if forward else 'back')
            far = self._list(self._pose(length if forward else 0.0, d, 'along' if forward else 'back'))
            humans.append(self._human(f'ped_{index + 1}', rng.choice(meshes), pose, 'adult', rng.uniform(*FLOW_SPEED), [far, self._list(pose)], 'repeat'))
        return humans

    def _objects(self, occupied: list[list[float]]) -> list[dict[str, typing.Any]]:
        c = self.config
        if c.obstacles == 0:
            return []
        rng = random.Random(self._seed + 1)
        bands = self._bands_of()
        keepout = shapely.union_all(
            [
                self._plan.route.buffer(c.sidewalk_width / 2),
                *(rect.buffer(CROSSWALK_KEEPOUT) for rect in self._plan.crosswalks),
                *(shapely.Point(pose[0], pose[1]).buffer(PED_KEEPOUT) for pose in occupied),
            ]
        )
        taken: list[tuple[shapely.Point, float]] = []
        objects = []
        for index in range(c.obstacles):
            category = rng.choice(sorted(FURNITURE))
            model, zone, radius = FURNITURE[category]
            room = bands[zone].buffer(-radius).difference(keepout)
            spot = self._spot(rng, room, taken, radius)
            if spot is None:
                self.warnings.append(Note(row=0, col=0, text=f'no room for {category} {index + 1}'))
                continue
            taken.append((spot, radius))
            yaw = self._tangent(spot) if category == 'bench' else rng.uniform(-math.pi, math.pi)
            objects.append({'name': f'{category}_{index + 1}', 'model': model, 'pose': [round(spot.x, 2), round(spot.y, 2), round(yaw, 3)]})
        return objects

    @staticmethod
    def _spot(rng: random.Random, room: shapely.geometry.base.BaseGeometry, taken: list[tuple[shapely.Point, float]], radius: float) -> shapely.Point | None:
        if room.is_empty:
            return None
        minx, miny, maxx, maxy = room.bounds
        for _ in range(SPOT_TRIES):
            point = shapely.Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
            if room.contains(point) and all(point.distance(other) >= radius + size + FURNITURE_GAP for other, size in taken):
                return point
        return None

    def _tangent(self, point: shapely.Point) -> float:
        axis = min(self._plan.axes, key=point.distance)
        s = axis.project(point)
        before, after = axis.interpolate(max(s - 0.5, 0.0)), axis.interpolate(min(s + 0.5, axis.length))
        return math.atan2(after.y - before.y, after.x - before.x)
