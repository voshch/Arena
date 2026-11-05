"""Procedural parametric concourse world generator with semantic dock sockets."""

from __future__ import annotations

import dataclasses
import enum
import math
import random
from typing import ClassVar, Literal, cast, final

import numpy as np
import shapely
import shapely.affinity
import shapely.ops
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from typing_extensions import override

from arena_simulation_setup.shared import Door, Elevator, Position
from arena_simulation_setup.tree.World import World
from arena_simulation_setup.tree.World.Scenario import SCENARIO_FORMAT

from . import BaseConfiguration, WorldGeneratorImpl
from .layout import Area, Layout, Region, compile_layout

SCENARIO = "default"

ANCHOR_EPSILON_DS = 0.05
MERGE_GAP_M = 0.2
ELEVATOR_AXIS_TOLERANCE_DEG = 10.0
ELEVATOR_END_TOLERANCE_M = 0.05
BUFFER_PAD_M = 0.05
TRUNK_DOCK_WIDTH_RATIO = 0.85
BRANCH_DOCK_DEPTH_RATIO = 0.45
BRANCH_DOCK_WIDTH_RATIO = 0.45
DEFAULT_ROOM_DOOR_WIDTH_M = 1.2
DEFAULT_ABSORPTION_RADIUS_M = 0.5
DEFAULT_ACTIVE_WINDOW_END_S = 999999.0

DEFAULT_OPPOSING_FLOW_RATE = 0.4
DEFAULT_CROSS_FLOW_RATE = 0.3
DEFAULT_FLOW_VELOCITY_MIN = 1.0
DEFAULT_FLOW_VELOCITY_MAX = 1.5
DEFAULT_CROSS_FLOW_VELOCITY_MIN = 0.9
DEFAULT_CROSS_FLOW_VELOCITY_MAX = 1.4


class ConcourseShape(str, enum.Enum):
    """Concourse spine geometry archetype."""

    STRAIGHT = "straight"
    SMOOTH_S_BEND = "smooth_s_bend"
    SMOOTH_L_BEND = "smooth_l_bend"
    S_BEND = "s_bend"
    L_BEND = "l_bend"


class BranchSide(str, enum.Enum):
    """Lateral side relative to spine forward heading."""

    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


class PillarShape(str, enum.Enum):
    """Cross-sectional shape of structural pillars."""

    CYLINDER = "cylinder"
    RECTANGLE = "rectangle"


class RobotTaskMode(str, enum.Enum):
    """Robot navigation mission mode."""

    MAIN_THROUGH = "main_through"
    TURN_INTO_BRANCH = "turn_into_branch"
    ENTER_ROOM = "enter_room"


@dataclasses.dataclass(frozen=True)
class SpineAnchor:
    """Local Frenet frame and half-width at a spine arc-length station."""

    point: tuple[float, float]
    tangent: tuple[float, float]
    normal: tuple[float, float]
    half_width: float


@dataclasses.dataclass(frozen=True)
class BranchPlacement:
    """Calculated position and orientation for a branch side attachment."""

    branch_index: int
    side: BranchSide
    door_width: float | None
    anchor: SpineAnchor
    tip_point: tuple[float, float]
    direction: tuple[float, float]
    junction_distance: float


class SideBranchFeature(BaseModel):
    """Parametric side corridor or room attached at arc-length s along the spine."""

    at_s: float = Field(10.0, ge=1.0)
    width: float = Field(3.0, ge=0.8)
    length: float = Field(6.0, ge=1.0)
    side: BranchSide = Field(default=BranchSide.BOTH)
    angle_deg: float = Field(90.0, ge=15.0, le=165.0)
    is_room: bool = Field(False)
    door_width: float | None = Field(default=None, ge=0.5)
    jamb_thickness: float = Field(0.25, ge=0.05)
    door_activation_distance: float = Field(1.2, ge=0.5)

    @model_validator(mode="after")
    def validate_door_width(self) -> SideBranchFeature:
        """Ensure doorway width does not exceed branch corridor width."""
        if self.door_width is not None and self.door_width > self.width:
            raise ValueError(f"door_width ({self.door_width} m) cannot exceed branch width ({self.width} m)")
        return self


class ElevatorBayFeature(BaseModel):
    """Enclosed elevator compartment at spine start or end."""

    at_s: float = Field(0.0, ge=0.0)
    cabin_width: float = Field(2.5, ge=1.2)
    cabin_depth: float = Field(2.5, ge=1.2)
    door_width: float = Field(1.4, ge=0.8)
    destinations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_door_width(self) -> ElevatorBayFeature:
        """Ensure doorway width does not exceed elevator cabin width."""
        if self.door_width > self.cabin_width:
            raise ValueError(f"door_width ({self.door_width} m) cannot exceed cabin_width ({self.cabin_width} m)")
        return self


class PillarFeature(BaseModel):
    """Single structural column obstacle along the spine."""

    at_s: float = Field(default=12.0, ge=0.5)
    lateral_offset: float = Field(default=0.0)
    lateral_ratio: float | None = Field(default=None, ge=-0.85, le=0.85)
    shape: PillarShape = Field(default=PillarShape.CYLINDER)
    radius: float = Field(default=0.35, ge=0.1)
    width: float = Field(default=0.6, ge=0.1)
    depth: float = Field(default=0.6, ge=0.1)
    wall_margin: float = Field(default=0.15, ge=0.05)


class ColonnadeFeature(BaseModel):
    """Repeated series of pillars along a spine interval."""

    start_s: float = Field(default=5.0, ge=0.5)
    end_s: float = Field(default=20.0, ge=1.0)
    spacing: float = Field(default=4.0, ge=1.0)
    lateral_offset: float = Field(default=0.0)
    lateral_ratio: float | None = Field(default=None, ge=-0.85, le=0.85)
    radius: float = Field(default=0.3, ge=0.1)
    wall_margin: float = Field(default=0.15, ge=0.05)

    @model_validator(mode="after")
    def validate_interval(self) -> ColonnadeFeature:
        """Ensure colonnade start arc-length does not exceed end arc-length."""
        if self.start_s > self.end_s:
            raise ValueError(f"start_s ({self.start_s}) cannot exceed end_s ({self.end_s})")
        return self


class FlowStreamSpec(BaseModel):
    """Continuous pedestrian crowd flow stream between dock socket zones."""

    name: str = Field(default="stream_0", description="Unique flow stream identifier")
    source_dock: str = Field(
        default="dock_start",
        description="Source dock socket zone name (e.g. dock_start, dock_end, dock_branch_0_left, dock_elevator_0)",
    )
    sink_dock: str = Field(
        default="dock_end",
        description="Target sink dock socket zone name (e.g. dock_start, dock_end, dock_branch_0_right)",
    )
    rate: float = Field(default=DEFAULT_OPPOSING_FLOW_RATE, ge=0.02, le=5.0, description="Pedestrian spawn rate in agents/second")
    agent_type: str = Field(default="adult", description="Humansim agent type: adult, elder, child, hurried, distracted, etc.")
    velocity_min: float = Field(default=DEFAULT_FLOW_VELOCITY_MIN, ge=0.2, description="Minimum desired speed in m/s")
    velocity_max: float = Field(default=DEFAULT_FLOW_VELOCITY_MAX, ge=0.2, description="Maximum desired speed in m/s")
    behavior_tree: str = Field(default="default", description="Behavior tree name from configs/hunav/behavior_trees: default, BTRegularNav.xml, BTScaredNav.xml, etc.")
    active_window: tuple[float, float] = Field(default=(0.0, DEFAULT_ACTIVE_WINDOW_END_S), description="Active time window (t_start, t_end)")
    max_concurrent: int = Field(default=-1, description="Max concurrent agents alive (-1 for unlimited)")
    max_total: int = Field(default=-1, description="Total agents to spawn over episode (-1 for unlimited)")

    @model_validator(mode="after")
    def validate_active_window(self) -> FlowStreamSpec:
        """Ensure active window has 0 <= t0 < t1."""
        t0, t1 = self.active_window
        if t0 < 0.0 or t0 >= t1:
            raise ValueError(f"active_window requires 0 <= t0 < t1, got ({t0}, {t1})")
        return self

    @model_validator(mode="after")
    def validate_velocity_range(self) -> FlowStreamSpec:
        """Ensure minimum desired velocity does not exceed maximum desired velocity."""
        if self.velocity_min > self.velocity_max:
            raise ValueError(f"velocity_min ({self.velocity_min} m/s) cannot exceed velocity_max ({self.velocity_max} m/s)")
        return self


class ConcourseConfiguration(BaseConfiguration):
    """Configuration for the parametric concourse generator.

    Generates concourse geometry alongside semantic dock socket zones:
    - 'dock_start' & 'dock_end': Terminals at the start and end of the main concourse trunk.
    - 'dock_branch_<idx>_<side>': Socket at the extremity of each side branch/corridor/room.
    - 'dock_elevator_<idx>': Socket inside each elevator cabin.

    Use these socket zone names in 'flows' or as 'ref: <dock_name>' in scenario 'regions'
    to attach pedestrian sources and sinks.
    """

    shape: ConcourseShape = Field(default=ConcourseShape.STRAIGHT, description="Spine geometry: straight, smooth_s_bend, smooth_l_bend, s_bend, l_bend")
    length: float = Field(
        24.0,
        ge=6.0,
        description="Longitudinal spine length in meters (nominal extent for smooth curve shapes; actual spine arc length may vary with curvature)",
    )
    width: float = Field(4.0, ge=1.5, description="Main concourse walkway width in meters")
    bend_angle_deg: float = Field(45.0, ge=10.0, le=120.0, description="Angle in degrees for L-bend and S-bend shapes")
    corner_fillet_radius: float = Field(0.0, ge=0.0, le=10.0, description="Spine corner fillet radius in meters for l_bend and s_bend (0 for sharp corners)")
    curve_samples: int = Field(24, ge=8, le=64, description="Sampling resolution for smooth curves")
    s_bend_lateral_ratio: float = Field(0.28, ge=0.05, le=1.0, description="Lateral shift ratio relative to length for smooth S-bends")
    dock_depth: float = Field(1.5, ge=0.5, le=4.0, description="Longitudinal depth in meters of terminal dock socket zones")
    branches: list[SideBranchFeature] = Field(default_factory=list, description="Side branches, connecting corridors, or doorway rooms")
    elevators: list[ElevatorBayFeature] = Field(default_factory=list, description="Enclosed elevator cabins at corridor ends")
    pillars: list[PillarFeature] = Field(default_factory=list, description="Individual structural columns along the spine")
    colonnades: list[ColonnadeFeature] = Field(default_factory=list, description="Repeated colonnade series along the spine")
    flows: list[FlowStreamSpec] = Field(
        default_factory=list,
        description="Crowd flow streams between dock sockets. Sockets are generated at dock_start, dock_end, dock_branch_<idx>_<side>, and dock_elevator_<idx>.",
    )
    robot_task_mode: RobotTaskMode = Field(default=RobotTaskMode.MAIN_THROUGH, description="Robot mission: main_through, turn_into_branch, enter_room")
    robot_target_branch: str | None = Field(default=None, description="Optional explicit dock name target for robot mission (e.g. dock_branch_0_left)")

    @field_validator("robot_target_branch", mode="before")
    @classmethod
    def normalize_empty_target_branch(cls, v: object) -> object:
        """Map empty string to None for ROS panel parameter compatibility."""
        if v == "":
            return None
        return v

    @model_validator(mode="after")
    def validate_features_within_bounds(self) -> ConcourseConfiguration:
        """Ensure all branch, elevator, pillar, and colonnade arc-length positions lie within nominal spine length."""
        for i, branch in enumerate(self.branches):
            if branch.at_s > self.length:
                raise ValueError(f"Branch {i} at_s ({branch.at_s} m) exceeds concourse length ({self.length} m)")
        for i, elevator in enumerate(self.elevators):
            if elevator.at_s > self.length:
                raise ValueError(f"Elevator {i} at_s ({elevator.at_s} m) exceeds concourse length ({self.length} m)")
        for i, pillar in enumerate(self.pillars):
            if pillar.at_s > self.length:
                raise ValueError(f"Pillar {i} at_s ({pillar.at_s} m) exceeds concourse length ({self.length} m)")
        for i, colonnade in enumerate(self.colonnades):
            if colonnade.end_s > self.length:
                raise ValueError(f"Colonnade {i} end_s ({colonnade.end_s} m) exceeds concourse length ({self.length} m)")
        names = [f.name for f in self.flows]
        if len(names) != len(set(names)):
            raise ValueError(f"Flow stream names must be unique, found duplicates: {names}")
        return self


def _oriented_box(
    center: tuple[float, float],
    length: float,
    width: float,
    dir_vec: tuple[float, float],
) -> Polygon:
    """Create a rectangular Polygon of given length and width oriented along dir_vec.

    length is measured along dir_vec; width is measured perpendicular to dir_vec.
    """
    dx, dy = dir_vec
    norm = math.hypot(dx, dy)
    if norm > 1e-9:
        dx, dy = dx / norm, dy / norm
    else:
        dx, dy = 1.0, 0.0
    px, py = -dy, dx
    hl = length / 2.0
    hw = width / 2.0
    cx, cy = center
    return Polygon(
        [
            (cx + dx * hl + px * hw, cy + dy * hl + py * hw),
            (cx - dx * hl + px * hw, cy - dy * hl + py * hw),
            (cx - dx * hl - px * hw, cy - dy * hl - py * hw),
            (cx + dx * hl - px * hw, cy + dy * hl - py * hw),
        ]
    )


@final
class WorldGeneratorConcourse(WorldGeneratorImpl):
    """Procedural concourse world and crowd scenario generator."""

    Configuration: ClassVar[type[BaseConfiguration]] = ConcourseConfiguration
    config: ConcourseConfiguration

    def __init__(self, configuration: dict[str, object], rng: random.Random) -> None:
        self.config = ConcourseConfiguration.model_validate(configuration)
        self._spine: LineString = LineString([(0.0, 0.0), (24.0, 0.0)])
        self._doors: list[Door] = []
        self._dock_metadata: dict[str, dict[str, object]] = {}
        self._semantic_zones: list[tuple[str, str, Polygon | MultiPolygon]] = []
        super().__init__(configuration, rng)

    @override
    def configure(self, configuration: dict[str, object]) -> None:
        """Validate and store the concourse configuration."""
        self.config = ConcourseConfiguration.model_validate(configuration)

    @override
    def compute(self) -> World:
        """Compile spine geometry, semantic zones, walls, and doors into a World."""
        return self._compile_world()

    @override
    def files(self) -> dict[str, bytes]:
        """Serialize default crowd flow scenario into the world archive."""
        if not self._dock_metadata:
            _ = self._compile_world()
        scenario_dict = self._compile_scenario_dict()
        yaml_bytes = yaml.safe_dump(scenario_dict, sort_keys=False).encode("utf-8")
        return {f"scenarios/{SCENARIO}/scenario.yaml": yaml_bytes}

    @override
    def params(self) -> dict[str, object]:
        """Bind the generated world to scenario mode on episode reset."""
        return {
            "modes": {"robots": "scenario", "obstacles": "scenario"},
            "params": {
                "robots": {"file": SCENARIO},
                "obstacles": {"file": SCENARIO},
            },
        }

    def _build_spine(self) -> LineString:
        """Build the 1D centerline spine from the configured shape archetype."""
        total_len = self.config.length
        shape = self.config.shape

        if shape == ConcourseShape.STRAIGHT:
            return LineString([(0.0, 0.0), (total_len, 0.0)])

        if shape == ConcourseShape.SMOOTH_S_BEND:
            lateral = total_len * self.config.s_bend_lateral_ratio
            xs = np.linspace(0.0, total_len, self.config.curve_samples)
            ys = [float(lateral * 0.5 * (1.0 - math.cos(math.pi * float(x) / total_len))) for x in xs]
            return LineString([(float(x), y) for x, y in zip(xs, ys, strict=True)])

        if shape == ConcourseShape.SMOOTH_L_BEND:
            half = total_len * 0.5
            ang = math.radians(self.config.bend_angle_deg)
            p2 = (half + half * math.cos(ang), half * math.sin(ang))
            return LineString(self._discretize_smooth_curve(((0.0, 0.0), (half, 0.0), p2)))

        if shape == ConcourseShape.L_BEND:
            half = total_len * 0.5
            ang = math.radians(self.config.bend_angle_deg)
            return LineString(self._fillet_polyline([(0.0, 0.0), (half, 0.0), (half + half * math.cos(ang), half * math.sin(ang))]))

        seg = total_len / 3.0
        ang = math.radians(self.config.bend_angle_deg)
        p1 = (seg, 0.0)
        p2 = (seg + seg * math.cos(ang), seg * math.sin(ang))
        p3 = (p2[0] + seg, p2[1])
        return LineString(self._fillet_polyline([(0.0, 0.0), p1, p2, p3]))

    def _fillet_polyline(self, pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Replace each interior vertex by a tangent arc of the configured fillet radius."""
        radius = self.config.corner_fillet_radius
        if radius <= 0.0:
            return pts
        out = [pts[0]]
        for a, b, c in zip(pts, pts[1:], pts[2:], strict=False):
            len_in = math.dist(a, b)
            len_out = math.dist(b, c)
            u1 = ((b[0] - a[0]) / len_in, (b[1] - a[1]) / len_in)
            u2 = ((c[0] - b[0]) / len_out, (c[1] - b[1]) / len_out)
            cross = u1[0] * u2[1] - u1[1] * u2[0]
            turn = math.atan2(abs(cross), u1[0] * u2[0] + u1[1] * u2[1])
            if turn < 1e-6:
                out.append(b)
                continue
            tangent_len = min(radius * math.tan(turn / 2.0), len_in / 2.0, len_out / 2.0)
            arc_radius = tangent_len / math.tan(turn / 2.0)
            side = math.copysign(1.0, cross)
            start = (b[0] - u1[0] * tangent_len, b[1] - u1[1] * tangent_len)
            center = (start[0] - u1[1] * side * arc_radius, start[1] + u1[0] * side * arc_radius)
            start_angle = math.atan2(start[1] - center[1], start[0] - center[0])
            steps = max(2, round(self.config.curve_samples * turn / math.pi))
            for k in range(steps + 1):
                angle = start_angle + side * turn * k / steps
                out.append((center[0] + arc_radius * math.cos(angle), center[1] + arc_radius * math.sin(angle)))
        out.append(pts[-1])
        return out

    def _discretize_smooth_curve(
        self,
        corners: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Sample a quadratic Bezier curve across three control points."""
        p0, p1, p2 = corners
        samples = self.config.curve_samples
        pts: list[tuple[float, float]] = []
        for i in range(samples + 1):
            t = i / samples
            x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
            y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
            pts.append((x, y))
        return pts

    def _anchor_at(self, s: float) -> SpineAnchor:
        """Evaluate local Frenet frame and half-width at arc-length s."""
        total = self._spine.length
        s_clamped = max(0.0, min(total, s))
        pt = self._spine.interpolate(s_clamped)
        s_prev = max(0.0, s_clamped - ANCHOR_EPSILON_DS)
        s_next = min(total, s_clamped + ANCHOR_EPSILON_DS)
        p_prev = self._spine.interpolate(s_prev)
        p_next = self._spine.interpolate(s_next)
        dx = p_next.x - p_prev.x
        dy = p_next.y - p_prev.y
        norm = math.hypot(dx, dy) or 1.0
        tx, ty = dx / norm, dy / norm
        return SpineAnchor(
            point=(pt.x, pt.y),
            tangent=(tx, ty),
            normal=(-ty, tx),
            half_width=self.config.width / 2.0,
        )

    def _buffered_main_corridor(self, half_w: float) -> Polygon:
        """Compute the base polygon covering the full main concourse corridor."""
        join_style = "round" if self.config.corner_fillet_radius > 0 else "mitre"
        return self._spine.buffer(half_w, cap_style="flat", join_style=join_style)

    def _validate_actual_spine_bounds(self) -> None:
        """Validate feature attachment stations against the evaluated spine arc length."""
        actual_len = self._spine.length
        shape_name = self.config.shape.value
        for i, branch in enumerate(self.config.branches):
            if branch.at_s > actual_len:
                raise ValueError(f"Branch {i} at_s ({branch.at_s} m) exceeds actual spine length ({actual_len:.2f} m for shape '{shape_name}')")
        for i, elevator in enumerate(self.config.elevators):
            if elevator.at_s > actual_len:
                raise ValueError(f"Elevator {i} at_s ({elevator.at_s} m) exceeds actual spine length ({actual_len:.2f} m for shape '{shape_name}')")
            if elevator.at_s > ELEVATOR_END_TOLERANCE_M and abs(elevator.at_s - actual_len) > ELEVATOR_END_TOLERANCE_M:
                raise ValueError(f"Elevator {i} at_s ({elevator.at_s} m) must be 0 (spine start) or {actual_len:.2f} m (spine end)")
        for i, pillar in enumerate(self.config.pillars):
            if pillar.at_s > actual_len:
                raise ValueError(f"Pillar {i} at_s ({pillar.at_s} m) exceeds actual spine length ({actual_len:.2f} m for shape '{shape_name}')")
        for i, colonnade in enumerate(self.config.colonnades):
            if colonnade.end_s > actual_len:
                raise ValueError(f"Colonnade {i} end_s ({colonnade.end_s} m) exceeds actual spine length ({actual_len:.2f} m for shape '{shape_name}')")

    def _compile_world(self) -> World:
        """Assemble semantic zones, dock sockets, walls, and doors into a World via Layout IR."""
        self._spine = self._build_spine()
        self._validate_actual_spine_bounds()
        self._doors = []
        self._elevators = []
        self._dock_metadata = {}
        self._semantic_zones = []
        self._pillar_obstacles = []

        half_w = self.config.width / 2.0
        total_len = self._spine.length

        self._partition_main_spine(half_w, total_len)
        self._build_side_branches(half_w)
        self._build_elevators(total_len)
        self._apply_pillar_obstacles(half_w)
        self._build_trunk_dock_sockets()
        self._validate_dock_pillar_clearance()

        # Validate dock socket references early so compute() (preview) and files() (save) fail fast
        self._validate_dock_references()

        corridor_poly = self._buffered_main_corridor(half_w)
        if self._pillar_obstacles:
            corridor_poly = corridor_poly.difference(shapely.unary_union(self._pillar_obstacles))
        branch_polys = [poly for name, _, poly in self._semantic_zones if name.startswith(("room_", "branch_"))]
        elev_polys = [poly for name, _, poly in self._semantic_zones if name.startswith("dock_elevator_")]
        free_space = cast("Polygon | MultiPolygon", shapely.unary_union([corridor_poly, *branch_polys, *elev_polys]))
        areas = [Area(polygon=free_space)]
        docks = [Region(name=z_name, polygon=z_poly, description=z_desc) for z_name, z_desc, z_poly in self._semantic_zones if z_name.startswith("dock_")]
        others = [Region(name=z_name, polygon=z_poly, description=z_desc) for z_name, z_desc, z_poly in self._semantic_zones if not z_name.startswith("dock_")]
        world, self.diagnostics = compile_layout(Layout(areas=areas, regions=[*docks, *others]))
        # Doors and elevators have no Layout IR form per design contract (see README.md:37 and hallway.py:119)
        world.all_levels[0].doors.extend(self._doors)
        world.all_levels[0].elevators.extend(self._elevators)

        # Verify that all generated dock socket zones survive layout compilation
        surviving_zones = {z.name for z in world.all_levels[0].zones}
        for dock_name in self._dock_metadata:
            if dock_name not in surviving_zones:
                raise ValueError(f"Dock socket zone '{dock_name}' was completely covered or lost during layout compilation")

        return world

    def _partition_main_spine(self, half_w: float, total_len: float) -> None:
        """Slice the main concourse spine into corridor and intersection zones."""
        main_corridor_poly = self._buffered_main_corridor(half_w)
        join_style = "round" if self.config.corner_fillet_radius > 0 else "mitre"

        intervals: list[tuple[float, float, str]] = []
        for branch in sorted(self.config.branches, key=lambda b: b.at_s):
            b_half = branch.width / 2.0
            s_lo = max(0.0, branch.at_s - b_half)
            s_hi = min(total_len, branch.at_s + b_half)
            intervals.append((s_lo, s_hi, "intersection"))

        merged_intervals: list[tuple[float, float, str]] = []
        for s_lo, s_hi, kind in intervals:
            if merged_intervals and s_lo <= merged_intervals[-1][1] + MERGE_GAP_M:
                prev_lo, prev_hi, _ = merged_intervals[-1]
                merged_intervals[-1] = (prev_lo, max(prev_hi, s_hi), "intersection")
            else:
                merged_intervals.append((s_lo, s_hi, kind))

        cursor = 0.0
        seg_slices: list[tuple[float, float, str]] = []
        for s_lo, s_hi, kind in merged_intervals:
            if s_lo - cursor > MERGE_GAP_M:
                seg_slices.append((cursor, s_lo, "corridor"))
            seg_slices.append((s_lo, s_hi, kind))
            cursor = s_hi
        if total_len - cursor > MERGE_GAP_M:
            seg_slices.append((cursor, total_len, "corridor"))
        if not seg_slices:
            seg_slices.append((0.0, total_len, "corridor"))

        seg_idx = 0
        int_idx = 0
        for s0, s1, z_type in seg_slices:
            sub_line = shapely.ops.substring(self._spine, s0, s1)
            sub_poly = sub_line.buffer(half_w + BUFFER_PAD_M, cap_style="flat", join_style=join_style).intersection(main_corridor_poly)
            if sub_poly.is_empty:
                continue
            geom = cast("Polygon | MultiPolygon", sub_poly)
            if z_type == "intersection":
                self._semantic_zones.append((f"intersection_{int_idx}", f"Concourse Intersection Junction {int_idx}", geom))
                int_idx += 1
            else:
                self._semantic_zones.append((f"corridor_seg_{seg_idx}", f"Main Concourse Segment {seg_idx}", geom))
                seg_idx += 1

    def _build_side_branches(self, half_w: float) -> None:
        """Attach side corridors or enclosed doorway rooms along the spine with dock sockets."""
        main_poly = self._buffered_main_corridor(half_w)

        for idx, branch in enumerate(self.config.branches):
            anchor = self._anchor_at(branch.at_s)
            sides = [BranchSide.LEFT, BranchSide.RIGHT] if branch.side == BranchSide.BOTH else [branch.side]

            for side in sides:
                outside_branch, tip_pt, dir_vec = self._build_branch_polygon(branch, anchor, side)
                outside_clipped = cast("Polygon | MultiPolygon", outside_branch.difference(main_poly))
                door_w = branch.door_width if branch.door_width is not None else (min(DEFAULT_ROOM_DOOR_WIDTH_M, branch.width) if branch.is_room else None)

                placement = BranchPlacement(
                    branch_index=idx,
                    side=side,
                    door_width=door_w,
                    anchor=anchor,
                    tip_point=tip_pt,
                    direction=dir_vec,
                    junction_distance=self._branch_junction_distance(branch, anchor),
                )

                if door_w is not None and door_w < branch.width:
                    outside_clipped = self._cut_doorway_jambs(outside_clipped, branch, placement)

                if not outside_clipped.is_empty:
                    z_prefix = "room" if branch.is_room else "branch"
                    z_desc = "Adjacent Room Space" if branch.is_room else "Side Branch Corridor"
                    self._semantic_zones.append((f"{z_prefix}_{idx}_{side.value}", f"{z_desc} {idx} ({side.value})", outside_clipped))

                self._register_branch_dock_socket(branch, placement)

    @staticmethod
    def _branch_junction_distance(branch: SideBranchFeature, anchor: SpineAnchor) -> float:
        """Distance along the branch axis from the spine to the cross-section clear of the corridor."""
        rad = math.radians(branch.angle_deg)
        return (anchor.half_width + branch.width / 2.0 * abs(math.cos(rad))) / math.sin(rad)

    def _build_branch_polygon(
        self,
        branch: SideBranchFeature,
        anchor: SpineAnchor,
        side: BranchSide,
    ) -> tuple[Polygon, tuple[float, float], tuple[float, float]]:
        """Compute the buffered polygon, tip coordinates, and direction vector of a branch."""
        sign = 1.0 if side == BranchSide.LEFT else -1.0
        rad = math.radians(branch.angle_deg)
        tx, ty = anchor.tangent
        nx, ny = anchor.normal
        dir_x = tx * math.cos(rad) + sign * nx * math.sin(rad)
        dir_y = ty * math.cos(rad) + sign * ny * math.sin(rad)

        px, py = anchor.point
        reach = self._branch_junction_distance(branch, anchor) + branch.length
        tip_x = px + dir_x * reach
        tip_y = py + dir_y * reach
        branch_line = LineString([(px, py), (tip_x, tip_y)])
        outside_branch = branch_line.buffer(branch.width / 2.0, cap_style="flat", join_style="mitre")
        return outside_branch, (tip_x, tip_y), (dir_x, dir_y)

    def _cut_doorway_jambs(
        self,
        outside_branch: Polygon | MultiPolygon,
        branch: SideBranchFeature,
        placement: BranchPlacement,
    ) -> Polygon | MultiPolygon:
        """Subtract doorway wall jambs from the branch entrance."""
        assert placement.door_width is not None
        dir_x, dir_y = placement.direction
        door_w = placement.door_width
        px, py = placement.anchor.point
        p_junc = (px + dir_x * placement.junction_distance, py + dir_y * placement.junction_distance)
        trans_x, trans_y = -dir_y, dir_x
        half_bw = branch.width / 2.0
        half_dw = door_w / 2.0
        jamb_len = max(0.05, half_bw - half_dw)
        j1_off = -(half_bw + half_dw) / 2.0
        j2_off = (half_bw + half_dw) / 2.0
        rot_deg = math.degrees(math.atan2(trans_y, trans_x))

        jamb_boxes: list[Polygon] = []
        for offset in (j1_off, j2_off):
            cx = p_junc[0] + trans_x * offset
            cy = p_junc[1] + trans_y * offset
            box_raw = shapely.box(
                cx - jamb_len / 2.0,
                cy - branch.jamb_thickness / 2.0,
                cx + jamb_len / 2.0,
                cy + branch.jamb_thickness / 2.0,
            )
            jamb_boxes.append(shapely.affinity.rotate(box_raw, rot_deg, origin=(cx, cy)))

        return cast("Polygon | MultiPolygon", outside_branch.difference(shapely.unary_union(jamb_boxes)))

    def _register_branch_dock_socket(
        self,
        branch: SideBranchFeature,
        placement: BranchPlacement,
    ) -> None:
        """Register Door aperture and dock socket zone at the branch terminal."""
        idx = placement.branch_index
        side = placement.side
        door_w = placement.door_width
        tip_x, tip_y = placement.tip_point
        dir_x, dir_y = placement.direction
        px, py = placement.anchor.point
        p_junc = (px + dir_x * placement.junction_distance, py + dir_y * placement.junction_distance)
        trans_x, trans_y = -dir_y, dir_x

        if door_w is not None:
            half_dw = door_w / 2.0
            d_start = Position(x=round(p_junc[0] - trans_x * half_dw, 3), y=round(p_junc[1] - trans_y * half_dw, 3))
            d_end = Position(x=round(p_junc[0] + trans_x * half_dw, 3), y=round(p_junc[1] + trans_y * half_dw, 3))
            act_dist = branch.door_activation_distance
            self._doors.append(
                Door(
                    name=f"door_branch_{idx}_{side.value}",
                    start=d_start,
                    end=d_end,
                    kind="sliding",
                    activation_distance=(act_dist, act_dist),
                )
            )

        dock_name = f"dock_branch_{idx}_{side.value}"
        depth = min(self.config.dock_depth, branch.length * BRANCH_DOCK_DEPTH_RATIO)
        dock_cx = tip_x - dir_x * (depth / 2.0)
        dock_cy = tip_y - dir_y * (depth / 2.0)
        dock_poly = _oriented_box(
            (dock_cx, dock_cy),
            length=depth,
            width=branch.width * (2.0 * BRANCH_DOCK_WIDTH_RATIO),
            dir_vec=(dir_x, dir_y),
        )
        self._semantic_zones.append((dock_name, f"Dock Socket for Branch {idx} ({side.value})", dock_poly))

        yaw_into = math.atan2(dir_y, dir_x)
        self._dock_metadata[dock_name] = {
            "nav_pose": [round(dock_cx, 2), round(dock_cy, 2), round(yaw_into, 3)],
            "is_room": branch.is_room,
        }

    def _build_elevators(self, total_len: float) -> None:
        """Construct enclosed elevator passenger compartments as dock sockets and proper Elevator entities."""
        for idx, elev in enumerate(self.config.elevators):
            anchor = self._anchor_at(elev.at_s)
            px, py = anchor.point
            tx, ty = anchor.tangent
            along_x = abs(tx) >= abs(ty)
            off_axis_deg = math.degrees(math.atan2(min(abs(tx), abs(ty)), max(abs(tx), abs(ty))))
            if off_axis_deg > ELEVATOR_AXIS_TOLERANCE_DEG:
                raise ValueError(f"Elevator {idx} spine tangent is {off_axis_deg:.1f} deg off the nearest axis, cabins need at most {ELEVATOR_AXIS_TOLERANCE_DEG} deg")
            snapped = (math.copysign(1.0, tx), 0.0) if along_x else (0.0, math.copysign(1.0, ty))
            extrude_dir = (-snapped[0], -snapped[1]) if elev.at_s < total_len / 2.0 else snapped
            cabin_cx = px + extrude_dir[0] * (elev.cabin_depth / 2.0)
            cabin_cy = py + extrude_dir[1] * (elev.cabin_depth / 2.0)
            size_x, size_y = (elev.cabin_depth, elev.cabin_width) if along_x else (elev.cabin_width, elev.cabin_depth)
            cabin_box = shapely.box(cabin_cx - size_x / 2.0, cabin_cy - size_y / 2.0, cabin_cx + size_x / 2.0, cabin_cy + size_y / 2.0)
            dock_name = f"dock_elevator_{idx}"
            self._semantic_zones.append((dock_name, f"Elevator Cabin Dock Socket {idx}", cabin_box))
            yaw_out = math.atan2(-extrude_dir[1], -extrude_dir[0])
            self._dock_metadata[dock_name] = {
                "nav_pose": [round(cabin_cx, 2), round(cabin_cy, 2), round(yaw_out, 3)],
                "is_room": False,
            }

            # Door faces outward from the cabin back into the concourse corridor
            door_dx = -extrude_dir[0]
            door_dy = -extrude_dir[1]
            door_side: Literal["+x", "-x", "+y", "-y"]
            if abs(door_dx) >= abs(door_dy):
                door_side = "+x" if door_dx > 0 else "-x"
            else:
                door_side = "+y" if door_dy > 0 else "-y"

            self._elevators.append(
                Elevator(
                    name=f"elevator_{idx}",
                    position=Position(round(cabin_cx, 2), round(cabin_cy, 2), 0.0),
                    size=[round(size_x, 2), round(size_y, 2), 2.5],
                    door_side=door_side,
                    destinations=list(elev.destinations),
                )
            )

    def _apply_pillar_obstacles(self, half_w: float) -> None:
        """Subtract pillar obstacles from semantic zone geometries."""
        all_pillars: list[PillarFeature] = list(self.config.pillars)
        for col in self.config.colonnades:
            dist = col.start_s
            while dist <= col.end_s:
                all_pillars.append(
                    PillarFeature(
                        at_s=dist,
                        lateral_offset=col.lateral_offset,
                        lateral_ratio=col.lateral_ratio,
                        shape=PillarShape.CYLINDER,
                        radius=col.radius,
                        wall_margin=col.wall_margin,
                    )
                )
                dist += col.spacing

        pillar_obstacles: list[Polygon] = []
        for p in all_pillars:
            anchor = self._anchor_at(p.at_s)
            px, py = anchor.point
            nx, ny = anchor.normal
            transverse_dist = p.lateral_offset if p.lateral_ratio is None else p.lateral_ratio * max(0.1, half_w - p.radius - p.wall_margin)
            cx = px + nx * transverse_dist
            cy = py + ny * transverse_dist

            if p.shape == PillarShape.CYLINDER:
                pillar_obstacles.append(Point(cx, cy).buffer(p.radius))
            else:
                pillar_obstacles.append(shapely.box(cx - p.width / 2.0, cy - p.depth / 2.0, cx + p.width / 2.0, cy + p.depth / 2.0))

        self._pillar_obstacles = pillar_obstacles
        if pillar_obstacles:
            combined = shapely.ops.unary_union(pillar_obstacles)
            updated_zones: list[tuple[str, str, Polygon | MultiPolygon]] = []
            for name, desc, poly in self._semantic_zones:
                diff = poly.difference(combined)
                if not diff.is_empty:
                    updated_zones.append((name, desc, cast("Polygon | MultiPolygon", diff)))
            self._semantic_zones = updated_zones

    def _build_trunk_dock_sockets(self) -> None:
        """Create start and end dock socket zones along the concourse trunk."""
        depth = self.config.dock_depth
        half_w = self.config.width / 2.0 * TRUNK_DOCK_WIDTH_RATIO

        anchor_start = self._anchor_at(0.0)
        p_start = anchor_start.point
        t0 = anchor_start.tangent
        yaw0 = math.atan2(t0[1], t0[0])

        dock_s_cx = p_start[0] + t0[0] * (depth / 2.0)
        dock_s_cy = p_start[1] + t0[1] * (depth / 2.0)
        dock_s_poly = _oriented_box(
            (dock_s_cx, dock_s_cy),
            length=depth,
            width=half_w * 2.0,
            dir_vec=t0,
        )
        self._semantic_zones.append(("dock_start", "Terminal Dock Socket Start", dock_s_poly))
        self._dock_metadata["dock_start"] = {
            "nav_pose": [round(dock_s_cx, 2), round(dock_s_cy, 2), round(yaw0, 3)],
        }

        anchor_end = self._anchor_at(self._spine.length)
        p_end = anchor_end.point
        tn = anchor_end.tangent
        yawn = math.atan2(tn[1], tn[0])
        yaw_end_norm = math.atan2(math.sin(yawn + math.pi), math.cos(yawn + math.pi))

        dock_e_cx = p_end[0] - tn[0] * (depth / 2.0)
        dock_e_cy = p_end[1] - tn[1] * (depth / 2.0)
        dock_e_poly = _oriented_box(
            (dock_e_cx, dock_e_cy),
            length=depth,
            width=half_w * 2.0,
            dir_vec=tn,
        )
        self._semantic_zones.append(("dock_end", "Terminal Dock Socket End", dock_e_poly))
        self._dock_metadata["dock_end"] = {
            "nav_pose": [round(dock_e_cx, 2), round(dock_e_cy, 2), round(yaw_end_norm, 3)],
        }

    def _validate_dock_pillar_clearance(self) -> None:
        """Ensure pillars neither split a dock socket into pieces nor cover its nav pose."""
        if not self._pillar_obstacles:
            return
        pillars = shapely.unary_union(self._pillar_obstacles)
        for name, _, poly in self._semantic_zones:
            if name not in self._dock_metadata:
                continue
            remaining = poly.difference(pillars)
            pieces = [part for part in shapely.get_parts(remaining) if part.area > 1e-6]
            if len(pieces) > 1:
                raise ValueError(f"Pillars split dock socket '{name}' into {len(pieces)} pieces")
            x, y = cast("list[float]", self._dock_metadata[name]["nav_pose"])[:2]
            if pillars.intersects(Point(x, y)):
                raise ValueError(f"A pillar covers the nav pose of dock socket '{name}'")

    def _validate_dock_references(self) -> None:
        """Ensure all dock references in flows and robot target resolve to valid socket zones."""
        available_docks = set(self._dock_metadata.keys())
        if self.config.robot_target_branch is not None:
            if self.config.robot_target_branch not in available_docks:
                raise ValueError(f"robot_target_branch '{self.config.robot_target_branch}' not found in dock sockets: {sorted(available_docks)}")
        for spec in self.config.flows:
            if spec.source_dock == spec.sink_dock:
                raise ValueError(f"Flow '{spec.name}' source_dock and sink_dock are both '{spec.source_dock}'")
            if spec.source_dock not in available_docks:
                raise ValueError(f"Flow '{spec.name}' source_dock '{spec.source_dock}' not found in dock sockets: {sorted(available_docks)}")
            if spec.sink_dock not in available_docks:
                raise ValueError(f"Flow '{spec.name}' sink_dock '{spec.sink_dock}' not found in dock sockets: {sorted(available_docks)}")

    def _resolve_robot_mission(self) -> tuple[list[float], list[float]]:
        """Determine robot start and goal poses using dock sockets."""
        start_pose = list(cast("list[float]", self._dock_metadata["dock_start"]["nav_pose"]))
        end_pose = list(cast("list[float]", self._dock_metadata["dock_end"]["nav_pose"]))
        mode = self.config.robot_task_mode

        target_key = self.config.robot_target_branch
        if target_key is not None:
            target_pose = list(cast("list[float]", self._dock_metadata[target_key]["nav_pose"]))
            return start_pose, target_pose

        if mode == RobotTaskMode.MAIN_THROUGH:
            return start_pose, end_pose

        branch_docks = [k for k in self._dock_metadata if k.startswith("dock_branch_")]
        if not branch_docks:
            return start_pose, end_pose

        if mode == RobotTaskMode.ENTER_ROOM:
            room_docks = [k for k in branch_docks if self._dock_metadata[k].get("is_room")]
            target_key = room_docks[0] if room_docks else branch_docks[0]
        else:
            target_key = branch_docks[0]

        target_pose = list(cast("list[float]", self._dock_metadata[target_key]["nav_pose"]))
        return start_pose, target_pose

    def _compile_rate_profile(self, spec: FlowStreamSpec) -> list[dict[str, float]]:
        """Construct keyframes holding constant spawn rate throughout the active window."""
        t0, t1 = spec.active_window
        is_infinite = t1 >= DEFAULT_ACTIVE_WINDOW_END_S

        if t0 <= 0.0 and is_infinite:
            return [{"t": 0.0, "rate": spec.rate}]
        if t0 <= 0.0:
            return [
                {"t": 0.0, "rate": spec.rate},
                {"t": t1, "rate": spec.rate},
                {"t": t1, "rate": 0.0},
            ]
        if is_infinite:
            return [
                {"t": 0.0, "rate": 0.0},
                {"t": t0, "rate": 0.0},
                {"t": t0, "rate": spec.rate},
            ]
        return [
            {"t": 0.0, "rate": 0.0},
            {"t": t0, "rate": 0.0},
            {"t": t0, "rate": spec.rate},
            {"t": t1, "rate": spec.rate},
            {"t": t1, "rate": 0.0},
        ]

    def _compile_scenario_dict(self) -> dict[str, object]:
        """Build format 2 scenario referencing dock socket zones directly."""
        regions: dict[str, dict[str, object]] = {}

        streams = list(self.config.flows)
        if not streams:
            streams.append(
                FlowStreamSpec(
                    name="main_opposing",
                    source_dock="dock_end",
                    sink_dock="dock_start",
                    rate=DEFAULT_OPPOSING_FLOW_RATE,
                    agent_type="adult",
                    velocity_min=DEFAULT_FLOW_VELOCITY_MIN,
                    velocity_max=DEFAULT_FLOW_VELOCITY_MAX,
                )
            )
            branch_docks = [k for k in self._dock_metadata if k.startswith("dock_branch_")]
            if len(branch_docks) >= 2:
                streams.append(
                    FlowStreamSpec(
                        name="cross_flow",
                        source_dock=branch_docks[0],
                        sink_dock=branch_docks[1],
                        rate=DEFAULT_CROSS_FLOW_RATE,
                        agent_type="adult",
                        velocity_min=DEFAULT_CROSS_FLOW_VELOCITY_MIN,
                        velocity_max=DEFAULT_CROSS_FLOW_VELOCITY_MAX,
                    )
                )

        # Register sink regions first in deterministic sorted order
        used_sinks = {s.sink_dock for s in streams}
        for sink_dock in sorted(used_sinks):
            sink_reg_name = f"{sink_dock}_sink"
            regions[sink_reg_name] = {
                "type": "sink",
                "ref": sink_dock,
                "config": {"capacity": -1, "absorption_radius": DEFAULT_ABSORPTION_RADIUS_M},
            }

        # Register source regions with agent profile and sink affinity
        for spec in streams:
            src_reg_name = f"{spec.source_dock}_source_{spec.name}"
            sink_reg_name = f"{spec.sink_dock}_sink"
            regions[src_reg_name] = {
                "type": "source",
                "ref": spec.source_dock,
                "config": {
                    "rate_profile": self._compile_rate_profile(spec),
                    "max_concurrent": spec.max_concurrent,
                    "max_total": spec.max_total,
                    "agent": {
                        "agent_type": spec.agent_type,
                        "desired_velocity": {"min": spec.velocity_min, "max": spec.velocity_max},
                        "behavior_tree": spec.behavior_tree,
                        "sink_affinity": [{"sink": sink_reg_name, "weight": 1.0}],
                    },
                },
            }

        robot_start, robot_goal = self._resolve_robot_mission()
        return {
            "format": SCENARIO_FORMAT,
            "robots": [{"start": robot_start, "phases": [{"goto": robot_goal}]}],
            "regions": regions,
            "timeline": [],
            "conditions": [],
        }
