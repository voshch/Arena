from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Hashable, Sequence
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import attrs
import numpy as np
from numpy.typing import NDArray

from .acoustic_room_spec import AcousticRoomSpec
from .material_catalog import (
    AcousticMaterial,
    AcousticMaterialCatalog,
)

if TYPE_CHECKING:
    import pyroomacoustics


Position3D = tuple[float, float, float]


@runtime_checkable
class PointLike(Protocol):
    """Structural stand-in for geometry_msgs/Point, so ROS stays out of here."""

    x: float
    y: float
    z: float


class PyroomacousticsUnavailableError(RuntimeError):
    """Raised when pyroomacoustics is required but not installed."""


@attrs.frozen
class PyroomacousticsConfig:
    """Controls construction of pyroomacoustics rooms.

    The defaults use the image-source method for direct sound and early
    reflections. Ray tracing is disabled because the current Arena
    material catalog does not yet contain calibrated scattering data.
    """

    sample_rate_hz: int = 44100

    # Image-source reflection order. Three is a reasonable starting point
    # for interactive early-reflection rendering.
    max_order: int = 3

    air_absorption: bool = True
    temperature_c: float = 20.0
    relative_humidity_percent: float = 50.0

    # Enable only after scattering values have been calibrated.
    ray_tracing: bool = False

    # Randomized ISM can reduce sweeping echoes in high-order simulations.
    use_randomized_ism: bool = False
    max_random_displacement_m: float = 0.08

    # Minimum-phase RIRs cannot be combined with ray tracing.
    minimum_phase: bool = False
    cache_position_quantization_m: float = 0.10
    cache_size: int = 512

    def __attrs_post_init__(self) -> None:
        if (
            isinstance(self.sample_rate_hz, bool)
            or not isinstance(self.sample_rate_hz, int)
            or self.sample_rate_hz <= 0
        ):
            raise ValueError(
                "sample_rate_hz must be a positive integer"
            )

        if (
            isinstance(self.max_order, bool)
            or not isinstance(self.max_order, int)
            or self.max_order < -1
        ):
            raise ValueError(
                "max_order must be an integer greater than or equal to -1"
            )

        if self.max_order == -1 and not self.ray_tracing:
            raise ValueError(
                "max_order=-1 disables the image-source method; "
                "ray_tracing must then be enabled"
            )

        if not math.isfinite(self.temperature_c):
            raise ValueError("temperature_c must be finite")

        if self.temperature_c <= -273.15:
            raise ValueError(
                "temperature_c must be above absolute zero"
            )

        if not math.isfinite(
            self.relative_humidity_percent
        ):
            raise ValueError(
                "relative_humidity_percent must be finite"
            )

        if not (
            0.0
            <= self.relative_humidity_percent
            <= 100.0
        ):
            raise ValueError(
                "relative_humidity_percent must be between 0 and 100"
            )

        if (
            not math.isfinite(
                self.max_random_displacement_m
            )
            or self.max_random_displacement_m < 0.0
        ):
            raise ValueError(
                "max_random_displacement_m must be finite "
                "and non-negative"
            )

        if self.minimum_phase and self.ray_tracing:
            raise ValueError(
                "minimum_phase cannot be combined with ray_tracing"
            )

        if self.use_randomized_ism and self.max_order < 0:
            raise ValueError(
                "randomized ISM requires max_order >= 0"
            )
        if self.cache_position_quantization_m <= 0.0:
            raise ValueError(
                "cache_position_quantization_m must be positive"
            )
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")


@attrs.frozen
class BuiltPyroom:
    """A pyroomacoustics room and information about its construction."""

    room: Any
    specification: AcousticRoomSpec

    # Material IDs that were absent from the YAML catalog and therefore
    # resolved through its default entry.
    fallback_material_ids: tuple[str, ...]


@attrs.frozen
class RoomImpulseResponse:
    """One source-to-listener room impulse response."""

    samples: NDArray[np.float64]
    sample_rate_hz: int

    # pyroomacoustics' fractional-delay filter introduces this global
    # delay. Preserve it so downstream synchronization can account for it.
    global_delay_samples: int

    fallback_material_ids: tuple[str, ...]

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate_hz


def _load_pyroomacoustics() -> ModuleType:
    """Load pyroomacoustics without making it a package import dependency."""

    try:
        return import_module("pyroomacoustics")
    except ImportError as exc:
        raise PyroomacousticsUnavailableError(
            "pyroomacoustics is required for acoustic RIR rendering "
            "but is not installed in the current Python environment"
        ) from exc


def _coefficient_description(
    material: AcousticMaterial,
    property_name: str,
) -> str:
    fallback_note = (
        "; resolved through catalog default"
        if material.used_default
        else ""
    )

    return (
        f"Arena material {material.material_id}: "
        f"{material.canonical_name}; {property_name}"
        f"{fallback_note}"
    )


class PyroomMaterialAdapter:
    """Converts Arena acoustic materials to pyroomacoustics materials.

    Transmission loss is intentionally not forwarded. A pyroomacoustics
    Material represents reflection absorption and scattering, not
    transmission through a wall assembly. Arena's occlusion/metadata path
    remains responsible for transmission loss.
    """

    def __init__(
        self,
        catalog: AcousticMaterialCatalog,
    ) -> None:
        self._catalog = catalog

    @property
    def catalog(self) -> AcousticMaterialCatalog:
        return self._catalog

    def resolve(
        self,
        material_id: str,
    ) -> AcousticMaterial:
        return self._catalog.get(material_id)

    def convert(
        self,
        material_or_id: AcousticMaterial | str,
        *,
        pyroomacoustics_module: ModuleType | None = None,
    ) -> pyroomacoustics.Material:
        """Create a fresh pyroomacoustics Material.

        A fresh object is returned for each call because pyroomacoustics
        may resample a Material's band data while constructing a room.
        Reusing one mutable instance across differently configured rooms
        could therefore produce surprising results.
        """

        if isinstance(material_or_id, AcousticMaterial):
            material = material_or_id
        else:
            material = self.resolve(material_or_id)

        pra = (
            pyroomacoustics_module
            if pyroomacoustics_module is not None
            else _load_pyroomacoustics()
        )

        frequencies = list(
            material.center_frequencies_hz
        )

        energy_absorption = {
            "description": _coefficient_description(
                material,
                "energy absorption",
            ),
            "coeffs": list(material.absorption),
            "center_freqs": frequencies,
        }

        scattering = {
            "description": _coefficient_description(
                material,
                "scattering",
            ),
            "coeffs": list(material.scattering),
            "center_freqs": frequencies,
        }

        return pra.Material(
            energy_absorption=energy_absorption,
            scattering=scattering,
        )


class PyroomacousticsAdapter:
    """Builds pyroomacoustics rooms from isolated Arena room specs."""

    def __init__(
        self,
        material_catalog: AcousticMaterialCatalog,
        config: PyroomacousticsConfig | None = None,
    ) -> None:
        self._config = config or PyroomacousticsConfig()
        self._materials = PyroomMaterialAdapter(
            material_catalog
        )
        self._rir_cache: OrderedDict[
            tuple[Hashable, ...],
            RoomImpulseResponse,
        ] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def config(self) -> PyroomacousticsConfig:
        return self._config

    @property
    def materials(self) -> PyroomMaterialAdapter:
        return self._materials

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        return self._cache_misses

    @property
    def cache_entries(self) -> int:
        return len(self._rir_cache)

    def build_room(
        self,
        specification: AcousticRoomSpec,
    ) -> BuiltPyroom:
        """Convert one Arena zone specification into a 3-D room."""

        self._validate_room_specification(specification)

        pra = _load_pyroomacoustics()

        corners = np.asarray(
            specification.corners_xy,
            dtype=np.float64,
        ).T

        resolved_boundary_materials = [
            self._materials.resolve(material_id)
            for material_id
            in specification.boundary_material_ids
        ]

        boundary_materials = [
            self._materials.convert(
                material,
                pyroomacoustics_module=pra,
            )
            for material in resolved_boundary_materials
        ]

        floor_material = self._materials.resolve(
            specification.floor_material_id
        )
        ceiling_material = self._materials.resolve(
            specification.ceiling_material_id
        )

        room = pra.Room.from_corners(
            corners,
            fs=self._config.sample_rate_hz,
            max_order=self._config.max_order,
            materials=boundary_materials,
            temperature=self._config.temperature_c,
            humidity=(
                self._config.relative_humidity_percent
            ),
            air_absorption=self._config.air_absorption,
            ray_tracing=self._config.ray_tracing,
            use_rand_ism=(
                self._config.use_randomized_ism
            ),
            max_rand_disp=(
                self._config.max_random_displacement_m
            ),
            min_phase=self._config.minimum_phase,
        )

        room.extrude(
            specification.ceiling_height_m,
            materials={
                "floor": self._materials.convert(
                    floor_material,
                    pyroomacoustics_module=pra,
                ),
                "ceiling": self._materials.convert(
                    ceiling_material,
                    pyroomacoustics_module=pra,
                ),
            },
        )

        all_resolved_materials = [
            *resolved_boundary_materials,
            floor_material,
            ceiling_material,
        ]

        fallback_material_ids = tuple(
            dict.fromkeys(
                material.material_id
                for material in all_resolved_materials
                if material.used_default
            )
        )

        return BuiltPyroom(
            room=room,
            specification=specification,
            fallback_material_ids=(
                fallback_material_ids
            ),
        )

    def compute_rir(
        self,
        specification: AcousticRoomSpec,
        *,
        source_position_m: Sequence[float] | object,
        listener_position_m: Sequence[float] | object,
    ) -> RoomImpulseResponse:
        """Compute one source-to-listener room impulse response.

        Both positions must contain x, y, and z coordinates in the same
        world frame as the room specification. The z coordinate should
        represent the emitter/listener acoustic height rather than the
        agent's ground-contact position.
        """

        source = self._position_xyz(
            source_position_m,
            name="source_position_m",
        )
        listener = self._position_xyz(
            listener_position_m,
            name="listener_position_m",
        )

        self._validate_vertical_position(
            source,
            name="source_position_m",
            ceiling_height_m=(
                specification.ceiling_height_m
            ),
        )
        self._validate_vertical_position(
            listener,
            name="listener_position_m",
            ceiling_height_m=(
                specification.ceiling_height_m
            ),
        )

        quantization = self._config.cache_position_quantization_m
        key = (
            specification,
            tuple(int(round(value / quantization)) for value in source),
            tuple(int(round(value / quantization)) for value in listener),
        )
        cached = self._rir_cache.pop(key, None)
        if cached is not None:
            self._rir_cache[key] = cached
            self._cache_hits += 1
            return cached
        self._cache_misses += 1

        built = self.build_room(specification)
        room = built.room

        # pyroomacoustics checks whether both points lie inside the room.
        room.add_source(np.asarray(source, dtype=np.float64))
        room.add_microphone(
            np.asarray(listener, dtype=np.float64)
        )

        room.compute_rir()

        if (
            room.rir is None
            or not room.rir
            or not room.rir[0]
        ):
            raise RuntimeError(
                "pyroomacoustics did not produce an RIR"
            )

        samples = np.asarray(
            room.rir[0][0],
            dtype=np.float64,
        ).copy()

        pra = _load_pyroomacoustics()
        fractional_delay_length = int(
            pra.constants.get("frac_delay_length")
        )

        result = RoomImpulseResponse(
            samples=samples,
            sample_rate_hz=self._config.sample_rate_hz,
            global_delay_samples=(
                fractional_delay_length // 2
            ),
            fallback_material_ids=(
                built.fallback_material_ids
            ),
        )
        self._rir_cache[key] = result
        while len(self._rir_cache) > self._config.cache_size:
            self._rir_cache.popitem(last=False)
        return result

    @staticmethod
    def _position_xyz(
        value: Sequence[float] | PointLike,
        *,
        name: str,
    ) -> Position3D:
        if isinstance(value, PointLike):
            raw = (
                value.x,
                value.y,
                value.z,
            )
        else:
            try:
                raw = tuple(value)
            except TypeError as exc:
                raise TypeError(
                    f"{name} must be a 3-element sequence or "
                    "an object with x, y, and z attributes"
                ) from exc

        if len(raw) != 3:
            raise ValueError(
                f"{name} must contain exactly three coordinates"
            )

        try:
            position = (
                float(raw[0]),
                float(raw[1]),
                float(raw[2]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} contains a non-numeric coordinate"
            ) from exc

        if not all(math.isfinite(value) for value in position):
            raise ValueError(
                f"{name} contains a non-finite coordinate"
            )

        return position

    @staticmethod
    def _validate_vertical_position(
        position: Position3D,
        *,
        name: str,
        ceiling_height_m: float,
    ) -> None:
        z = position[2]

        # Points directly on a boundary are invalid/ambiguous for the
        # image-source visibility calculations.
        if not 0.0 < z < ceiling_height_m:
            raise ValueError(
                f"{name} z={z} must be strictly between "
                f"0 and the ceiling height {ceiling_height_m}"
            )

    @staticmethod
    def _validate_room_specification(
        specification: AcousticRoomSpec,
    ) -> None:
        corners = np.asarray(
            specification.corners_xy,
            dtype=np.float64,
        )

        if corners.ndim != 2 or corners.shape[1] != 2:
            raise ValueError(
                "room corners must form an N-by-2 array"
            )

        if len(corners) < 3:
            raise ValueError(
                "a room requires at least three corners"
            )

        if not np.isfinite(corners).all():
            raise ValueError(
                "room corners contain non-finite coordinates"
            )

        if (
            len(specification.boundary_material_ids)
            != len(corners)
        ):
            raise ValueError(
                "one boundary material is required per room edge"
            )

        # Prevent pyroomacoustics from reversing a clockwise polygon,
        # which would break the association between edges and materials.
        x = corners[:, 0]
        y = corners[:, 1]

        signed_double_area = float(
            np.sum(
                x * np.roll(y, -1)
                - np.roll(x, -1) * y
            )
        )

        if signed_double_area <= 0.0:
            raise ValueError(
                "room corners must be ordered counter-clockwise"
            )

        if (
            not math.isfinite(
                specification.ceiling_height_m
            )
            or specification.ceiling_height_m <= 0.0
        ):
            raise ValueError(
                "ceiling height must be finite and positive"
            )
