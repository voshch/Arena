from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math

import numpy as np
from scipy.signal import fftconvolve

from .acoustic_world_graph import AcousticPortal, AcousticWorldGraph, Position3D
from .pyroomacoustics_adapter import PyroomacousticsAdapter, RoomImpulseResponse


@dataclass(frozen=True)
class PortalCouplingConfig:
    portal_inset_m: float = 0.03
    portal_loss_db: float = 3.0
    source_room_early_window_sec: float = 0.08
    maximum_rir_duration_sec: float = 2.0
    position_quantization_m: float = 0.10
    cache_size: int = 256

    def __post_init__(self) -> None:
        if self.portal_inset_m <= 0.0:
            raise ValueError("portal_inset_m must be positive")
        if self.portal_loss_db < 0.0:
            raise ValueError("portal_loss_db cannot be negative")
        if self.source_room_early_window_sec <= 0.0:
            raise ValueError("source_room_early_window_sec must be positive")
        if self.maximum_rir_duration_sec <= 0.0:
            raise ValueError("maximum_rir_duration_sec must be positive")
        if self.position_quantization_m <= 0.0:
            raise ValueError("position_quantization_m must be positive")
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")


@dataclass(frozen=True)
class PortalCouplingResult:
    rir: RoomImpulseResponse
    portal: AcousticPortal
    source_portal_position: Position3D
    listener_portal_position: Position3D
    cache_hits: int
    cache_misses: int


class OneDoorRirCoupler:
    """Compose two room RIRs through one explicit, adjacent-zone door."""

    def __init__(
        self,
        adapter: PyroomacousticsAdapter,
        graph: AcousticWorldGraph,
        *,
        world_name: str,
        config: PortalCouplingConfig | None = None,
    ) -> None:
        self._adapter = adapter
        self._graph = graph
        self._world_name = world_name
        self._config = config or PortalCouplingConfig()
        self._cache: OrderedDict[tuple[object, ...], RoomImpulseResponse] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        return self._cache_misses

    @property
    def cache_entries(self) -> int:
        return len(self._cache)

    def compute(
        self,
        *,
        source_zone: str,
        listener_zone: str,
        source_position_m: Position3D,
        listener_position_m: Position3D,
    ) -> PortalCouplingResult:
        portal = self._graph.direct_portal(
            source_zone,
            listener_zone,
            source_xy=source_position_m[:2],
            listener_xy=listener_position_m[:2],
        )
        if portal is None:
            raise LookupError(
                f"no one-door portal connects {source_zone!r} and {listener_zone!r}"
            )

        source_room = self._graph.room(source_zone)
        listener_room = self._graph.room(listener_zone)
        if source_room is None or listener_room is None:
            raise LookupError("portal endpoint has no acoustic room specification")

        portal_height = min(
            0.5 * portal.height_m,
            source_room.ceiling_height_m - 0.01,
            listener_room.ceiling_height_m - 0.01,
        )
        source_portal = self._graph.position_inside_portal(
            portal,
            source_zone,
            inset_m=self._config.portal_inset_m,
            height_m=portal_height,
        )
        listener_portal = self._graph.position_inside_portal(
            portal,
            listener_zone,
            inset_m=self._config.portal_inset_m,
            height_m=portal_height,
        )

        source_rir = self._cached_room_rir(
            source_room.zone_name, source_position_m, source_portal
        )
        listener_rir = self._cached_room_rir(
            listener_room.zone_name, listener_portal, listener_position_m
        )
        if source_rir.sample_rate_hz != listener_rir.sample_rate_hz:
            raise ValueError("portal endpoint RIR sample rates differ")

        first = np.asarray(source_rir.samples, dtype=np.float64)
        second = np.asarray(listener_rir.samples, dtype=np.float64)
        if first.size == 0 or second.size == 0:
            raise ValueError("portal endpoint RIR is empty")

        source_peak = int(np.argmax(np.abs(first)))
        early_count = int(
            math.ceil(
                self._config.source_room_early_window_sec
                * source_rir.sample_rate_hz
            )
        )
        first = first[: min(first.size, max(source_peak + 1, early_count))]
        coupled = fftconvolve(first, second, mode="full")
        coupled *= 10.0 ** (-self._config.portal_loss_db / 20.0)
        maximum_count = int(
            self._config.maximum_rir_duration_sec * source_rir.sample_rate_hz
        )
        coupled = np.asarray(coupled[:maximum_count], dtype=np.float64)
        fallback_ids = tuple(
            dict.fromkeys(
                (*source_rir.fallback_material_ids, *listener_rir.fallback_material_ids)
            )
        )
        return PortalCouplingResult(
            rir=RoomImpulseResponse(
                samples=coupled,
                sample_rate_hz=source_rir.sample_rate_hz,
                global_delay_samples=(
                    source_rir.global_delay_samples
                    + listener_rir.global_delay_samples
                ),
                fallback_material_ids=fallback_ids,
            ),
            portal=portal,
            source_portal_position=source_portal,
            listener_portal_position=listener_portal,
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
        )

    def _cached_room_rir(
        self,
        zone_name: str,
        source: Position3D,
        listener: Position3D,
    ) -> RoomImpulseResponse:
        quantization = self._config.position_quantization_m

        def quantize(position: Position3D) -> tuple[int, int, int]:
            return tuple(int(round(value / quantization)) for value in position)

        key = (
            self._world_name,
            zone_name,
            quantize(source),
            quantize(listener),
        )
        cached = self._cache.pop(key, None)
        if cached is not None:
            self._cache[key] = cached
            self._cache_hits += 1
            return cached

        room = self._graph.room(zone_name)
        if room is None:
            raise LookupError(f"no room specification for zone {zone_name!r}")
        rir = self._adapter.compute_rir(
            room,
            source_position_m=source,
            listener_position_m=listener,
        )
        self._cache[key] = rir
        self._cache_misses += 1
        while len(self._cache) > self._config.cache_size:
            self._cache.popitem(last=False)
        return rir
