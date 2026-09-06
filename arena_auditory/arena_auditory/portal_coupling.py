from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Hashable

import attrs
import numpy as np
from scipy.signal import fftconvolve

from .acoustic_world_graph import (
    AcousticPortal,
    AcousticPortalRoute,
    AcousticWorldGraph,
    Position3D,
)
from .pyroomacoustics_adapter import PyroomacousticsAdapter, RoomImpulseResponse


@attrs.frozen
class PortalCouplingConfig:
    portal_inset_m: float = 0.03
    portal_loss_db: float = 3.0
    source_room_early_window_sec: float = 0.08
    maximum_rir_duration_sec: float = 2.0
    position_quantization_m: float = 0.10
    cache_size: int = 256
    opening_portal_loss_db: float = 0.5
    max_portal_hops: int = 4
    route_distance_loss_db_per_m: float = 0.05

    def __attrs_post_init__(self) -> None:
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
        if self.opening_portal_loss_db < 0.0:
            raise ValueError("opening_portal_loss_db cannot be negative")
        if self.max_portal_hops <= 0:
            raise ValueError("max_portal_hops must be positive")
        if self.route_distance_loss_db_per_m < 0.0:
            raise ValueError("route_distance_loss_db_per_m cannot be negative")


@attrs.frozen
class PortalCouplingResult:
    rir: RoomImpulseResponse
    portal: AcousticPortal
    source_portal_position: Position3D
    listener_portal_position: Position3D
    cache_hits: int
    cache_misses: int
    route: AcousticPortalRoute | None = None
    portal_positions: tuple[Position3D, ...] = ()
    applied_portal_loss_db: float = 0.0


class MultiPortalRirCoupler:
    """Compose room-local RIRs along an acoustically weighted portal route."""

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
        self._cache: OrderedDict[tuple[Hashable, ...], RoomImpulseResponse] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._route_cache: OrderedDict[
            tuple[Hashable, ...],
            PortalCouplingResult,
        ] = OrderedDict()
        self._route_cache_hits = 0
        self._route_cache_misses = 0

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        return self._cache_misses

    @property
    def cache_entries(self) -> int:
        return len(self._cache)

    @property
    def route_cache_entries(self) -> int:
        return len(self._route_cache)

    @property
    def route_cache_hits(self) -> int:
        return self._route_cache_hits

    @property
    def route_cache_misses(self) -> int:
        return self._route_cache_misses

    def compute(
        self,
        *,
        source_zone: str,
        listener_zone: str,
        source_position_m: Position3D,
        listener_position_m: Position3D,
    ) -> PortalCouplingResult:
        route = self._graph.find_portal_route(
            source_zone,
            listener_zone,
            source_xy=source_position_m[:2],
            listener_xy=listener_position_m[:2],
            max_portals=self._config.max_portal_hops,
            distance_loss_db_per_m=(self._config.route_distance_loss_db_per_m),
            default_door_loss_db=self._config.portal_loss_db,
            default_opening_loss_db=self._config.opening_portal_loss_db,
        )
        if route is None or not route.portals:
            raise LookupError(f"no portal route connects {source_zone!r} and {listener_zone!r}")
        route_key = self._route_cache_key(
            route,
            source_position_m,
            listener_position_m,
        )
        cached_route = self._route_cache.pop(route_key, None)
        if cached_route is not None:
            self._route_cache[route_key] = cached_route
            self._route_cache_hits += 1
            return attrs.evolve(
                cached_route,
                cache_hits=self._cache_hits,
                cache_misses=self._cache_misses,
            )
        self._route_cache_misses += 1

        endpoints: list[tuple[Position3D, Position3D]] = []
        flat_positions: list[Position3D] = []
        for index, portal in enumerate(route.portals):
            before_zone = route.zones[index]
            after_zone = route.zones[index + 1]
            before_room = self._graph.room(before_zone)
            after_room = self._graph.room(after_zone)
            if before_room is None or after_room is None:
                raise LookupError("portal endpoint has no acoustic room specification")
            portal_height = min(
                0.5 * portal.height_m,
                before_room.ceiling_height_m - 0.01,
                after_room.ceiling_height_m - 0.01,
            )
            before = self._graph.position_inside_portal(
                portal,
                before_zone,
                inset_m=self._config.portal_inset_m,
                height_m=portal_height,
            )
            after = self._graph.position_inside_portal(
                portal,
                after_zone,
                inset_m=self._config.portal_inset_m,
                height_m=portal_height,
            )
            endpoints.append((before, after))
            flat_positions.append(before)

        room_rirs: list[RoomImpulseResponse] = []
        room_rirs.append(self._cached_room_rir(source_zone, source_position_m, endpoints[0][0]))
        for index, zone_name in enumerate(route.zones[1:-1], start=1):
            room_rirs.append(
                self._cached_room_rir(
                    zone_name,
                    endpoints[index - 1][1],
                    endpoints[index][0],
                )
            )
        room_rirs.append(
            self._cached_room_rir(
                listener_zone,
                endpoints[-1][1],
                listener_position_m,
            )
        )
        sample_rates = {rir.sample_rate_hz for rir in room_rirs}
        if len(sample_rates) != 1:
            raise ValueError("portal route RIR sample rates differ")
        sample_rate = room_rirs[0].sample_rate_hz
        early_count = int(math.ceil(self._config.source_room_early_window_sec * sample_rate))
        segments: list[np.ndarray] = []
        for index, rir in enumerate(room_rirs):
            samples = np.asarray(rir.samples, dtype=np.float64)
            if samples.size == 0:
                raise ValueError("portal route contains an empty room RIR")
            if index < len(room_rirs) - 1:
                peak = int(np.argmax(np.abs(samples)))
                samples = samples[: min(samples.size, max(peak + 1, early_count))]
            segments.append(samples)

        coupled = segments[0]
        maximum_count = int(self._config.maximum_rir_duration_sec * sample_rate)
        for segment in segments[1:]:
            coupled = fftconvolve(coupled, segment, mode="full")
            coupled = coupled[:maximum_count]
        applied_loss = sum((portal.loss_db if portal.loss_db is not None else (self._config.opening_portal_loss_db if portal.portal_kind == "opening" else self._config.portal_loss_db)) for portal in route.portals)
        coupled *= 10.0 ** (-applied_loss / 20.0)
        coupled = np.asarray(coupled[:maximum_count], dtype=np.float64)
        fallback_ids = tuple(dict.fromkeys(material_id for rir in room_rirs for material_id in rir.fallback_material_ids))
        result = PortalCouplingResult(
            rir=RoomImpulseResponse(
                samples=coupled,
                sample_rate_hz=sample_rate,
                global_delay_samples=sum(rir.global_delay_samples for rir in room_rirs),
                fallback_material_ids=fallback_ids,
            ),
            portal=route.portals[0],
            source_portal_position=endpoints[0][0],
            listener_portal_position=endpoints[-1][1],
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
            route=route,
            portal_positions=tuple(flat_positions),
            applied_portal_loss_db=applied_loss,
        )
        self._route_cache[route_key] = result
        while len(self._route_cache) > self._config.cache_size:
            self._route_cache.popitem(last=False)
        return result

    def _route_cache_key(
        self,
        route: AcousticPortalRoute,
        source: Position3D,
        listener: Position3D,
    ) -> tuple[Hashable, ...]:
        quantization = self._config.position_quantization_m

        def quantize(position: Position3D) -> tuple[int, int, int]:
            return tuple(int(round(value / quantization)) for value in position)

        return (
            self._world_name,
            tuple(portal.portal_id for portal in route.portals),
            quantize(source),
            quantize(listener),
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


class OneDoorRirCoupler(MultiPortalRirCoupler):
    """Backward-compatible coupler restricted to one portal."""

    def __init__(
        self,
        adapter: PyroomacousticsAdapter,
        graph: AcousticWorldGraph,
        *,
        world_name: str,
        config: PortalCouplingConfig | None = None,
    ) -> None:
        effective = config or PortalCouplingConfig()
        if effective.max_portal_hops != 1:
            effective = PortalCouplingConfig(
                portal_inset_m=effective.portal_inset_m,
                portal_loss_db=effective.portal_loss_db,
                source_room_early_window_sec=(effective.source_room_early_window_sec),
                maximum_rir_duration_sec=effective.maximum_rir_duration_sec,
                position_quantization_m=effective.position_quantization_m,
                cache_size=effective.cache_size,
                opening_portal_loss_db=effective.opening_portal_loss_db,
                max_portal_hops=1,
                route_distance_loss_db_per_m=(effective.route_distance_loss_db_per_m),
            )
        super().__init__(
            adapter,
            graph,
            world_name=world_name,
            config=effective,
        )
