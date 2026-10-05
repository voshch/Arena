from __future__ import annotations

import math
from typing import TYPE_CHECKING

import attrs
import numpy as np
from numpy.typing import NDArray

from .acoustic_room_spec import (
    AcousticBoundarySpec,
    AcousticRoomSpec,
)
from .pyroomacoustics_adapter import RoomImpulseResponse

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from mpl_toolkits.mplot3d import Axes3D

Position3D = tuple[float, float, float]


@attrs.frozen
class AcousticPlotSnapshot:
    room_specs: tuple[AcousticRoomSpec, ...]
    source_position_m: Position3D
    listener_position_m: Position3D
    rir: RoomImpulseResponse
    backend: str
    source_zone: str
    listener_zone: str
    traversed_zones: tuple[str, ...] = ()
    portal_positions_m: tuple[Position3D, ...] = ()
    label: str = ""


def make_static_room_spec(
    corners_xy: tuple[tuple[float, float], ...],
    *,
    ceiling_height_m: float,
    wall_material_id: str,
    floor_material_id: str,
    ceiling_material_id: str,
) -> AcousticRoomSpec:
    if len(corners_xy) < 3:
        raise ValueError("static room requires at least three corners")
    boundary = tuple(
        AcousticBoundarySpec(
            start=start,
            end=corners_xy[(index + 1) % len(corners_xy)],
            material_id=wall_material_id,
            kind="wall",
        )
        for index, start in enumerate(corners_xy)
    )
    return AcousticRoomSpec(
        zone_name="static_room",
        boundary=boundary,
        floor_material_id=floor_material_id,
        ceiling_material_id=ceiling_material_id,
        ceiling_height_m=ceiling_height_m,
    )


class AcousticPlotDashboard:
    """Display room geometry and one source-to-listener RIR."""

    def __init__(self, *, energy_bin_ms: float, early_window_sec: float) -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        self._energy_bin_ms = max(float(energy_bin_ms), 0.1)
        self._early_window_sec = max(float(early_window_sec), 0.0)
        self._figure = plt.figure(figsize=(14.0, 8.5))
        self._figure.canvas.manager.set_window_title("Arena pyroomacoustics diagnostics")
        self._figure.suptitle("Waiting for an acoustic RIR")
        plt.show(block=False)

    @property
    def is_open(self) -> bool:
        return self._plt.fignum_exists(self._figure.number)

    def update(self, snapshot: AcousticPlotSnapshot) -> None:
        if not self.is_open:
            return
        self._figure.clear()
        geometry = self._figure.add_subplot(2, 2, 1, projection="3d")
        rir_axis = self._figure.add_subplot(2, 2, 2)
        decay_axis = self._figure.add_subplot(2, 2, 3)
        energy_axis = self._figure.add_subplot(2, 2, 4)

        self._plot_geometry(geometry, snapshot)
        samples, physical_time_ms, peak_index = self._plot_rir(
            rir_axis,
            snapshot,
        )
        self._plot_decay(decay_axis, samples, physical_time_ms)
        self._plot_energy(
            energy_axis,
            samples,
            physical_time_ms,
            peak_index,
            snapshot.rir.sample_rate_hz,
        )

        route = snapshot.traversed_zones or (
            snapshot.source_zone,
            snapshot.listener_zone,
        )
        route_text = " -> ".join(dict.fromkeys(route))
        title = snapshot.backend
        if route_text:
            title += f" | {route_text}"
        if snapshot.label:
            title += f" | {snapshot.label}"
        self._figure.suptitle(title)
        self._figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        self._figure.canvas.draw_idle()
        self._figure.canvas.flush_events()

    def close(self) -> None:
        self._plt.close(self._figure)

    def pump_events(self) -> None:
        if self.is_open:
            self._figure.canvas.flush_events()

    @staticmethod
    def _plot_geometry(axis: Axes3D, snapshot: AcousticPlotSnapshot) -> None:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        selected_zones = set(snapshot.traversed_zones)
        selected_zones.update(zone for zone in (snapshot.source_zone, snapshot.listener_zone) if zone)
        rooms = tuple(room for room in snapshot.room_specs if not selected_zones or room.zone_name in selected_zones)
        if not rooms:
            rooms = snapshot.room_specs

        x_values: list[float] = []
        y_values: list[float] = []
        maximum_height = 0.0
        kind_colors = {
            "wall": "#6c757d",
            "door": "#f4a261",
            "opening": "#2a9d8f",
        }
        for room in rooms:
            height = room.ceiling_height_m
            maximum_height = max(maximum_height, height)
            corners = room.corners_xy
            x_values.extend(point[0] for point in corners)
            y_values.extend(point[1] for point in corners)
            floor = [(x, y, 0.0) for x, y in corners]
            axis.add_collection3d(
                Poly3DCollection(
                    [floor],
                    facecolor="#457b9d",
                    edgecolor="#1d3557",
                    alpha=0.08,
                )
            )
            for segment in room.boundary:
                sx, sy = segment.start
                ex, ey = segment.end
                wall = [
                    (sx, sy, 0.0),
                    (ex, ey, 0.0),
                    (ex, ey, height),
                    (sx, sy, height),
                ]
                color = kind_colors[segment.kind]
                axis.add_collection3d(
                    Poly3DCollection(
                        [wall],
                        facecolor=color,
                        edgecolor=color,
                        alpha=0.16 if segment.kind == "wall" else 0.25,
                    )
                )
            center_x = sum(point[0] for point in corners) / len(corners)
            center_y = sum(point[1] for point in corners) / len(corners)
            axis.text(center_x, center_y, 0.05, room.zone_name, fontsize=8)

        source = snapshot.source_position_m
        listener = snapshot.listener_position_m
        route = (source, *snapshot.portal_positions_m, listener)
        axis.plot(
            [point[0] for point in route],
            [point[1] for point in route],
            [point[2] for point in route],
            color="#7b2cbf",
            linewidth=2.0,
            label="acoustic route",
        )
        axis.scatter(*source, color="#e63946", s=65, label="source")
        axis.scatter(*listener, color="#1d4ed8", s=65, label="listener")
        if snapshot.portal_positions_m:
            axis.scatter(
                [point[0] for point in snapshot.portal_positions_m],
                [point[1] for point in snapshot.portal_positions_m],
                [point[2] for point in snapshot.portal_positions_m],
                color="#f4a261",
                marker="s",
                s=45,
                label="portal",
            )

        x_values.extend((source[0], listener[0]))
        y_values.extend((source[1], listener[1]))
        if x_values and y_values:
            x_min, x_max = min(x_values), max(x_values)
            y_min, y_max = min(y_values), max(y_values)
            span = max(x_max - x_min, y_max - y_min, 1.0)
            x_mid = 0.5 * (x_min + x_max)
            y_mid = 0.5 * (y_min + y_max)
            axis.set_xlim(x_mid - 0.55 * span, x_mid + 0.55 * span)
            axis.set_ylim(y_mid - 0.55 * span, y_mid + 0.55 * span)
            axis.set_box_aspect((1.0, 1.0, max(maximum_height / span, 0.25)))
        axis.set_zlim(0.0, max(maximum_height, source[2], listener[2], 1.0))
        axis.set_xlabel("world x [m]")
        axis.set_ylabel("world y [m]")
        axis.set_zlabel("height [m]")
        axis.set_title("3D acoustic room geometry")
        axis.legend(loc="upper right", fontsize=8)

    @staticmethod
    def _plot_rir(
        axis: Axes,
        snapshot: AcousticPlotSnapshot,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
        samples = np.asarray(snapshot.rir.samples, dtype=np.float64)
        if samples.size == 0 or not np.isfinite(samples).all():
            raise ValueError("RIR contains no finite samples")
        peak_amplitude = float(np.max(np.abs(samples)))
        if peak_amplitude <= 0.0:
            raise ValueError("RIR has no non-zero samples")
        peak_index = int(np.argmax(np.abs(samples)))
        delay = int(snapshot.rir.global_delay_samples)
        physical_time_ms = (np.arange(samples.size, dtype=np.float64) - delay) / float(snapshot.rir.sample_rate_hz) * 1000.0
        normalized = samples / peak_amplitude
        dominant_time_ms = float(physical_time_ms[peak_index])
        axis.plot(physical_time_ms, normalized, color="#6a1b9a", linewidth=0.8)
        axis.axvline(
            dominant_time_ms,
            color="#e63946",
            linestyle="--",
            linewidth=1.0,
            label=f"dominant arrival {dominant_time_ms:.2f} ms",
        )
        gain_db = 20.0 * math.log10(max(peak_amplitude, 1e-15))
        axis.set_title(f"RIR waveform (peak transfer {gain_db:.1f} dB, filter delay {delay} samples)")
        axis.set_xlabel("physical time [ms]")
        axis.set_ylabel("peak-normalized amplitude")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        return samples, physical_time_ms, peak_index

    @staticmethod
    def _plot_decay(
        axis: Axes,
        samples: NDArray[np.float64],
        physical_time_ms: NDArray[np.float64],
    ) -> None:
        energy = samples * samples
        decay = np.cumsum(energy[::-1])[::-1]
        decay /= max(float(decay[0]), 1e-30)
        decay_db = 10.0 * np.log10(np.maximum(decay, 1e-12))
        axis.plot(physical_time_ms, decay_db, color="#264653", linewidth=1.2)
        for level in (-5.0, -25.0, -35.0, -60.0):
            axis.axhline(level, color="#adb5bd", linewidth=0.6, linestyle=":")
        axis.set_ylim(-80.0, 2.0)
        axis.set_title("Schroeder energy decay")
        axis.set_xlabel("physical time [ms]")
        axis.set_ylabel("remaining energy [dB]")
        axis.grid(alpha=0.25)

    def _plot_energy(
        self,
        axis: Axes,
        samples: NDArray[np.float64],
        physical_time_ms: NDArray[np.float64],
        peak_index: int,
        sample_rate_hz: int,
    ) -> None:
        bin_samples = max(
            int(round(self._energy_bin_ms * sample_rate_hz / 1000.0)),
            1,
        )
        starts = np.arange(0, samples.size, bin_samples)
        energy = np.add.reduceat(samples * samples, starts)
        maximum = max(float(np.max(energy)), 1e-30)
        energy_db = 10.0 * np.log10(np.maximum(energy / maximum, 1e-12))
        centers = np.asarray([physical_time_ms[min(int(start + bin_samples // 2), samples.size - 1)] for start in starts])
        peak_time_ms = float(physical_time_ms[peak_index])
        early_end_ms = peak_time_ms + self._early_window_sec * 1000.0
        colors = [("#e63946" if abs(center - peak_time_ms) <= 0.5 * self._energy_bin_ms else "#f4a261" if peak_time_ms < center <= early_end_ms else "#457b9d" if center > early_end_ms else "#adb5bd") for center in centers]
        axis.bar(
            centers,
            energy_db,
            width=0.9 * self._energy_bin_ms,
            color=colors,
        )
        axis.set_ylim(-80.0, 2.0)
        axis.set_title(f"RIR energy in {self._energy_bin_ms:g} ms bins")
        axis.set_xlabel("physical time [ms]")
        axis.set_ylabel("relative energy [dB]")
        axis.grid(axis="y", alpha=0.25)
