from __future__ import annotations

import hashlib
import traceback
from collections.abc import Hashable
from concurrent.futures import Future

import rclpy
from builtin_interfaces.msg import Time
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.parameter import Parameter
from task_generator_msgs.msg import ContinuousHeardSoundState, EpisodeRecord

from task_generator.auditory.asset_lib import AcousticAsset, CachedSample
from task_generator.auditory.audio_playback_node import SoundPlaybackNode
from task_generator.auditory.procedural_audio import LoopingSampleRenderSource


class EnvironmentSoundPlaybackNode(SoundPlaybackNode):
    def __init__(self, **kwargs: object) -> None:
        self._environment_sources: dict[
            tuple[str, str], LoopingSampleRenderSource
        ] = {}
        self._environment_assets: dict[tuple[str, str], AcousticAsset] = {}
        self._environment_rir_signatures: dict[
            tuple[str, str], tuple[Hashable, ...]
        ] = {}
        self._environment_programs: dict[
            tuple[str, str], tuple[str, str, int, int]
        ] = {}
        self._environment_pending: dict[
            tuple[str, str],
            tuple[
                Future[CachedSample],
                AcousticAsset,
                ContinuousHeardSoundState,
            ],
        ] = {}
        self._completed_programs: set[tuple[str, str, int, int]] = set()
        super().__init__("environment_sound_playback", "environment", **kwargs)
        self.declare_parameter("enable_environment_playback", True)
        self.declare_parameter("environment_rir_crossfade_sec", 0.1)
        self._mixer.set_bus_enabled(
            "environment",
            bool(self.get_parameter("enable_environment_playback").value),
        )
        self._environment_poll_timer = self.create_timer(
            0.01,
            self._poll_environment_loads,
        )

    def _on_set_parameters(
        self,
        parameters: list[Parameter],
    ) -> SetParametersResult:
        routing_changed = any(
            parameter.name == "listener_id"
            for parameter in parameters
        )
        result = super()._on_set_parameters(parameters)
        if not result.successful:
            return result
        if routing_changed:
            self._clear_environment_sources()
        for parameter in parameters:
            if parameter.name != "enable_environment_playback":
                continue
            if parameter.type_ != Parameter.Type.BOOL:
                return SetParametersResult(
                    successful=False,
                    reason="enable_environment_playback must be a boolean",
                )
            self._mixer.set_bus_enabled("environment", bool(parameter.value))
        return result

    def _cb_episode(self, msg: EpisodeRecord) -> None:
        previous_episode = self._episode_id
        super()._cb_episode(msg)
        if self._episode_id == previous_episode:
            return
        self._clear_environment_sources()

    def _clear_environment_sources(self) -> None:
        for future, _, _ in self._environment_pending.values():
            future.cancel()
        self._environment_sources.clear()
        self._environment_assets.clear()
        self._environment_rir_signatures.clear()
        self._environment_programs.clear()
        self._environment_pending.clear()
        self._completed_programs.clear()

    def _cb_continuous_heard_sound(
        self,
        msg: ContinuousHeardSoundState,
    ) -> None:
        if msg.source_backend != "wav_loop":
            return
        if not self._matches_listener(msg.listener_id):
            return
        if self._use_rir and not self._room_specs:
            return

        source_key = (str(msg.listener_id), str(msg.source_id))
        program_key = self._program_key(msg)
        source = self._environment_sources.get(source_key)
        if (
            source is not None
            and self._environment_programs.get(source_key) != program_key
        ):
            self._environment_sources.pop(source_key, None)
            self._environment_assets.pop(source_key, None)
            self._environment_rir_signatures.pop(source_key, None)
            self._environment_programs.pop(source_key, None)
            source = None
        if source is not None and source.finished:
            if not msg.loop:
                self._completed_programs.add(program_key)
            self._environment_sources.pop(source_key, None)
            self._environment_assets.pop(source_key, None)
            self._environment_rir_signatures.pop(source_key, None)
            self._environment_programs.pop(source_key, None)
            source = None

        pending = self._environment_pending.get(source_key)
        if pending is not None:
            self._environment_pending[source_key] = (
                pending[0],
                pending[1],
                msg,
            )
            return

        if source is None:
            if not msg.active or program_key in self._completed_programs:
                return
            asset_id = str(msg.asset_id).strip() or str(msg.sound_type).strip()
            selected = self._catalog.select(
                asset_id,
                episode_seed=self._episode_seed,
                agent_id=self._system_numeric_id(msg.system_id),
                occurrence=0,
            )
            if selected is None:
                self.get_logger().warning(
                    f"no environment acoustic asset for {asset_id!r}"
                )
                return
            asset, sample_spec = selected
            future = self._asset_loader.submit(self._catalog.load, sample_spec)
            self._environment_pending[source_key] = (future, asset, msg)
            return

        asset = self._environment_assets[source_key]
        self._update_environment_source(source_key, source, asset, msg)
        if not msg.active and not msg.loop:
            self._completed_programs.add(program_key)

    def _poll_environment_loads(self) -> None:
        for source_key, pending in tuple(self._environment_pending.items()):
            future, asset, msg = pending
            if not future.done():
                continue
            self._environment_pending.pop(source_key, None)
            if future.cancelled() or not msg.active:
                continue
            try:
                sample = future.result()
                elapsed_ns = max(
                    self.get_clock().now().nanoseconds
                    - self._time_nanoseconds(msg.program_start_time),
                    0,
                )
                start_frame = int(elapsed_ns * sample.sample_rate / 1e9)
                if not msg.loop and start_frame >= len(sample.samples):
                    self._completed_programs.add(self._program_key(msg))
                    continue
                source = LoopingSampleRenderSource(
                    sample,
                    block_size=int(self.get_parameter("block_size").value),
                    loop=bool(msg.loop),
                    start_frame=start_frame,
                    rir_crossfade_seconds=float(
                        self.get_parameter(
                            "environment_rir_crossfade_sec"
                        ).value
                    ),
                )
                self._environment_sources[source_key] = source
                self._environment_assets[source_key] = asset
                self._environment_programs[source_key] = self._program_key(msg)
                self._mixer.add_render_source(
                    source,
                    voice_id=f"{source_key[0]}|{source_key[1]}",
                    bus="environment",
                )
                self._update_environment_source(
                    source_key,
                    source,
                    asset,
                    msg,
                )
                self._played_events += 1
            except Exception:
                self.get_logger().error(
                    "environment acoustic asset load failed:\n"
                    f"{traceback.format_exc()}"
                )

    def _update_environment_source(
        self,
        source_key: tuple[str, str],
        source: LoopingSampleRenderSource,
        asset: AcousticAsset,
        msg: ContinuousHeardSoundState,
    ) -> None:
        signature = self._continuous_rir_signature(msg)
        impulse = None
        active = bool(msg.active and msg.audible)
        if (
            active
            and self._use_rir
            and signature != self._environment_rir_signatures.get(source_key)
        ):
            try:
                impulse, _ = self._compute_normalized_rir(msg)
                self._environment_rir_signatures[source_key] = signature
            except Exception as exc:
                self.get_logger().warning(
                    f"environment RIR unavailable for {msg.source_id!r}: "
                    f"{exc}"
                )
        has_rir = (
            impulse is not None
            or source_key in self._environment_rir_signatures
        )
        gain_db = (
            float(msg.received_volume_db)
            - float(asset.reference_level_db)
            + float(asset.playback_gain_db)
        )
        active = bool(
            active
            and gain_db >= self._minimum_playback_gain_db
            and (
                has_rir
                or not self._use_rir
                or self._rir_dry_fallback
            )
        )
        source.update(
            gain_db=gain_db,
            active=active,
            impulse=impulse,
            rir_signature=signature if impulse is not None else None,
        )

    @staticmethod
    def _time_nanoseconds(stamp: Time) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    @classmethod
    def _program_key(
        cls,
        msg: ContinuousHeardSoundState,
    ) -> tuple[str, str, int, int]:
        return (
            str(msg.listener_id),
            str(msg.source_id),
            int(msg.program_start_time.sec),
            int(msg.program_start_time.nanosec),
        )

    @staticmethod
    def _system_numeric_id(system_id: str) -> int:
        digest = hashlib.blake2b(system_id.encode(), digest_size=4).digest()
        return int.from_bytes(digest, "little") & 0x7FFFFFFF


def main() -> None:
    rclpy.init()
    node = EnvironmentSoundPlaybackNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
