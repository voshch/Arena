from __future__ import annotations

import threading
from typing import Protocol

import attrs
import numpy as np
import sounddevice as sd

from task_generator.auditory.asset_lib import CachedSample


class RenderSource(Protocol):
    """Anything the mixer can pull blocks from."""

    @property
    def finished(self) -> bool: ...

    def render(self, frames: int) -> np.ndarray: ...


@attrs.define
class Voice:
    sample: CachedSample
    position: int = 0
    loop: bool = False
    gain: float = 1.0
    voice_id: str | None = None
    loop_sample: CachedSample | None = None
    outro_sample: CachedSample | None = None
    stage: str = "single"
    stop_requested: bool = False
    bus: str = "main"


@attrs.define
class RenderVoice:
    source: RenderSource
    voice_id: str
    bus: str = "main"


class AudioMixer:
    def __init__(self, *, channels: int = 2, master_gain_db: float = 0.0, output: sd.OutputStream | None = None) -> None:
        self._channels = channels
        self._voices: list[Voice] = []
        self._render_voices: list[RenderVoice] = []
        self._bus_enabled: dict[str, bool] = {}
        self._lock = threading.Lock()
        self._master_gain = 10.0 ** (master_gain_db / 20.0)
        self._callback_count = 0
        self._last_output_peak = 0.0
        self._last_status = ""
        self._status_count = 0
        self._stream = output
        if self._stream is not None:
            self._stream.start()

    @classmethod
    def open(cls, *, sample_rate: int = 44100, channels: int = 2, block_size: int = 2048, device: str | int | None = None, master_gain_db: float = 0.0) -> AudioMixer:
        mixer = cls(channels=channels, master_gain_db=master_gain_db)
        try:
            sd.query_devices(device, "output")
            stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype="float32",
                blocksize=block_size,
                device=device,
                callback=mixer._callback,
            )
        except (ValueError, sd.PortAudioError) as exc:
            available = [
                f"{index}: {description['name']}"
                for index, description in enumerate(sd.query_devices())
                if int(description["max_output_channels"]) > 0
            ]
            requested = "default" if device is None else repr(device)
            raise RuntimeError(
                f"cannot open requested audio output {requested}: {exc}; "
                f"available outputs={available or ['none']}"
            ) from exc
        mixer._stream = stream
        stream.start()
        return mixer

    @property
    def active(self) -> bool:
        return self._stream is not None and bool(self._stream.active)

    @property
    def device(self) -> int | None:
        return None if self._stream is None else self._stream.device

    @property
    def voice_count(self) -> int:
        with self._lock:
            return len(self._voices) + len(self._render_voices)

    @property
    def callback_count(self) -> int:
        return self._callback_count

    @property
    def last_output_peak(self) -> float:
        return self._last_output_peak

    @property
    def last_status(self) -> str:
        return self._last_status

    @property
    def status_count(self) -> int:
        return self._status_count

    def play(self, sample: CachedSample, *, loop: bool = False, gain_db: float = 0.0, voice_id: str | None = None, bus: str = "main") -> None:
        voice = Voice(
            sample=sample,
            loop=loop,
            gain=10.0 ** (gain_db / 20.0),
            voice_id=voice_id,
            bus=bus,
        )
        with self._lock:
            if voice_id is not None:
                self._voices = [
                    active
                    for active in self._voices
                    if active.voice_id != voice_id
                ]
            self._voices.append(voice)

    def add_render_source(self, source: RenderSource, *, voice_id: str, bus: str = "main") -> None:
        with self._lock:
            self._render_voices = [
                active
                for active in self._render_voices
                if active.voice_id != voice_id
            ]
            self._render_voices.append(
                RenderVoice(source=source, voice_id=voice_id, bus=bus)
            )

    def play_looping_sequence(
        self,
        intro: CachedSample,
        loop_sample: CachedSample,
        outro: CachedSample,
        *,
        voice_id: str,
        gain_db: float = 0.0,
        bus: str = "main",
    ) -> None:
        """Play intro once, loop the middle, then play outro when stopped."""
        voice = Voice(
            sample=intro,
            gain=10.0 ** (gain_db / 20.0),
            voice_id=voice_id,
            loop_sample=loop_sample,
            outro_sample=outro,
            stage="intro",
            bus=bus,
        )
        with self._lock:
            # A duplicate start event replaces the previous motor voice rather
            # than adding another permanent loop to the mix.
            self._voices = [
                active for active in self._voices
                if active.voice_id != voice_id
            ]
            self._voices.append(voice)

    def stop(self, voice_id: str) -> bool:
        """Request a keyed sequence to leave its loop and play its outro."""
        with self._lock:
            for voice in self._voices:
                if voice.voice_id == voice_id:
                    voice.stop_requested = True
                    return True
        return False

    def stop_all(self) -> None:
        with self._lock:
            self._voices.clear()
            self._render_voices.clear()

    def set_bus_enabled(self, bus: str, enabled: bool) -> None:
        with self._lock:
            self._bus_enabled[bus] = enabled

    def close(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()

    def _callback(self, outdata: np.ndarray, frames: int, _time_info: sd.CallbackTimeInfo, status: sd.CallbackFlags) -> None:
        self._callback_count += 1
        if status:
            self._last_status = str(status)
            self._status_count += 1
        outdata.fill(0.0)

        with self._lock:
            active: list[Voice] = []

            for voice in self._voices:
                written = 0
                bus_enabled = self._bus_enabled.get(voice.bus, True)

                if voice.sample.samples.size == 0 or len(voice.sample.samples) == 0:
                    raise ValueError(f"empty WAV file: {voice.sample.path}")

                while written < frames:
                    remaining = len(voice.sample.samples) - voice.position
                    count = min(frames - written, remaining)

                    if count > 0 and bus_enabled:
                        outdata[written:written + count] += (
                            voice.sample.samples[
                                voice.position:voice.position + count
                            ]
                            * voice.gain
                        )
                    voice.position += count
                    written += count

                    if voice.position >= len(voice.sample.samples):
                        if not self._advance_voice(voice):
                            break

                if self._voice_is_active(voice):
                    active.append(voice)

            self._voices = active

            active_renderers: list[RenderVoice] = []
            for voice in self._render_voices:
                rendered = np.asarray(
                    voice.source.render(frames), dtype=np.float32
                )
                if rendered.shape != outdata.shape:
                    raise ValueError(
                        f"render source {voice.voice_id!r} returned "
                        f"{rendered.shape}, expected {outdata.shape}"
                    )
                if self._bus_enabled.get(voice.bus, True):
                    outdata += rendered
                if not bool(voice.source.finished):
                    active_renderers.append(voice)
            self._render_voices = active_renderers

        outdata *= self._master_gain
        np.clip(outdata, -1.0, 1.0, out=outdata)
        self._last_output_peak = float(np.max(np.abs(outdata)))

    @staticmethod
    def _advance_voice(voice: Voice) -> bool:
        """Move a voice to its next segment after its current sample ends."""
        if voice.stage == "intro":
            next_sample = (
                voice.outro_sample
                if voice.stop_requested else voice.loop_sample
            )
            if next_sample is None:
                return False
            voice.sample = next_sample
            voice.position = 0
            voice.stage = "outro" if voice.stop_requested else "loop"
            return True

        if voice.stage == "loop":
            if voice.stop_requested:
                if voice.outro_sample is None:
                    return False
                voice.sample = voice.outro_sample
                voice.stage = "outro"
            voice.position = 0
            return True

        if voice.stage == "outro":
            return False

        if voice.loop:
            if voice.stop_requested:
                return False
            voice.position = 0
            return True
        return False

    @staticmethod
    def _voice_is_active(voice: Voice) -> bool:
        if voice.position < len(voice.sample.samples):
            return True
        if (
            voice.stage == "single"
            and voice.loop
            and voice.stop_requested
        ):
            return False
        return voice.stage in {"intro", "loop"} or voice.loop
