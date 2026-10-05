"""Offline driver for the four-microphone renderer.

The live node is one driver of ``arena_auditory.render_core``.  This is the
other one: it reads the per-block render trace out of a recorded episode and
replays it through the same ``render_block``, so what it writes is the audio the
navigation stack heard, sample for sample.  No DSP lives here.

Nothing in this path talks to ROS, so episodes are plain independent processes
and the whole corpus renders in parallel.
"""

from __future__ import annotations

import os

# The BLAS thread pin decides reduction order and is only read when numpy is
# first imported, so it has to be set before the import below.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import multiprocessing
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import yaml
from scipy.io import wavfile

from arena_auditory import render_core
from arena_auditory.asset_lib import AcousticAssetCatalog

TRACE_TOPIC_SUFFIX = "/audio/diagnostics/render_inputs"
RAW_TOPIC_SUFFIX = "/audio/raw_array"
MOTOR_TOPIC_SUFFIX = "/audio/stem_motor"

MODES = ("render", "verify", "remix")


@dataclass(frozen=True)
class EpisodeTrace:
    """One recording's render trace and the audio the live node published."""

    robot: str
    sample_rate: int
    block_size: int
    blocks: tuple[str, ...]
    raw: tuple[np.ndarray, ...]
    motor: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class Divergence:
    stream: str
    block: int
    channel: int
    sample: int
    replayed: float
    recorded: float


@dataclass(frozen=True)
class EpisodeJob:
    path: Path
    mode: str
    assets: Path
    sounds: Path
    out_dir: Path | None = None
    robot: str = ""
    attenuation_db: float = 0.0
    sample_rate: int = 0
    block_size: int = 0


@dataclass(frozen=True)
class EpisodeReport:
    path: Path
    mode: str
    blocks: int = 0
    output: Path | None = None
    divergence: Divergence | None = None
    clipped_blocks: tuple[int, ...] = ()
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.divergence is None


def audio_block(
    topic: str,
    *,
    encoding: str,
    interleaved: bool,
    channel_count: int,
    frame_count: int,
    data: npt.ArrayLike,
) -> np.ndarray:
    """Deinterleave one recorded ``AudioFrame`` payload into a channels-first block."""
    if encoding != "32FC1" or not interleaved:
        raise ValueError(f"{topic}: expected an interleaved 32FC1 AudioFrame, got encoding={encoding!r} interleaved={interleaved!r}")
    values = np.asarray(data, dtype="<f4")
    if channel_count != render_core.CHANNELS or frame_count <= 0 or values.size != channel_count * frame_count:
        raise ValueError(f"{topic}: malformed AudioFrame dimensions ({channel_count} x {frame_count}, {values.size} values)")
    return np.ascontiguousarray(values.reshape(frame_count, channel_count).T)


def read_episode_trace(path: Path, *, robot: str = "", full_audio: bool = True) -> EpisodeTrace:
    """Pull the render trace and the published audio out of one recording.

    Decoding an ``AudioFrame`` payload costs more than rendering the block it
    holds, so ``full_audio=False`` keeps only the first block of each audio
    stream, which is all the block geometry needs.
    """
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    blocks: list[str] = []
    raw: list[np.ndarray] = []
    motor: list[np.ndarray] = []
    robots: set[str] = set()
    sample_rates: set[int] = set()

    with path.open("rb") as source:
        reader = make_reader(source, decoder_factories=[DecoderFactory()])
        for _, channel, _, message in reader.iter_decoded_messages(log_time_order=True):
            topic = "/" + channel.topic.strip("/")
            for suffix in (TRACE_TOPIC_SUFFIX, RAW_TOPIC_SUFFIX, MOTOR_TOPIC_SUFFIX):
                if not topic.endswith(suffix):
                    continue
                name = topic[: -len(suffix)].rsplit("/", 1)[-1]
                if robot and name != robot:
                    continue
                robots.add(name)
                if suffix == TRACE_TOPIC_SUFFIX:
                    blocks.append(str(message.data))
                else:
                    sample_rates.add(int(message.sample_rate))
                    stream = raw if suffix == RAW_TOPIC_SUFFIX else motor
                    if full_audio or not stream:
                        stream.append(
                            audio_block(
                                topic,
                                encoding=str(message.encoding),
                                interleaved=bool(message.interleaved),
                                channel_count=int(message.channel_count),
                                frame_count=int(message.frame_count),
                                data=message.data,
                            )
                        )

    if len(robots) > 1:
        raise ValueError(f"{path}: recording carries several robots {sorted(robots)}, select one with --robot")
    if not blocks:
        raise ValueError(f"{path}: no render trace found on */{TRACE_TOPIC_SUFFIX.strip('/')}")
    if len(sample_rates) > 1:
        raise ValueError(f"{path}: recorded audio mixes sample rates {sorted(sample_rates)}")
    recorded = raw or motor
    return EpisodeTrace(
        robot=next(iter(robots), ""),
        sample_rate=next(iter(sample_rates), 0),
        block_size=int(recorded[0].shape[1]) if recorded else 0,
        blocks=tuple(blocks),
        raw=tuple(raw),
        motor=tuple(motor),
    )


def asset_resolver(assets: Path, sounds: Path, sample_rate: int) -> Callable[[str], np.ndarray]:
    """Rebuild the node's ``asset_key -> mono samples`` mapping from the catalog."""
    catalog = AcousticAssetCatalog(assets, sounds, output_sample_rate=sample_rate, output_channels=1)
    specs = {str(variant.sample_id): variant for asset_id in yaml.safe_load(Path(assets).read_text()).get("assets", {}) for variant in catalog.require(str(asset_id)).variants}

    def resolve(asset_key: str) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(catalog.load(specs[asset_key]).samples[:, 0], dtype=np.float32))

    return resolve


def decode_blocks(texts: Iterable[str], resolve: Callable[[str], np.ndarray]) -> Iterator[render_core.RenderInputs]:
    for text in texts:
        yield render_core.render_inputs_from_json(text, resolve)


def replay(
    blocks: Iterable[render_core.RenderInputs],
    params: render_core.RenderParams,
    *,
    state: render_core.RenderState | None = None,
) -> Iterator[tuple[render_core.RenderInputs, render_core.RenderResult]]:
    """Drive ``render_block`` over a trace, carrying state exactly as the node does."""
    carried = render_core.RenderState() if state is None else state
    for block in blocks:
        # A live catch-up skip advances the cursor without rendering, so the
        # recorded block index is the only authority on where a block sits.
        carried.cursor = block.block_index * params.block_size
        yield block, render_core.render_block(carried, block, params)


def render_results(results: Iterable[tuple[render_core.RenderInputs, render_core.RenderResult]]) -> np.ndarray:
    return _join([result.raw for _, result in results])


def verify_results(
    results: Iterable[tuple[render_core.RenderInputs, render_core.RenderResult]],
    raw: Sequence[np.ndarray],
    motor: Sequence[np.ndarray],
) -> Divergence | None:
    """Compare a replay against the recorded streams, first mismatch wins."""
    for index, (block, result) in enumerate(results):
        if index < len(raw):
            divergence = first_divergence("raw_array", block.block_index, result.raw, raw[index])
            if divergence is not None:
                return divergence
        if index < len(motor):
            divergence = first_divergence("stem_motor", block.block_index, result.motor, motor[index])
            if divergence is not None:
                return divergence
    return None


def remix_results(
    results: Iterable[tuple[render_core.RenderInputs, render_core.RenderResult]],
    attenuation_db: float,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Re-balance ego noise block by block, listing the blocks the clip spoiled."""
    blocks: list[np.ndarray] = []
    clipped: list[int] = []
    for inputs, result in results:
        if result.clipped_samples:
            clipped.append(inputs.block_index)
        blocks.append(remix_block(result.raw, result.motor, attenuation_db))
    return _join(blocks), tuple(clipped)


def remix_block(raw: np.ndarray, motor: np.ndarray, attenuation_db: float) -> np.ndarray:
    """``(raw - motor)`` is the pedestrian stem, and the motor comes back attenuated.

    Only exact for a block the renderer did not clip: past the clip the raw
    block no longer holds the sum the motor stem was subtracted from.
    """
    return (raw - motor) + 10.0 ** (-attenuation_db / 20.0) * motor


def first_divergence(stream: str, block: int, replayed: np.ndarray, recorded: np.ndarray) -> Divergence | None:
    if replayed.shape != recorded.shape:
        raise ValueError(f"block {block} {stream}: replayed {replayed.shape} against recorded {recorded.shape}")
    if np.array_equal(replayed, recorded):
        return None
    channel, sample = (int(index) for index in np.argwhere(replayed != recorded)[0])
    return Divergence(
        stream=stream,
        block=block,
        channel=channel,
        sample=sample,
        replayed=float(replayed[channel, sample]),
        recorded=float(recorded[channel, sample]),
    )


def write_wav(path: Path, channels: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sample_rate, np.ascontiguousarray(channels.T, dtype=np.float32))


def _join(blocks: Sequence[np.ndarray]) -> np.ndarray:
    if not blocks:
        return np.zeros((render_core.CHANNELS, 0), dtype=np.float32)
    return np.concatenate(blocks, axis=1)


def render_episode(job: EpisodeJob) -> EpisodeReport:
    """Worker entry point: one recording in, one report (and maybe a wav) out."""
    try:
        return _render_episode(job)
    except Exception as exc:
        return EpisodeReport(path=job.path, mode=job.mode, error=f"{type(exc).__name__}: {exc}")


def _render_episode(job: EpisodeJob) -> EpisodeReport:
    trace = read_episode_trace(job.path, robot=job.robot, full_audio=job.mode == "verify")
    sample_rate = job.sample_rate or trace.sample_rate
    block_size = job.block_size or trace.block_size
    if sample_rate <= 0 or block_size <= 0:
        raise ValueError(f"{job.path}: no recorded audio to read the block geometry from, pass --sample-rate and --block-size")
    params = render_core.RenderParams(
        block_size=block_size,
        sample_rate=sample_rate,
        resolve=asset_resolver(job.assets, job.sounds, sample_rate),
    )
    results = replay(decode_blocks(trace.blocks, params.resolve), params)

    if job.mode == "verify":
        if not trace.raw and not trace.motor:
            raise ValueError(f"{job.path}: nothing to verify against, neither raw_array nor stem_motor was recorded")
        for name, recorded in (("raw_array", trace.raw), ("stem_motor", trace.motor)):
            if recorded and len(recorded) != len(trace.blocks):
                raise ValueError(f"{job.path}: {len(trace.blocks)} trace blocks against {len(recorded)} recorded {name} blocks")
        return EpisodeReport(
            path=job.path,
            mode=job.mode,
            blocks=len(trace.blocks),
            divergence=verify_results(results, trace.raw, trace.motor),
        )

    output = (job.out_dir or job.path.parent) / _output_name(job)
    if job.mode == "remix":
        audio, clipped = remix_results(results, job.attenuation_db)
        write_wav(output, audio, sample_rate)
        return EpisodeReport(path=job.path, mode=job.mode, blocks=len(trace.blocks), output=output, clipped_blocks=clipped)
    audio = render_results(results)
    write_wav(output, audio, sample_rate)
    return EpisodeReport(path=job.path, mode=job.mode, blocks=len(trace.blocks), output=output)


def _output_name(job: EpisodeJob) -> str:
    if job.mode == "remix":
        return f"{job.path.stem}.remix_{job.attenuation_db:g}db.wav"
    return f"{job.path.stem}.render.wav"


def _package_share() -> Path:
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory("arena_auditory"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="auditory_offline_render",
        description="Replay recorded episodes through the live four-microphone renderer.",
    )
    parser.add_argument(
        "mode",
        choices=MODES,
        help="render: write the replayed four-channel audio to a wav. verify: assert the replay is bit-identical to the recorded raw_array and stem_motor, which holds only for a recording made with enabled=true and mute_all=false. remix: write the audio with the motor stem attenuated by --attenuate-db.",
    )
    parser.add_argument("recordings", nargs="+", type=Path, metavar="RECORDING.mcap", help="episode recordings, rendered one per worker process")
    parser.add_argument("--out", type=Path, default=None, help="directory for render/remix wavs (default: beside each recording)")
    parser.add_argument("--attenuate-db", type=float, default=0.0, metavar="DB", help="remix: attenuation applied to the motor stem, in dB (0 reproduces the original mix)")
    parser.add_argument("--robot", default="", help="robot name, needed only when a recording carries more than one")
    parser.add_argument("--assets", type=Path, default=None, help="acoustic asset catalog (default: the installed config/acoustic_assets.yaml)")
    parser.add_argument("--sounds", type=Path, default=None, help="directory of asset wavs (default: the installed sounds/)")
    parser.add_argument("--sample-rate", type=int, default=0, metavar="HZ", help="override the sample rate read from the recorded audio")
    parser.add_argument("--block-size", type=int, default=0, metavar="FRAMES", help="override the block size read from the recorded audio")
    parser.add_argument("--jobs", type=int, default=0, metavar="N", help="worker processes (default: one per CPU, capped at the number of recordings)")
    return parser.parse_args()


def _run(jobs: Sequence[EpisodeJob], workers: int) -> list[EpisodeReport]:
    if workers <= 1:
        return [render_episode(job) for job in jobs]
    # Spawn, not fork: a fresh interpreter reads the thread pin above before it
    # imports numpy, which a forked child would inherit already configured.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(render_episode, jobs))


def main() -> None:
    args = _parse_args()
    assets = args.assets or _package_share() / "config" / "acoustic_assets.yaml"
    sounds = args.sounds or _package_share() / "sounds"
    jobs = [
        EpisodeJob(
            path=recording,
            mode=args.mode,
            assets=assets,
            sounds=sounds,
            out_dir=args.out,
            robot=args.robot,
            attenuation_db=args.attenuate_db,
            sample_rate=args.sample_rate,
            block_size=args.block_size,
        )
        for recording in args.recordings
    ]
    workers = args.jobs if args.jobs > 0 else min(len(jobs), os.cpu_count() or 1)

    failed = 0
    for report in _run(jobs, workers):
        if report.error:
            failed += 1
            print(f"FAIL   {report.path}: {report.error}", file=sys.stderr)
            continue
        if report.divergence is not None:
            failed += 1
            divergence = report.divergence
            print(f"DIFFER {report.path}: {divergence.stream} block {divergence.block} channel {divergence.channel} sample {divergence.sample} replayed {divergence.replayed!r} recorded {divergence.recorded!r}", file=sys.stderr)
            continue
        detail = f" -> {report.output}" if report.output else ""
        print(f"OK     {report.path}: {report.blocks} blocks{detail}")
        if report.clipped_blocks:
            print(f"       {len(report.clipped_blocks)} clipped block(s), remix is approximate there, first {report.clipped_blocks[:8]}", file=sys.stderr)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
