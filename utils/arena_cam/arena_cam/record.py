"""Encode captured viewport frames to a video through an ffmpeg pipe, and stills to PPM (P6)."""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import typing
from datetime import datetime
from pathlib import Path

if typing.TYPE_CHECKING:
    from collections.abc import Iterable

    from sensor_msgs.msg import Image


def resolve_path(out: str) -> Path:
    """A bare name lands under $ARENA_DATA_DIR/recordings, an explicit path is kept. No suffix means .mp4."""
    path = Path(out).expanduser()
    if not (path.is_absolute() or os.sep in out):
        base = os.environ.get("ARENA_DATA_DIR")
        path = (Path(base) if base else Path.cwd()) / "recordings" / out
    return path if path.suffix else path.with_suffix(".mp4")


def default_name(stem: str) -> str:
    """`<stem>_<YYYYmmdd-HHMMSS>`, colon-free so it survives NTFS, exFAT and URLs."""
    return f"{stem}_{datetime.now():%Y%m%d-%H%M%S}"


def screenshots_dir() -> Path:
    """$ARENA_SCREENSHOTS_DIR, else $ARENA_DATA_DIR/screenshots, the dir the sim GUIs also drop shots into."""
    explicit = os.environ.get("ARENA_SCREENSHOTS_DIR")
    if explicit:
        return Path(explicit).expanduser()
    base = os.environ.get("ARENA_DATA_DIR")
    return (Path(base) if base else Path.cwd()) / "screenshots"


def record_path(out: str) -> Path:
    """Resolve the output file and create its directory."""
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("cam: record needs ffmpeg on PATH")
    path = resolve_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def tagged(path: Path, tag: str) -> Path:
    """`tour.mp4` -> `tour-sim.mp4`, one file per recorded camera."""
    return path.with_stem(f"{path.stem}-{tag}")


def claim(paths: Iterable[Path], force: bool = False) -> None:
    """Refuse existing files so a take is never clobbered, `force` overrides that."""
    taken = [str(path) for path in paths if path.exists()]
    if taken and not force:
        raise FileExistsError(f"cam: refusing to overwrite {', '.join(taken)}, pass -f/--force")


def rgb_bytes(image: Image) -> bytes:
    """A captured frame as tightly packed rgb24 rows."""
    step, row = image.step, image.width * 3
    data = bytes(image.data)
    if step != row:  # strip any per-row padding the renderer added
        data = b"".join(data[r * step : r * step + row] for r in range(image.height))
    return data


def encode_ppm(image: Image) -> bytes:
    """A captured frame as binary PPM (P6)."""
    return f"P6\n{image.width} {image.height}\n255\n".encode() + rgb_bytes(image)


def next_still(directory: Path) -> Path:
    """The first free `still_*.ppm` in a directory, so stills accumulate across sessions."""
    taken = {path.name for path in directory.glob("still_*.ppm")}
    n = 0
    while f"still_{n:05d}.ppm" in taken:
        n += 1
    return directory / f"still_{n:05d}.ppm"


class Recorder:
    """Pipes `ViewportCapture` frames into an ffmpeg encoding the (already-prepared) output file."""

    def __init__(self, path: str, fps: float) -> None:
        self.path = Path(path)
        self.fps = float(fps)
        self.n = 0
        self._size: tuple[int, int] | None = None
        self._ffmpeg: subprocess.Popen | None = None

    def write(self, image: Image) -> None:
        self.write_rgb(image.width, image.height, rgb_bytes(image))

    def write_rgb(self, width: int, height: int, data: bytes) -> None:
        """Feed one rgb24 frame, the first one fixes the frame size and starts ffmpeg."""
        if self._ffmpeg is None:
            self._size = (width, height)
            source = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-framerate", f"{self.fps:g}", "-i", "-"]
            # yuv420p needs even dimensions
            sink = ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", str(self.path)]
            self._ffmpeg = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-nostats", *source, *sink], stdin=subprocess.PIPE)
        elif (width, height) != self._size:
            raise ValueError(f"frame size changed mid-take ({self._size[0]}x{self._size[1]} to {width}x{height}), keep the viewport size fixed")
        self._ffmpeg.stdin.write(data)
        self.n += 1

    def close(self) -> bool:
        """Finish the file, False if ffmpeg failed or no frame ever arrived."""
        if self._ffmpeg is None:
            return False
        with contextlib.suppress(BrokenPipeError):
            self._ffmpeg.stdin.close()
        return self._ffmpeg.wait() == 0
