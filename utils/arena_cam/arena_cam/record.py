"""Write captured viewport frames to a numbered PPM (P6) sequence."""

from __future__ import annotations

import os
import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    from sensor_msgs.msg import Image


def resolve_dir(out_dir: str) -> Path:
    """A bare name lands under $ARENA_DATA_DIR/recordings, an explicit path is kept."""
    path = Path(out_dir).expanduser()
    if path.is_absolute() or os.sep in out_dir:
        return path
    base = os.environ.get("ARENA_DATA_DIR")
    return (Path(base) if base else Path.cwd()) / "recordings" / out_dir


def record_dir(out_dir: str, force: bool = False) -> Path:
    """Resolve and prepare the output directory.

    Refuse a non-empty directory so a take is never clobbered; `force` overrides
    that, clearing our prior `frame_*.ppm` so a shorter re-run leaves no stale tail.
    """
    path = resolve_dir(out_dir)
    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(f"cam: {path} already exists, pass -f/--force to overwrite")
        for frame in path.glob("frame_*.ppm"):
            frame.unlink()
    path.mkdir(parents=True, exist_ok=True)
    return path


def encode_ppm(image: Image) -> bytes:
    """A captured frame as binary PPM (P6), so writing needs no encoder dependency."""
    width, height, step = image.width, image.height, image.step
    row = width * 3
    data = bytes(image.data)
    if step != row:  # strip any per-row padding the renderer added
        data = b"".join(data[r * step : r * step + row] for r in range(height))
    return f"P6\n{width} {height}\n255\n".encode() + data


def next_still(directory: Path) -> Path:
    """The first free `still_*.ppm` in a directory, so stills accumulate across sessions."""
    taken = {path.name for path in directory.glob("still_*.ppm")}
    n = 0
    while f"still_{n:05d}.ppm" in taken:
        n += 1
    return directory / f"still_{n:05d}.ppm"


class Recorder:
    """Numbers `ViewportCapture` frames into the (already-prepared) output directory."""

    def __init__(self, out_dir: str, fps: float) -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.n = 0

    def write(self, image: Image) -> None:
        (self.dir / f"frame_{self.n:05d}.ppm").write_bytes(encode_ppm(image))
        self.n += 1
