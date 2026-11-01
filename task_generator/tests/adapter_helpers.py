from __future__ import annotations

import csv
from pathlib import Path


def write_sequence(path: Path, walkers: list[tuple[int, int, int, tuple[float, float], tuple[float, float]]]) -> None:
    """A SocNavBench sequence file: rows frame, id, y, x, one column per sample, a walker per straight line."""
    columns = []
    for ident, first, last, (x0, y0), (x1, y1) in walkers:
        for frame in range(first, last + 1):
            u = (frame - first) / (last - first)
            columns.append((frame, ident, round(y0 + u * (y1 - y0), 4), round(x0 + u * (x1 - x0), 4)))
    columns.sort()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(zip(*columns, strict=True))
