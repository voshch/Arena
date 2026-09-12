"""Stick-figure renders of wire-format frame sequences (matplotlib, inspection only)."""

from __future__ import annotations

import typing
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import skeleton as S

if typing.TYPE_CHECKING:
    from matplotlib.axes import Axes

VIEWS = {
    "3q": np.array([1.0, 0.9, 0.35]),
    "front": np.array([1.0, 0.0, 0.0]),
    "side": np.array([0.0, 1.0, 0.0]),
    "top": np.array([0.0, 0.0, 1.0]),
}


def _camera(view: str) -> tuple[np.ndarray, np.ndarray]:
    eye = S.unit(VIEWS[view])
    up_ref = np.array([0.0, 0.0, 1.0]) if abs(eye[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    right = S.unit(np.cross(up_ref, eye))
    return right, np.cross(eye, right)


def _project(p: np.ndarray, view: str) -> np.ndarray:
    r, u = _camera(view)
    return np.array([p @ r, p @ u])


def _plt() -> typing.Any:  # noqa: ANN401 - matplotlib.pyplot module
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def draw_pose(ax: Axes, pos: dict, view: str = "3q", *, target: np.ndarray | None = None, color: str = "#2b6cb0", lw: float = 1.6) -> None:
    for a, b in S.STICK_EDGES:
        pa, pb = _project(pos[a], view), _project(pos[b], view)
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], "-", lw=lw, color=color, solid_capstyle="round")
    h = _project(pos["head"], view)
    ax.plot([h[0]], [h[1]], "o", ms=lw * 4.2, color=color)
    if target is not None:
        t = _project(np.asarray(target, dtype=float), view)
        ax.plot([t[0]], [t[1]], "*", ms=8, color="#dd6b20")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _style(ax: Axes, view: str, center: tuple[float, float] = (0.0, 1.05), span: float = 1.0) -> None:
    c = np.array([center[0], 0.0, center[1]]) if view != "top" else np.array([center[0], 0.0, 0.0])
    cc = _project(c, view)
    ax.set_xlim(cc[0] - span, cc[0] + span)
    ax.set_ylim(cc[1] - span, cc[1] + span)


def filmstrip(frames: Sequence[dict], body: S.Body, out_path: str | Path, *, n: int = 12, views: Sequence[str] = ("3q", "side"), target: np.ndarray | None = None, title: str = "") -> Path:
    """n evenly spaced frames, one row per view."""
    plt = _plt()
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).round().astype(int)
    fig, axes = plt.subplots(len(views), len(idx), figsize=(2.4 * len(idx), 2.8 * len(views)))
    axes = np.atleast_2d(axes)
    for r, view in enumerate(views):
        for c, i in enumerate(idx):
            pos, _ = S.fk(frames[i]["angles"], body)
            ax = axes[r, c]
            draw_pose(ax, pos, view, target=target)
            _style(ax, view)
            if r == 0:
                ax.set_title(f"{frames[i].get('t', i / 20.0):.2f}s", fontsize=7)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def video(frames: Sequence[dict], body: S.Body, out_path: str | Path, *, fps: float = 20.0, view: str = "3q", target: np.ndarray | None = None) -> Path:
    """MP4 of the sequence (needs ffmpeg on PATH)."""
    plt = _plt()
    from matplotlib import animation

    fig, ax = plt.subplots(figsize=(3.2, 3.2))

    def frame(i: int) -> list:
        ax.cla()
        pos, _ = S.fk(frames[i]["angles"], body)
        draw_pose(ax, pos, view, target=target)
        _style(ax, view)
        ax.set_title(f"{frames[i].get('t', i / fps):.2f}s", fontsize=8)
        return []

    anim = animation.FuncAnimation(fig, frame, frames=len(frames), interval=1000.0 / fps, blit=False)
    out = Path(out_path)
    anim.save(out, writer=animation.FFMpegWriter(fps=fps))
    plt.close(fig)
    return out
