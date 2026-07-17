"""Episode-scoped keyed RNG.

A single per-episode seed fans out into independent, reproducible substreams
keyed by a stable label, so neither draw order nor concurrent scheduling can
affect any individual stream.
"""

from __future__ import annotations

import hashlib
import typing

import numpy as np

_MASK = 0x7FFFFFFFFFFFFFFF
_T = typing.TypeVar("_T")


def stable_int(key: object) -> int:
    """Process-independent int from a key, never builtin hash()."""
    if isinstance(key, int):
        return key & _MASK
    digest = hashlib.blake2b(str(key).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") & _MASK


class EpisodeRng:
    """Per-episode keyed RNG, re-rooted each reset via reseed()."""

    def __init__(self) -> None:
        self._root: int = 0
        self._streams: dict[tuple[object, ...], np.random.Generator] = {}
        self._values: dict[tuple[object, ...], object] = {}

    def reseed(self, seed: int) -> None:
        """Re-root every stream on a new per-episode seed."""
        self._root = seed & _MASK
        self._streams = {}
        self._values = {}

    def stream(self, *key: object) -> np.random.Generator:
        """Return the independent generator for a stable key, cached per episode."""
        if key not in self._streams:
            seq = np.random.SeedSequence([self._root, *(stable_int(k) for k in key)])
            self._streams[key] = np.random.default_rng(seq)
        return self._streams[key]

    def once(self, *key: object, factory: typing.Callable[[], _T]) -> _T:
        """Compute ``factory()`` at most once per episode for ``key``, caching the result.

        Use this for per-episode decisions that may be requested from more than one
        call site during a single reset (e.g. the crowd driver, which is drawn in the
        spawn path that can fire once per batch or once per obstacle). The cached value
        is cleared on ``reseed`` so the next episode redraws — draw order within the
        episode cannot change it, matching the stream() reproducibility contract.
        """
        if key not in self._values:
            self._values[key] = factory()
        return self._values[key]
