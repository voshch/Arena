"""Sample cursor pinned to a clock: how many blocks a render owes, and how many to skip."""

from __future__ import annotations

import attrs


@attrs.define
class RenderCursor:
    """Blocks owed to a clock. ``start_ns`` anchors sample zero on the first ``owed`` call."""

    block_ns: int
    max_catchup: int
    start_ns: int | None = None
    rendered: int = 0
    skipped: int = 0

    def owed(self, now_ns: int) -> tuple[int, int]:
        """(blocks to render now, blocks to skip) so that rendered + skipped covers ``now_ns``."""
        if self.start_ns is None:
            self.start_ns = now_ns
        covered = (now_ns - self.start_ns) // self.block_ns + 1
        owed = covered - self.rendered - self.skipped
        if owed <= 0:
            return 0, 0
        render = min(owed, self.max_catchup)
        skip = owed - render
        self.rendered += render
        self.skipped += skip
        return render, skip
