from __future__ import annotations

import arena_runtime_msgs.msg
import attrs
import builtin_interfaces.msg


def _extent_eq(a: arena_runtime_msgs.msg.WorldExtent, b: arena_runtime_msgs.msg.WorldExtent) -> bool:
    return a.x_min == b.x_min and a.y_min == b.y_min and a.x_max == b.x_max and a.y_max == b.y_max


@attrs.define
class Placement:
    reference: tuple[float, float]
    slot_extent: tuple[float, float]
    prespawn: tuple[float, float]


@attrs.define
class EnvRecord:
    env_id: int
    fqn: str
    extent: arena_runtime_msgs.msg.WorldExtent = attrs.Factory(arena_runtime_msgs.msg.WorldExtent)
    placed: bool = False
    reference: tuple[float, float] = (0.0, 0.0)
    slot_extent: tuple[float, float] = (0.0, 0.0)
    prespawn: tuple[float, float] = (0.0, 0.0)
    ready: bool = False
    draining: bool = False
    last_heartbeat: builtin_interfaces.msg.Time = attrs.Factory(builtin_interfaces.msg.Time)


@attrs.define
class _Shelf:
    """Row in the shelf packer: cursor advances along x, height fixed at first placement."""

    y: float
    height: float
    cursor: float = 0.0


class EnvRegistry:
    """Identity + deferred per-extent placement.

    `reserve()` allocates an env_id and namespace without committing to a slot.
    `place()` runs the shelf packer to size and position a slot for the requested rectangle.
    `unplace()` frees the footprint so it can be repacked (used on rect change or eviction).
    """

    def __init__(self, slot_buffer: float = 5.0) -> None:
        self._records: dict[int, EnvRecord] = {}
        self._free: list[int] = []
        self._next_id: int = 0
        self._slot_buffer = slot_buffer
        self._shelves: list[_Shelf] = []

    def reserve(
        self,
        requested_env_id: int | None = None,
        requested_ns: str | None = None,
        *,
        now: builtin_interfaces.msg.Time,
    ) -> tuple[int, str]:
        if requested_env_id is not None:
            if requested_env_id in self._records:
                raise ValueError(f"env_id {requested_env_id} already in use")
            if requested_env_id in self._free:
                self._free.remove(requested_env_id)
            if requested_env_id >= self._next_id:
                self._next_id = requested_env_id + 1
            env_id = requested_env_id
        else:
            free_candidates = [i for i in self._free if not self._is_draining(i)]
            if free_candidates:
                env_id = free_candidates[0]
                self._free.remove(env_id)
            else:
                env_id = self._next_id
                self._next_id += 1

        namespace = requested_ns.lstrip("/") if requested_ns else f"arena/env_{env_id}/task_generator_node"

        self._records[env_id] = EnvRecord(
            env_id=env_id,
            fqn=f"/{namespace}",
            last_heartbeat=now,
        )
        return env_id, namespace

    def place(
        self,
        env_id: int,
        extent: arena_runtime_msgs.msg.WorldExtent,
    ) -> Placement:
        """Pack a slot sized to the bbox + buffer ring. Idempotent on identical extent."""
        record = self._records.get(env_id)
        if record is None:
            raise ValueError(f"env_id {env_id} not reserved")

        if record.placed and _extent_eq(record.extent, extent):
            return Placement(
                reference=record.reference,
                slot_extent=record.slot_extent,
                prespawn=record.prespawn,
            )

        if record.placed:
            self._unplace_record(record)

        bbox_w = max(0.0, extent.x_max - extent.x_min)
        bbox_h = max(0.0, extent.y_max - extent.y_min)
        bbox_cx = (extent.x_min + extent.x_max) / 2.0
        bbox_cy = (extent.y_min + extent.y_max) / 2.0
        buffer = self._slot_buffer
        slot_w = bbox_w + 2.0 * buffer
        slot_h = bbox_h + 2.0 * buffer

        x, y = self._pack(slot_w, slot_h)

        slot_cx = x + slot_w / 2.0
        slot_cy = y + slot_h / 2.0

        ref_x = slot_cx - bbox_cx
        ref_y = slot_cy - bbox_cy
        prespawn_x = ref_x + extent.x_max + buffer / 2.0
        prespawn_y = ref_y + bbox_cy

        record.extent = extent
        record.reference = (ref_x, ref_y)
        record.slot_extent = (slot_w, slot_h)
        record.prespawn = (prespawn_x, prespawn_y)
        record.placed = True

        return Placement(
            reference=record.reference,
            slot_extent=record.slot_extent,
            prespawn=record.prespawn,
        )

    def unplace(self, env_id: int) -> None:
        record = self._records.get(env_id)
        if record is None or not record.placed:
            return
        self._unplace_record(record)

    def _unplace_record(self, record: EnvRecord) -> None:
        record.placed = False
        record.reference = (0.0, 0.0)
        record.slot_extent = (0.0, 0.0)
        record.prespawn = (0.0, 0.0)
        self._reflow()

    def _reflow(self) -> None:
        """Rebuild shelves and recompute reference/prespawn for every placed record, in env_id order."""
        placed = sorted(
            (r for r in self._records.values() if r.placed),
            key=lambda r: r.env_id,
        )
        self._shelves = []
        for r in placed:
            slot_w, slot_h = r.slot_extent
            x, y = self._pack(slot_w, slot_h)
            slot_cx = x + slot_w / 2.0
            slot_cy = y + slot_h / 2.0
            bbox_cx = (r.extent.x_min + r.extent.x_max) / 2.0
            bbox_cy = (r.extent.y_min + r.extent.y_max) / 2.0
            buffer = self._slot_buffer
            r.reference = (slot_cx - bbox_cx, slot_cy - bbox_cy)
            r.prespawn = (
                r.reference[0] + r.extent.x_max + buffer / 2.0,
                r.reference[1] + bbox_cy,
            )

    def _pack(self, w: float, h: float) -> tuple[float, float]:
        """First-fit shelf pack. Shelves grow along +x; new shelves stack along +y."""
        for shelf in self._shelves:
            if h <= shelf.height + 1e-6 and shelf.cursor + w <= self._row_width_target(shelf):
                x = shelf.cursor
                y = shelf.y
                shelf.cursor += w
                return x, y

        if self._shelves:
            last = self._shelves[-1]
            new_y = last.y + last.height
        else:
            new_y = 0.0
        shelf = _Shelf(y=new_y, height=h, cursor=w)
        self._shelves.append(shelf)
        return 0.0, new_y

    def _row_width_target(self, shelf: _Shelf) -> float:
        """Per-shelf width budget; grows as a near-square so layout doesn't degenerate to a strip."""
        total_height = sum(s.height for s in self._shelves) or shelf.height
        return max(total_height, shelf.cursor + shelf.height)

    def _is_draining(self, env_id: int) -> bool:
        record = self._records.get(env_id)
        return record is not None and record.draining

    def start_eviction(self, env_id: int) -> bool:
        """Mark slot as DRAINING. Returns False if already draining or not found."""
        record = self._records.get(env_id)
        if record is None or record.draining:
            return False
        record.draining = True
        return True

    def complete_eviction(self, env_id: int) -> None:
        """Free the slot after purge completes. ID becomes reusable."""
        self.unplace(env_id)
        self._records.pop(env_id, None)
        if env_id not in self._free:
            self._free.append(env_id)
            self._free.sort()

    def free(self, env_id: int) -> None:
        self.unplace(env_id)
        self._records.pop(env_id, None)
        if env_id not in self._free:
            self._free.append(env_id)
            self._free.sort()

    def get(self, env_id: int) -> EnvRecord | None:
        return self._records.get(env_id)

    def items(self) -> list[tuple[int, EnvRecord]]:
        return list(self._records.items())

    def update_heartbeat(self, env_id: int, stamp: builtin_interfaces.msg.Time) -> None:
        record = self._records.get(env_id)
        if record is not None:
            record.last_heartbeat = stamp

    def update_ready(self, env_id: int, ready: bool) -> None:
        record = self._records.get(env_id)
        if record is not None:
            record.ready = ready

    def snapshot(self) -> list[arena_runtime_msgs.msg.EnvRecord]:
        result: list[arena_runtime_msgs.msg.EnvRecord] = []
        for record in self._records.values():
            if record.draining:
                continue
            msg = arena_runtime_msgs.msg.EnvRecord()
            msg.env_id = record.env_id
            msg.fqn = record.fqn
            msg.extent = record.extent
            msg.reference = list(record.reference)
            msg.slot_extent = list(record.slot_extent)
            msg.prespawn = list(record.prespawn)
            msg.placed = record.placed
            msg.ready = record.ready
            msg.last_heartbeat = record.last_heartbeat
            result.append(msg)
        return result


def sweep_verdict(
    record: EnvRecord,
    *,
    elapsed: float,
    has_reset_hold: bool,
    process_alive: bool | None,
    heartbeat_timeout: float,
    reset_timeout: float,
    bootstrap_timeout: float,
) -> str | None:
    """Return the eviction reason, or None to keep the env alive."""
    if record.draining:
        return None
    if process_alive is False:
        return "process_exited"
    if not record.ready:
        return "bootstrap_timeout" if elapsed > bootstrap_timeout else None
    budget = reset_timeout if has_reset_hold else heartbeat_timeout
    return "heartbeat_timeout" if elapsed > budget else None
