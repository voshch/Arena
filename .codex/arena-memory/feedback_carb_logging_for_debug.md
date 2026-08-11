---
name: carb logging for debug
description: In arena_isaac code, use carb.log_* for debug output, never print().
type: feedback
originSessionId: 55c84bef-4608-4339-805b-db8e06fd2757
---
In arena_isaac (any Isaac-hosted Python), route debug/diagnostic output through `carb.log_info` / `carb.log_warn` / `carb.log_error`. Never use `print()`.

**Why:** Isaac's log pipeline is timestamped, channel-aware, and routable (stream to file, filter by channel, etc.). `print()` bypasses all of that and clutters stdout in a format inconsistent with the rest of Isaac's output.

**How to apply:** When adding temporary diagnostic output to pin down a bug (or permanent info/warn messages), use `carb.log_warn` for visibility-on-default and `carb.log_info` for normal progress. If the user reports they can't see `log_info` output, that's a log-level filter issue — address it by upgrading to `log_warn` or fixing the channel level, not by falling back to `print()`.
