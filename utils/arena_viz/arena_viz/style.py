"""Vendor-neutral display styling. Each visualizer interprets these in its own idiom."""

from __future__ import annotations

import json

import attrs


@attrs.frozen
class StyleSpec:
    color: tuple[int, int, int] | None = None
    alpha: float = 1.0
    line_width: float = 0.05
    enabled: bool = True
    decay: float = 0.0
    latched: bool = False
    extra: dict[str, object] = attrs.field(factory=dict)

    @classmethod
    def from_json(cls, s: str) -> StyleSpec:
        if not s:
            return cls()
        raw = json.loads(s)
        color = raw.get("color")
        if color is not None:
            color = tuple(int(c) for c in color)
            if len(color) != 3:
                raise ValueError(f"StyleSpec.color must be a 3-tuple, got {color!r}")
        return cls(
            color=color,
            alpha=float(raw.get("alpha", 1.0)),
            line_width=float(raw.get("line_width", 0.05)),
            enabled=bool(raw.get("enabled", True)),
            decay=float(raw.get("decay", 0.0)),
            latched=bool(raw.get("latched", False)),
            extra=dict(raw.get("extra", {})),
        )

    def to_json(self) -> str:
        payload: dict[str, object] = {
            "alpha": self.alpha,
            "line_width": self.line_width,
            "enabled": self.enabled,
            "decay": self.decay,
            "latched": self.latched,
        }
        if self.color is not None:
            payload["color"] = list(self.color)
        if self.extra:
            payload["extra"] = self.extra
        return json.dumps(payload)
