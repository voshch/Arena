"""Load a declarative shot YAML file into a `Camera` timeline.

Schema (see cam/README.md):

    params:                            # optional defaults for ${...} references
      target: [0, 0, 0.5]
    projection: perspective            # optional, desugars to a leading step
    reference: { entity: env_0/jackal, mode: yaw }   # optional, desugars to a leading step
    timeline:
      - look:  { eye: [8, 8, 6], target: "${target}", fov: 1.0 }
      - orbit: { radius: 4, elevation_deg: 30, sweep_deg: 360, duration: 8, ease: inout }
      - establishing: { radius: 8 }    # a step may reference another shot

Each step is a single-key map {name: params} routed through `Camera.add`, so every
verb and every registered shot is usable here with no per-step code. A file is just
a shot loaded by path rather than by name. Angles are radians except `sweep_deg` /
`start_deg`; fov is radians.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .camera import Camera
from .client import TargetSelection
from .shots import desugar, substitute


def load_shot(path: str | Path, targets: TargetSelection) -> Camera:
    """Parse a shot YAML file into a ready-to-play `Camera`."""
    spec = yaml.safe_load(Path(path).read_text()) or {}
    args = spec.get("params", {})
    cam = Camera(targets)
    for step in desugar(spec):
        ((name, raw),) = step.items()
        cam.add(name, substitute(raw, args))
    return cam
