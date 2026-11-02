"""Load a camera file by path into a `Camera`.

Schema (see cam/README.md):

    params:                            # optional defaults for ${...} references
      target: [0, 0, 0.5]
    projection: perspective            # optional, desugars to a leading step
    reference: { entity: env_0/jackal/base_link, mode: yaw }   # optional, desugars to a leading step
    timeline:
      - look:  { eye: [8, 8, 6], target: "${target}", fov: 1.0 }
      - orbit: { radius: 4, elevation_deg: 30, sweep_deg: 360, duration: 8, ease: inout }
      - establishing: { radius: 8 }    # a step may reference another shot

Each step is a single-key map {name: params}, a verb or a camera asset, and a file
may hold a `cameras:` rig instead of a timeline. A file is just a camera asset loaded
by path rather than by name, its nested names resolve through the global tiers.
Angles are radians except `sweep_deg` / `start_deg`; fov is radians.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .camera import Camera
from .client import TargetSelection


def load_shot(path: str | Path, targets: TargetSelection, params: dict | None = None) -> Camera:
    """Parse a camera file into a ready-to-play `Camera`."""
    from arena_simulation_setup.tree import Identifier
    from arena_simulation_setup.tree.assets.Camera import collect

    root = str(path)
    spec = yaml.safe_load(Path(path).read_text()) or {}
    return Camera(targets).load(root, Identifier._run_sync(collect(root, spec)), params)
