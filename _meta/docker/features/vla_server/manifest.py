#!/usr/bin/env python3
"""Emit the vla_server weight manifest as `<model>\t<repo>\t<weights>` lines for the download loop."""

import os
import sys

import yaml

_FEATURE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_YAML = os.path.join(_FEATURE_DIR, "models.yaml")


def main() -> None:
    if sys.argv[1:2] != ["manifest"]:
        print(f"usage: {sys.argv[0]} manifest", file=sys.stderr)
        sys.exit(2)
    with open(_MODELS_YAML) as f:
        models = yaml.safe_load(f)
    for key, row in models.items():
        print(f"{key}\t{row['repo']}\t{row['weights']}")


if __name__ == "__main__":
    main()
