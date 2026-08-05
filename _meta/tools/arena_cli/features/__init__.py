"""Python feature groups, used only while the script they mirror is unchanged."""

import hashlib
import importlib
import sys
from collections.abc import Callable
from types import ModuleType

from common import _reg_add, _reg_has, _reg_pull, _reg_remove, _reg_resolve


def load(name: str) -> ModuleType | None:
    """Return the feature's python module (COMMANDS + DESCRIPTION), or None to fall back to its script."""
    try:
        mod = importlib.import_module(f"features.{name}")
    except ImportError:
        return None
    path = _reg_resolve(name)
    if path is None:
        return None
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != mod.SCRIPT_SHA256:
        return None
    return mod


def default_install(name: str, update: Callable[[], int]) -> int:
    """Pull repos, register, run update, rolling back the registration on failure."""
    if _reg_has(name):
        print(f"{name} is already installed.")
        return 0
    _reg_pull(name)
    _reg_add(name)
    if update() == 0:
        print(f"Installed {name} successfully.")
        return 0
    _reg_remove(name)
    print(f"Install of {name} failed during update.", file=sys.stderr)
    return 1
