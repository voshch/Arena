#!/usr/bin/env python3
"""Fail if any tracked markdown doc contains a dead or absolute local link."""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

VENDORED = {"deps"}

MD_LINK = re.compile(r"\[[^\]]*\]\(\s*([^)\s]+?)(?:\s+\"[^\"]*\")?\s*\)")
HTML_REF = re.compile(r"(?:src|href)=\"([^\"]+)\"")


def iter_targets(line: str):
    for pattern in (MD_LINK, HTML_REF):
        for match in pattern.finditer(line):
            yield match.group(1)


def is_external(target: str) -> bool:
    return "://" in target or target.startswith(("mailto:", "data:", "#"))


def unverifiable(resolved: pathlib.Path, root: pathlib.Path) -> bool:
    # An empty existing ancestor is an uninitialized submodule (sparse CI checkout).
    ancestor = resolved
    while not ancestor.exists():
        ancestor = ancestor.parent
    if ancestor.is_dir() and not any(ancestor.iterdir()):
        return True
    return root not in resolved.parents


def tracked_docs(root: pathlib.Path) -> list[pathlib.Path]:
    """Every .md git tracks under root, including initialized submodules."""
    listing = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--recurse-submodules", "--", "*.md"],
        check=True,
        capture_output=True,
    ).stdout
    return [
        root / name
        for name in listing.decode().split("\0")
        if name and VENDORED.isdisjoint(pathlib.PurePosixPath(name).parts)
    ]


def check(root: pathlib.Path) -> list[str]:
    errors = []
    for entry in tracked_docs(root):
        if entry.is_file():
            for lineno, line in enumerate(entry.read_text(errors="replace").splitlines(), 1):
                for target in iter_targets(line):
                    if is_external(target):
                        continue
                    where = f"{entry.relative_to(root)}:{lineno}"
                    if target.startswith("/"):
                        errors.append(f"{where}: absolute link {target}")
                        continue
                    plain = target.split("#")[0]
                    if not plain:
                        continue
                    resolved = (entry.parent / plain).resolve()
                    if resolved.exists() or unverifiable(resolved, root):
                        continue
                    errors.append(f"{where}: dead link {target}")
    return errors


def main() -> int:
    if len(sys.argv) > 1:
        root = pathlib.Path(sys.argv[1]).resolve()
    else:
        root = pathlib.Path(__file__).resolve().parents[2]
    errors = check(root)
    for error in errors:
        print(error)
    if errors:
        print(f"{len(errors)} dead doc links")
        return 1
    print("all ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
