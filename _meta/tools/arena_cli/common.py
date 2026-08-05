"""Shared plumbing for arena CLI modules."""

import dataclasses
import os
import sys
from collections.abc import Callable
from typing import NoReturn


class CLIError(Exception):
    """User-facing CLI failure, printed as `Error: <message>`."""


@dataclasses.dataclass
class Verb:
    """One CLI verb. `run` receives the raw argv after the verb name and returns an exit code (None means 0).

    `passthrough` verbs forward unknown tokens verbatim, so -h/--help only
    triggers help when it appears before the first non-option token.
    """

    name: str
    run: Callable[[list[str]], int | None]
    short: str
    help: str
    hidden: bool = False
    passthrough: bool = False


def make_verb(name: str, run: Callable[[list[str]], int | None], *, hidden: bool = False, passthrough: bool = False, help_text: str | None = None) -> Verb:
    text = (help_text if help_text is not None else run.__doc__) or ""
    text = "\n".join(line.strip() for line in text.replace("\b", "").strip().splitlines())
    short = text.splitlines()[0] if text else ""
    return Verb(name, run, short, text, hidden, passthrough)


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise CLIError(f"${name} is not set, run 'source arena' from the workspace first")
    return value


def _exec(*argv: str) -> NoReturn:
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        os.execvp(argv[0], list(argv))
    except OSError as e:
        raise CLIError(f"{argv[0]}: {e}") from e


def _run(*argv: str) -> int:
    import subprocess

    return subprocess.run(list(argv), check=False).returncode


# mirror of _feature_registry in _meta/tools/source


def _reg_list() -> list[str]:
    try:
        with open(_env("INSTALLED")) as f:
            return [line.strip() for line in f if line.strip()]
    except OSError:
        return []


def _reg_has(name: str) -> bool:
    return name in _reg_list()


def _reg_add(name: str) -> None:
    if not _reg_has(name):
        with open(_env("INSTALLED"), "a") as f:
            f.write(name + "\n")


def _reg_remove(name: str) -> None:
    kept = [n for n in _reg_list() if n != name]
    with open(_env("INSTALLED"), "w") as f:
        f.writelines(n + "\n" for n in kept)


def _reg_resolve(name: str) -> str | None:
    base = os.path.join(_env("ARENA_FEATURES_DIR"), name)
    if os.path.isfile(base):
        return base
    main_file = os.path.join(base, "main")
    if os.path.isfile(main_file):
        return main_file
    return None


def _reg_pull(name: str) -> None:
    repos = os.path.join(_env("ARENA_DIR"), "_meta", "repos", f"{name}.repos")
    if not os.path.isfile(repos):
        return
    rc = _run("vcs", "import", "--input", repos, "--shallow", "--recursive", "--ff", "--add-existing", os.path.join(_env("ARENA_WS_DIR"), "src"))
    if rc:
        print(f"failed to pull all {name} repos, ignoring", file=sys.stderr)


def _features_line() -> str:
    names = []
    for entry in sorted(os.listdir(_env("ARENA_FEATURES_DIR"))):
        if _reg_resolve(entry):
            names.append(entry + ("*" if _reg_has(entry) else ""))
    return " ".join(names)


def _feature_subshell(path: str, argv: tuple[str, ...]) -> int:
    import shlex

    src = _env("SOURCE_FILE")
    sh = os.environ.get("shell", "bash")
    cmd = f"source {shlex.quote(src)} > /dev/null 2>&1 && {sh} {shlex.quote(path)} {shlex.join(argv)}"
    return _run(os.environ.get("SHELL", "/bin/bash"), "-c", cmd)


def _feature_dispatch(name: str, argv: tuple[str, ...]) -> int:
    path = _reg_resolve(name)
    if path is None:
        raise CLIError(f"unknown feature '{name}' (available: {_features_line()})")
    verb = argv[0] if argv else ""
    if verb in ("update", "launch") and not _reg_has(name):
        raise CLIError(f"{name} is not installed, run 'arena feature {name} install' first")
    return _feature_subshell(path, argv)


_script_help_cache: dict[str, str] = {}


def _script_help(path: str) -> str:
    if path not in _script_help_cache:
        import subprocess

        try:
            proc = subprocess.run(["bash", path, "help"], capture_output=True, text=True, timeout=5, check=False)
            _script_help_cache[path] = (proc.stdout or proc.stderr).strip()
        except OSError:
            _script_help_cache[path] = ""
    return _script_help_cache[path]


def _script_desc(path: str) -> str:
    import re

    lines = [line.strip() for line in _script_help(path).splitlines() if line.strip()]
    for line in lines:
        if not line.startswith("Usage:"):
            return line
    if lines:
        verbs = re.search(r"<(.+)>", lines[0])
        if verbs:
            return verbs.group(1).replace("|", " ")
    return ""
