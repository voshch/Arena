"""Shared plumbing for arena CLI modules."""

import dataclasses
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from complete import Spec


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
    complete: "Spec | None" = None


def make_verb(name: str, run: Callable[[list[str]], int | None], *, hidden: bool = False, passthrough: bool = False, help_text: str | None = None, complete: "Spec | None" = None) -> Verb:
    text = (help_text if help_text is not None else run.__doc__) or ""
    text = "\n".join(line.strip() for line in text.replace("\b", "").strip().splitlines())
    short = text.splitlines()[0] if text else ""
    return Verb(name, run, short, text, hidden, passthrough, complete)


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


def _git_ssh_command() -> str:
    """ssh that fails fast on unreachable hosts and never blocks on a prompt without a tty."""
    opts = "-o ConnectTimeout=5" if sys.stdin.isatty() else "-o ConnectTimeout=5 -o BatchMode=yes"
    return f"ssh {opts}"


def _run(*argv: str) -> int:
    import subprocess

    return subprocess.run(list(argv), check=False).returncode


def _cli(*argv: str, env: dict[str, str] | None = None) -> int:
    """Re-run an arena verb in a subprocess."""
    import subprocess

    return subprocess.run(
        [sys.executable, os.path.join(_env("TOOLS_DIR"), "arena_cli", "__main__.py"), *argv],
        cwd=_env("ARENA_WS_DIR"),
        env=env,
        check=False,
    ).returncode


def _resourced(cmd: str) -> int:
    """Run a shell command in a freshly re-sourced arena environment."""
    import shlex
    import subprocess

    src = _env("SOURCE_FILE")
    return subprocess.run(
        [os.environ.get("SHELL", "/bin/bash"), "-c", f"source {shlex.quote(src)} > /dev/null 2>&1 && {cmd}"],
        cwd=_env("ARENA_WS_DIR"),
        check=False,
    ).returncode


# installed-features registry, backed by the $INSTALLED file


def _reg_list() -> list[str]:
    try:
        with open(_env("INSTALLED")) as f:
            return [line.strip() for line in f if line.strip()]
    except OSError:
        return []


def _reg_has(name: str) -> bool:
    return name in _reg_list()


def _reg_require(name: str) -> None:
    if not _reg_has(name):
        raise CLIError(f"{name} is not installed, run 'arena feature {name} install' first")


def _reg_add(name: str) -> None:
    if not _reg_has(name):
        with open(_env("INSTALLED"), "a") as f:
            f.write(name + "\n")


def _reg_remove(name: str) -> None:
    kept = [n for n in _reg_list() if n != name]
    with open(_env("INSTALLED"), "w") as f:
        f.writelines(n + "\n" for n in kept)


def _reg_pull(name: str) -> int:
    import subprocess

    repos = os.path.join(_env("ARENA_DIR"), "_meta", "repos", f"{name}.repos")
    if not os.path.isfile(repos):
        return 0
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    rc = subprocess.run(["vcs", "import", "--input", repos, "--shallow", "--recursive", "--ff", "--add-existing", os.path.join(_env("ARENA_WS_DIR"), "src")], env=env, check=False).returncode
    if rc:
        print(f"failed to pull all {name} repos, ignoring", file=sys.stderr)
    return rc
