"""Build the mkdocs nav from the repo's READMEs, folders as sections."""

import re
import subprocess
from pathlib import Path

import mkdocs_gen_files

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT_REPO_URL = "https://github.com/voshch/Arena"

EXCLUDE_SEGMENTS = {
    ".git",
    ".github",
    ".venv",
    ".ruff_cache",
    ".pytest_cache",
    "node_modules",
    "_assets",
    "_meta",
    ".docs",
    "deps",
}

NAMED_DOCS = {"bringup.md", "driving.md", "authoring.md", "services.md"}


def excluded(rel: Path) -> bool:
    return any(seg in EXCLUDE_SEGMENTS for seg in rel.parts)


def included_markdown():
    for path in REPO_ROOT.rglob("*.md"):
        rel = path.relative_to(REPO_ROOT)
        if excluded(rel):
            continue
        name = path.name.lower()
        if name == "readme.md" or name in NAMED_DOCS:
            yield rel


included = sorted(included_markdown())
included_set = set(included)
documented = {rel.parent for rel in included if rel.name.lower() == "readme.md"}


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def _subrepos() -> list[tuple[Path, str, str]]:
    """(path, https url, checked-out sha) per submodule, deepest paths first."""
    listing = _git(
        ["config", "-f", ".gitmodules", "--get-regexp", r"submodule\..*\.path"],
        REPO_ROOT,
    )
    repos = []
    for line in listing.splitlines():
        key, _, path = line.partition(" ")
        name = key.removeprefix("submodule.").removesuffix(".path")
        url = _git(["config", "-f", ".gitmodules", f"submodule.{name}.url"], REPO_ROOT).removesuffix(".git")
        sha = _git(["rev-parse", "HEAD"], REPO_ROOT / path)
        repos.append((Path(path), url, sha))
    repos.sort(key=lambda entry: len(entry[0].parts), reverse=True)
    return repos


SUBREPOS = _subrepos()
ROOT_SHA = _git(["rev-parse", "HEAD"], REPO_ROOT)

MD_LINK = re.compile(r"(!?)(\[[^\]]*\]\()([^()\s]+)(\))")
HTML_SRC = re.compile(r'(src=")([^"]+)(")')


def _format(url: str, ref: str, rel: Path, is_dir: bool, image: bool) -> str:
    if image:
        owner_repo = url.removeprefix("https://github.com/")
        return f"https://raw.githubusercontent.com/{owner_repo}/{ref}/{rel.as_posix()}"
    kind = "tree" if is_dir else "blob"
    return f"{url}/{kind}/{ref}/{rel.as_posix()}"


def github_url(rel: Path, is_dir: bool, image: bool) -> str:
    for sub_path, url, sha in SUBREPOS:
        if rel.is_relative_to(sub_path):
            return _format(url, sha, rel.relative_to(sub_path), is_dir, image)
    return _format(ROOT_REPO_URL, ROOT_SHA, rel, is_dir, image)


def rewrite_target(target: str, src_dir: Path, image: bool) -> str:
    """Point page links at the generated index.md, everything else at GitHub."""
    if "://" in target or target.startswith(("mailto:", "#")):
        return target
    path, sep, frag = target.partition("#")
    resolved = (src_dir / path).resolve()
    try:
        rel = resolved.relative_to(REPO_ROOT)
    except ValueError:
        return target
    if path.endswith("README.md") and rel.parent in documented:
        return path.removesuffix("README.md") + "index.md" + sep + frag
    if rel in included_set:
        return target
    return github_url(rel, resolved.is_dir(), image) + sep + frag


def rewrite_links(text: str, src_dir: Path) -> str:
    def md(match: re.Match) -> str:
        bang, opening, target, closing = match.groups()
        return bang + opening + rewrite_target(target, src_dir, bool(bang)) + closing

    def src(match: re.Match) -> str:
        opening, target, closing = match.groups()
        return opening + rewrite_target(target, src_dir, image=True) + closing

    return HTML_SRC.sub(src, MD_LINK.sub(md, text))


def nav_parts(folder: Path) -> list:
    parts, accum = [], Path(".")
    for seg in folder.parts:
        accum = accum / seg
        if accum in documented:
            parts.append(seg)
    return parts


nav = mkdocs_gen_files.Nav()
seen = set()

for rel in included:
    folder = rel.parent
    is_readme = rel.name.lower() == "readme.md"

    if is_readme:
        doc_path = Path("index.md") if folder == Path(".") else folder / "index.md"
    else:
        doc_path = rel

    parts = nav_parts(folder)
    if not is_readme:
        parts.append(rel.stem.capitalize())
    if not parts:
        parts = ["Home"]

    key = tuple(parts)
    if key in seen:
        parts = list(folder.parts) + ([] if is_readme else [rel.stem.capitalize()])
        key = tuple(parts)
    seen.add(key)

    nav[key] = doc_path.as_posix()

    source = (REPO_ROOT / rel).read_text(encoding="utf-8")
    with mkdocs_gen_files.open(doc_path, "w") as fd:
        fd.write(rewrite_links(source, (REPO_ROOT / rel).parent))
    mkdocs_gen_files.set_edit_path(doc_path, rel.as_posix())

with mkdocs_gen_files.open("SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
