"""Per-package summary of colcon test results (JUnit XML under build/*/test_results)."""

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

GREEN = "\033[1;32m"
RED = "\033[1;31m"
YELLOW = "\033[1;33m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

MAX_FAILS_PER_PKG = 10


def _collect_pkg(pkg_dir: "Path") -> tuple[int, int, int, int, float, list[tuple[str, str, str, str]]]:
    import xml.etree.ElementTree as ET

    tests = failures = errors = skipped = 0
    elapsed = 0.0
    fails: list[tuple[str, str, str, str]] = []
    for xml in pkg_dir.rglob("*.xml"):
        try:
            root = ET.parse(xml).getroot()
        except (ET.ParseError, OSError):
            continue
        if root.tag not in ("testsuite", "testsuites"):
            continue
        suites = root.findall(".//testsuite") if root.tag == "testsuites" else [root]
        for s in suites:
            tests += int(s.get("tests") or 0)
            failures += int(s.get("failures") or 0)
            errors += int(s.get("errors") or 0)
            skipped += int(s.get("skipped") or 0)
            try:
                elapsed += float(s.get("time") or 0)
            except ValueError:
                pass
            for tc in s.findall("testcase"):
                for kind in ("failure", "error"):
                    el = tc.find(kind)
                    if el is None:
                        continue
                    msg = (el.get("message") or "").strip().splitlines()
                    fails.append((
                        kind,
                        tc.get("classname", ""),
                        tc.get("name", ""),
                        msg[0] if msg else "",
                    ))
    return tests, failures, errors, skipped, elapsed, fails


def _fmt(n: int, color: str) -> str:
    return f"{color}{n:>5}{RESET}" if n else f"{DIM}{n:>5}{RESET}"


def summarize(argv: list[str]) -> int:
    """Summarize colcon test results under a build dir. argv[0] is the prog name."""
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir", nargs="?", default="build")
    ap.add_argument("--packages", nargs="*", default=None)
    args = ap.parse_args(argv[1:])

    build = Path(args.build_dir)
    if not build.is_dir():
        print(f"no build dir: {build}", file=sys.stderr)
        return 2

    if args.packages:
        pkg_dirs = [build / p for p in args.packages if (build / p).is_dir()]
    else:
        pkg_dirs = sorted(p for p in build.iterdir() if p.is_dir())

    rows: list[tuple[str, int, int, int, int, float]] = []
    all_fails: dict[str, list[tuple[str, str, str, str]]] = {}
    for pkg_dir in pkg_dirs:
        tests, failures, errors, skipped, elapsed, fails = _collect_pkg(pkg_dir)
        if tests or failures or errors:
            rows.append((pkg_dir.name, tests, failures, errors, skipped, elapsed))
            if fails:
                all_fails[pkg_dir.name] = fails

    if not rows:
        if args.packages:
            missing_build = [p for p in args.packages if not (build / p).is_dir()]
            print(f"{YELLOW}no JUnit XML found under {build}{RESET}", file=sys.stderr)
            print(f"  checked {len(args.packages)} package(s): {' '.join(args.packages)}", file=sys.stderr)
            if missing_build:
                print(f"  {DIM}not built (no build/<pkg>): {' '.join(missing_build)}{RESET}", file=sys.stderr)
        else:
            print("no JUnit XML found under", build)
        return 0

    if all_fails:
        print()
        print(f"{BOLD}{RED}=== failures ==={RESET}")
        for pkg, items in all_fails.items():
            print(f"\n{BOLD}{pkg}{RESET} ({len(items)})")
            for kind, classname, name, msg in items[:MAX_FAILS_PER_PKG]:
                where = f"{classname}::{name}" if classname else name
                print(f"  {RED}{kind:<7}{RESET} {where}")
                if msg:
                    print(f"          {DIM}{msg[:200]}{RESET}")
            if len(items) > MAX_FAILS_PER_PKG:
                rest = len(items) - MAX_FAILS_PER_PKG
                print(f"  {DIM}... and {rest} more — colcon test-result --verbose --test-result-base build/{pkg}{RESET}")

    name_w = max(len(r[0]) for r in rows + [("package", 0, 0, 0, 0, 0.0)])
    width = name_w + 2 + 5 + 1 + 5 + 1 + 5 + 1 + 5 + 2 + 7 + 2 + 8

    print()
    print(f"{BOLD}{'package':<{name_w}}  {'pass':>5} {'fail':>5} {'err':>5} {'skip':>5}  {'time(s)':>7}  status{RESET}")
    print(DIM + "-" * width + RESET)

    tp = tf = te = ts = 0
    tt = 0.0
    for pkg, tests, failures, errors, skipped, elapsed in rows:
        passed = max(0, tests - failures - errors - skipped)
        bad = failures + errors
        status = f"{GREEN}OK{RESET}" if bad == 0 else f"{RED}FAIL{RESET}"
        print(f"{pkg:<{name_w}}  {_fmt(passed, GREEN)} {_fmt(failures, RED)} {_fmt(errors, RED)} {_fmt(skipped, YELLOW)}  {elapsed:>7.1f}  {status}")
        tp += passed
        tf += failures
        te += errors
        ts += skipped
        tt += elapsed

    print(DIM + "-" * width + RESET)
    grand = f"{GREEN}OK{RESET}" if (tf + te) == 0 else f"{RED}FAIL ({tf + te}){RESET}"
    print(f"{BOLD}{'TOTAL':<{name_w}}{RESET}  {_fmt(tp, GREEN)} {_fmt(tf, RED)} {_fmt(te, RED)} {_fmt(ts, YELLOW)}  {tt:>7.1f}  {grand}")
    print()

    return 1 if (tf + te) else 0


if __name__ == "__main__":
    sys.exit(summarize(sys.argv))
