"""Tests for the colcon test-result summary against real JUnit files."""

from pathlib import Path

import pytest

from arena_cli.testsum import summarize

SUITE = '<testsuite name="s" tests="{tests}" failures="{failures}" errors="0" skipped="0" time="1.0"/>'


def _results(build: Path, pkg: str, tests: int, failures: int) -> None:
    out = build / pkg / "test_results" / pkg
    out.mkdir(parents=True)
    (out / "results.xml").write_text(SUITE.format(tests=tests, failures=failures))


def test_summary_skips_dot_dirs_in_build(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _results(tmp_path, "pkg", tests=3, failures=0)
    _results(tmp_path, ".uv-cache", tests=5, failures=5)
    _results(tmp_path, ".forks", tests=5, failures=5)
    rc = summarize(["testsum", str(tmp_path)])
    out = capsys.readouterr().out
    assert "pkg" in out
    assert ".uv-cache" not in out
    assert ".forks" not in out
    assert rc == 0
