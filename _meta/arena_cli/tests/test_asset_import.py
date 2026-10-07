"""Tests for the argv the import verb hands to arena_assets."""

import pytest

from arena_cli.asset import _IMPORT, IMPORT_KINDS, _import_args, _parse
from arena_cli.common import CLIError


def parse(argv: list[str]) -> list[str]:
    pos, opts = _parse(argv, ("KIND", "SOURCE"), flags=("--all",), valued=("--out",), optional=2, variadic=True)
    return _import_args(pos, opts)


def test_bare_lists_sources() -> None:
    assert parse([]) == []


def test_kind_alone_lists_its_sources() -> None:
    assert parse(["object"]) == ["object"]


def test_kind_and_source_list_items() -> None:
    assert parse(["object", "gso"]) == ["object", "gso"]


def test_items_and_out_forward_in_order() -> None:
    assert parse(["object", "gso", "a", "b", "--out", "/tmp/x"]) == ["object", "gso", "a", "b", "--out", "/tmp/x"]


def test_all_forwards_flag() -> None:
    assert parse(["--all", "object", "gso"]) == ["object", "gso", "--all"]


def test_all_with_items_is_refused() -> None:
    with pytest.raises(CLIError, match="not both"):
        parse(["object", "gso", "a", "--all"])


def test_options_without_kind_and_source_are_refused() -> None:
    with pytest.raises(CLIError, match="KIND and a SOURCE"):
        parse(["--all"])
    with pytest.raises(CLIError, match="KIND and a SOURCE"):
        parse(["object", "--all"])


def test_completion_offers_every_importer_kind() -> None:
    assert tuple(_IMPORT.table()) == IMPORT_KINDS


def test_out_needs_a_value() -> None:
    with pytest.raises(CLIError, match="needs a value"):
        parse(["object", "gso", "--out"])


def test_unknown_option_lists_options() -> None:
    with pytest.raises(CLIError, match="--all, --out"):
        parse(["object", "gso", "--bogus"])


def test_parse_without_variadic_still_rejects_extras() -> None:
    with pytest.raises(CLIError, match="expected KIND NAME"):
        _parse(["a", "b", "c"], ("KIND", "NAME"))
