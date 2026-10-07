"""Tests for the argv the import verb hands to arena_assets."""

import json
import os
import time

import pytest

from arena_cli.asset import _IMPORT, IMPORT_KINDS, ITEM_CANDIDATES, _import_args, _parse, import_listing, listed_items
from arena_cli.common import CLIError
from arena_cli.complete import Context, _expand


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


def candidates(words: list[str], cur: str = "") -> list[str]:
    ctx = Context("/nonexistent", {"values": {"asset.import.world": {"barn": "challenge worlds", "houseexpo": "floor plans"}}})
    return [value for value, _, _ in _expand(_IMPORT, words, cur, ctx).items if not value.startswith("--")]


def cache_items(root: str, source: str, items: list[str], expires_at: float | None) -> str:
    path = os.path.join(root, ".import", source, "items.world.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"items": items, "fetched_at": 0.0, "expires_at": expires_at}, f)
    with open(path + ".lock", "w"):
        pass
    return path


@pytest.fixture
def assets_dir(tmp_path):
    saved = os.environ.get("ARENA_ASSETS_DIR")
    os.environ["ARENA_ASSETS_DIR"] = str(tmp_path)
    yield str(tmp_path)
    os.environ.pop("ARENA_ASSETS_DIR", None)
    if saved is not None:
        os.environ["ARENA_ASSETS_DIR"] = saved


def test_source_position_offers_the_sources_of_the_kind(assets_dir) -> None:
    assert candidates(["world"]) == ["barn", "houseexpo"]


def test_item_position_offers_the_cached_items_by_prefix(assets_dir) -> None:
    cache_items(assets_dir, "barn", ["0", "1", "10", "11", "2"], None)
    assert candidates(["world", "barn"]) == ["0", "1", "10", "11", "2"]
    assert candidates(["world", "barn"], "1") == ["1", "10", "11"]
    assert candidates(["world", "barn", "10"], "1") == ["1", "11"]
    assert candidates(["world", "barn", "--out", "/tmp/x"], "2") == ["2"]


def test_item_position_caps_a_long_list(assets_dir) -> None:
    cache_items(assets_dir, "houseexpo", [f"{index:032x}" for index in range(2000)], time.time() + 60)
    assert len(candidates(["world", "houseexpo"])) == ITEM_CANDIDATES


def test_item_position_without_a_cache_offers_nothing(assets_dir) -> None:
    assert candidates(["world", "barn"]) == []


def test_cached_list_expiry(assets_dir) -> None:
    now = time.time()
    path = cache_items(assets_dir, "barn", ["0"], now - 1)
    assert listed_items(path, now) == (["0"], True)
    path = cache_items(assets_dir, "barn", ["0"], now + 1)
    assert listed_items(path, now) == (["0"], False)
    path = cache_items(assets_dir, "barn", ["0"], None)
    assert listed_items(path, now + 1e9) == (["0"], False)
    assert listed_items(os.path.join(assets_dir, "missing.json"), now) == ([], True)


def test_listing_path_follows_the_importer_cache_layout(assets_dir) -> None:
    assert import_listing("world", "barn") == os.path.join(assets_dir, ".import", "barn", "items.world.json")
    del os.environ["ARENA_ASSETS_DIR"]
    assert import_listing("world", "barn") is None
