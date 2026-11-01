from __future__ import annotations

import pytest

from arena_simulation_setup.kg import ANY, LITERAL, PACKAGES_DIR, Vocabulary, load_package


def test_shipped_packages_are_the_six_known_versions():
    assert sorted(p.stem for p in PACKAGES_DIR.glob("*.toml")) == ["access@1", "core@1", "core@2", "robot@1", "social@1", "space@1"]


def test_space_states_a_room_purpose():
    vocab = Vocabulary(["core@2", "space@1"])
    assert vocab.verbs["has_purpose"] == {"from": ["room"], "to": ["purpose"]}
    assert vocab.is_auto("purpose")
    assert vocab.is_propless("purpose")


def test_core2_adds_placeable_kinds_and_placed_prop():
    vocab = Vocabulary(["core@2"])
    for kind in ("light", "sound", "microphone"):
        assert vocab.props(kind) == ["label"]
    assert "placed" in vocab.props("obj")
    assert vocab.verbs["contains"]["from"] == ["floor", "room"]
    assert vocab.verbs["contains"]["to"] == ["room", "obj", "door", "wall", "elevator", "stair", "light", "sound", "microphone"]


def test_core2_connects_floor_joins_elevators_and_stairs():
    vocab = Vocabulary(["core@2"])
    assert vocab.verbs["connects_floor"] == {"from": ["elevator", "stair"], "to": ["elevator", "stair"]}
    assert vocab.is_propless("stair")
    assert "stair" not in vocab.authorable()


def test_core2_matches_core1_outside_obj_contains_connects_floor_and_new_kinds():
    core1, core2 = load_package(PACKAGES_DIR / "core@1.toml"), load_package(PACKAGES_DIR / "core@2.toml")
    assert core2.constraints == core1.constraints
    assert {k: v for k, v in core2.verbs.items() if k not in ("contains", "connects_floor")} == {k: v for k, v in core1.verbs.items() if k not in ("contains", "connects_floor")}
    assert {k: v for k, v in core2.nodes.items() if k not in ("obj", "stair", "light", "sound", "microphone")} == {k: v for k, v in core1.nodes.items() if k != "obj"}


def test_bare_package_name_resolves_to_newest_version():
    assert [p.key for p in Vocabulary(["core", "social"]).packages] == ["core@2", "social@1"]


def test_unknown_package_names_the_available_ones():
    with pytest.raises(ValueError, match=r"unknown package 'core@9' \(available: access@1, core@1, core@2"):
        Vocabulary(["core@9"])


def test_access_without_social_reports_missing_requirement():
    with pytest.raises(ValueError, match=r"access@1 requires 'social@1'"):
        Vocabulary(["core@2", "access@1"])


def test_two_core_versions_conflict_on_shared_words():
    with pytest.raises(ValueError, match=r"defined by both core@1 and core@2"):
        Vocabulary(["core@1", "core@2"])


def test_access_with_social_loads_restricted_to():
    vocab = Vocabulary(["core@2", "social@1", "access@1"])
    assert vocab.verbs["restricted_to"] == {"from": ["room", "region"], "to": ["role"]}
    assert vocab.constraints["no_entry"]["arity"] == 2


def test_auto_propless_and_authorable_flags():
    vocab = Vocabulary(["core@2", "social@1", "robot@1"])
    assert vocab.is_auto("role")
    assert vocab.is_propless("role")
    assert not vocab.is_auto("room")
    assert vocab.is_propless("elevator")
    assert "wall" not in vocab.authorable()
    assert "robot" not in vocab.authorable()
    assert "room" in vocab.authorable()


def test_signature_markers_any_and_literal():
    vocab = Vocabulary(["core@2"])
    assert vocab.verbs["attr"] == {"from": ANY, "to": LITERAL}
    assert vocab.verbs["note"] == {"from": ANY, "to": LITERAL}


@pytest.mark.parametrize(
    ("value", "expected"),
    [("room:kitchen", True), ("obj:icu/bed_1", True), ("shape: L-shaped", False), ("kitchen", False), ("con:x", False), (3, False)],
)
def test_is_entity_ref(value: object, expected: bool):
    assert Vocabulary(["core@2"]).is_entity_ref(value) is expected


@pytest.mark.parametrize(
    ("toml", "message"),
    [
        ('name = "extra"\nversion = 2\n', r"declares package 'extra@2'"),
        ('name = "extra"\nversion = 1\n[verbs.hugs]\nfrom = ["room"]\n', r"verbs.hugs misses required key\(s\) 'to'"),
        ('name = "extra"\nversion = 1\n[nodes.tag]\nprops = ["label"]\npropless = true\n', r"nodes.tag is propless"),
    ],
)
def test_malformed_package_is_rejected(tmp_path, toml: str, message: str):
    path = tmp_path / "extra@1.toml"
    path.write_text(toml)
    with pytest.raises(ValueError, match=message):
        load_package(path)


def test_vocabulary_from_custom_directory(tmp_path):
    (tmp_path / "base@1.toml").write_text('name = "base"\nversion = 1\n[nodes.room]\nprops = []\n')
    (tmp_path / "extra@1.toml").write_text('name = "extra"\nversion = 1\nrequires = ["base@1"]\n[verbs.near]\nfrom = ["room"]\nto = ["room"]\n')
    vocab = Vocabulary(["base@1", "extra@1"], directory=tmp_path)
    assert list(vocab.nodes) == ["room"]
    assert list(vocab.verbs) == ["near"]
