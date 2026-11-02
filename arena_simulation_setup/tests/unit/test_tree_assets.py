from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from arena_simulation_setup.tree import DynamicPaths
from arena_simulation_setup.tree.assets.Human import HumanIdentifier, HumanView
from arena_simulation_setup.tree.assets.Material import Material, MaterialIdentifier
from arena_simulation_setup.tree.assets.Object import ObjectIdentifier, ObjectView
from arena_simulation_setup.utils.models import ModelType, ModelWrapper


def test_object_identifier_load_sdf_only(tmp_path: Path) -> None:
    model_name = "test_model"
    model_dir = tmp_path / model_name
    model_dir.mkdir()
    sdf_file = model_dir / f"{model_name}.sdf"
    sdf_file.write_text("<sdf></sdf>")

    ident = ObjectIdentifier(model_name)
    view = ident.load(model_dir)
    assert isinstance(view, ObjectView)
    assert isinstance(view.model, ModelWrapper)
    assert view.model.name == model_name


def test_object_identifier_load_empty_dir(tmp_path: Path) -> None:
    model_name = "empty_model"
    model_dir = tmp_path / model_name
    model_dir.mkdir()

    ident = ObjectIdentifier(model_name)
    view = ident.load(model_dir)
    assert isinstance(view, ObjectView)
    assert isinstance(view.model, ModelWrapper)


def test_object_identifier_load_nested_sdf(tmp_path: Path) -> None:
    model_name = "nested_model"
    model_dir = tmp_path / model_name
    nested_dir = model_dir / f"{model_name}.sdf"
    nested_dir.mkdir(parents=True)
    (nested_dir / f"{model_name}.sdf").write_text("<sdf/>")

    ident = ObjectIdentifier(model_name)
    view = ident.load(model_dir)
    assert isinstance(view, ObjectView)
    assert isinstance(view.model, ModelWrapper)


def test_pedestrian_identifier_load_sdf(tmp_path: Path) -> None:
    model_name = "pedestrian_test"
    model_dir = tmp_path / model_name
    model_dir.mkdir()
    (model_dir / f"{model_name}.sdf").write_text("<sdf/>")

    ident = HumanIdentifier(model_name)
    view = ident.load(model_dir)
    assert isinstance(view, HumanView)
    assert isinstance(view.model, ModelWrapper)
    assert view.model.name == model_name


def test_a_human_model_resolves_to_the_rig_sdf_when_the_bundle_has_one(tmp_path: Path) -> None:
    bundle = tmp_path / "alice"
    (bundle / "rig").mkdir(parents=True)
    (bundle / "alice.sdf").write_text("<sdf/>")
    view = HumanIdentifier("alice").load(bundle)
    assert asyncio.run(view.model.get(ModelType.SDF)).path == bundle / "alice.sdf"
    (bundle / "rig" / "alice.sdf").write_text("<sdf/>")
    assert asyncio.run(HumanIdentifier("alice").load(bundle).model.get(ModelType.SDF)).path == bundle / "rig" / "alice.sdf"


def test_material_identifier_load_no_tint(tmp_path: Path) -> None:
    mat_name = "TestMat"
    mat_dir = tmp_path / mat_name
    mat_dir.mkdir()
    mdl_path = mat_dir / f"{mat_name}.mdl"
    mdl_path.write_text("\n")

    ident = MaterialIdentifier(mat_name)
    mat = ident.load(mat_dir)
    assert isinstance(mat, Material)
    assert mat.name == mat_name


def test_material_identifier_load_tint_no_textures(tmp_path: Path) -> None:
    mat_name = "TintedMat"
    mat_dir = tmp_path / mat_name
    mat_dir.mkdir()
    mdl_path = mat_dir / f"{mat_name}.mdl"
    mdl_path.write_text("\n")

    ident = MaterialIdentifier(f"{mat_name}?tint=rgb(1,0,0)")
    mat = ident.load(mat_dir)
    assert isinstance(mat, Material)


def _human_bundle(root: Path, name: str, tags: list[str] | None) -> None:
    bundle = root / "Common" / "Human" / name
    bundle.mkdir(parents=True)
    (bundle / f"{name}.sdf").write_text("<sdf/>")
    if tags is not None:
        (bundle / "annotation.yaml").write_text("tags:\n" + "".join(f"- {tag}\n" for tag in tags))


def test_human_view_reads_annotation_tags(tmp_path: Path) -> None:
    _human_bundle(tmp_path, "tagged", ["human::mobility::wheelchair", "domain::common"])
    _human_bundle(tmp_path, "bare", None)
    assert HumanView(tmp_path / "Common" / "Human" / "tagged").tags == {"human::mobility::wheelchair", "domain::common"}
    assert HumanView(tmp_path / "Common" / "Human" / "bare").tags == frozenset()


def test_human_identifier_tagged_lists_bundles_carrying_every_tag(tmp_path: Path) -> None:
    _human_bundle(tmp_path, "zed_chair", ["probe::mobility::chair", "probe::age::old"])
    _human_bundle(tmp_path, "amy_chair", ["probe::mobility::chair"])
    _human_bundle(tmp_path, "walker", ["probe::age::old"])
    _human_bundle(tmp_path, "bare", None)
    previous = DynamicPaths.ARENA.path
    DynamicPaths.ARENA.path = tmp_path
    try:
        chairs = asyncio.run(HumanIdentifier.tagged(["probe::mobility::chair"]))
        old_chairs = asyncio.run(HumanIdentifier.tagged(["probe::mobility::chair", "probe::age::old"]))
        nobody = asyncio.run(HumanIdentifier.tagged(["probe::mobility::scooter"]))
    finally:
        DynamicPaths.ARENA.path = previous
    assert [ident.name for ident in chairs] == ["amy_chair", "zed_chair"]
    assert [ident.name for ident in old_chairs] == ["zed_chair"]
    assert nobody == []


def test_material_default_floor():
    m = Material.default('floor')
    assert isinstance(m, MaterialIdentifier)


def test_material_default_wall():
    m = Material.default('wall')
    assert isinstance(m, MaterialIdentifier)


def test_material_default_door():
    m = Material.default('door')
    assert isinstance(m, MaterialIdentifier)


def test_material_default_unknown_fallback():
    m = Material.default('some_unknown_context')
    assert isinstance(m, MaterialIdentifier)
    assert m.name == "Marble"


def test_object_view_reads_placed_affordance_slots(tmp_path: Path) -> None:
    model_dir = tmp_path / "bench"
    model_dir.mkdir()
    (model_dir / "annotation.yaml").write_text("affordances:\n  sit:\n    - {bbox: [[-0.5, 0.5], [0.0, 0.4], [0.0, 0.45]], yaw: -y}\n    - {}\n")
    slots = ObjectIdentifier("bench").load(model_dir).affordances
    assert [(slot.kind, slot.index, slot.bbox[2]) for slot in slots] == [("sit", 0, (0.0, 0.45))]
    empty = tmp_path / "box"
    empty.mkdir()
    assert ObjectIdentifier("box").load(empty).affordances == []


CABINET_LINKS = {
    "body": [[-0.5, 0.5], [-0.3, 0.3], [0.0, 1.0]],
    "door": [[-0.5, 0.5], [-0.32, -0.3], [0.0, 0.5]],
    "drawer": [[-0.45, 0.45], [-0.3, 0.25], [0.6, 0.8]],
}
CABINET_SWEPT = [[-0.5, 0.52], [-1.32, 0.3], [0.0, 1.0]]


def _cabinet(root: Path) -> Path:
    bundle = root / "cabinet"
    articulation = {
        "links": {"body": ["body"], "door": ["door"], "drawer": ["drawer"], "knob": []},
        "joints": [
            {"name": "door", "type": "revolute", "parent": "body", "child": "door", "origin": [0.5, -0.3, 0.0], "axis": [0, 0, 1], "limit": [0.0, 1.57]},
            {"name": "drawer", "type": "prismatic", "parent": "body", "child": "drawer", "axis": [0, -1, 0], "limit": [0.0, 0.4]},
            {"name": "knob", "type": "continuous", "parent": "door", "child": "knob", "origin": [0.0, -0.32, 0.25], "axis": [0, 1, 0]},
        ],
        "states": {"closed": {"door": 0.0, "drawer": 0.0}, "open": {"door": 1.57, "drawer": 0.4}},
        "swept": CABINET_SWEPT,
    }
    for link, box in CABINET_LINKS.items():
        (bundle / "links" / link).mkdir(parents=True)
        (bundle / "links" / link / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": box}))
    (bundle / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": CABINET_LINKS["body"], "articulation": articulation}))
    return bundle


def test_object_view_parses_its_articulation_and_opens_link_bundles(tmp_path: Path) -> None:
    view = ObjectView(_cabinet(tmp_path))
    assert view.articulation is not None
    assert view.articulation.root == "body"
    assert [joint.name for joint in view.articulation.movable] == ["door", "drawer", "knob"]
    door = view.link("door")
    assert door.path == view.path / "links" / "door"
    assert door.model.name == "door"
    assert door.articulation is None
    assert door.bbox is not None
    assert [*door.bbox[0], *door.bbox[1]] == pytest.approx([1.0, 0.02, 0.5, 0.0, -0.31, 0.25])


def test_object_view_link_without_a_bundle_names_the_bundles_there_are(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="has no link bundle 'knob', links: body, door, drawer"):
        ObjectView(_cabinet(tmp_path)).link("knob")
    rigid = tmp_path / "crate"
    rigid.mkdir()
    with pytest.raises(FileNotFoundError, match="has no link bundle 'lid', links: none"):
        ObjectView(rigid).link("lid")


def test_object_view_bounds_cover_the_swing_while_bbox_stays_at_rest(tmp_path: Path) -> None:
    view = ObjectView(_cabinet(tmp_path))
    assert view.extent == ((-0.5, 0.52), (-1.32, 0.3), (0.0, 1.0))
    assert view.bounds == [(-0.5, -1.32), (-0.5, 0.3), (0.52, 0.3), (0.52, -1.32)]
    assert view.bbox == ((1.0, 0.6, 1.0), (0.0, 0.0, 0.5))


def test_rigid_object_view_has_no_articulation_and_bounds_at_rest(tmp_path: Path) -> None:
    rigid = tmp_path / "crate"
    rigid.mkdir()
    (rigid / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": [[-0.2, 0.2], [-0.1, 0.1], [0.0, 0.3]]}))
    view = ObjectView(rigid)
    assert view.articulation is None
    assert view.bounds == [(-0.2, -0.1), (-0.2, 0.1), (0.2, 0.1), (0.2, -0.1)]
