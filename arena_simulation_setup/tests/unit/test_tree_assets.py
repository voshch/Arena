from __future__ import annotations

from pathlib import Path

from arena_simulation_setup.tree.assets.Human import HumanIdentifier, HumanView
from arena_simulation_setup.tree.assets.Material import Material, MaterialIdentifier
from arena_simulation_setup.tree.assets.Object import ObjectIdentifier, ObjectView
from arena_simulation_setup.utils.models import ModelWrapper


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
