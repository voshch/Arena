from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from arena_simulation_setup.utils.models import Model, ModelType, ModelWrapper
from arena_simulation_setup.utils.models.obj import ModelProvider_OBJ
from arena_simulation_setup.utils.models.sdf import ModelProvider_SDF
from arena_simulation_setup.utils.models.usd import ModelProvider_USD
from arena_simulation_setup.utils.models.yaml import ModelProvider_YAML


def test_sdf_load_flat_file(tmp_path):
    model_dir = tmp_path / "mymodel"
    model_dir.mkdir()
    (model_dir / "mymodel.sdf").write_text("<sdf/>")

    model = asyncio.run(ModelProvider_SDF.load(model_dir, "mymodel", None))
    assert model.type == ModelType.SDF
    assert model.name == "mymodel"
    assert "<sdf/>" in model.description


def test_sdf_load_nested_dir(tmp_path):
    model_dir = tmp_path / "mymodel2"
    nested = model_dir / "mymodel2.sdf"
    nested.mkdir(parents=True)
    (nested / "mymodel2.sdf").write_text("<sdf>nested</sdf>")

    model = asyncio.run(ModelProvider_SDF.load(model_dir, "mymodel2", None))
    assert "nested" in model.description


def test_sdf_load_not_found_raises(tmp_path):
    model_dir = tmp_path / "missing"
    model_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        asyncio.run(ModelProvider_SDF.load(model_dir, "missing_model", None))


def test_usd_load_usdz(tmp_path):
    (tmp_path / "test.usdz").touch()
    model = asyncio.run(ModelProvider_USD.load(tmp_path, "test", None))
    assert model.type == ModelType.USD


def test_usd_load_usd(tmp_path):
    (tmp_path / "test.usd").touch()
    model = asyncio.run(ModelProvider_USD.load(tmp_path, "test", None))
    assert model.type == ModelType.USD


def test_usd_load_usda(tmp_path):
    (tmp_path / "test.usda").touch()
    model = asyncio.run(ModelProvider_USD.load(tmp_path, "test", None))
    assert model.type == ModelType.USD


def test_usd_load_usdc(tmp_path):
    (tmp_path / "test.usdc").touch()
    model = asyncio.run(ModelProvider_USD.load(tmp_path, "test", None))
    assert model.type == ModelType.USD


def test_usd_load_none_present_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        asyncio.run(ModelProvider_USD.load(tmp_path, "no_usd_model", None))


def test_usd_convertable_returns_sdf():
    result = ModelProvider_USD.convertable()
    assert ModelType.SDF in result


def test_usd_convert_non_sdf_returns_none(tmp_path):
    model = Model(type=ModelType.YAML, name="x", description="", path=tmp_path / "x.yaml")
    result = asyncio.run(ModelProvider_USD.convert(tmp_path, model, None))
    assert result is None


def test_yaml_load(tmp_path):
    (tmp_path / "mymodel.yaml").write_text("key: value\n")
    model = asyncio.run(ModelProvider_YAML.load(tmp_path, "mymodel", None))
    assert model.type == ModelType.YAML
    assert "key" in model.description


def test_yaml_load_not_found_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        asyncio.run(ModelProvider_YAML.load(tmp_path, "nonexistent", None))


def test_model_wrapper_get_falls_through_to_the_next_type_with_a_file(tmp_path):
    (tmp_path / "chair.sdf").write_text("<sdf/>")
    wrapper = ModelWrapper("chair", {**ModelProvider_SDF.asdict(tmp_path, "chair"), **ModelProvider_OBJ.asdict(tmp_path, "chair")})

    assert asyncio.run(wrapper.get((ModelType.OBJ, ModelType.SDF))).type is ModelType.SDF
    with pytest.raises(FileNotFoundError):
        asyncio.run(wrapper.get(ModelType.OBJ))


def test_model_wrapper_serialize():
    w = ModelWrapper("my_model", {})
    assert w.serialize() == "my_model"


def test_model_wrapper_parse_str_returns_empty():
    w = ModelWrapper.parse("anything")
    assert w.name == "__EMPTY"


def test_model_wrapper_parse_instance_returns_self():
    w = ModelWrapper.EMPTY()
    w2 = ModelWrapper.parse(w)
    assert w2 is w


def test_model_wrapper_parse_invalid_raises():
    with pytest.raises((TypeError, Exception)):
        ModelWrapper.parse(12345)


def test_model_wrapper_clone_is_new_instance():
    w = ModelWrapper("orig", {})
    c = w.clone()
    assert c is not w
    assert c.name == "orig"


def test_model_type_values():
    assert ModelType.UNKNOWN.value == ""
    assert ModelType.URDF.value == "urdf"
    assert ModelType.SDF.value == "sdf"
    assert ModelType.YAML.value == "yaml"
    assert ModelType.USD.value == "usd"


def test_model_empty_fields():
    m = Model.EMPTY()
    assert m.type == ModelType.UNKNOWN
    assert m.name == "__EMPTY"
    assert m.path is not None


def test_model_replace_name():
    m = Model.EMPTY()
    m2 = m.replace(name="new_name")
    assert m2.name == "new_name"
    assert m2.type == m.type


def test_model_replace_type():
    m = Model.EMPTY()
    m2 = m.replace(type=ModelType.SDF)
    assert m2.type == ModelType.SDF
    assert m2.name == m.name


def test_model_mapper_always_returns_self():
    m = Model.EMPTY()
    other = Model(type=ModelType.SDF, name="other", description="d", path=None)
    assert m.mapper(other) is m


def test_model_wrapper_constant_single_type():
    m = Model(type=ModelType.SDF, name="box", description="<sdf/>", path=None)
    w = ModelWrapper.Constant("box", {ModelType.SDF: m})
    result = asyncio.run(w.get(ModelType.SDF))
    assert result is m


def test_model_wrapper_constant_multi_type():
    m_sdf = Model(type=ModelType.SDF, name="box", description="<sdf/>", path=None)
    m_yaml = Model(type=ModelType.YAML, name="box", description="key: v", path=None)
    w = ModelWrapper.Constant("box", {ModelType.SDF: m_sdf, ModelType.YAML: m_yaml})
    assert asyncio.run(w.get(ModelType.SDF)) is m_sdf
    assert asyncio.run(w.get(ModelType.YAML)) is m_yaml


def test_model_wrapper_constant_missing_type_raises():
    m = Model(type=ModelType.SDF, name="box", description="", path=None)
    w = ModelWrapper.Constant("box", {ModelType.SDF: m})
    with pytest.raises(FileNotFoundError):
        asyncio.run(w.get(ModelType.USD))


def test_model_wrapper_from_model_name():
    m = Model(type=ModelType.URDF, name="robot", description="", path=None)
    w = ModelWrapper.from_model(m)
    assert w.name == "robot"


def test_model_wrapper_from_model_get():
    m = Model(type=ModelType.URDF, name="robot", description="", path=None)
    w = ModelWrapper.from_model(m)
    result = asyncio.run(w.get(ModelType.URDF))
    assert result is m


def test_model_wrapper_from_model_wrong_type_raises():
    m = Model(type=ModelType.URDF, name="robot", description="", path=None)
    w = ModelWrapper.from_model(m)
    with pytest.raises(FileNotFoundError):
        asyncio.run(w.get(ModelType.SDF))


def test_model_wrapper_get_any_picks_first_available():
    m_sdf = Model(type=ModelType.SDF, name="box", description="", path=None)
    m_yaml = Model(type=ModelType.YAML, name="box", description="", path=None)
    w = ModelWrapper.Constant("box", {ModelType.SDF: m_sdf, ModelType.YAML: m_yaml})
    result = asyncio.run(w.get())
    assert result in (m_sdf, m_yaml)


def test_model_wrapper_get_collection_filter():
    m_sdf = Model(type=ModelType.SDF, name="box", description="", path=None)
    m_yaml = Model(type=ModelType.YAML, name="box", description="", path=None)
    w = ModelWrapper.Constant("box", {ModelType.SDF: m_sdf, ModelType.YAML: m_yaml})
    result = asyncio.run(w.get([ModelType.YAML]))
    assert result is m_yaml


def test_model_wrapper_get_no_matching_loader_raises():
    w = ModelWrapper("empty", {})
    with pytest.raises(FileNotFoundError):
        asyncio.run(w.get(ModelType.SDF))


def test_model_wrapper_override_replaces_loader():
    original = Model(type=ModelType.SDF, name="orig", description="orig_desc", path=None)
    replaced = Model(type=ModelType.SDF, name="replaced", description="new_desc", path=None)
    w = ModelWrapper.Constant("box", {ModelType.SDF: original})
    w.override(ModelType.SDF, lambda m: replaced)
    result = asyncio.run(w.get(ModelType.SDF))
    assert result is replaced


def test_model_wrapper_override_stacks_transforms():
    base = Model(type=ModelType.SDF, name="base", description="desc", path=None)
    w = ModelWrapper.Constant("box", {ModelType.SDF: base})
    w.override(ModelType.SDF, lambda m: m.replace(name="step1"))
    w.override(ModelType.SDF, lambda m: m.replace(name=m.name + "_step2"))
    result = asyncio.run(w.get(ModelType.SDF))
    assert result.name == "step1_step2"


def test_model_wrapper_override_noload_skips_original():
    replaced = Model(type=ModelType.SDF, name="noload", description="", path=None)
    w = ModelWrapper.Constant("box", {ModelType.SDF: Model(type=ModelType.SDF, name="orig", description="", path=None)})
    w.override(ModelType.SDF, lambda m: replaced, noload=True)
    result = asyncio.run(w.get(ModelType.SDF))
    assert result is replaced


def test_model_wrapper_override_missing_type_raises_on_get():
    replaced = Model(type=ModelType.USD, name="r", description="", path=None)
    w = ModelWrapper("box", {})
    w.override(ModelType.USD, lambda m: replaced)
    with pytest.raises((FileNotFoundError, ValueError)):
        asyncio.run(w.get(ModelType.USD))


def test_model_wrapper_override_noload_missing_type_returns_override():
    replaced = Model(type=ModelType.USD, name="r", description="", path=None)
    w = ModelWrapper("box", {})
    w.override(ModelType.USD, lambda m: replaced, noload=True)
    result = asyncio.run(w.get(ModelType.USD))
    assert result is replaced
