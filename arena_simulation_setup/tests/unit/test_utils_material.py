from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from arena_simulation_setup.utils.material import Color, ImgUtil, MdlUtil


# ---------------------------------------------------------------------------
# Color
# ---------------------------------------------------------------------------


def test_color_hex_string():
    c = Color("#ff0000")
    assert c is not None


def test_color_rgb_tuple():
    c = Color("rgb(1.0,0.0,0.0)")
    assert c is not None


def test_color_named():
    c = Color("red")
    assert c is not None


def test_color_invalid_raises():
    with pytest.raises(Exception):
        Color("not_a_color_xyz_12345")


# ---------------------------------------------------------------------------
# ImgUtil.tint
# ---------------------------------------------------------------------------


def _make_rgba_image(w: int = 8, h: int = 8) -> Image.Image:
    img = Image.new("RGBA", (w, h), (128, 128, 128, 255))
    return img


def test_imutil_tint_image_input():
    img = _make_rgba_image()
    result = ImgUtil.tint(img, "red")
    assert result.mode == "RGB"
    assert result.size == (8, 8)


def test_imutil_tint_path_input(tmp_path):
    img = _make_rgba_image()
    img_path = tmp_path / "test.png"
    img.save(img_path)
    result = ImgUtil.tint(img_path, "blue")
    assert result.mode == "RGB"


def test_imutil_tint_preserves_nonuniform_alpha():
    img = Image.new("RGBA", (2, 1), (128, 128, 128, 255))
    img.putpixel((1, 0), (128, 128, 128, 0))
    result = ImgUtil.tint(img, "red")
    assert result.mode == "RGBA"
    assert [result.getpixel((x, 0))[3] for x in (0, 1)] == [255, 0]


def test_imutil_tint_rgba_strength_affects_result():
    img = Image.new("RGBA", (4, 4), (0, 0, 0, 255))
    result_light = ImgUtil.tint(img, "rgba(1.0,1.0,1.0,0.1)")
    result_heavy = ImgUtil.tint(img, "rgba(1.0,1.0,1.0,0.9)")
    r_light = result_light.getpixel((0, 0))[0]
    r_heavy = result_heavy.getpixel((0, 0))[0]
    assert r_heavy > r_light


# ---------------------------------------------------------------------------
# MdlUtil.diffuse_texture_paths
# ---------------------------------------------------------------------------


def test_mdlutil_diffuse_texture_paths_found(tmp_path):
    mdl_content = 'diffuse_texture: texture_2d("./textures/albedo.png")\n'
    mdl_file = tmp_path / "mat.mdl"
    mdl_file.write_text(mdl_content)
    util = MdlUtil(mdl_file)
    paths = list(util.diffuse_texture_paths)
    assert len(paths) == 1
    assert paths[0].name == "albedo.png"


def test_mdlutil_diffuse_texture_paths_multiple(tmp_path):
    mdl_content = (
        'diffuse_texture: texture_2d("./tex1.png")\n'
        'diffuse_texture: texture_2d("./tex2.png")\n'
    )
    mdl_file = tmp_path / "mat.mdl"
    mdl_file.write_text(mdl_content)
    paths = list(MdlUtil(mdl_file).diffuse_texture_paths)
    assert len(paths) == 2


def test_mdlutil_diffuse_texture_paths_empty(tmp_path):
    mdl_file = tmp_path / "mat.mdl"
    mdl_file.write_text("# no textures\nsome_other_field: value\n")
    paths = list(MdlUtil(mdl_file).diffuse_texture_paths)
    assert paths == []


def test_mdlutil_diffuse_texture_paths_invalid_lines(tmp_path):
    mdl_file = tmp_path / "mat.mdl"
    mdl_file.write_text("diffuse_texture: not_a_texture_2d(blah)\n")
    paths = list(MdlUtil(mdl_file).diffuse_texture_paths)
    assert paths == []


# ---------------------------------------------------------------------------
# MdlUtil.texture
# ---------------------------------------------------------------------------


def test_mdlutil_texture_diffuse_slot(tmp_path):
    mdl_file = tmp_path / "mat.mdl"
    mdl_file.write_text('diffuse_texture: texture_2d("./Mat/Mat_BaseColor.png")\n')
    tex = MdlUtil(mdl_file).texture("diffuse_texture")
    assert tex is not None
    assert tex.name == "Mat_BaseColor.png"
    assert tex.parent == tmp_path / "Mat"


def test_mdlutil_texture_normal_slot(tmp_path):
    mdl_content = (
        'diffuse_texture: texture_2d("./Mat/Mat_BaseColor.png")\n'
        'normalmap_texture: texture_2d("./Mat/Mat_N.png")\n'
    )
    mdl_file = tmp_path / "mat.mdl"
    mdl_file.write_text(mdl_content)
    assert MdlUtil(mdl_file).texture("normalmap_texture").name == "Mat_N.png"


def test_mdlutil_texture_empty_slot_returns_none(tmp_path):
    mdl_content = (
        'diffuse_texture: texture_2d("./Mat/Mat_BaseColor.png")\n'
        "normalmap_texture: texture_2d()\n"
    )
    mdl_file = tmp_path / "mat.mdl"
    mdl_file.write_text(mdl_content)
    assert MdlUtil(mdl_file).texture("normalmap_texture") is None


def test_mdlutil_texture_absent_slot_returns_none(tmp_path):
    mdl_file = tmp_path / "mat.mdl"
    mdl_file.write_text('diffuse_texture: texture_2d("./Mat/Mat_BaseColor.png")\n')
    assert MdlUtil(mdl_file).texture("ORM_texture") is None
