"""parse_scene_models against gz.msgs.Scene textproto as `gz service` prints it."""

from __future__ import annotations

from arena_runtime.gz_scene import parse_scene_models

_SCENE = """ambient {
  r: 0.4
  a: 1
}
shadows: true
model {
  name: "env_0/jackal"
  id: 4
  pose {
    position {
      x: 1
      y: 2
    }
    orientation {
      w: 1
    }
  }
  link {
    id: 5
    name: "base_link"
    pose {
      position {
      }
    }
  }
}
model {
  id: 12
  name: "env_0/wall_2"
  link { id: 13 name: "link" }
}
origin_visual: true
"""


def test_top_level_models_only():
    assert parse_scene_models(_SCENE) == {"env_0/jackal": 4, "env_0/wall_2": 12}


def test_field_order_does_not_matter():
    assert parse_scene_models('model {\n  id: 7\n  name: "a"\n}\n') == {"a": 7}


def test_duplicate_name_keeps_newest_id():
    scene = 'model {\n  name: "env_0/jackal"\n  id: 4\n}\nmodel {\n  name: "env_0/jackal"\n  id: 529\n}\nmodel {\n  name: "env_0/jackal"\n  id: 3\n}\n'
    assert parse_scene_models(scene) == {"env_0/jackal": 529}


def test_empty_world():
    assert parse_scene_models("ambient {\n  r: 0.4\n}\ngrid: true\n") == {}


def test_escaped_quote_in_name():
    assert parse_scene_models('model {\n  name: "we\\"ird"\n  id: 1\n}\n') == {'we\\"ird': 1}
