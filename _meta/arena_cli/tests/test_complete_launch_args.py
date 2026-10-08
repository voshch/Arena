"""Unit tests for launch-argument completion of backend launch files."""

from arena_cli import cli
from arena_cli.complete import Context, _expand

MANIFEST = {
    "launch": {
        "task_generator/task_generator.launch.py": {"acoustics": {"desc": "acoustic simulator", "default": "none"}},
        "arena_auditory/arena_auditory.launch.py": {
            "namespace": {"desc": "", "default": ""},
            "env.ns": {"desc": "", "default": ""},
            "hearing": {"desc": "", "default": "none"},
            "array.spec": {"desc": "microphone array", "default": ""},
            "viz.enabled": {"desc": "markers", "default": "true"},
        },
        "arena_hearing/hearing.launch.py": {
            "env.ns": {"desc": "", "default": ""},
            "tg_node": {"desc": "", "default": ""},
            "frontend": {"desc": "", "default": "bus"},
            "policy": {"desc": "", "default": "full"},
            "robot.hearing.belief.tau_s": {"desc": "belief decay", "default": ""},
        },
    }
}


def _keys(cur: str) -> set[str]:
    return {value for value, _desc, _group in _expand(cli.ENV_ARGS, [], cur, Context("", MANIFEST)).items}


def test_env_args_offer_the_backend_keys_under_their_launch_prefix() -> None:
    keys = _keys("")
    assert {"acoustics:=", "auditory.array.spec:=", "auditory.viz.enabled:=", "robot.hearing.belief.tau_s:="} <= keys


def test_env_args_leave_out_the_keys_the_dispatch_sets() -> None:
    keys = _keys("")
    assert not {"namespace:=", "env.ns:=", "hearing:=", "auditory.namespace:=", "auditory.env.ns:=", "auditory.hearing:=", "tg_node:=", "frontend:=", "policy:="} & keys


def test_prefixed_backend_bool_key_completes_its_values() -> None:
    assert {"true", "false"} <= _keys("auditory.viz.enabled:=")
