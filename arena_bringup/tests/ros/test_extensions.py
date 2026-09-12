from __future__ import annotations

import json
import logging
import textwrap
from types import SimpleNamespace

import pytest


def _make_context():
    import launch

    return launch.LaunchContext()


def _set_rules(ctx, rules: list[tuple[str, str]]) -> None:
    ctx.launch_configurations["NodeLogLevelExtension_log_level"] = json.dumps(rules)


def _node(fqn: str):
    return SimpleNamespace(node_name=fqn)


class TestParseLogLevelSpec:
    def test_bare_scalar(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import parse_log_level_spec

        assert parse_log_level_spec("debug") == ("replace", [("**/*", "debug")])

    def test_inline_default_only(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import parse_log_level_spec

        assert parse_log_level_spec("{info}") == ("replace", [("**/*", "info")])

    def test_inline_rules_with_default(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import parse_log_level_spec

        spec = "{**/nav2*/**:fatal, /dummy/node:warn, info}"
        assert parse_log_level_spec(spec) == (
            "replace",
            [
                ("**/nav2*/**", "fatal"),
                ("/dummy/node", "warn"),
                ("**/*", "info"),
            ],
        )

    def test_inline_no_default_is_allowed(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import parse_log_level_spec

        assert parse_log_level_spec("{/foo:debug}") == ("replace", [("/foo", "debug")])

    def test_inline_default_must_be_last(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import parse_log_level_spec

        with pytest.raises(ValueError, match="bare log_level"):
            parse_log_level_spec("{info, /foo:debug}")

    def test_inline_rejects_warning_alias(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import parse_log_level_spec

        with pytest.raises(ValueError, match="invalid log level"):
            parse_log_level_spec("{warning}")

    def test_inline_empty_braces(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import parse_log_level_spec

        with pytest.raises(ValueError, match="empty"):
            parse_log_level_spec("{}")

    def test_prepend_form(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import parse_log_level_spec

        assert parse_log_level_spec("+[**/nav2*/**:error]") == (
            "prepend",
            [("**/nav2*/**", "error")],
        )

    def test_prepend_multiple_rules(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import parse_log_level_spec

        assert parse_log_level_spec("+[/foo:debug, /bar/**:warn]") == (
            "prepend",
            [("/foo", "debug"), ("/bar/**", "warn")],
        )

    def test_append_form(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import parse_log_level_spec

        assert parse_log_level_spec("[/foo:debug]+") == (
            "append",
            [("/foo", "debug")],
        )

    def test_merge_form_rejects_bare_level(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import parse_log_level_spec

        with pytest.raises(ValueError, match="must be"):
            parse_log_level_spec("+[warn]")

    def test_yaml_file(self, tmp_path) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import parse_log_level_spec

        f = tmp_path / "lvl.yaml"
        f.write_text(
            textwrap.dedent("""
            default: warn
            rules:
              - { match: '**/nav2*/**', level: fatal }
              - { match: '/dummy/node', level: error }
        """)
        )
        assert parse_log_level_spec(str(f)) == (
            "replace",
            [
                ("**/nav2*/**", "fatal"),
                ("/dummy/node", "error"),
                ("**/*", "warn"),
            ],
        )

    def test_yaml_file_no_default(self, tmp_path) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import parse_log_level_spec

        f = tmp_path / "lvl.yaml"
        f.write_text("rules: [{match: '/foo', level: debug}]\n")
        assert parse_log_level_spec(str(f)) == ("replace", [("/foo", "debug")])

    def test_yaml_file_rejects_bad_level(self, tmp_path) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import parse_log_level_spec

        f = tmp_path / "lvl.yaml"
        f.write_text("default: warning\n")
        with pytest.raises(ValueError):
            parse_log_level_spec(str(f))

    def test_unknown_value_errors(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import parse_log_level_spec

        with pytest.raises(ValueError, match="not a known level"):
            parse_log_level_spec("verbose")

    def test_empty_value_returns_empty(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import parse_log_level_spec

        assert parse_log_level_spec("") == ("replace", [])


class TestGlobToRegex:
    def test_double_star_matches_zero_segments(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import _glob_to_regex

        rx = _glob_to_regex("**/nav2*/**")
        assert rx.match("nav2x")
        assert rx.match("nav2x/y")
        assert rx.match("a/nav2x")
        assert rx.match("a/nav2x/y/z")
        assert not rx.match("nav3x")
        assert not rx.match("a/foo/bar")

    def test_anchored_pattern(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import _glob_to_regex

        rx = _glob_to_regex("/dummy/node")
        assert rx.match("dummy/node")
        assert not rx.match("env_0/dummy/node")
        assert not rx.match("dummy/node/child")

    def test_single_star_within_segment(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import _glob_to_regex

        rx = _glob_to_regex("/env_*/jackal")
        assert rx.match("env_0/jackal")
        assert rx.match("env_42/jackal")
        assert not rx.match("env_0/foo/jackal")

    def test_default_pattern(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import _glob_to_regex

        rx = _glob_to_regex("**/*")
        assert rx.match("foo")
        assert rx.match("a/b/c")

    def test_middle_double_star_collapses(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import _glob_to_regex

        rx = _glob_to_regex("/foo/**/bar")
        assert rx.match("foo/bar")
        assert rx.match("foo/x/bar")
        assert rx.match("foo/x/y/bar")
        assert not rx.match("foo/bar/baz")


class TestMatchLevel:
    def test_first_match_wins(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import _match_level

        rules = [("/env_0/**", "debug"), ("**/*", "warn")]
        assert _match_level(rules, "/env_0/jackal") == "debug"
        assert _match_level(rules, "/env_1/jackal") == "warn"

    def test_no_match_returns_none(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import _match_level

        assert _match_level([("/foo", "debug")], "/bar") is None

    def test_strips_leading_slash_from_fqn(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import _match_level

        assert _match_level([("foo", "info")], "/foo") == "info"


class TestNodeLogLevelExtension:
    def test_applies_default_rule(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import NodeLogLevelExtension

        ctx = _make_context()
        _set_rules(ctx, [("**/*", "debug")])
        ext = NodeLogLevelExtension()
        args, _ = ext.prepare_for_execute(ctx, {}, _node("/anything"))
        assert len(args) == 2
        assert args[0][0].text == "--log-level"
        assert args[1][0].text == "debug"

    def test_applies_specific_rule(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import NodeLogLevelExtension

        ctx = _make_context()
        _set_rules(
            ctx,
            [
                ("**/nav2*/**", "fatal"),
                ("/dummy/node", "warn"),
                ("**/*", "info"),
            ],
        )
        ext = NodeLogLevelExtension()
        args, _ = ext.prepare_for_execute(ctx, {}, _node("/env_0/jackal/nav2_controller"))
        assert args[1][0].text == "fatal"

    def test_no_match_emits_no_args(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import NodeLogLevelExtension

        ctx = _make_context()
        _set_rules(ctx, [("/foo", "debug")])
        ext = NodeLogLevelExtension()
        args, _ = ext.prepare_for_execute(ctx, {}, _node("/bar"))
        assert args == []

    def test_no_config_emits_no_args(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import NodeLogLevelExtension

        ctx = _make_context()
        ext = NodeLogLevelExtension()
        args, _ = ext.prepare_for_execute(ctx, {}, _node("/foo"))
        assert args == []

    def test_returns_ros_specific_unchanged(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import NodeLogLevelExtension

        ctx = _make_context()
        _set_rules(ctx, [("**/*", "info")])
        extra = {"k": "v"}
        ext = NodeLogLevelExtension()
        _, ros_specific = ext.prepare_for_execute(ctx, extra, _node("/foo"))
        assert ros_specific == extra

    def test_first_match_wins_in_extension(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import NodeLogLevelExtension

        ctx = _make_context()
        _set_rules(ctx, [("**/*", "warn"), ("/foo", "debug")])
        ext = NodeLogLevelExtension()
        args, _ = ext.prepare_for_execute(ctx, {}, _node("/foo"))
        assert args[1][0].text == "warn"


class TestSetGlobalLogLevelAction:
    def test_str_to_level_debug(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import SetGlobalLogLevelAction

        assert SetGlobalLogLevelAction.str_to_level("debug") == logging.DEBUG

    def test_str_to_level_info(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import SetGlobalLogLevelAction

        assert SetGlobalLogLevelAction.str_to_level("info") == logging.INFO

    def test_str_to_level_warn(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import SetGlobalLogLevelAction

        assert SetGlobalLogLevelAction.str_to_level("warn") == logging.WARN

    def test_str_to_level_error(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import SetGlobalLogLevelAction

        assert SetGlobalLogLevelAction.str_to_level("error") == logging.ERROR

    def test_str_to_level_fatal(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import SetGlobalLogLevelAction

        assert SetGlobalLogLevelAction.str_to_level("fatal") == logging.FATAL

    def test_str_to_level_unknown_returns_notset(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import SetGlobalLogLevelAction

        assert SetGlobalLogLevelAction.str_to_level("unknown_level") == logging.NOTSET

    def test_execute_bare_scalar(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import SetGlobalLogLevelAction

        ctx = _make_context()
        SetGlobalLogLevelAction("info").execute(ctx)
        assert json.loads(ctx.launch_configurations["NodeLogLevelExtension_log_level"]) == [
            ["**/*", "info"],
        ]

    def test_execute_inline_rules(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import SetGlobalLogLevelAction

        ctx = _make_context()
        SetGlobalLogLevelAction("{**/nav2*/**:fatal, /dummy/node:warn, info}").execute(ctx)
        assert json.loads(ctx.launch_configurations["NodeLogLevelExtension_log_level"]) == [
            ["**/nav2*/**", "fatal"],
            ["/dummy/node", "warn"],
            ["**/*", "info"],
        ]

    def test_execute_overwrites_existing(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import SetGlobalLogLevelAction

        ctx = _make_context()
        SetGlobalLogLevelAction("debug").execute(ctx)
        SetGlobalLogLevelAction("error").execute(ctx)
        assert json.loads(ctx.launch_configurations["NodeLogLevelExtension_log_level"]) == [
            ["**/*", "error"],
        ]

    def test_execute_prepend(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import SetGlobalLogLevelAction

        ctx = _make_context()
        SetGlobalLogLevelAction("warn").execute(ctx)
        SetGlobalLogLevelAction("+[**/nav2*/**:error]").execute(ctx)
        assert json.loads(ctx.launch_configurations["NodeLogLevelExtension_log_level"]) == [
            ["**/nav2*/**", "error"],
            ["**/*", "warn"],
        ]

    def test_execute_append(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import SetGlobalLogLevelAction

        ctx = _make_context()
        SetGlobalLogLevelAction("warn").execute(ctx)
        SetGlobalLogLevelAction("[**/nav2*/**:error]+").execute(ctx)
        assert json.loads(ctx.launch_configurations["NodeLogLevelExtension_log_level"]) == [
            ["**/*", "warn"],
            ["**/nav2*/**", "error"],
        ]

    def test_execute_prepend_with_no_existing_seeds_base(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import SetGlobalLogLevelAction

        ctx = _make_context()
        SetGlobalLogLevelAction("+[/foo:debug]").execute(ctx)
        assert json.loads(ctx.launch_configurations["NodeLogLevelExtension_log_level"]) == [
            ["/foo", "debug"],
            ["**/*", "warn"],
        ]

    def test_execute_append_with_no_existing_seeds_base(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import SetGlobalLogLevelAction

        ctx = _make_context()
        SetGlobalLogLevelAction("[/foo:debug]+").execute(ctx)
        assert json.loads(ctx.launch_configurations["NodeLogLevelExtension_log_level"]) == [
            ["**/*", "warn"],
            ["/foo", "debug"],
        ]

    def test_execute_prepend_with_custom_base(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import SetGlobalLogLevelAction

        ctx = _make_context()
        SetGlobalLogLevelAction("+[/foo:debug]", base="info").execute(ctx)
        assert json.loads(ctx.launch_configurations["NodeLogLevelExtension_log_level"]) == [
            ["/foo", "debug"],
            ["**/*", "info"],
        ]

    def test_execute_seed_only_when_existing_empty(self) -> None:
        from arena_bringup.extensions.NodeLogLevelExtension import SetGlobalLogLevelAction

        ctx = _make_context()
        SetGlobalLogLevelAction("error").execute(ctx)
        SetGlobalLogLevelAction("+[/foo:debug]").execute(ctx)
        assert json.loads(ctx.launch_configurations["NodeLogLevelExtension_log_level"]) == [
            ["/foo", "debug"],
            ["**/*", "error"],
        ]
