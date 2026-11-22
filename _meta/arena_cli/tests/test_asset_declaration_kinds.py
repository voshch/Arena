"""The plot and query kinds of `arena asset` bundle in the benchmark family, next to suites and reports."""

from arena_cli.asset import KINDS


def test_plot_and_query_kinds_are_bundles_in_the_benchmark_family() -> None:
    for kind in ("plot", "query"):
        assert KINDS[kind].bundle
        assert KINDS[kind].providers == "BENCHMARK_PROVIDERS"
        assert KINDS[kind].module == KINDS["manifest"].module
