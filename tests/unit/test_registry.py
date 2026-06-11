from __future__ import annotations

import pytest

import woracle.testing.plugins  # noqa: F401  (registers first-party components)
from woracle.errors import PluginError
from woracle.registry import (
    _REGISTRY,
    available,
    get,
    load_entry_points,
    register,
)


def test_first_party_components_registered() -> None:
    assert "blob.color" in available("grounder")
    assert {"binding_health", "permanence", "motion_sanity"} <= set(available("gate_signal"))
    assert {"progress.goal_distance", "success.predicates"} <= set(available("channel"))


def test_get_unknown_is_helpful() -> None:
    with pytest.raises(PluginError, match="available:"):
        get("channel", "does.not.exist")


def test_unknown_kind_rejected() -> None:
    with pytest.raises(PluginError):
        register("flavor", "x")
    with pytest.raises(PluginError):
        get("flavor", "x")


def test_duplicate_registration_rejected() -> None:
    @register("channel", "_test.dup")
    class A:
        pass

    try:
        with pytest.raises(PluginError, match="already registered"):

            @register("channel", "_test.dup")
            class B:
                pass

        # idempotent re-registration of the SAME class is fine (re-imports)
        register("channel", "_test.dup")(A)
    finally:
        _REGISTRY["channel"].pop("_test.dup", None)


def test_entry_point_loader_is_defensive(monkeypatch) -> None:
    class BrokenEP:
        name = "broken_plugin"

        def load(self):
            raise ImportError("dependency hell")

    class GoodEP:
        name = "good_plugin"

        def load(self):
            return lambda: None

    monkeypatch.setattr("importlib.metadata.entry_points", lambda group: [BrokenEP(), GoodEP()])
    report = load_entry_points()
    assert report.loaded == ["good_plugin"]
    assert "broken_plugin" in report.failed
    assert "ImportError" in report.failed["broken_plugin"]


def test_entry_point_loader_disable_env(monkeypatch) -> None:
    monkeypatch.setenv("WORACLE_DISABLE_PLUGINS", "1")
    report = load_entry_points()
    assert report.skipped and not report.loaded
