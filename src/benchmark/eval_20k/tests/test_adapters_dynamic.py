"""Dynamic adapter registry: ``benchmark.eval_20k.adapters.<lane>.Adapter``
takes precedence when present and well-formed; absent modules/attributes fall
back to the built-in L05/reflection registry; broken adapters surface as
explicit NOT_SUPPORTED with the precise reason — never a crash, never a
silent score.
"""
from __future__ import annotations

import importlib as importlib_module
import sys
import types

import pytest

from benchmark.eval_20k.lanes import (
    L05Adapter,
    LaneAdapter,
    LaneNotSupported,
    get_adapter,
)


def _install_fake_module(monkeypatch: pytest.MonkeyPatch, lane_lower: str,
                         adapter_cls: type | None = None) -> None:
    name = "benchmark.eval_20k.adapters.%s" % lane_lower
    module = types.ModuleType(name)
    if adapter_cls is not None:
        module.Adapter = adapter_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, module)


def test_dynamic_adapter_takes_precedence_and_receives_lane_kw(monkeypatch):
    class Adapter(LaneAdapter):
        lane_id = "L05"

        def build_verifier(self):
            raise LaneNotSupported("fake-l05")

    _install_fake_module(monkeypatch, "l05", Adapter)
    adapter = get_adapter("L05", {"index_dir": "/tmp/idx"})
    assert type(adapter) is Adapter  # shadows the built-in L05Adapter
    assert adapter.lane_kw == {"index_dir": "/tmp/idx"}


def test_module_without_adapter_attribute_falls_back(monkeypatch):
    _install_fake_module(monkeypatch, "l05")  # no Adapter attribute
    assert isinstance(get_adapter("L05"), L05Adapter)


def test_absent_module_falls_back_to_reflection_placeholder():
    # no adapters/l99.py exists and none is faked: explicit NOT_SUPPORTED
    adapter = get_adapter("L99")
    supported, _, reason = adapter.support()
    assert not supported
    assert "no harness adapter" in reason


def test_broken_adapter_module_is_explicit_not_supported(monkeypatch):
    real_import_module = importlib_module.import_module

    def fake_import(name, package=None):
        if name.endswith("adapters.l13"):
            raise ModuleNotFoundError(
                "No module named 'definitely_missing_dep'",
                name="definitely_missing_dep")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib_module, "import_module", fake_import)
    adapter = get_adapter("L13")
    supported, _, reason = adapter.support()
    assert not supported
    assert "definitely_missing_dep" in reason


def test_non_laneadapter_class_is_explicit_not_supported(monkeypatch):
    _install_fake_module(monkeypatch, "l14", adapter_cls=object)
    adapter = get_adapter("L14")
    supported, _, reason = adapter.support()
    assert not supported
    assert "not a LaneAdapter" in reason


def test_constructor_failure_is_explicit_not_supported(monkeypatch):
    class Adapter(LaneAdapter):
        lane_id = "L15"

        def __init__(self, lane_kw=None):
            raise RuntimeError("needs a store")

        def build_verifier(self):  # pragma: no cover - never reached
            raise AssertionError

    _install_fake_module(monkeypatch, "l15", Adapter)
    adapter = get_adapter("L15")
    supported, _, reason = adapter.support()
    assert not supported
    assert "construction failed" in reason
    assert "needs a store" in reason
