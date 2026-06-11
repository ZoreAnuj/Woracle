from __future__ import annotations

import os

import pytest

from woracle.errors import StoreError
from woracle.store import ContentStore


def _writer(text: str):
    def write(payload_dir: str) -> None:
        with open(os.path.join(payload_dir, "data.txt"), "w") as f:
            f.write(text)

    return write


def test_key_is_order_invariant_and_explicit() -> None:
    k1 = ContentStore.key_for(
        stage="ground",
        component="g",
        component_version="1",
        inputs={"a": "x", "b": "y"},
        params={"p": 1},
    )
    k2 = ContentStore.key_for(
        stage="ground",
        component="g",
        component_version="1",
        inputs={"b": "y", "a": "x"},
        params={"p": 1},
    )
    assert k1 == k2 and len(k1) == 64


def test_version_bump_misses_cache() -> None:
    base = dict(stage="s", component="c", inputs={"i": "d"}, params={})
    assert ContentStore.key_for(component_version="1.0", **base) != ContentStore.key_for(
        component_version="1.1", **base
    )


def test_param_change_misses_cache() -> None:
    base = dict(stage="s", component="c", component_version="1", inputs={"i": "d"})
    assert ContentStore.key_for(params={"thr": 0.5}, **base) != ContentStore.key_for(
        params={"thr": 0.6}, **base
    )


def test_unjsonable_key_fields_are_loud() -> None:
    with pytest.raises(StoreError):
        ContentStore.key_for(
            stage="s",
            component="c",
            component_version="1",
            inputs={"i": "d"},
            params={"fn": object()},  # type: ignore[dict-item]
        )


def test_put_get_roundtrip(tmp_path) -> None:
    store = ContentStore(str(tmp_path / "store"))
    key = ContentStore.key_for(stage="s", component="c", component_version="1", inputs={"i": "d"})
    assert not store.has(key)
    entry, hit = store.get_or_create(key, _writer("hello"), meta={"m": 1})
    assert not hit
    with open(os.path.join(entry.payload_dir, "data.txt")) as f:
        assert f.read() == "hello"
    entry2, hit2 = store.get_or_create(key, _writer("SHOULD NOT RUN"))
    assert hit2
    with open(os.path.join(entry2.payload_dir, "data.txt")) as f:
        assert f.read() == "hello"  # second writer never executed


def test_failed_writer_leaves_no_entry(tmp_path) -> None:
    store = ContentStore(str(tmp_path / "store"))
    key = ContentStore.key_for(stage="s", component="c", component_version="1", inputs={"i": "e"})

    def bad_writer(payload_dir: str) -> None:
        with open(os.path.join(payload_dir, "partial.txt"), "w") as f:
            f.write("partial")
        raise RuntimeError("producer crashed")

    with pytest.raises(RuntimeError):
        store.put(key, bad_writer)
    assert not store.has(key)  # no half-written entry masquerading as a hit


def test_malformed_key_rejected(tmp_path) -> None:
    store = ContentStore(str(tmp_path / "store"))
    with pytest.raises(StoreError):
        store.has("not-a-sha")
