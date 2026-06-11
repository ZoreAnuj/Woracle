"""Shared fixtures. Unit tests are hermetic: no network (pytest-socket via
addopts), no GPU, no checkpoints — blobworld provides ground truth."""

from __future__ import annotations

import os

import pytest

from woracle.contracts import TaskSpec
from woracle.testing.blobworld import blob_spec, write_dataset


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_SLOW") != "1":
        skip = pytest.mark.skip(reason="slow test: set RUN_SLOW=1")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def spec() -> TaskSpec:
    return blob_spec()


@pytest.fixture(scope="session")
def blob_dataset(tmp_path_factory) -> str:
    """One shared blobworld dataset per test session (deterministic, seed=0)."""
    root = tmp_path_factory.mktemp("blobds")
    write_dataset(
        str(root),
        kinds={"success": 1, "fail_miss": 1, "fail_drop": 1, "vanish": 1, "random": 1},
        seed=0,
    )
    return str(root)
