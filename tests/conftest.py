"""Shared pytest configuration.

* `slow` tests (real models, minutes of compute) are skipped unless
  `--run-slow` is passed or `LDW_RUN_SLOW=1` is set.
* Fixtures for offline testing live here (keys, specs, fake targets).
"""

from __future__ import annotations

import os

import pytest


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False, help="run tests marked `slow`")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow") or os.environ.get("LDW_RUN_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="slow test: pass --run-slow or set LDW_RUN_SLOW=1")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
