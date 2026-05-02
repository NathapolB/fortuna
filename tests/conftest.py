"""pytest configuration — shared fixtures and patches."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def patch_icloud_check(monkeypatch):
    """Disable iCloud path check during tests — test files live in tmp dirs."""
    monkeypatch.setattr("fortuna.config.check_not_icloud", lambda: None)
    monkeypatch.setattr("fortuna.store.check_not_icloud", lambda: None)
