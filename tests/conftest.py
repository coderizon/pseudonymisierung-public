"""Shared fixtures for tests."""

from __future__ import annotations

import os
import tempfile

# Set DATA_DIR before any app module is imported (config.py runs os.makedirs at import time)
_test_data_dir = tempfile.mkdtemp()
os.environ.setdefault("DATA_DIR", _test_data_dir)

import pytest


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Redirect DB and encryption key to a temp directory for each test."""
    db_path = str(tmp_path / "mappings.db")
    key_path = str(tmp_path / ".encryption_key")

    # Patch in both config and mapping_store (the latter imports DB_PATH at module level)
    monkeypatch.setattr("app.config.DB_PATH", db_path)
    monkeypatch.setattr("app.config.ENCRYPTION_KEY_PATH", key_path)
    monkeypatch.setattr("app.mapping_store.DB_PATH", db_path)
    monkeypatch.setattr("app.mapping_store.ENCRYPTION_KEY_PATH", key_path)

    # Reset module-level singletons so each test gets a fresh key/db
    import app.mapping_store as ms
    ms._fernet = None
    ms._hmac_key = None
