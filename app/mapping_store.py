"""Encrypted mapping store using SQLite + Fernet symmetric encryption."""

from __future__ import annotations

import base64
import hashlib
import hmac as hmac_mod
import json
import os
import sqlite3
from pathlib import Path

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from app.config import DB_PATH, ENCRYPTION_KEY_PATH


def _get_or_create_key() -> bytes:
    """Load or generate the Fernet encryption key."""
    key_path = Path(ENCRYPTION_KEY_PATH)
    if key_path.exists():
        return key_path.read_bytes().strip()
    key = Fernet.generate_key()
    key_path.write_bytes(key)
    os.chmod(str(key_path), 0o600)
    return key


_fernet: Fernet | None = None
_hmac_key: bytes | None = None


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        _fernet = Fernet(_get_or_create_key())
    return _fernet


def _get_hmac_key() -> bytes:
    """Derive a separate HMAC key from the Fernet key via HKDF."""
    global _hmac_key
    if _hmac_key is None:
        raw_key = base64.urlsafe_b64decode(_get_or_create_key())
        _hmac_key = HKDF(
            algorithm=SHA256(),
            length=32,
            salt=None,
            info=b"occurrence-hmac-v1",
        ).derive(raw_key)
    return _hmac_key


def _occurrence_hmac(casing: str, filename: str, block_source: str) -> str:
    """Deterministic digest for deduplication. Reveals equality, not content."""
    payload = f"{casing}\x00{filename}\x00{block_source}".encode("utf-8")
    return hmac_mod.new(_get_hmac_key(), payload, hashlib.sha256).hexdigest()


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mappings (
            project_id TEXT NOT NULL,
            category TEXT NOT NULL,
            original_encrypted TEXT NOT NULL,
            pseudonym TEXT NOT NULL,
            context TEXT DEFAULT '',
            UNIQUE(project_id, pseudonym)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS occurrence_variants (
            mapping_rowid INTEGER NOT NULL,
            casing_encrypted TEXT NOT NULL,
            filename_encrypted TEXT NOT NULL,
            block_source_encrypted TEXT NOT NULL,
            occurrence_hmac TEXT NOT NULL,
            UNIQUE(mapping_rowid, occurrence_hmac)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_project
        ON mappings(project_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_occurrence_mapping
        ON occurrence_variants(mapping_rowid)
    """)
    conn.commit()
    return conn


def _encrypt(text: str) -> str:
    return _get_fernet().encrypt(text.encode("utf-8")).decode("ascii")


def _decrypt(token: str) -> str:
    return _get_fernet().decrypt(token.encode("ascii")).decode("utf-8")


class MappingStore:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self._db = _get_db()
        self._cache: dict[str, str] = {}
        self._reverse_cache: dict[str, str] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        cursor = self._db.execute(
            "SELECT original_encrypted, pseudonym FROM mappings WHERE project_id = ?",
            (self.project_id,),
        )
        for enc_original, pseudonym in cursor.fetchall():
            original = _decrypt(enc_original)
            self._cache[original.lower()] = pseudonym
            self._reverse_cache[pseudonym] = original

    def get_pseudonym(self, original: str) -> str | None:
        return self._cache.get(original.lower())

    def get_original(self, pseudonym: str) -> str | None:
        return self._reverse_cache.get(pseudonym)

    def save_mapping(
        self,
        original: str,
        pseudonym: str,
        category: str,
        context: str = "",
        document_filename: str = "",
        block_source: str = "",
    ) -> None:
        key = original.lower()
        if key not in self._cache:
            enc_original = _encrypt(original)
            self._db.execute(
                "INSERT OR IGNORE INTO mappings (project_id, category, original_encrypted, pseudonym, context) VALUES (?, ?, ?, ?, ?)",
                (self.project_id, category, enc_original, pseudonym, context),
            )
            self._db.commit()
            self._cache[key] = pseudonym
            self._reverse_cache[pseudonym] = original
        if document_filename:
            self.save_occurrence(original, original, document_filename, block_source)

    def save_occurrence(
        self,
        original: str,
        casing: str,
        document_filename: str,
        block_source: str,
    ) -> None:
        """Record a casing/location variant for an existing mapping."""
        pseudonym = self._cache.get(original.lower())
        if pseudonym is None:
            return
        digest = _occurrence_hmac(casing, document_filename, block_source)
        row = self._db.execute(
            "SELECT rowid FROM mappings WHERE project_id = ? AND pseudonym = ?",
            (self.project_id, pseudonym),
        ).fetchone()
        if row is None:
            return
        mapping_rowid = row[0]
        self._db.execute(
            "INSERT OR IGNORE INTO occurrence_variants "
            "(mapping_rowid, casing_encrypted, filename_encrypted, block_source_encrypted, occurrence_hmac) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                mapping_rowid,
                _encrypt(casing),
                _encrypt(document_filename),
                _encrypt(block_source),
                digest,
            ),
        )
        self._db.commit()

    def get_occurrences(self, pseudonym: str) -> list[dict]:
        """Return all occurrence variants for a given pseudonym."""
        cursor = self._db.execute(
            "SELECT ov.casing_encrypted, ov.filename_encrypted, ov.block_source_encrypted "
            "FROM occurrence_variants ov "
            "JOIN mappings m ON ov.mapping_rowid = m.rowid "
            "WHERE m.project_id = ? AND m.pseudonym = ?",
            (self.project_id, pseudonym),
        )
        results = []
        for enc_casing, enc_filename, enc_block in cursor.fetchall():
            results.append({
                "casing": _decrypt(enc_casing),
                "document_filename": _decrypt(enc_filename),
                "block_source": _decrypt(enc_block),
            })
        return results

    def get_all_mappings(self) -> list[dict]:
        cursor = self._db.execute(
            "SELECT category, original_encrypted, pseudonym, context FROM mappings WHERE project_id = ?",
            (self.project_id,),
        )
        results = []
        for category, enc_original, pseudonym, context in cursor.fetchall():
            results.append({
                "category": category,
                "original": _decrypt(enc_original),
                "pseudonym": pseudonym,
                "context": context,
            })
        return results

    def get_next_pseudonym(self, category: str) -> str:
        """Generate the next pseudonym for a category, e.g. PERSON_A, PERSON_B, ..."""
        existing = [
            p for p in self._cache.values()
            if p.startswith(category.upper() + "_")
        ]
        index = len(existing)
        if index < 26:
            suffix = chr(ord("A") + index)
        else:
            suffix = str(index + 1)
        return f"{category.upper()}_{suffix}"

    def close(self) -> None:
        self._db.close()
