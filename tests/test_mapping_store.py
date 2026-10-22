"""Tests for the mapping store, including occurrence variant tracking."""

from __future__ import annotations

from app.mapping_store import MappingStore, _occurrence_hmac


class TestOccurrenceVariants:
    def test_save_and_retrieve_occurrence(self):
        store = MappingStore("proj1")
        store.save_mapping("Max Müller", "PERSON_A", "PERSON")
        store.save_occurrence("Max Müller", "Max Müller", "report.pdf", "page 1")

        occurrences = store.get_occurrences("PERSON_A")
        assert len(occurrences) == 1
        assert occurrences[0]["casing"] == "Max Müller"
        assert occurrences[0]["document_filename"] == "report.pdf"
        assert occurrences[0]["block_source"] == "page 1"
        store.close()

    def test_dedup_via_hmac(self):
        store = MappingStore("proj1")
        store.save_mapping("Max Müller", "PERSON_A", "PERSON")
        store.save_occurrence("Max Müller", "Max Müller", "report.pdf", "page 1")
        store.save_occurrence("Max Müller", "Max Müller", "report.pdf", "page 1")

        occurrences = store.get_occurrences("PERSON_A")
        assert len(occurrences) == 1
        store.close()

    def test_different_casings_stored_separately(self):
        store = MappingStore("proj1")
        store.save_mapping("Max Müller", "PERSON_A", "PERSON")
        store.save_occurrence("Max Müller", "Max Müller", "report.pdf", "page 1")
        store.save_occurrence("Max Müller", "MAX MÜLLER", "report.pdf", "page 2")
        store.save_occurrence("Max Müller", "max müller", "summary.docx", "paragraph 3")

        occurrences = store.get_occurrences("PERSON_A")
        assert len(occurrences) == 3
        casings = {o["casing"] for o in occurrences}
        assert casings == {"Max Müller", "MAX MÜLLER", "max müller"}
        store.close()

    def test_same_casing_different_blocks(self):
        store = MappingStore("proj1")
        store.save_mapping("Acme GmbH", "ORG_CLIENT_A", "ORG_CLIENT")
        store.save_occurrence("Acme GmbH", "Acme GmbH", "report.pdf", "page 1")
        store.save_occurrence("Acme GmbH", "Acme GmbH", "report.pdf", "page 5")

        occurrences = store.get_occurrences("ORG_CLIENT_A")
        assert len(occurrences) == 2
        store.close()

    def test_hmac_deterministic(self):
        h1 = _occurrence_hmac("Max", "file.pdf", "page 1")
        h2 = _occurrence_hmac("Max", "file.pdf", "page 1")
        assert h1 == h2

    def test_hmac_differs_for_different_input(self):
        h1 = _occurrence_hmac("Max", "file.pdf", "page 1")
        h2 = _occurrence_hmac("MAX", "file.pdf", "page 1")
        assert h1 != h2

    def test_occurrence_fields_encrypted_in_db(self):
        store = MappingStore("proj1")
        store.save_mapping("Max Müller", "PERSON_A", "PERSON")
        store.save_occurrence("Max Müller", "Max Müller", "report.pdf", "page 1")

        # Read raw DB values
        row = store._db.execute(
            "SELECT casing_encrypted, filename_encrypted, block_source_encrypted "
            "FROM occurrence_variants"
        ).fetchone()
        assert row is not None
        # Encrypted values must not match plaintext
        assert row[0] != "Max Müller"
        assert row[1] != "report.pdf"
        assert row[2] != "page 1"
        store.close()

    def test_save_mapping_with_occurrence(self):
        store = MappingStore("proj1")
        store.save_mapping(
            "Max Müller", "PERSON_A", "PERSON",
            document_filename="report.pdf", block_source="page 1",
        )

        occurrences = store.get_occurrences("PERSON_A")
        assert len(occurrences) == 1
        assert occurrences[0]["casing"] == "Max Müller"
        store.close()

    def test_save_mapping_existing_still_records_occurrence(self):
        """Even when the canonical mapping already exists, a new occurrence is recorded."""
        store = MappingStore("proj1")
        store.save_mapping("Max Müller", "PERSON_A", "PERSON")
        # Second call with occurrence context, mapping already cached
        store.save_mapping(
            "Max Müller", "PERSON_A", "PERSON",
            document_filename="other.pdf", block_source="page 3",
        )

        occurrences = store.get_occurrences("PERSON_A")
        assert len(occurrences) == 1
        assert occurrences[0]["document_filename"] == "other.pdf"
        store.close()

    def test_no_occurrence_without_filename(self):
        """If document_filename is empty, no occurrence is recorded via save_mapping."""
        store = MappingStore("proj1")
        store.save_mapping("Max Müller", "PERSON_A", "PERSON")

        occurrences = store.get_occurrences("PERSON_A")
        assert len(occurrences) == 0
        store.close()

    def test_occurrences_scoped_to_project(self):
        store1 = MappingStore("proj1")
        store1.save_mapping("Max Müller", "PERSON_A", "PERSON")
        store1.save_occurrence("Max Müller", "Max Müller", "report.pdf", "page 1")
        store1.close()

        store2 = MappingStore("proj2")
        store2.save_mapping("Max Müller", "PERSON_A", "PERSON")
        store2.save_occurrence("Max Müller", "Max Müller", "other.pdf", "page 2")

        occurrences = store2.get_occurrences("PERSON_A")
        assert len(occurrences) == 1
        assert occurrences[0]["document_filename"] == "other.pdf"
        store2.close()
