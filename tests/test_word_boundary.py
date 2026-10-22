"""Tests for word boundary filtering (is_word_boundary_match)."""

from __future__ import annotations

from app.pseudonymizer import is_word_boundary_match


# ---------------------------------------------------------------------------
# Word boundary filter
# ---------------------------------------------------------------------------

class TestWordBoundaryMatch:

    def test_standalone_word_matches(self):
        assert is_word_boundary_match("Die AG hat entschieden.", "AG")

    def test_partial_word_rejected(self):
        assert not is_word_boundary_match("Der Betrag ist hoch.", "AG")

    def test_partial_word_rejected_gmbh(self):
        assert not is_word_boundary_match("FirmaGmbH ist bekannt.", "GmbH")

    def test_word_at_end_of_text(self):
        assert is_word_boundary_match("Firma GmbH", "GmbH")

    def test_word_at_start_of_text(self):
        assert is_word_boundary_match("GmbH ist registriert.", "GmbH")

    def test_case_insensitive(self):
        assert is_word_boundary_match("Die firma ag hat entschieden.", "AG")

    def test_multi_word_entity(self):
        assert is_word_boundary_match("Dr. Thomas Mueller leitet das Projekt.", "Dr. Thomas Mueller")

    def test_entity_with_special_chars(self):
        assert is_word_boundary_match("Kontakt: m.weber@firma.de", "m.weber@firma.de")

    def test_not_present_at_all(self):
        assert not is_word_boundary_match("Kein Treffer hier.", "XYZ")
