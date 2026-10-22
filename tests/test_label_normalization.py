"""Tests for app.label_normalization: strict bracket stripping."""

from __future__ import annotations

from app.label_normalization import strict_strip_surrounding_brackets


def test_canonical_shape_is_stripped_and_well_formed():
    stripped, ok = strict_strip_surrounding_brackets("[Software-Gigant]")
    assert stripped == "Software-Gigant"
    assert ok is True


def test_bare_label_is_returned_unchanged_and_well_formed():
    stripped, ok = strict_strip_surrounding_brackets("Software-Gigant")
    assert stripped == "Software-Gigant"
    assert ok is True


def test_asymmetric_leading_bracket_is_malformed():
    stripped, ok = strict_strip_surrounding_brackets("[Software-Gigant")
    assert stripped == "[Software-Gigant"
    assert ok is False


def test_asymmetric_trailing_bracket_is_malformed():
    stripped, ok = strict_strip_surrounding_brackets("Software-Gigant]")
    assert stripped == "Software-Gigant]"
    assert ok is False


def test_nested_brackets_are_malformed():
    stripped, ok = strict_strip_surrounding_brackets("[foo [bar]]")
    assert stripped == "[foo [bar]]"
    assert ok is False


def test_multiple_bracket_pairs_are_malformed():
    stripped, ok = strict_strip_surrounding_brackets("[foo]-[bar]")
    assert stripped == "[foo]-[bar]"
    assert ok is False


def test_empty_string_is_well_formed_and_unchanged():
    stripped, ok = strict_strip_surrounding_brackets("")
    assert stripped == ""
    assert ok is True


def test_empty_bracket_pair_is_stripped_to_empty():
    stripped, ok = strict_strip_surrounding_brackets("[]")
    assert stripped == ""
    assert ok is True


def test_whitespace_only_label_is_well_formed():
    stripped, ok = strict_strip_surrounding_brackets("   ")
    assert stripped == "   "
    assert ok is True
