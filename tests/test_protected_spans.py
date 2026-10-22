"""Tests for app.protected_spans: timestamp regex + merged span list."""

from __future__ import annotations

import re

from app.protected_spans import (
    TIMESTAMP_PATTERN,
    PROTECTED_PATTERNS,
    build_protected_spans,
)


def test_timestamp_pattern_matches_hhmmss():
    matches = [m.group() for m in TIMESTAMP_PATTERN.finditer("Beginn um 00:01:33, Ende um 01:05:30.")]
    assert matches == ["00:01:33", "01:05:30"]


def test_timestamp_pattern_matches_mmss():
    matches = [m.group() for m in TIMESTAMP_PATTERN.finditer("Laufzeit 04:30 bis 12:45.")]
    assert matches == ["04:30", "12:45"]


def test_timestamp_pattern_does_not_match_bare_numbers():
    matches = [m.group() for m in TIMESTAMP_PATTERN.finditer("Betrag 33 Euro, Zahl 500000.")]
    assert matches == []


def test_timestamp_pattern_does_not_match_partial_colons():
    matches = [m.group() for m in TIMESTAMP_PATTERN.finditer("Ratio 3:4, Versionsnummer 1:5.")]
    assert matches == []


def test_build_protected_spans_returns_sorted_tuples():
    text = "Um 00:01:33 und spaeter 00:00:05 dann 01:05:30."
    spans = build_protected_spans(text, PROTECTED_PATTERNS)
    assert all(isinstance(s, tuple) and len(s) == 2 for s in spans)
    assert all(s[0] <= s[1] for s in spans)
    assert spans == sorted(spans)
    assert len(spans) == 3


def test_build_protected_spans_merges_overlapping():
    # Two adjacent matches (2,4) and (4,6) — they touch. The merge collapses them into (2,6).
    pattern_two = re.compile(r"AB")
    text2 = "xxABAByy"
    spans2 = build_protected_spans(text2, [pattern_two])
    assert spans2 == [(2, 6)]


def test_build_protected_spans_empty_text_returns_empty_list():
    assert build_protected_spans("", PROTECTED_PATTERNS) == []


def test_build_protected_spans_no_match_returns_empty_list():
    assert build_protected_spans("no timestamps here at all", PROTECTED_PATTERNS) == []


def test_timestamp_pattern_over_protects_two_digit_ratios():
    """Intentional, documented limitation: two-digit 'ratios' that share the same
    shape as timestamps are matched and therefore protected. This is a conservative
    bias: the pattern protects legitimate meeting timestamps at the cost of possibly
    over-protecting e.g. page references ('Seite 5:23') or ratios ('10:20'). See
    BUG-017 for the critical use case. If this over-protection ever costs us in
    the field, tighten the pattern here — this test will then fail as a signal.
    """
    # Ratio shape: two-digit colon two-digit — matches.
    assert [m.group() for m in TIMESTAMP_PATTERN.finditer("Anteil 10:20 des Marktes.")] == ["10:20"]
    # Page reference shape (single-digit first part, two-digit second part) — also matches.
    assert [m.group() for m in TIMESTAMP_PATTERN.finditer("siehe Seite 5:23.")] == ["5:23"]
