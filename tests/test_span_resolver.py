"""Tests for app.span_resolver: chunk-to-occurrence resolution + longest-match + protected filter."""

from __future__ import annotations

from app.pseudonymizer import DetectedEntity, EntityOccurrence
from app.span_resolver import (
    apply_longest_match_policy,
    filter_protected,
    resolve_occurrences_in_chunk,
)


def _e(text: str, category: str = "PERSON") -> DetectedEntity:
    return DetectedEntity(text=text, category=category, context="")


def test_resolve_single_match_returns_one_occurrence_with_absolute_offsets():
    chunk = "Dr. Müller leitet das Projekt."
    chunk_start = 100
    entity = _e("Dr. Müller")

    occs = resolve_occurrences_in_chunk(entity, chunk, chunk_start)
    assert len(occs) == 1
    occ = occs[0]
    assert occ.entity is entity
    assert occ.start == chunk_start + 0
    assert occ.end == chunk_start + len("Dr. Müller")


def test_resolve_multiple_matches_emits_one_occurrence_each():
    chunk = "Müller traf Müller am Montag."
    entity = _e("Müller")

    occs = resolve_occurrences_in_chunk(entity, chunk, 0)
    assert len(occs) == 2
    assert occs[0].start == 0
    assert occs[1].start == chunk.index("Müller", 1)


def test_resolve_zero_matches_returns_empty_list():
    chunk = "keine Treffer hier"
    entity = _e("Müller")

    assert resolve_occurrences_in_chunk(entity, chunk, 0) == []


def test_resolve_substring_only_rejection():
    # "AG" inside "Betrag" must not match (word-boundary rule).
    chunk = "Der Betrag war 100 Euro."
    entity = _e("AG", category="ORG_CLIENT")

    assert resolve_occurrences_in_chunk(entity, chunk, 0) == []


def test_resolve_case_insensitive_but_preserves_source_span():
    chunk = "MÜLLER war da, müller auch."
    entity = _e("Müller")

    occs = resolve_occurrences_in_chunk(entity, chunk, 0)
    assert len(occs) == 2
    # Spans reference the chunk positions regardless of case.
    assert chunk[occs[0].start : occs[0].end] == "MÜLLER"
    assert chunk[occs[1].start : occs[1].end] == "müller"


def test_resolve_chunk_offset_arithmetic():
    chunk = "Dr. Müller"
    entity = _e("Dr. Müller")

    occs = resolve_occurrences_in_chunk(entity, chunk, 500)
    assert occs[0].start == 500
    assert occs[0].end == 500 + len("Dr. Müller")


def test_longest_match_policy_drops_substring_entity():
    # "Müller" at (4, 10) is a proper substring of "Dr. Müller" at (0, 10). Drop the shorter.
    long_entity = _e("Dr. Müller")
    short_entity = _e("Müller")
    long_occ = EntityOccurrence(entity=long_entity, start=0, end=10)
    short_occ = EntityOccurrence(entity=short_entity, start=4, end=10)

    survivors = apply_longest_match_policy([long_occ, short_occ])
    assert survivors == [long_occ]


def test_longest_match_policy_keeps_non_overlapping_occurrences():
    e1 = _e("Müller")
    e2 = _e("Schmidt")
    occ1 = EntityOccurrence(entity=e1, start=0, end=6)
    occ2 = EntityOccurrence(entity=e2, start=20, end=27)

    survivors = apply_longest_match_policy([occ1, occ2])
    assert sorted(survivors, key=lambda o: o.start) == sorted([occ1, occ2], key=lambda o: o.start)


def test_longest_match_policy_keeps_disjoint_occurrences_of_same_text():
    # "Müller" at (0, 6) and "Müller" at (30, 36). Both kept: no substring relationship.
    e = _e("Müller")
    occ1 = EntityOccurrence(entity=e, start=0, end=6)
    occ2 = EntityOccurrence(entity=e, start=30, end=36)

    survivors = apply_longest_match_policy([occ1, occ2])
    assert sorted(survivors, key=lambda o: o.start) == sorted([occ1, occ2], key=lambda o: o.start)


def test_filter_protected_drops_occurrence_inside_protected_span():
    e = _e("33", category="BUSINESS_METRIC")
    occ = EntityOccurrence(entity=e, start=6, end=8)  # inside "00:01:33"

    protected = [(0, 8)]
    survivors = filter_protected([occ], protected)
    assert survivors == []


def test_filter_protected_keeps_occurrence_outside_protected_spans():
    e = _e("33", category="BUSINESS_METRIC")
    occ = EntityOccurrence(entity=e, start=50, end=52)

    protected = [(0, 8)]  # timestamp span
    survivors = filter_protected([occ], protected)
    assert survivors == [occ]


def test_filter_protected_empty_protected_list_is_noop():
    e = _e("Müller")
    occ = EntityOccurrence(entity=e, start=5, end=11)

    assert filter_protected([occ], []) == [occ]


def test_longest_match_policy_preserves_partial_overlap():
    """Two spans share characters but neither is a proper substring of the other: both survive.

    Locks in the policy contract that only engulfed spans are dropped, not merely overlapping ones.
    """
    e1 = _e("first", category="PERSON")
    e2 = _e("second", category="PERSON")
    occ1 = EntityOccurrence(entity=e1, start=0, end=6)
    occ2 = EntityOccurrence(entity=e2, start=4, end=10)

    survivors = apply_longest_match_policy([occ1, occ2])
    assert sorted(survivors, key=lambda o: o.start) == [occ1, occ2]


def test_longest_match_policy_empty_input_returns_empty_list():
    assert apply_longest_match_policy([]) == []


def test_longest_match_policy_single_occurrence_returns_singleton():
    e = _e("lonely")
    occ = EntityOccurrence(entity=e, start=0, end=6)
    result = apply_longest_match_policy([occ])
    assert result == [occ]
    # Returns a new list, not the same reference (callers sometimes mutate the result).
    assert result is not [occ]
