"""Tests for pseudonymize_by_span: span-addressed replacement in reverse order."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.pseudonymizer import (
    DetectedEntity,
    EntityOccurrence,
    pseudonymize_by_span,
)


class _StubStore:
    """Minimal MappingStore stand-in for unit tests. Signature mirrors MappingStore.save_mapping."""

    def __init__(self):
        self._map: dict[str, str] = {}
        self._counter: dict[str, int] = {}

    def get_pseudonym(self, original: str) -> str | None:
        return self._map.get(original.lower())

    def get_next_pseudonym(self, prefix: str) -> str:
        self._counter[prefix] = self._counter.get(prefix, 0) + 1
        return f"{prefix}_{chr(ord('A') + self._counter[prefix] - 1)}"

    def save_mapping(
        self,
        original: str,
        pseudonym: str,
        category: str,
        context: str = "",
        document_filename: str = "",
        block_source: str = "",
    ) -> None:
        self._map[original.lower()] = pseudonym


def _mk_config():
    """Config stub: returns a CategoryDef-like object with 'pseudonymize' action and an uppercase prefix."""
    cat_def = MagicMock()
    cat_def.action = "pseudonymize"
    cat_def.pseudonym_prefix = None
    config = MagicMock()
    config.get = MagicMock(return_value=cat_def)
    return config


def test_pseudonymize_by_span_single_occurrence_replaces_at_exact_span():
    text = "Dr. Müller leitet."
    entity = DetectedEntity(text="Dr. Müller", category="PERSON", context="")
    occ = EntityOccurrence(entity=entity, start=0, end=10)

    store = _StubStore()
    result = pseudonymize_by_span(text, [occ], store, _mk_config())
    assert result == "[PERSON_A] leitet."


def test_pseudonymize_by_span_multiple_occurrences_same_pseudonym():
    text = "Müller traf Müller."
    entity = DetectedEntity(text="Müller", category="PERSON", context="")
    occ1 = EntityOccurrence(entity=entity, start=0, end=6)
    occ2 = EntityOccurrence(entity=entity, start=12, end=18)

    store = _StubStore()
    result = pseudonymize_by_span(text, [occ1, occ2], store, _mk_config())
    assert result == "[PERSON_A] traf [PERSON_A]."


def test_pseudonymize_by_span_reverse_order_preserves_earlier_offsets():
    # If we applied substitutions in forward order, the first replacement would shift
    # the second's offsets and corrupt the output. Reverse order is the correctness test.
    text = "ab Müller cd Müller ef"
    entity = DetectedEntity(text="Müller", category="PERSON", context="")
    occ1 = EntityOccurrence(entity=entity, start=3, end=9)
    occ2 = EntityOccurrence(entity=entity, start=13, end=19)

    store = _StubStore()
    result = pseudonymize_by_span(text, [occ1, occ2], store, _mk_config())
    assert result == "ab [PERSON_A] cd [PERSON_A] ef"


def test_pseudonymize_by_span_different_entities_get_distinct_pseudonyms():
    text = "Müller und Schmidt."
    e1 = DetectedEntity(text="Müller", category="PERSON", context="")
    e2 = DetectedEntity(text="Schmidt", category="PERSON", context="")
    occ1 = EntityOccurrence(entity=e1, start=0, end=6)
    occ2 = EntityOccurrence(entity=e2, start=11, end=18)

    store = _StubStore()
    result = pseudonymize_by_span(text, [occ1, occ2], store, _mk_config())
    assert result == "[PERSON_A] und [PERSON_B]."


def test_pseudonymize_by_span_klasse1_description_used_when_provided():
    text = "AeroTech Solutions GmbH ist Mandant."
    entity = DetectedEntity(text="AeroTech Solutions GmbH", category="ORG_CLIENT", context="")
    occ = EntityOccurrence(entity=entity, start=0, end=23)

    store = _StubStore()
    k1 = {"aerotech solutions gmbh": "mittelstaendisches IT-Unternehmen"}
    result = pseudonymize_by_span(text, [occ], store, _mk_config(), klasse1_descriptions=k1)
    assert result == "[mittelstaendisches IT-Unternehmen] ist Mandant."


def test_pseudonymize_by_span_suppress_action_replaces_with_marker():
    text = "Kontakt: +49 30 12345678 erreichbar."
    entity = DetectedEntity(text="+49 30 12345678", category="CONTACT", context="")
    occ = EntityOccurrence(entity=entity, start=9, end=24)

    cat_def = MagicMock()
    cat_def.action = "suppress"
    cat_def.pseudonym_prefix = None
    config = MagicMock()
    config.get = MagicMock(return_value=cat_def)

    store = _StubStore()
    result = pseudonymize_by_span(text, [occ], store, config)
    assert result == "Kontakt: [KONTAKTDATEN ENTFERNT] erreichbar."


def test_pseudonymize_by_span_no_occurrences_returns_text_unchanged():
    text = "nothing to do here."
    store = _StubStore()
    assert pseudonymize_by_span(text, [], store, _mk_config()) == text


def test_pseudonymize_by_span_same_span_duplicate_first_wins():
    """Two distinct entities (different category) at the same (start, end) — the first
    occurrence wins; the second is silently dropped. This locks in the policy flagged
    by apply_longest_match_policy's docstring: callers rely on span-addressed
    substitution not operating on already-substituted text.
    """
    text = "Müller ist da."
    e_person = DetectedEntity(text="Müller", category="PERSON", context="")
    e_org = DetectedEntity(text="Müller", category="ORG_CLIENT", context="")
    occ_first = EntityOccurrence(entity=e_person, start=0, end=6)
    occ_second = EntityOccurrence(entity=e_org, start=0, end=6)  # identical span

    store = _StubStore()
    result = pseudonymize_by_span(text, [occ_first, occ_second], store, _mk_config())
    # The PERSON (first in the list) wins; ORG_CLIENT is silently dropped.
    assert result == "[PERSON_A] ist da."
    # Only the winning entity's mapping was recorded — ORG_CLIENT_* was never allocated.
    assert store._counter.get("PERSON") == 1
    assert "ORG_CLIENT" not in store._counter
