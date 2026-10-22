"""Tests for Klasse 1 type-preserving generalization."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.klasse1_transformer import (
    KLASSE1_GENERALIZE_CATEGORIES,
    compute_klasse1_descriptions,
    _fallback_description,
)
from app.mapping_store import MappingStore
from app.pseudonymizer import DetectedEntity, pseudonymize_by_span
from app.span_resolver import resolve_occurrences_in_chunk


class TestComputeKlasse1Descriptions:
    @pytest.mark.asyncio
    async def test_returns_descriptions_for_all_entities(self):
        entities = [
            {"text": "AeroTech Solutions GmbH", "category": "ORG_CLIENT", "context": "AeroTech Solutions GmbH ist ein IT-Unternehmen"},
            {"text": "Freiburg", "category": "LOCATION", "context": "Standort Freiburg"},
            {"text": "Abteilung Digital Innovation", "category": "ORG_UNIT", "context": "Die Abteilung Digital Innovation"},
        ]

        mock_response = {
            "descriptions": [
                {"original": "AeroTech Solutions GmbH", "description": "mittelstaendisches IT-Unternehmen"},
                {"original": "Freiburg", "description": "Standort Sueddeutschland"},
                {"original": "Abteilung Digital Innovation", "description": "Fachbereich Digitalisierung"},
            ]
        }

        with patch("app.klasse1_transformer.chat_json", new_callable=AsyncMock, return_value=mock_response):
            result = await compute_klasse1_descriptions(entities)

        assert result["aerotech solutions gmbh"] == "mittelstaendisches IT-Unternehmen"
        assert result["freiburg"] == "Standort Sueddeutschland"
        assert result["abteilung digital innovation"] == "Fachbereich Digitalisierung"

    @pytest.mark.asyncio
    async def test_ignores_non_klasse1_categories(self):
        entities = [
            {"text": "Dr. Schmidt", "category": "PERSON", "context": "Dr. Schmidt leitet"},
            {"text": "Freiburg", "category": "LOCATION", "context": "in Freiburg"},
        ]

        mock_response = {
            "descriptions": [
                {"original": "Freiburg", "description": "Standort Sueddeutschland"},
            ]
        }

        with patch("app.klasse1_transformer.chat_json", new_callable=AsyncMock, return_value=mock_response):
            result = await compute_klasse1_descriptions(entities)

        assert "dr. schmidt" not in result
        assert result["freiburg"] == "Standort Sueddeutschland"

    @pytest.mark.asyncio
    async def test_deduplicates_entities(self):
        entities = [
            {"text": "Freiburg", "category": "LOCATION", "context": "in Freiburg"},
            {"text": "freiburg", "category": "LOCATION", "context": "nach freiburg"},
        ]

        mock_response = {
            "descriptions": [
                {"original": "Freiburg", "description": "Standort Sueddeutschland"},
            ]
        }

        with patch("app.klasse1_transformer.chat_json", new_callable=AsyncMock, return_value=mock_response) as mock_chat:
            result = await compute_klasse1_descriptions(entities)
            # Should only send one entity to LLM
            call_args = mock_chat.call_args[0][0]
            user_msg = call_args[1]["content"]
            assert user_msg.count("LOCATION") == 1

        assert result["freiburg"] == "Standort Sueddeutschland"

    @pytest.mark.asyncio
    async def test_empty_entities_returns_empty(self):
        result = await compute_klasse1_descriptions([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_no_klasse1_entities_returns_empty(self):
        entities = [
            {"text": "Dr. Schmidt", "category": "PERSON", "context": ""},
        ]
        result = await compute_klasse1_descriptions(entities)
        assert result == {}

    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self):
        entities = [
            {"text": "AeroTech GmbH", "category": "ORG_CLIENT", "context": ""},
            {"text": "Freiburg", "category": "LOCATION", "context": ""},
        ]

        with patch("app.klasse1_transformer.chat_json", new_callable=AsyncMock, side_effect=Exception("LLM down")):
            result = await compute_klasse1_descriptions(entities)

        assert result["aerotech gmbh"] == "Unternehmen"
        assert result["freiburg"] == "Standort"

    @pytest.mark.asyncio
    async def test_fallback_for_missing_descriptions(self):
        entities = [
            {"text": "AeroTech GmbH", "category": "ORG_CLIENT", "context": ""},
            {"text": "Freiburg", "category": "LOCATION", "context": ""},
        ]

        # LLM only returns one description
        mock_response = {
            "descriptions": [
                {"original": "AeroTech GmbH", "description": "IT-Unternehmen"},
            ]
        }

        with patch("app.klasse1_transformer.chat_json", new_callable=AsyncMock, return_value=mock_response):
            result = await compute_klasse1_descriptions(entities)

        assert result["aerotech gmbh"] == "IT-Unternehmen"
        assert result["freiburg"] == "Standort"  # fallback


class TestFallbackDescription:
    def test_known_categories(self):
        assert _fallback_description("ORG_CLIENT") == "Unternehmen"
        assert _fallback_description("ORG_UNIT") == "Organisationseinheit"
        assert _fallback_description("LOCATION") == "Standort"

    def test_unknown_category(self):
        assert _fallback_description("UNKNOWN") == "UNKNOWN"


class TestPseudonymizeBySpanWithKlasse1:
    def test_uses_description_instead_of_token(self):
        text = "AeroTech Solutions GmbH hat in Freiburg ein Buero."
        entities = [
            DetectedEntity(text="AeroTech Solutions GmbH", category="ORG_CLIENT", context=""),
            DetectedEntity(text="Freiburg", category="LOCATION", context=""),
        ]
        descriptions = {
            "aerotech solutions gmbh": "mittelstaendisches IT-Unternehmen",
            "freiburg": "Standort Sueddeutschland",
        }
        store = MappingStore("test-k1")
        occurrences = []
        for entity in entities:
            occurrences.extend(resolve_occurrences_in_chunk(entity, text, chunk_start=0))
        result = pseudonymize_by_span(text, occurrences, store, klasse1_descriptions=descriptions)
        store.close()

        assert "[mittelstaendisches IT-Unternehmen]" in result
        assert "[Standort Sueddeutschland]" in result
        assert "AeroTech" not in result
        assert "Freiburg" not in result

    def test_person_still_gets_token(self):
        text = "Dr. Schmidt arbeitet bei AeroTech GmbH."
        entities = [
            DetectedEntity(text="Dr. Schmidt", category="PERSON", context=""),
            DetectedEntity(text="AeroTech GmbH", category="ORG_CLIENT", context=""),
        ]
        descriptions = {
            "aerotech gmbh": "IT-Unternehmen",
        }
        store = MappingStore("test-k1-person")
        occurrences = []
        for entity in entities:
            occurrences.extend(resolve_occurrences_in_chunk(entity, text, chunk_start=0))
        result = pseudonymize_by_span(text, occurrences, store, klasse1_descriptions=descriptions)
        store.close()

        assert "[PERSON_A]" in result
        assert "[IT-Unternehmen]" in result

    def test_consistent_mapping_across_occurrences(self):
        text = "AeroTech GmbH expandiert. AeroTech GmbH stellt ein."
        entities = [
            DetectedEntity(text="AeroTech GmbH", category="ORG_CLIENT", context=""),
        ]
        descriptions = {
            "aerotech gmbh": "IT-Unternehmen",
        }
        store = MappingStore("test-k1-consistent")
        occurrences = []
        for entity in entities:
            occurrences.extend(resolve_occurrences_in_chunk(entity, text, chunk_start=0))
        result = pseudonymize_by_span(text, occurrences, store, klasse1_descriptions=descriptions)
        store.close()

        assert result.count("[IT-Unternehmen]") == 2

    def test_without_descriptions_falls_back_to_token(self):
        text = "AeroTech GmbH expandiert."
        entities = [
            DetectedEntity(text="AeroTech GmbH", category="ORG_CLIENT", context=""),
        ]
        store = MappingStore("test-k1-fallback")
        occurrences = []
        for entity in entities:
            occurrences.extend(resolve_occurrences_in_chunk(entity, text, chunk_start=0))
        result = pseudonymize_by_span(text, occurrences, store)
        store.close()

        assert "[ORG_CLIENT_A]" in result

    def test_description_stored_in_mapping(self):
        text = "AeroTech GmbH expandiert."
        entities = [
            DetectedEntity(text="AeroTech GmbH", category="ORG_CLIENT", context="ctx"),
        ]
        descriptions = {"aerotech gmbh": "IT-Unternehmen"}
        store = MappingStore("test-k1-store")
        occurrences = []
        for entity in entities:
            occurrences.extend(resolve_occurrences_in_chunk(entity, text, chunk_start=0))
        pseudonymize_by_span(text, occurrences, store, klasse1_descriptions=descriptions)

        # Verify the mapping stores the description, not a token
        assert store.get_pseudonym("AeroTech GmbH") == "IT-Unternehmen"
        store.close()


# ---------------------------------------------------------------------------
# BUG-019: strict-strip + self-repair loop for Klasse 1 LLM output
# ---------------------------------------------------------------------------

import logging as _logging_for_bug_019


@pytest.mark.asyncio
async def test_klasse1_strips_canonical_bracketed_llm_output():
    entities = [
        {"text": "ACME GmbH", "category": "ORG_CLIENT", "context": ""},
    ]
    with patch(
        "app.klasse1_transformer.chat_json",
        new=AsyncMock(return_value={
            "descriptions": [
                {"original": "ACME GmbH", "description": "[mittelstaendisches Unternehmen]"},
            ],
        }),
    ):
        result = await compute_klasse1_descriptions(entities)

    assert result == {"acme gmbh": "mittelstaendisches Unternehmen"}


@pytest.mark.asyncio
async def test_klasse1_triggers_repair_on_malformed_bracket_shape():
    entities = [
        {"text": "ACME GmbH", "category": "ORG_CLIENT", "context": ""},
    ]
    call_count = {"n": 0}

    async def _fake_chat_json(_messages):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"descriptions": [
                {"original": "ACME GmbH", "description": "[Unternehmen-[branche]]"},
            ]}
        return {"description": "mittelstaendisches Unternehmen"}

    with patch("app.klasse1_transformer.chat_json", new=_fake_chat_json):
        result = await compute_klasse1_descriptions(entities)

    assert result == {"acme gmbh": "mittelstaendisches Unternehmen"}
    assert call_count["n"] == 2


@pytest.mark.asyncio
async def test_klasse1_falls_back_on_double_repair_failure(caplog):
    from app.klasse1_transformer import _fallback_description

    entities = [
        {"text": "ACME GmbH", "category": "ORG_CLIENT", "context": ""},
    ]
    expected_fallback = _fallback_description("ORG_CLIENT")

    call_count = {"n": 0}

    async def _fake_chat_json(_messages):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"descriptions": [
                {"original": "ACME GmbH", "description": "[Unternehmen-[branche]]"},
            ]}
        return {"description": "[still-[bad]]"}

    with caplog.at_level(_logging_for_bug_019.WARNING, logger="app.klasse1_transformer"):
        with patch("app.klasse1_transformer.chat_json", new=_fake_chat_json):
            result = await compute_klasse1_descriptions(entities)

    assert result == {"acme gmbh": expected_fallback}
    assert call_count["n"] == 2
