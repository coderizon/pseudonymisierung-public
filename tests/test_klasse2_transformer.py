"""Tests for Klasse 2 structured generalization."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.klasse2_transformer import (
    KLASSE2_HANDLED_CATEGORIES,
    FALLBACK_DESCRIPTIONS,
    TAXONOMY_CATEGORIES,
    LLM_CATEGORIES,
    BANDING_CATEGORY,
    _apply_banding_result,
    _apply_taxonomy_result,
    _generalize_generic_number,
    _generalize_monetary,
    _dedup_entities,
    compute_klasse2_descriptions,
)
from app.mapping_store import MappingStore
from app.pseudonymizer import DetectedEntity, pseudonymize_by_span
from app.span_resolver import resolve_occurrences_in_chunk
from app.taxonomy_loader import load_taxonomy, load_banding, load_percentage_rounder

TAXONOMIES_DIR = Path(__file__).resolve().parent.parent / "config" / "taxonomies"


class TestDedup:
    def test_deduplicates_by_lowercase(self):
        entities = [
            {"text": "Aviation AI", "category": "INDUSTRY", "context": ""},
            {"text": "aviation ai", "category": "INDUSTRY", "context": ""},
        ]
        result = _dedup_entities(entities)
        assert len(result) == 1
        assert result[0]["text"] == "Aviation AI"

    def test_preserves_different_entities(self):
        entities = [
            {"text": "Aviation AI", "category": "INDUSTRY", "context": ""},
            {"text": "Pharma-Logistik", "category": "INDUSTRY", "context": ""},
        ]
        result = _dedup_entities(entities)
        assert len(result) == 2


class TestApplyTaxonomyResult:
    def test_nace_level1_lookup(self):
        classifications = [
            {"original": "Aviation AI", "category": "INDUSTRY", "code": "J"},
        ]
        result = _apply_taxonomy_result(classifications, None)
        assert result["aviation ai"] == "Information und Kommunikation"

    def test_isco_level2_lookup(self):
        classifications = [
            {"original": "VP European Fleet Sales", "category": "FUNCTION_TITLE", "code": "33"},
        ]
        result = _apply_taxonomy_result(classifications, None)
        # Code 33 at level 2 should have a label
        assert "33" not in result.get("vp european fleet sales", "")  # the code should not be the label itself
        assert result["vp european fleet sales"] != ""

    def test_unknown_code_uses_fallback(self):
        classifications = [
            {"original": "Something", "category": "INDUSTRY", "code": "ZZZ"},
        ]
        result = _apply_taxonomy_result(classifications, None)
        assert result["something"] == FALLBACK_DESCRIPTIONS["INDUSTRY"]

    def test_empty_classifications(self):
        result = _apply_taxonomy_result([], None)
        assert result == {}

    def test_generalize_from_deeper_level(self):
        # If LLM returns a level-2 code for INDUSTRY, should walk up to level 1
        classifications = [
            {"original": "IT-Dienstleistungen", "category": "INDUSTRY", "code": "62"},
        ]
        result = _apply_taxonomy_result(classifications, None)
        # Code 62 is under section J
        assert result["it-dienstleistungen"] == "Information und Kommunikation"


class TestApplyBandingResult:
    def test_headcount_small_team(self):
        metrics = [
            {"original": "12 FTE", "value": 12, "metric_type": "headcount", "unit": "FTE"},
        ]
        result = _apply_banding_result(metrics)
        assert "12 fte" in result
        label = result["12 fte"]
        assert label  # should have a non-empty label

    def test_headcount_large(self):
        metrics = [
            {"original": "~100 FTE", "value": 100, "metric_type": "headcount", "unit": "FTE"},
        ]
        result = _apply_banding_result(metrics)
        assert "~100 fte" in result

    def test_revenue_banding(self):
        metrics = [
            {"original": "EUR 720K Einsparung", "value": 720000, "metric_type": "revenue", "unit": "EUR"},
        ]
        result = _apply_banding_result(metrics)
        assert "eur 720k einsparung" in result

    def test_percentage_rounding(self):
        metrics = [
            {"original": "14.7% Marge", "value": 14.7, "metric_type": "percentage", "unit": "%"},
        ]
        result = _apply_banding_result(metrics)
        label = result["14.7% marge"]
        assert "15" in label  # 5er-Rundung: 14.7 -> ca. 15%

    def test_other_metric_type(self):
        metrics = [
            {"original": "23 Projekte", "value": 23, "metric_type": "other", "unit": ""},
        ]
        result = _apply_banding_result(metrics)
        assert "23 projekte" in result

    def test_monetary_type_not_revenue(self):
        metrics = [
            {"original": "3000 EUR", "value": 3000, "metric_type": "monetary", "unit": "EUR"},
        ]
        result = _apply_banding_result(metrics)
        assert "3000 eur" in result
        # Should NOT produce a company-size label like "Kleinstunternehmen"
        assert "unternehmen" not in result["3000 eur"].lower()
        assert "Betrag" in result["3000 eur"]

    def test_invalid_value_uses_fallback(self):
        metrics = [
            {"original": "viele", "value": "invalid", "metric_type": "headcount", "unit": ""},
        ]
        result = _apply_banding_result(metrics)
        assert result["viele"] == FALLBACK_DESCRIPTIONS["BUSINESS_METRIC"]


class TestGeneralizeMonetary:
    def test_small_amount(self):
        assert _generalize_monetary(500) == "dreistelliger Betrag"

    def test_thousands(self):
        assert _generalize_monetary(3000) == "vierstelliger Betrag"

    def test_tens_of_thousands(self):
        assert _generalize_monetary(50000) == "fuenfstelliger Betrag"

    def test_hundreds_of_thousands(self):
        assert _generalize_monetary(720000) == "sechsstelliger Betrag"

    def test_millions(self):
        assert _generalize_monetary(2_500_000) == "niedriger Millionenbetrag"


class TestGeneralizeGenericNumber:
    def test_small_numbers(self):
        assert _generalize_generic_number(5) == "wenige"

    def test_dozens(self):
        assert _generalize_generic_number(23) == "mehrere Dutzend"

    def test_near_hundred(self):
        assert _generalize_generic_number(80) == "knapp hundert"

    def test_hundreds(self):
        assert _generalize_generic_number(340) == "mehrere hundert"

    def test_thousands(self):
        result = _generalize_generic_number(1500)
        assert "ca." in result


class TestComputeKlasse2Descriptions:
    @pytest.mark.asyncio
    async def test_empty_entities_returns_empty(self):
        result = await compute_klasse2_descriptions([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_non_klasse2_entities_returns_empty(self):
        entities = [
            {"text": "Dr. Schmidt", "category": "PERSON", "context": ""},
            {"text": "Freiburg", "category": "LOCATION", "context": ""},
        ]
        result = await compute_klasse2_descriptions(entities)
        assert result == {}

    @pytest.mark.asyncio
    async def test_taxonomy_classification_flow(self):
        entities = [
            {"text": "Aviation AI", "category": "INDUSTRY", "context": "Aviation AI Startup"},
        ]

        mock_taxonomy_response = {
            "classifications": [
                {"original": "Aviation AI", "category": "INDUSTRY", "code": "J"},
            ]
        }

        with patch("app.klasse2_transformer.chat_json", new_callable=AsyncMock, return_value=mock_taxonomy_response):
            result = await compute_klasse2_descriptions(entities)

        assert result["aviation ai"] == "Information und Kommunikation"

    @pytest.mark.asyncio
    async def test_banding_flow(self):
        entities = [
            {"text": "~100 FTE", "category": "BUSINESS_METRIC", "context": "Team von ~100 FTE"},
        ]

        mock_banding_response = {
            "metrics": [
                {"original": "~100 FTE", "value": 100, "metric_type": "headcount", "unit": "FTE"},
            ]
        }

        with patch("app.klasse2_transformer.chat_json", new_callable=AsyncMock, return_value=mock_banding_response):
            result = await compute_klasse2_descriptions(entities)

        assert "~100 fte" in result
        assert result["~100 fte"]  # non-empty label

    @pytest.mark.asyncio
    async def test_llm_generalization_flow(self):
        entities = [
            {"text": "SAP S/4HANA", "category": "PRODUCT", "context": "nutzt SAP S/4HANA als ERP"},
        ]

        mock_llm_response = {
            "descriptions": [
                {"original": "SAP S/4HANA", "description": "ERP-System"},
            ]
        }

        with patch("app.klasse2_transformer.chat_json", new_callable=AsyncMock, return_value=mock_llm_response):
            result = await compute_klasse2_descriptions(entities)

        assert result["sap s/4hana"] == "ERP-System"

    @pytest.mark.asyncio
    async def test_mixed_categories(self):
        """Test that entities from different tiers are all handled."""
        entities = [
            {"text": "Aviation AI", "category": "INDUSTRY", "context": ""},
            {"text": "~100 FTE", "category": "BUSINESS_METRIC", "context": ""},
            {"text": "SAP S/4HANA", "category": "PRODUCT", "context": ""},
        ]

        # chat_json is called 3 times: taxonomy, banding, LLM
        responses = [
            # Taxonomy classification
            {"classifications": [{"original": "Aviation AI", "category": "INDUSTRY", "code": "J"}]},
            # Banding extraction
            {"metrics": [{"original": "~100 FTE", "value": 100, "metric_type": "headcount", "unit": "FTE"}]},
            # LLM generalization
            {"descriptions": [{"original": "SAP S/4HANA", "description": "ERP-System"}]},
        ]

        with patch("app.klasse2_transformer.chat_json", new_callable=AsyncMock, side_effect=responses):
            result = await compute_klasse2_descriptions(entities)

        assert len(result) == 3
        assert "aviation ai" in result
        assert "~100 fte" in result
        assert "sap s/4hana" in result

    @pytest.mark.asyncio
    async def test_fallback_on_taxonomy_failure(self):
        entities = [
            {"text": "Aviation AI", "category": "INDUSTRY", "context": ""},
        ]

        with patch("app.klasse2_transformer.chat_json", new_callable=AsyncMock, side_effect=Exception("LLM down")):
            result = await compute_klasse2_descriptions(entities)

        assert result["aviation ai"] == FALLBACK_DESCRIPTIONS["INDUSTRY"]

    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self):
        entities = [
            {"text": "SAP S/4HANA", "category": "PRODUCT", "context": ""},
        ]

        with patch("app.klasse2_transformer.chat_json", new_callable=AsyncMock, side_effect=Exception("LLM down")):
            result = await compute_klasse2_descriptions(entities)

        assert result["sap s/4hana"] == FALLBACK_DESCRIPTIONS["PRODUCT"]

    @pytest.mark.asyncio
    async def test_llm_batch_failure_retries_individually(self):
        """When the batch call fails, entities are retried one by one."""
        entities = [
            {"text": "SAP S/4HANA", "category": "PRODUCT", "context": ""},
            {"text": "Amazon", "category": "ORG_PARTNER", "context": ""},
        ]

        # First call (batch) fails, then individual calls succeed
        responses = [
            Exception("batch JSON broken"),
            {"descriptions": [{"original": "SAP S/4HANA", "description": "ERP-System"}]},
            {"descriptions": [{"original": "Amazon", "description": "globaler Technologiekonzern"}]},
        ]

        with patch("app.klasse2_transformer.chat_json", new_callable=AsyncMock, side_effect=responses):
            result = await compute_klasse2_descriptions(entities)

        assert result["sap s/4hana"] == "ERP-System"
        assert result["amazon"] == "globaler Technologiekonzern"

    @pytest.mark.asyncio
    async def test_skips_contract_and_quasi_id(self):
        entities = [
            {"text": "CC-047", "category": "CONTRACT", "context": ""},
            {"text": "IT + Sueddeutschland", "category": "QUASI_ID", "context": ""},
        ]

        result = await compute_klasse2_descriptions(entities)
        assert result == {}


class TestPseudonymizeBySpanWithKlasse2:
    def test_uses_klasse2_description_instead_of_token(self):
        text = "Die Branche ist Aviation AI und das System ist SAP S/4HANA."
        entities = [
            DetectedEntity(text="Aviation AI", category="INDUSTRY", context=""),
            DetectedEntity(text="SAP S/4HANA", category="PRODUCT", context=""),
        ]
        k2_descriptions = {
            "aviation ai": "Informations- und Kommunikationssektor",
            "sap s/4hana": "ERP-System",
        }
        store = MappingStore("test-k2")
        occurrences = []
        for entity in entities:
            occurrences.extend(resolve_occurrences_in_chunk(entity, text, chunk_start=0))
        result = pseudonymize_by_span(text, occurrences, store, klasse2_descriptions=k2_descriptions)
        store.close()

        assert "[Informations- und Kommunikationssektor]" in result
        assert "[ERP-System]" in result
        assert "Aviation AI" not in result
        assert "SAP S/4HANA" not in result

    def test_klasse1_takes_precedence_over_klasse2(self):
        """If an entity somehow has both K1 and K2 descriptions, K1 wins."""
        text = "AeroTech GmbH expandiert."
        entities = [
            DetectedEntity(text="AeroTech GmbH", category="ORG_CLIENT", context=""),
        ]
        k1 = {"aerotech gmbh": "IT-Unternehmen"}
        k2 = {"aerotech gmbh": "should-not-appear"}
        store = MappingStore("test-k1-over-k2")
        occurrences = []
        for entity in entities:
            occurrences.extend(resolve_occurrences_in_chunk(entity, text, chunk_start=0))
        result = pseudonymize_by_span(text, occurrences, store, klasse1_descriptions=k1, klasse2_descriptions=k2)
        store.close()

        assert "[IT-Unternehmen]" in result
        assert "should-not-appear" not in result

    def test_without_k2_descriptions_falls_back_to_token(self):
        text = "Die Branche ist Aviation AI."
        entities = [
            DetectedEntity(text="Aviation AI", category="INDUSTRY", context=""),
        ]
        store = MappingStore("test-k2-fallback")
        occurrences = []
        for entity in entities:
            occurrences.extend(resolve_occurrences_in_chunk(entity, text, chunk_start=0))
        result = pseudonymize_by_span(text, occurrences, store)
        store.close()

        assert "[INDUSTRY_A]" in result

    def test_k2_description_stored_in_mapping(self):
        text = "Die Branche ist Aviation AI."
        entities = [
            DetectedEntity(text="Aviation AI", category="INDUSTRY", context="ctx"),
        ]
        k2 = {"aviation ai": "Informationssektor"}
        store = MappingStore("test-k2-store")
        occurrences = []
        for entity in entities:
            occurrences.extend(resolve_occurrences_in_chunk(entity, text, chunk_start=0))
        pseudonymize_by_span(text, occurrences, store, klasse2_descriptions=k2)

        assert store.get_pseudonym("Aviation AI") == "Informationssektor"
        store.close()

    def test_consistent_mapping_across_occurrences(self):
        text = "Aviation AI expandiert. Aviation AI wachst."
        entities = [
            DetectedEntity(text="Aviation AI", category="INDUSTRY", context=""),
        ]
        k2 = {"aviation ai": "Informationssektor"}
        store = MappingStore("test-k2-consistent")
        occurrences = []
        for entity in entities:
            occurrences.extend(resolve_occurrences_in_chunk(entity, text, chunk_start=0))
        result = pseudonymize_by_span(text, occurrences, store, klasse2_descriptions=k2)
        store.close()

        assert result.count("[Informationssektor]") == 2


# ---------------------------------------------------------------------------
# BUG-019: defensive strict-strip at Tier 1a (taxonomy) and Tier 1b (banding)
# ---------------------------------------------------------------------------


def test_apply_banding_result_strips_canonical_surrounding_brackets(monkeypatch):
    """Defensive: if a band.label_de ever ships with brackets, strip them."""
    from app import klasse2_transformer as k2t

    class _BracketedBand:
        label_de = "[Kleinstunternehmen]"

    class _FakeBanding:
        def classify(self, _value):
            return _BracketedBand()

    monkeypatch.setattr(k2t, "get_banding", lambda _name: _FakeBanding())

    result = k2t._apply_banding_result([
        {"original": "Umsatz 1 Mio", "value": 1_000_000, "metric_type": "revenue"},
    ])

    assert result["umsatz 1 mio"] == "Kleinstunternehmen"


def test_apply_taxonomy_result_strips_canonical_surrounding_brackets(monkeypatch):
    """Defensive: if a taxonomy label ever ships with brackets, strip them."""
    from app import klasse2_transformer as k2t
    from app.category_config import get_config

    class _BracketedEntry:
        level = 1
        label_de = "[IT-Dienstleistungen]"
        label_en = "[IT services]"

    class _FakeTaxonomy:
        def lookup(self, _code):
            return _BracketedEntry()

    monkeypatch.setattr(k2t, "get_taxonomy", lambda _name: _FakeTaxonomy())

    result = k2t._apply_taxonomy_result(
        [{"original": "ACME GmbH", "category": "INDUSTRY", "code": "62.01"}],
        config=get_config(),
    )

    assert result["acme gmbh"] == "IT-Dienstleistungen"


# ---------------------------------------------------------------------------
# BUG-019: strict-strip + self-repair loop at Tier 2 (_generalize_with_llm)
# ---------------------------------------------------------------------------

import logging as _logging_for_bug_019  # avoid clobbering if an earlier import

from app.category_config import get_config as _get_config_for_bug_019
from app.klasse2_transformer import _generalize_with_llm


@pytest.mark.asyncio
async def test_generalize_with_llm_strips_canonical_bracketed_output():
    entities = [
        {"text": "Microsoft", "category": "ORG_PARTNER", "context": ""},
    ]
    with patch(
        "app.klasse2_transformer.chat_json",
        new=AsyncMock(return_value={
            "descriptions": [
                {"original": "Microsoft", "description": "[Software-Gigant]"},
            ],
        }),
    ):
        result = await _generalize_with_llm(entities, config=_get_config_for_bug_019())

    assert result == {"microsoft": "Software-Gigant"}


@pytest.mark.asyncio
async def test_generalize_with_llm_triggers_repair_on_malformed_bracket_shape():
    entities = [
        {"text": "Microsoft", "category": "ORG_PARTNER", "context": ""},
    ]
    call_count = {"n": 0}

    async def _fake_chat_json(_messages):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"descriptions": [
                {"original": "Microsoft", "description": "[Software-[brand]-Gigant]"},
            ]}
        return {"description": "Software-Gigant"}

    with patch("app.klasse2_transformer.chat_json", new=_fake_chat_json):
        result = await _generalize_with_llm(entities, config=_get_config_for_bug_019())

    assert result == {"microsoft": "Software-Gigant"}
    assert call_count["n"] == 2


@pytest.mark.asyncio
async def test_generalize_with_llm_falls_back_on_double_repair_failure(caplog):
    entities = [
        {"text": "Microsoft", "category": "ORG_PARTNER", "context": ""},
    ]
    expected_fallback = FALLBACK_DESCRIPTIONS.get("ORG_PARTNER", "ORG_PARTNER")

    call_count = {"n": 0}

    async def _fake_chat_json(_messages):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"descriptions": [
                {"original": "Microsoft", "description": "[Software-[brand]-Gigant]"},
            ]}
        return {"description": "[still-[malformed]]"}

    with caplog.at_level(_logging_for_bug_019.WARNING, logger="app.klasse2_transformer"):
        with patch("app.klasse2_transformer.chat_json", new=_fake_chat_json):
            result = await _generalize_with_llm(entities, config=_get_config_for_bug_019())

    assert result == {"microsoft": expected_fallback}
    assert call_count["n"] == 2
    assert any("bug-019" in rec.message.lower() or "bracket" in rec.message.lower()
               for rec in caplog.records)
