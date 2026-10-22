"""Integration tests for the BUG-019 double-bracket-nesting fix."""

from __future__ import annotations

import logging

from app.mapping_store import MappingStore
from app.pseudonymizer import (
    DetectedEntity,
    EntityOccurrence,
    pseudonymize_by_span,
)


def test_sink_guard_prevents_double_wrap_when_description_already_bracketed(
    caplog,
):
    """If a description somehow enters pseudonymize_by_span already bracketed,
    the sink guard must emit it exactly once (no [[...]]) and log a warning."""
    text = "Microsoft ist ein grosser Anbieter."
    entity = DetectedEntity(text="Microsoft", category="ORG_PARTNER", context="")
    occ = EntityOccurrence(entity=entity, start=0, end=len("Microsoft"))

    store = MappingStore("test-bug-019")
    klasse2_descriptions = {"microsoft": "[Software-Gigant]"}

    with caplog.at_level(logging.WARNING, logger="app.pseudonymizer"):
        result = pseudonymize_by_span(
            text=text,
            occurrences=[occ],
            store=store,
            klasse2_descriptions=klasse2_descriptions,
        )

    assert "[[Software-Gigant]]" not in result
    assert "[Software-Gigant]" in result
    assert result.count("[Software-Gigant]") == 1
    assert any(
        "already bracketed" in rec.message.lower()
        or "double-wrap" in rec.message.lower()
        or "bug-019" in rec.message.lower()
        for rec in caplog.records
    )


import pytest
from unittest.mock import AsyncMock, patch

from app.klasse2_transformer import compute_klasse2_descriptions


@pytest.mark.asyncio
async def test_end_to_end_no_double_brackets_in_pseudonymized_output():
    """Bracketed LLM description -> strict strip -> no `[[...]]` in final output."""
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
        k2_descriptions = await compute_klasse2_descriptions(entities, config=None)

    text = "Microsoft ist ein grosser Anbieter."
    entity = DetectedEntity(text="Microsoft", category="ORG_PARTNER", context="")
    occ = EntityOccurrence(entity=entity, start=0, end=len("Microsoft"))
    store = MappingStore("test-bug-019-e2e")

    result = pseudonymize_by_span(
        text=text,
        occurrences=[occ],
        store=store,
        klasse2_descriptions=k2_descriptions,
    )

    assert "[[" not in result
    assert "]]" not in result
    assert "[Software-Gigant]" in result
