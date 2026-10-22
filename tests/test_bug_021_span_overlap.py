"""Tests for the BUG-021 span-overlap prefix-entities fix."""

from __future__ import annotations

import logging

from app.mapping_store import MappingStore
from app.pseudonymizer import (
    DetectedEntity,
    EntityOccurrence,
    pseudonymize_by_span,
)
from app.span_resolver import apply_longest_match_policy


def _occ(text: str, category: str, start: int, end: int) -> EntityOccurrence:
    entity = DetectedEntity(text=text, category=category, context="")
    return EntityOccurrence(entity=entity, start=start, end=end)


def test_longest_match_drops_bug021_prefix_overlap_shape():
    """Regression guard for the NGO observation: two occurrences sharing a
    start, different ends, where the shorter is a proper substring of the
    longer. The shorter must be dropped."""
    prefix = _occ("Microsoft", "ORG_PARTNER", 4, 13)
    composite = _occ("Microsoft-Support-Dienstleister", "ORG_PARTNER", 4, 35)

    survivors = apply_longest_match_policy([composite, prefix])

    assert survivors == [composite]


def test_longest_match_emits_debug_log_on_drop(caplog):
    """Observability: dropping an occurrence must emit a DEBUG log line that
    identifies the dropped entity, its span, and the engulfing span."""
    prefix = _occ("Microsoft", "ORG_PARTNER", 4, 13)
    composite = _occ("Microsoft-Support-Dienstleister", "ORG_PARTNER", 4, 35)

    with caplog.at_level(logging.DEBUG, logger="app.span_resolver"):
        apply_longest_match_policy([composite, prefix])

    messages = [rec.message for rec in caplog.records]
    assert any(
        "Microsoft" in msg and "(4, 13)" in msg and "(4, 35)" in msg
        for msg in messages
    ), f"Expected a DEBUG log identifying the dropped (4, 13) span inside (4, 35); got: {messages}"


def test_sink_guard_prevents_span_overlap_corruption(caplog):
    """If a proper-substring overlap reaches pseudonymize_by_span, the sink
    guard must drop the shorter occurrence and emit a WARNING, producing
    clean output (no splice collision)."""
    text = "mit Microsoft-Support-Dienstleister."
    prefix = _occ("Microsoft", "ORG_PARTNER", 4, 13)
    composite = _occ("Microsoft-Support-Dienstleister", "ORG_PARTNER", 4, 35)

    store = MappingStore("test-bug-021")
    klasse2_descriptions = {
        "microsoft": "Software-Gigant",
        "microsoft-support-dienstleister": "Software-Support-Dienstleister",
    }

    with caplog.at_level(logging.WARNING, logger="app.pseudonymizer"):
        result = pseudonymize_by_span(
            text=text,
            occurrences=[composite, prefix],
            store=store,
            klasse2_descriptions=klasse2_descriptions,
        )

    assert result == "mit [Software-Support-Dienstleister]."
    assert "[Software-Gigant]" not in result
    assert any(
        "bug-021" in rec.message.lower()
        or "proper substring" in rec.message.lower()
        or "sink guard" in rec.message.lower()
        for rec in caplog.records
    )


def test_orchestrator_filters_overlap_after_pass2():
    """The orchestrator must call apply_longest_match_policy on the verifier
    survivor list before handing it to pseudonymize_by_span. Without this
    call, Pass 2 MISSED or CORRECT decisions can introduce proper-substring
    overlaps that the sink guard would catch too late (with a WARN)."""
    import inspect

    from app import orchestrator as _orchestrator_module

    source = inspect.getsource(_orchestrator_module.process_document)

    verify_index = source.find("verify_occurrences(")
    filter_index = source.find("apply_longest_match_policy", verify_index)
    pseudonymize_index = source.find("pseudonymize_by_span(", verify_index)

    assert verify_index != -1, "process_document no longer calls verify_occurrences"
    assert filter_index != -1, (
        "process_document must call apply_longest_match_policy on the "
        "verifier survivor list after verify_occurrences returns."
    )
    assert filter_index < pseudonymize_index, (
        "apply_longest_match_policy must be called between verify_occurrences "
        "and pseudonymize_by_span so the filter operates on the verifier "
        "survivors before they reach the span-addressed splice."
    )


import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_end_to_end_prefix_overlap_produces_clean_output_without_sink_guard_warn(caplog):
    """Full-pipeline regression: Pass 1 detects the prefix alone; the verifier
    simulates a MISSED addition of the composite, which shares a start with
    the prefix and engulfs it. After Task 3 the post-Pass-2 filter drops the
    prefix before pseudonymize_by_span sees it, so the sink guard does not
    fire and the final output is the single composite token. If the
    post-Pass-2 filter is ever removed, the sink guard still saves the
    output but emits a WARNING, which this test asserts must NOT happen
    under normal operation."""
    from app.orchestrator import process_document

    text = "mit Microsoft-Support-Dienstleister.\n"
    data = text.encode("utf-8")

    async def _fake_warm_up():
        return None

    async def _fake_detect(_text, _config, chunk_timings=None):
        return [
            DetectedEntity(text="Microsoft", category="ORG_PARTNER", context=""),
        ]

    async def _fake_verify(**kwargs):
        # Simulate Pass 2 MISSED adding the composite occurrence that
        # engulfs the Pass 1 prefix occurrence at the same start offset.
        composite_entity = DetectedEntity(
            text="Microsoft-Support-Dienstleister",
            category="ORG_PARTNER",
            context="",
        )
        composite_occ = EntityOccurrence(entity=composite_entity, start=4, end=35)
        return list(kwargs["occurrences"]) + [composite_occ]

    async def _fake_k1(_entities, _config):
        return {}

    async def _fake_k2(_entities, _config):
        return {
            "microsoft": "Software-Gigant",
            "microsoft-support-dienstleister": "Software-Support-Dienstleister",
        }

    with (
        patch("app.orchestrator.detect_entities", new=_fake_detect),
        patch("app.orchestrator.verify_occurrences", new=_fake_verify),
        patch("app.orchestrator.compute_klasse1_descriptions", new=_fake_k1),
        patch("app.orchestrator.compute_klasse2_descriptions", new=_fake_k2),
        patch("app.orchestrator.warm_up", new=_fake_warm_up),
    ):
        with caplog.at_level(logging.WARNING):
            result = await process_document(
                data=data,
                filename="bug021_regression.txt",
                project_id="bug021-regression-test",
            )

    assert "[Software-Support-Dienstleister]" in result.pseudonymized_text
    assert "[Software-Gigant]" not in result.pseudonymized_text
    assert "[Software-Gigant]-Support-Dienstleister]" not in result.pseudonymized_text

    sink_guard_warnings = [
        rec for rec in caplog.records
        if rec.name == "app.pseudonymizer"
        and "bug-021" in rec.message.lower()
    ]
    assert not sink_guard_warnings, (
        "Sink guard fired, which means the post-Pass-2 filter did not run. "
        f"Warnings: {[rec.message for rec in sink_guard_warnings]}"
    )
