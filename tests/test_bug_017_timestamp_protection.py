"""Integration test: timestamps are not corrupted by BUSINESS_METRIC banding.

Fails on main before the span-first refactor (the regex sweep rewrites "33" inside
"00:01:33"). Passes after the refactor because the protected-span filter drops
occurrences whose span lies inside a recognized timestamp.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.orchestrator import process_document, JobStatus
from app.pseudonymizer import DetectedEntity


@pytest.fixture
def fixture_document_bytes():
    body = "Um 00:01:33 beginnt das Meeting. Das Budget von 33 Mio steigt."
    return body.encode("utf-8")


@pytest.mark.asyncio
async def test_timestamp_not_corrupted_by_business_metric_banding(fixture_document_bytes):
    # Pass 1 returns "33" as BUSINESS_METRIC (value-level; the LLM cannot distinguish
    # which occurrence it saw).
    pass1 = [DetectedEntity(text="33", category="BUSINESS_METRIC", context="Budget")]

    async def _fake_detect(text, config=None, chunk_timings=None):
        return pass1

    # Pass 2 receives occurrences already filtered through the protected-span filter,
    # so the occurrence inside the timestamp has already been dropped before this is called.
    # Return the input unchanged (all KEEP).
    async def _fake_verify_occurrences(
        chunks, occurrences, config=None, chunk_timings=None, protected_spans=None,
    ):
        return list(occurrences)

    with (
        patch("app.orchestrator.detect_entities", side_effect=_fake_detect),
        patch("app.orchestrator.verify_occurrences", side_effect=_fake_verify_occurrences),
        patch("app.orchestrator.warm_up", new_callable=AsyncMock, return_value=None),
        patch("app.orchestrator.compute_klasse2_descriptions", new_callable=AsyncMock, return_value={}),
        patch("app.orchestrator.compute_klasse1_descriptions", new_callable=AsyncMock, return_value={}),
    ):
        result = await process_document(
            fixture_document_bytes, "bug017.txt", "p",
        )

    assert result.status == JobStatus.SUCCESS

    # PRIMARY regression check (this is the BUG-017 contract):
    # the timestamp substring "00:01:33" must be intact in the final output.
    # If filter_protected (or any upstream piece of the protected-span mechanism)
    # is removed, the "33" occurrence inside the timestamp would be pseudonymized,
    # producing "00:01:[...]" and this assertion would fail. This is the assertion
    # that guards the fix.
    assert "00:01:33" in result.final_text

    # Secondary check: the legitimate BUSINESS_METRIC at "33 Mio" was substituted.
    # This one is sensitive to category_config: if the action for BUSINESS_METRIC
    # ever becomes "suppress" or "flag", "33 Mio" could disappear for reasons
    # unrelated to the BUG-017 fix. Kept as a smoke test that the pipeline still
    # does replacement at all; the primary check above is what actually pins BUG-017.
    assert "33 Mio" not in result.final_text
