"""Tests for verify_occurrences: value-level dedupe at the LLM boundary, decisions applied to occurrences."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.pseudonymizer import Chunk, DetectedEntity, EntityOccurrence


def _occ(text: str, category: str, start: int, end: int) -> EntityOccurrence:
    return EntityOccurrence(
        entity=DetectedEntity(text=text, category=category, context=""),
        start=start,
        end=end,
    )


@pytest.mark.asyncio
async def test_verify_occurrences_keep_maps_to_all_occurrences_sharing_value():
    from app.verifier import verify_occurrences

    # Three occurrences of "33" as BUSINESS_METRIC, same (text, category) key.
    occs = [
        _occ("33", "BUSINESS_METRIC", 10, 12),
        _occ("33", "BUSINESS_METRIC", 50, 52),
        _occ("33", "BUSINESS_METRIC", 90, 92),
    ]

    llm_response = {
        "verified": [{"text": "33", "category": "BUSINESS_METRIC", "status": "keep"}],
        "missed": [],
    }

    with patch("app.verifier.chat_json", new_callable=AsyncMock, return_value=llm_response):
        survivors = await verify_occurrences(
            chunks=[Chunk(text="x" * 200, start=0)],
            occurrences=occs,
        )

    assert len(survivors) == 3
    assert sorted(o.start for o in survivors) == [10, 50, 90]


@pytest.mark.asyncio
async def test_verify_occurrences_reject_drops_all_occurrences_sharing_value():
    from app.verifier import verify_occurrences

    full_text = "Trailer" + "x" * 93 + "Trailer"
    occs = [
        _occ("Trailer", "PROCESS_ARTIFACT", 0, 7),
        _occ("Trailer", "PROCESS_ARTIFACT", 100, 107),
    ]

    llm_response = {
        "verified": [{
            "text": "Trailer", "category": "PROCESS_ARTIFACT",
            "status": "reject", "reason": "film term",
        }],
        "missed": [],
    }

    with patch("app.verifier.chat_json", new_callable=AsyncMock, return_value=llm_response):
        survivors = await verify_occurrences(
            chunks=[Chunk(text=full_text, start=0)],
            occurrences=occs,
        )

    assert survivors == []


@pytest.mark.asyncio
async def test_verify_occurrences_dedupes_candidates_for_llm_call():
    """The LLM call receives one entry per unique (text, category), not per occurrence."""
    from app.verifier import verify_occurrences

    full_text = "Müller xxxxxxxx Müller xxxxxxxxxxxxxxxxx Müller"
    occs = [
        _occ("Müller", "PERSON", 0, 6),
        _occ("Müller", "PERSON", 16, 22),
        _occ("Müller", "PERSON", full_text.rfind("Müller"), full_text.rfind("Müller") + 6),
    ]

    sent_payloads: list[list[dict]] = []

    async def _capture(messages, **kwargs):
        user_content = messages[-1]["content"]
        import json
        import re as _re
        payload_match = _re.search(r"Kandidaten-Entitaeten:\s*(\[.*\])", user_content, _re.DOTALL)
        if payload_match:
            sent_payloads.append(json.loads(payload_match.group(1)))
        return {
            "verified": [{"text": "Müller", "category": "PERSON", "status": "keep"}],
            "missed": [],
        }

    with patch("app.verifier.chat_json", new=AsyncMock(side_effect=_capture)):
        await verify_occurrences(
            chunks=[Chunk(text=full_text, start=0)],
            occurrences=occs,
        )

    for payload in sent_payloads:
        keys = [(item["text"], item["category"]) for item in payload]
        assert len(keys) == len(set(keys)), "Verifier sent duplicate (text, category) entries"


@pytest.mark.asyncio
async def test_verify_occurrences_empty_input_returns_empty():
    from app.verifier import verify_occurrences

    survivors = await verify_occurrences(
        chunks=[Chunk(text="anything", start=0)],
        occurrences=[],
    )
    assert survivors == []


@pytest.mark.asyncio
async def test_verify_occurrences_correct_rewrites_entity_and_reresolves_in_chunk():
    """A CORRECT verdict rewrites the entity (text and/or category) and re-resolves
    occurrences via the span resolver. The original span is dropped in favor of the
    positions the corrected text matches in the chunk.
    """
    from app.verifier import verify_occurrences

    chunk_text = "Herr Dr. Müller war da. Dr. Müller ist Arzt."
    # Pass 1 found "Müller" (PERSON) at a single position.
    occs = [
        _occ("Müller", "PERSON", chunk_text.index("Müller"), chunk_text.index("Müller") + 6),
    ]

    # The LLM corrects the text to "Dr. Müller" (expanding the boundary).
    llm_response = {
        "verified": [{
            "text": "Müller", "category": "PERSON",
            "status": "correct",
            "corrected_text": "Dr. Müller",
            "corrected_category": "PERSON",
        }],
        "missed": [],
    }

    with patch("app.verifier.chat_json", new_callable=AsyncMock, return_value=llm_response):
        survivors = await verify_occurrences(
            chunks=[Chunk(text=chunk_text, start=0)],
            occurrences=occs,
        )

    # "Dr. Müller" appears twice in the chunk; re-resolution emits one occurrence per match.
    assert len(survivors) == 2
    assert all(s.entity.text == "Dr. Müller" for s in survivors)
    assert all(s.entity.category == "PERSON" for s in survivors)
    # Spans reference the corrected-text positions, not the original "Müller" position.
    assert [chunk_text[s.start : s.end] for s in survivors] == ["Dr. Müller", "Dr. Müller"]


@pytest.mark.asyncio
async def test_verify_occurrences_missed_adds_new_occurrences_via_resolver():
    """MISSED entries from the LLM resolve to one occurrence per word-boundary match
    in the chunk they were reported for.
    """
    from app.verifier import verify_occurrences

    chunk_text = "Berlin ist eine Stadt. In Berlin regnet es heute."
    # Pass 1 found nothing; verifier sees an empty candidate list but the LLM reports MISSED.
    occs: list[EntityOccurrence] = [
        # A placeholder so the function does not early-return on empty occurrences.
        # We give it a different category so the dedupe-for-llm payload is separate.
        _occ("placeholder", "PERSON", 0, 11),
    ]

    llm_response = {
        "verified": [{"text": "placeholder", "category": "PERSON", "status": "reject"}],
        "missed": [
            {"text": "Berlin", "category": "LOCATION", "context": "Stadt"},
        ],
    }

    with patch("app.verifier.chat_json", new_callable=AsyncMock, return_value=llm_response):
        survivors = await verify_occurrences(
            chunks=[Chunk(text=chunk_text, start=0)],
            occurrences=occs,
        )

    # The placeholder was rejected; two "Berlin" occurrences added via MISSED.
    assert len(survivors) == 2
    assert all(s.entity.text == "Berlin" for s in survivors)
    assert all(s.entity.category == "LOCATION" for s in survivors)
    assert sorted(s.start for s in survivors) == sorted(
        [chunk_text.index("Berlin"), chunk_text.rindex("Berlin")]
    )


@pytest.mark.asyncio
async def test_verify_occurrences_missed_filtered_through_protected_spans():
    """A MISSED entity whose resolved span falls inside a protected region is dropped.

    INV-19 contract: the Pass 2 LLM cannot re-introduce a substitution into a protected
    region by reporting the offending text as MISSED. This closes the gap flagged in
    the final holistic review — without it, BUG-017 could reappear through the MISSED
    path even though the Pass 1 filter caught the original occurrence.
    """
    from app.verifier import verify_occurrences

    chunk_text = "Um 00:01:33 beginnt das Meeting. Das Budget von 33 Mio steigt."
    # Pass 1 occurrence that survives (the real metric, outside the timestamp).
    occs = [
        _occ("33", "BUSINESS_METRIC", chunk_text.index("33 Mio"), chunk_text.index("33 Mio") + 2),
    ]
    # Protected span: the "00:01:33" timestamp.
    ts_start = chunk_text.index("00:01:33")
    protected = [(ts_start, ts_start + len("00:01:33"))]

    # LLM keeps the Pass 1 candidate AND reports "33" as MISSED again. Without the filter,
    # this would re-resolve to two positions (inside the timestamp and at "33 Mio"),
    # re-introducing the corruption.
    llm_response = {
        "verified": [{"text": "33", "category": "BUSINESS_METRIC", "status": "keep"}],
        "missed": [{"text": "33", "category": "BUSINESS_METRIC", "context": "recheck"}],
    }

    with patch("app.verifier.chat_json", new_callable=AsyncMock, return_value=llm_response):
        survivors = await verify_occurrences(
            chunks=[Chunk(text=chunk_text, start=0)],
            occurrences=occs,
            protected_spans=protected,
        )

    # Exactly one occurrence survives: the "33" at "33 Mio". No occurrence lands
    # inside the timestamp span.
    ts_span = range(*protected[0])
    for occ in survivors:
        assert occ.start not in ts_span, (
            f"Occurrence at {occ.start}–{occ.end} lands inside protected timestamp span "
            f"{protected[0]} — INV-19 violated."
        )
    # At least the original metric occurrence is preserved.
    assert any(s.start == chunk_text.index("33 Mio") for s in survivors)


@pytest.mark.asyncio
async def test_verify_occurrences_correct_filtered_through_protected_spans():
    """A CORRECT verdict cannot re-land a substitution inside a protected region.

    INV-19 contract for the CORRECT path. If the LLM rewrites an entity's text to
    something that re-resolves to a timestamp-embedded span, the filter drops it.
    """
    from app.verifier import verify_occurrences

    chunk_text = "Meeting at 00:01:33 mentioned item 33."
    # Pass 1 occurrence for "item" that survives (outside any protected region).
    item_start = chunk_text.index("item")
    occs = [
        _occ("item", "BUSINESS_METRIC", item_start, item_start + 4),
    ]
    ts_start = chunk_text.index("00:01:33")
    protected = [(ts_start, ts_start + len("00:01:33"))]

    # LLM "corrects" the entity text from "item" to "33". Without the filter, this would
    # re-resolve to two positions (inside the timestamp and after "item ").
    llm_response = {
        "verified": [{
            "text": "item", "category": "BUSINESS_METRIC",
            "status": "correct",
            "corrected_text": "33",
            "corrected_category": "BUSINESS_METRIC",
        }],
        "missed": [],
    }

    with patch("app.verifier.chat_json", new_callable=AsyncMock, return_value=llm_response):
        survivors = await verify_occurrences(
            chunks=[Chunk(text=chunk_text, start=0)],
            occurrences=occs,
            protected_spans=protected,
        )

    ts_span = range(*protected[0])
    for occ in survivors:
        assert occ.start not in ts_span, (
            f"CORRECT re-resolved occurrence at {occ.start}–{occ.end} landed inside "
            f"protected timestamp span {protected[0]} — INV-19 violated."
        )


@pytest.mark.asyncio
async def test_verify_occurrences_llm_failure_keeps_occurrences_as_fallback():
    """When the LLM call raises, surviving occurrences in that chunk are the raw
    pre-verification candidates (conservative fallback).
    """
    from app.verifier import verify_occurrences

    chunk_text = "Müller und Schmidt."
    occs = [
        _occ("Müller", "PERSON", 0, 6),
        _occ("Schmidt", "PERSON", 11, 18),
    ]

    async def _raise(_messages, **_kwargs):
        raise RuntimeError("simulated LLM outage")

    timings: list[dict] = []
    with patch("app.verifier.chat_json", new=AsyncMock(side_effect=_raise)):
        survivors = await verify_occurrences(
            chunks=[Chunk(text=chunk_text, start=0)],
            occurrences=occs,
            chunk_timings=timings,
        )

    # Both occurrences preserved untouched.
    assert survivors == occs
    # Timing entry recorded for the failed chunk with candidates/verified equal.
    assert len(timings) == 1
    assert timings[0]["candidates"] == 2
    assert timings[0]["verified"] == 2
