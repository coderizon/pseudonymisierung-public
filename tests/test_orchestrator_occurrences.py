"""Tests for occurrence variant recording in the orchestrator pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.mapping_store import MappingStore
from app.orchestrator import process_document, JobStatus
from app.pseudonymizer import DetectedEntity


@pytest.fixture
def sample_pdf_bytes():
    """Create a minimal PDF with two pages containing the same entity in different casings."""
    import fitz
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Report by Max Müller about the project.")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "MAX MÜLLER approved the final version.")
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def mock_entities():
    return [
        DetectedEntity(text="Max Müller", category="PERSON", context="consultant"),
    ]


@pytest.mark.asyncio
async def test_occurrences_recorded_per_block(sample_pdf_bytes, mock_entities):
    """Entities are recorded with their actual casing and block source."""

    async def _echo_verify(
        chunks, occurrences, config=None, chunk_timings=None, protected_spans=None,
    ):
        return list(occurrences)

    with (
        patch("app.orchestrator.warm_up", new_callable=AsyncMock),
        patch("app.orchestrator.detect_entities", new_callable=AsyncMock, return_value=mock_entities),
        patch("app.orchestrator.verify_occurrences", side_effect=_echo_verify),
    ):
        result = await process_document(sample_pdf_bytes, "report.pdf", "test_proj")

    assert result.status == JobStatus.SUCCESS

    store = MappingStore("test_proj")
    pseudonym = store.get_pseudonym("Max Müller")
    assert pseudonym is not None

    occurrences = store.get_occurrences(pseudonym)
    assert len(occurrences) == 2

    casings = {o["casing"] for o in occurrences}
    assert "Max Müller" in casings
    assert "MAX MÜLLER" in casings

    # Both point to the same document
    filenames = {o["document_filename"] for o in occurrences}
    assert filenames == {"report.pdf"}

    # Different pages
    sources = {o["block_source"] for o in occurrences}
    assert len(sources) == 2
    store.close()


@pytest.mark.asyncio
async def test_no_occurrences_on_failed_extraction():
    """If no text can be extracted, no occurrences are recorded."""
    empty_data = b"not a valid document"
    with (
        patch("app.orchestrator.warm_up", new_callable=AsyncMock),
        patch("app.orchestrator.detect_entities", new_callable=AsyncMock),
        patch("app.orchestrator.verify_occurrences", new_callable=AsyncMock),
    ):
        result = await process_document(empty_data, "broken.xyz", "test_proj")

    assert result.status == JobStatus.FAILED
    store = MappingStore("test_proj")
    assert store.get_all_mappings() == []
    store.close()
