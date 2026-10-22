"""Tests for clip_to_max_chunks helper (prerequisite for the --max-chunks dev flag)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from app.orchestrator import clip_to_max_chunks, process_document
from app.pseudonymizer import set_chunk_size


def test_clip_to_max_chunks_none_returns_full_text_unchanged():
    full = "first paragraph\n\nsecond paragraph\n\nthird paragraph"
    assert clip_to_max_chunks(full, max_chunks=None, max_chunk_size=100) == full


def test_clip_to_max_chunks_more_requested_than_available_returns_full_text():
    full = "a\n\nb\n\nc"
    # With chunk_size 100, the whole text fits in one chunk; requesting 5 returns all.
    assert clip_to_max_chunks(full, max_chunks=5, max_chunk_size=100) == full


def test_clip_to_max_chunks_truncates_to_first_n():
    # Force one chunk per paragraph by making chunk_size smaller than two paragraphs concatenated.
    full = "first paragraph\n\nsecond paragraph\n\nthird paragraph"
    result = clip_to_max_chunks(full, max_chunks=2, max_chunk_size=20)
    assert result == "first paragraph\n\nsecond paragraph"


def test_clip_to_max_chunks_deterministic():
    full = "alpha\n\nbeta\n\ngamma\n\ndelta"
    first = clip_to_max_chunks(full, max_chunks=2, max_chunk_size=10)
    second = clip_to_max_chunks(full, max_chunks=2, max_chunk_size=10)
    assert first == second


def test_clip_to_max_chunks_zero_returns_empty_string():
    full = "anything\n\ngoes\n\nhere"
    assert clip_to_max_chunks(full, max_chunks=0, max_chunk_size=10) == ""


@pytest.mark.asyncio
async def test_process_document_max_chunks_clips_text_seen_by_detect():
    """When max_chunks is set, detect_entities sees only the first N chunks' content."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    # Ten distinct paragraphs so chunking will produce many chunks at small chunk_size.
    body = "\n\n".join(f"Paragraph {i} text." for i in range(10))
    page.insert_text((72, 72), body)
    data = doc.tobytes()
    doc.close()

    captured: dict[str, str] = {}

    async def _capture_detect(text, config=None, chunk_timings=None):
        captured["text"] = text
        return []

    try:
        # Force one paragraph per chunk. Inside try so restoration in finally is airtight
        # even if the set_chunk_size call itself raises.
        set_chunk_size(30)
        with (
            patch("app.orchestrator.detect_entities", side_effect=_capture_detect),
            patch("app.orchestrator.verify_occurrences", new_callable=AsyncMock, return_value=[]),
            patch("app.orchestrator.warm_up", new_callable=AsyncMock, return_value=None),
        ):
            result = await process_document(data, "test.pdf", "p", max_chunks=3)

        assert "Paragraph 0" in captured["text"]
        assert "Paragraph 2" in captured["text"]
        assert "Paragraph 3" not in captured["text"]
        assert "Paragraph 9" not in captured["text"]
        # extracted_text on the result reflects the effective (clipped) text.
        assert result.extracted_text == captured["text"]
    finally:
        # Restore default chunk size so downstream tests aren't affected.
        set_chunk_size(2000)


@pytest.mark.asyncio
async def test_process_document_max_chunks_none_processes_full_text():
    """max_chunks=None (the default) leaves the pipeline unchanged."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    body = "\n\n".join(f"Paragraph {i} text." for i in range(5))
    page.insert_text((72, 72), body)
    data = doc.tobytes()
    doc.close()

    captured: dict[str, str] = {}

    async def _capture_detect(text, config=None, chunk_timings=None):
        captured["text"] = text
        return []

    try:
        set_chunk_size(30)
        with (
            patch("app.orchestrator.detect_entities", side_effect=_capture_detect),
            patch("app.orchestrator.verify_occurrences", new_callable=AsyncMock, return_value=[]),
            patch("app.orchestrator.warm_up", new_callable=AsyncMock, return_value=None),
        ):
            await process_document(data, "test.pdf", "p")  # max_chunks omitted

        for i in range(5):
            assert f"Paragraph {i}" in captured["text"]
    finally:
        set_chunk_size(2000)


def test_run_cli_accepts_max_chunks_flag_and_propagates():
    """--max-chunks N on `run` is passed to process_document as max_chunks=N."""
    from app.cli import cli

    runner = CliRunner()

    # Create a tiny fixture file on disk (the CLI requires exists=True).
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("paragraph one\n\nparagraph two\n\nparagraph three")
        tmp_path = f.name

    captured: dict = {}

    async def _fake_process(data, filename, project_id, max_chunks=None):
        captured["max_chunks"] = max_chunks
        from app.orchestrator import ProcessingResult, JobStatus
        return ProcessingResult(
            status=JobStatus.SUCCESS,
            original_filename=filename,
            extracted_text="ok",
            final_text="ok",
        )

    try:
        with patch("app.orchestrator.process_document", side_effect=_fake_process):
            result = runner.invoke(cli, ["run", tmp_path, "--max-chunks", "5"])
        assert result.exit_code == 0, result.output
        assert captured.get("max_chunks") == 5
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_run_cli_defaults_max_chunks_to_none_when_flag_omitted():
    from app.cli import cli

    runner = CliRunner()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("paragraph one")
        tmp_path = f.name

    captured: dict = {}

    async def _fake_process(data, filename, project_id, max_chunks=None):
        captured["max_chunks"] = max_chunks
        from app.orchestrator import ProcessingResult, JobStatus
        return ProcessingResult(
            status=JobStatus.SUCCESS,
            original_filename=filename,
            extracted_text="ok",
            final_text="ok",
        )

    try:
        with patch("app.orchestrator.process_document", side_effect=_fake_process):
            result = runner.invoke(cli, ["run", tmp_path])
        assert result.exit_code == 0, result.output
        assert captured.get("max_chunks") is None
    finally:
        Path(tmp_path).unlink(missing_ok=True)
