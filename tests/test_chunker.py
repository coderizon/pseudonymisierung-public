"""Tests for the new Chunk-returning chunk_text contract."""

from __future__ import annotations

from app.pseudonymizer import Chunk, chunk_text


def test_chunk_text_returns_list_of_chunks():
    result = chunk_text("short text", max_size=1000)
    assert len(result) == 1
    assert isinstance(result[0], Chunk)
    assert result[0].text == "short text"
    assert result[0].start == 0
    assert result[0].end == len("short text")


def test_chunk_text_preserves_original_full_text_by_concatenation_with_separator():
    full = "alpha\n\nbeta\n\ngamma"
    chunks = chunk_text(full, max_size=7)  # each paragraph alone fits; pairs do not
    # Each chunk.text is one of the paragraphs, in order.
    assert [c.text for c in chunks] == ["alpha", "beta", "gamma"]
    # Offsets reference positions in the original full text.
    for c in chunks:
        assert full[c.start : c.end] == c.text


def test_chunk_text_start_ends_are_non_decreasing():
    full = "\n\n".join(f"para{i}" for i in range(6))
    chunks = chunk_text(full, max_size=6)
    starts = [c.start for c in chunks]
    assert starts == sorted(starts)


def test_chunk_end_is_exclusive_and_matches_text_length():
    chunks = chunk_text("hello", max_size=10)
    assert chunks[0].end - chunks[0].start == len(chunks[0].text)


def test_chunk_text_single_chunk_when_input_fits():
    full = "fits in one chunk"
    chunks = chunk_text(full, max_size=1000)
    assert len(chunks) == 1
    assert chunks[0].text == full
    assert chunks[0].start == 0


def test_chunk_text_hard_cut_path_preserves_round_trip():
    """No paragraph/line/sentence boundary + input exceeds max_size forces the hard-cut
    fallback. Every chunk must still round-trip through its (start, end) offsets."""
    full = "a" * 50
    chunks = chunk_text(full, max_size=10)
    assert len(chunks) > 1
    for c in chunks:
        assert full[c.start : c.end] == c.text
    # The pieces should cover the whole text end-to-end.
    assert "".join(c.text for c in chunks) == full


def test_chunk_text_hard_cut_with_repeated_content_round_trips():
    """Hard-cut with repeated short content: text.find could find earlier occurrences of a
    repeated piece, but the cursor discipline in chunk_text must keep starts monotonic and
    round-trip every chunk.
    """
    full = "AAAB" * 8  # 32 chars of a repeated pattern
    chunks = chunk_text(full, max_size=4)
    assert len(chunks) >= 2
    starts = [c.start for c in chunks]
    assert starts == sorted(starts)
    for c in chunks:
        assert full[c.start : c.end] == c.text
