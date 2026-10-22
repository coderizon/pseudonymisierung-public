"""Protected spans: regions of the effective document where entity replacement must not occur.

Producing a single sorted, merged list of (start, end) tuples. Any entity occurrence whose
span intersects any protected span is dropped before replacement. Today we seed the pattern
list with timestamps only (BUG-017); adding a new protected pattern is a regex append.
"""

from __future__ import annotations

import re


# Matches HH:MM, H:MM, HH:MM:SS variants with word boundaries (two-digit
# minute/second required; hour/minute-prefix may be one or two digits).
# Conservative by design: the pattern is loose enough to catch meeting
# timestamps reliably, at the cost of matching same-shaped text such as
# two-digit ratios ("10:20") that could be legitimate content. See the
# "over-protection" test in tests/test_protected_spans.py.
TIMESTAMP_PATTERN = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")

# The authoritative list of patterns that produce protected regions.
PROTECTED_PATTERNS: list[re.Pattern[str]] = [TIMESTAMP_PATTERN]


def build_protected_spans(
    text: str, patterns: list[re.Pattern[str]]
) -> list[tuple[int, int]]:
    """Return a sorted, merged list of (start, end) spans where no entity replacement may land.

    Pure function: deterministic on inputs, no I/O.
    """
    if not text or not patterns:
        return []

    raw: list[tuple[int, int]] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            raw.append((match.start(), match.end()))

    return _merge_overlapping(sorted(raw))


def _merge_overlapping(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Collapse touching or overlapping spans. Input must be sorted by start."""
    if not spans:
        return []

    merged: list[tuple[int, int]] = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged
