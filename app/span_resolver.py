"""Span resolution: turn DetectedEntity into EntityOccurrence with full-text offsets.

Pure, deterministic, no I/O, no LLM. Three responsibilities:

1. `resolve_occurrences_in_chunk`: given a DetectedEntity and the chunk it was found in
   (plus the chunk's start offset in the effective document text), produce one
   EntityOccurrence per word-boundary-anchored, case-insensitive match.

2. `apply_longest_match_policy`: if two occurrences overlap and one is a proper substring
   of the other, drop the shorter.

3. `filter_protected`: drop occurrences whose span intersects any protected span.
"""

from __future__ import annotations

import logging
import re

from app.pseudonymizer import DetectedEntity, EntityOccurrence

logger = logging.getLogger(__name__)


def resolve_occurrences_in_chunk(
    entity: DetectedEntity,
    chunk_text: str,
    chunk_start: int,
) -> list[EntityOccurrence]:
    """Return one EntityOccurrence per case-insensitive, word-boundary match of entity.text
    inside chunk_text, with spans translated to absolute offsets (chunk_start + match_offset).
    """
    if not entity.text:
        return []

    pattern = re.compile(r"(?<!\w)" + re.escape(entity.text) + r"(?!\w)", re.IGNORECASE)
    return [
        EntityOccurrence(
            entity=entity,
            start=chunk_start + m.start(),
            end=chunk_start + m.end(),
        )
        for m in pattern.finditer(chunk_text)
    ]


def apply_longest_match_policy(
    occurrences: list[EntityOccurrence],
) -> list[EntityOccurrence]:
    """Drop occurrences whose span is a proper substring of another occurrence's span.

    A span (a, b) is a proper substring of (c, d) when c <= a and b <= d and (a, b) != (c, d).

    Notes for consumers (pseudonymize_by_span in particular):
    - Partial-overlap cases (two spans share characters but neither engulfs the other)
      are preserved intentionally. Both survive.
    - Equal-span occurrences produced by distinct entities (for example, the same surface
      text under two different categories) are also preserved. Downstream span-addressed
      substitution must tolerate same-position duplicates or resolve the conflict explicitly.
    """
    if len(occurrences) < 2:
        return list(occurrences)

    survivors: list[EntityOccurrence] = []
    for occ in occurrences:
        engulfing = next(
            (
                other for other in occurrences
                if other is not occ
                and _proper_substring((occ.start, occ.end), (other.start, other.end))
            ),
            None,
        )
        if engulfing is None:
            survivors.append(occ)
        else:
            logger.debug(
                "apply_longest_match_policy: dropping occurrence %r (%s) at "
                "span (%d, %d); engulfed by %r (%s) at span (%d, %d).",
                occ.entity.text, occ.entity.category, occ.start, occ.end,
                engulfing.entity.text, engulfing.entity.category,
                engulfing.start, engulfing.end,
            )
    return survivors


def filter_protected(
    occurrences: list[EntityOccurrence],
    protected_spans: list[tuple[int, int]],
) -> list[EntityOccurrence]:
    """Drop occurrences whose [start, end) intersects any protected span."""
    if not protected_spans:
        return list(occurrences)
    return [occ for occ in occurrences if not _intersects_any(occ.start, occ.end, protected_spans)]


def _proper_substring(inner: tuple[int, int], outer: tuple[int, int]) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1] and inner != outer


def _intersects_any(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    for s, e in spans:
        if start < e and s < end:
            return True
    return False
