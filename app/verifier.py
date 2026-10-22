"""Pass 2: LLM-based verification of NER candidates in context."""

from __future__ import annotations

import json
import logging
import time

from app.category_config import CategoryConfig, get_config, remap_category
from app.llm_client import chat_json
from app.pseudonymizer import Chunk, DetectedEntity, EntityOccurrence
from app.span_resolver import filter_protected, resolve_occurrences_in_chunk

logger = logging.getLogger(__name__)


async def verify_occurrences(
    chunks: list[Chunk],
    occurrences: list[EntityOccurrence],
    config: CategoryConfig | None = None,
    chunk_timings: list[dict] | None = None,
    protected_spans: list[tuple[int, int]] | None = None,
) -> list[EntityOccurrence]:
    """Verify candidate occurrences using the LLM in context. Returns surviving occurrences.

    The LLM call is value-level: candidates are deduped by (text, category) per chunk, and
    KEEP/REJECT decisions are applied back to all occurrences sharing that key in that chunk.
    CORRECT decisions rewrite the entity.text/category and re-resolve to occurrences in the
    chunk. MISSED entities (new Pass-2 additions) are resolved by string-matching the chunk
    text for word-boundary-anchored matches and emit one occurrence per match.

    protected_spans (optional) enforces INV-19 on occurrences that Pass 2 introduces:
    CORRECT-rewritten spans and MISSED-found spans are filtered against the protected-span
    list before joining the survivors, so the Pass-2 LLM cannot re-introduce a corruption
    into a timestamp or other protected region even if it reports the offending text as
    MISSED or a CORRECT target. Pass 1 occurrences are assumed to have been filtered by
    the caller before reaching this function; protected_spans is re-applied here only to
    the new Pass-2 material.
    """
    if config is None:
        config = get_config()
    if protected_spans is None:
        protected_spans = []

    if not occurrences:
        return []

    verification_prompt = config.build_verification_prompt()
    valid_categories = {c.name for c in config.ner_categories()}

    # Group occurrences by containing chunk. Each occurrence lives in exactly one chunk
    # under the normal chunker invariant. If an occurrence does not map to any chunk
    # (only reachable if chunk_text's invariant-violation fallback fired, see that
    # function's logger.error), we log and drop it so the caller can see the drift.
    by_chunk: dict[int, list[EntityOccurrence]] = {i: [] for i in range(len(chunks))}
    unassigned = 0
    for occ in occurrences:
        assigned = False
        for i, chunk in enumerate(chunks):
            if chunk.start <= occ.start < chunk.end:
                by_chunk[i].append(occ)
                assigned = True
                break
        if not assigned:
            unassigned += 1
    if unassigned:
        logger.warning(
            "verify_occurrences: %d occurrence(s) did not map to any chunk and were dropped; "
            "likely indicates a chunk_text invariant violation upstream.",
            unassigned,
        )

    survivors: list[EntityOccurrence] = []
    chunks_ok = 0
    chunks_repaired = 0
    chunks_failed = 0

    for i, chunk in enumerate(chunks):
        chunk_occs = by_chunk[i]
        if not chunk_occs:
            continue

        # Dedupe by (text.lower(), category) for the LLM call.
        unique_keys: dict[tuple[str, str], DetectedEntity] = {}
        for occ in chunk_occs:
            key = (occ.entity.text.lower(), occ.entity.category)
            if key not in unique_keys:
                unique_keys[key] = occ.entity

        candidate_list = [
            {"text": entity.text, "category": entity.category} for entity in unique_keys.values()
        ]

        t0 = time.time()
        try:
            result = await chat_json([
                {"role": "system", "content": verification_prompt},
                {"role": "user", "content": (
                    "Textabschnitt:\n\n"
                    f"{chunk.text}\n\n"
                    "Kandidaten-Entitaeten:\n"
                    f"{json.dumps(candidate_list, ensure_ascii=False)}"
                )},
            ])
            if result.get("_repaired"):
                chunks_repaired += 1
            else:
                chunks_ok += 1
        except Exception as e:
            chunks_failed += 1
            logger.error(
                "--- Verification (occurrences) chunk %d/%d FAILED (keeping occurrences): %s ---",
                i + 1, len(chunks), e,
            )
            # Fallback: keep all occurrences in this chunk on LLM failure (conservative;
            # avoids silent entity loss). Already-Pass-1-filtered, so no protected-span
            # re-filter needed here.
            survivors.extend(chunk_occs)
            if chunk_timings is not None:
                chunk_timings.append({
                    "chunk": i + 1,
                    "chars": len(chunk.text),
                    "candidates": len(chunk_occs),
                    "duration": round(time.time() - t0, 2),
                    "verified": len(chunk_occs),
                })
            continue

        # Build per-key decisions from the LLM verdicts.
        decisions: dict[tuple[str, str], tuple[str, DetectedEntity | None]] = {}
        for item in result.get("verified", []):
            orig_text = item.get("text", "")
            orig_cat = item.get("category", "")
            status = item.get("status", "keep")
            orig_key = (orig_text.lower(), orig_cat)

            if status == "reject":
                decisions[orig_key] = ("reject", None)
                continue
            if status == "correct":
                new_text = item.get("corrected_text", orig_text)
                new_cat_raw = item.get("corrected_category", orig_cat)
                new_cat = remap_category(new_cat_raw, valid_categories)
                if new_cat is None:
                    decisions[orig_key] = ("reject", None)
                    continue
                original = unique_keys.get(orig_key)
                new_entity = DetectedEntity(
                    text=new_text,
                    category=new_cat,
                    context=original.context if original else "",
                )
                decisions[orig_key] = ("correct", new_entity)
                continue
            # keep (default)
            remapped = remap_category(orig_cat, valid_categories)
            if remapped is None:
                decisions[orig_key] = ("reject", None)
                continue
            original = unique_keys.get(orig_key)
            kept_entity = (
                original
                if original and original.category == remapped
                else DetectedEntity(
                    text=orig_text,
                    category=remapped,
                    context=original.context if original else "",
                )
            )
            decisions[orig_key] = ("keep", kept_entity)

        chunk_survivors: list[EntityOccurrence] = []

        # Apply decisions to each occurrence.
        for occ in chunk_occs:
            key = (occ.entity.text.lower(), occ.entity.category)
            decision = decisions.get(key)
            if decision is None:
                # LLM did not reference this (text, category); treat as KEEP with no change.
                chunk_survivors.append(occ)
                continue
            action, new_entity = decision
            if action == "reject":
                continue
            if action == "keep" and new_entity is not None:
                chunk_survivors.append(
                    EntityOccurrence(entity=new_entity, start=occ.start, end=occ.end)
                )
                continue
            if action == "correct" and new_entity is not None:
                # Re-resolve the corrected entity text inside the chunk and run the
                # protected-span filter so a CORRECT rewrite cannot re-introduce a
                # substitution into a protected region (INV-19).
                new_occs = resolve_occurrences_in_chunk(new_entity, chunk.text, chunk.start)
                new_occs = filter_protected(new_occs, protected_spans)
                chunk_survivors.extend(new_occs)
                continue

        # MISSED entities: LLM says it found these additionally in this chunk.
        for item in result.get("missed", []):
            missed_text = item.get("text", "")
            missed_cat_raw = item.get("category", "")
            if not missed_text:
                continue
            remapped = remap_category(missed_cat_raw, valid_categories)
            if remapped is None:
                continue
            missed_entity = DetectedEntity(
                text=missed_text,
                category=remapped,
                context=item.get("context", ""),
            )
            # Apply the protected-span filter (INV-19) so the LLM cannot re-introduce a
            # corruption into a protected region via MISSED.
            missed_occs = resolve_occurrences_in_chunk(missed_entity, chunk.text, chunk.start)
            missed_occs = filter_protected(missed_occs, protected_spans)
            chunk_survivors.extend(missed_occs)

        survivors.extend(chunk_survivors)

        if chunk_timings is not None:
            chunk_timings.append({
                "chunk": i + 1,
                "chars": len(chunk.text),
                "candidates": len(chunk_occs),
                "duration": round(time.time() - t0, 2),
                "verified": len(chunk_survivors),
            })

    logger.warning(
        "=== Verification (occurrences) summary: %d chunks, %d OK, %d repaired, %d failed, %d verified occurrences ===",
        len(chunks), chunks_ok, chunks_repaired, chunks_failed, len(survivors),
    )
    return survivors
