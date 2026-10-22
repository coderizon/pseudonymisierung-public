"""Workflow orchestrator: chains ingestion -> NER -> pseudonymization -> generalization."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum

from app.category_config import CategoryConfig, get_config
from app.ingestion import ExtractedDocument, ingest_document
from app.klasse1_transformer import compute_klasse1_descriptions
from app.klasse2_transformer import compute_klasse2_descriptions
from app.mapping_store import MappingStore
from app.llm_client import warm_up
from app.pseudonymizer import (
    detect_entities,
    pseudonymize_by_span,
    DetectedEntity,
    EntityOccurrence,
)
from app.pseudonymizer import chunk_text as _chunk_text_for_clip
from app.pseudonymizer import chunk_text  # re-imported locally for pipeline use
# Imported as module to read the mutable MAX_CHUNK_SIZE global at call time;
# a `from ... import MAX_CHUNK_SIZE` would bind the initial value and miss
# runtime changes made via set_chunk_size().
import app.pseudonymizer as _pseudonymizer_module
from app.verifier import verify_occurrences
from app.sprecher import ergaenze_sprecher
from app.protected_spans import build_protected_spans, PROTECTED_PATTERNS
from app.span_resolver import (
    resolve_occurrences_in_chunk,
    apply_longest_match_policy,
    filter_protected,
)
logger = logging.getLogger(__name__)


def clip_to_max_chunks(
    full_text: str,
    max_chunks: int | None,
    max_chunk_size: int,
) -> str:
    """Return a prefix of full_text ending at the last kept chunk.

    With the span-first refactor, the chunker returns Chunks that carry their
    original offsets; the effective document is a true prefix of full_text,
    preserving the original inter-chunk separators.
    """
    if max_chunks is None:
        return full_text
    if max_chunks <= 0:
        return ""
    chunks = _chunk_text_for_clip(full_text, max_chunk_size)
    if len(chunks) <= max_chunks:
        return full_text
    return full_text[: chunks[max_chunks - 1].end]


class JobStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class StepResult:
    name: str
    label: str
    status: StepStatus = StepStatus.PENDING
    duration_seconds: float = 0.0
    detail: str = ""


@dataclass
class ProcessingResult:
    status: JobStatus
    original_filename: str
    # Per-step data
    extracted_text: str = ""
    extracted_blocks: list[dict] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    pass1_candidates: list[dict] = field(default_factory=list)
    pseudonymized_text: str = ""
    pre_generalization_text: str = ""
    final_text: str = ""
    # Summaries
    entities_found: int = 0
    entities_pseudonymized: int = 0
    warnings: list[str] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    mappings: list[dict] = field(default_factory=list)
    steps: list[StepResult] = field(default_factory=list)
    # Klasse 1 type-preserving descriptions (original_lower -> description)
    klasse1_descriptions: dict[str, str] = field(default_factory=dict)
    # Klasse 2 structured generalizations (original_lower -> description)
    klasse2_descriptions: dict[str, str] = field(default_factory=dict)
    # Per-chunk timing
    pass1_chunk_timings: list[dict] = field(default_factory=list)
    pass2_chunk_timings: list[dict] = field(default_factory=list)


async def process_document(
    data: bytes,
    filename: str,
    project_id: str,
    max_chunks: int | None = None,
) -> ProcessingResult:
    """Full pipeline: ingest -> detect entities -> pseudonymize -> generalize."""
    pipeline_start = time.time()
    config = get_config()
    result = ProcessingResult(
        status=JobStatus.FAILED,
        original_filename=filename,
    )
    store = MappingStore(project_id)

    # Define pipeline steps
    step_ingest = StepResult(name="ingest", label="Dokumenten-Ingestion")
    step_ner = StepResult(name="ner", label="Entitätserkennung (NER)")
    step_k1 = StepResult(name="klasse1", label="Klasse-1-Generalisierung")
    step_k2 = StepResult(name="klasse2", label="Klasse-2-Generalisierung (strukturiert)")
    step_pseudo = StepResult(name="pseudonymize", label="Pseudonymisierung")
    result.steps = [step_ingest, step_ner, step_k1, step_k2, step_pseudo]

    try:
        # Ensure model is loaded with correct context size before pipeline starts
        await warm_up()

        # Step 1: Ingest
        step_ingest.status = StepStatus.RUNNING
        t0 = time.time()
        logger.warning("=== Step 1/6: Dokumenten-Ingestion ===")
        extracted = ingest_document(data, filename)
        result.warnings.extend(extracted.warnings)
        result.extracted_text = extracted.full_text
        result.extracted_blocks = [
            {"text": b.text, "source": b.source} for b in extracted.blocks
        ]
        step_ingest.duration_seconds = round(time.time() - t0, 2)
        step_ingest.detail = f"{len(extracted.blocks)} Textblöcke, {len(result.extracted_text)} Zeichen"
        step_ingest.status = StepStatus.DONE

        if not result.extracted_text.strip():
            result.status = JobStatus.FAILED
            result.warnings.append("No text could be extracted from the document.")
            step_ner.status = StepStatus.FAILED
            step_pseudo.status = StepStatus.FAILED
            step_k1.status = StepStatus.SKIPPED
            step_k2.status = StepStatus.SKIPPED
            return result

        # Apply --max-chunks clipping if requested (dev flag; no-op when None).
        # Note: max_chunks=0 is degenerate and not the intended use — it yields
        # an empty effective text. The empty-text guard above runs on the ingested
        # text (pre-clipping), so downstream steps will run against the empty
        # string; they handle it gracefully but the job's entity count will be 0.
        if max_chunks is not None:
            effective_text = clip_to_max_chunks(
                result.extracted_text, max_chunks, _pseudonymizer_module.MAX_CHUNK_SIZE,
            )
            if effective_text != result.extracted_text:
                result.warnings.append(
                    f"--max-chunks={max_chunks} active: pipeline input clipped "
                    f"from {len(result.extracted_text)} to {len(effective_text)} chars."
                )
            result.extracted_text = effective_text

        # Step 2: Entity detection (two-pass) — span-first
        step_ner.status = StepStatus.RUNNING
        t0 = time.time()

        effective_text = result.extracted_text
        chunks = chunk_text(effective_text, _pseudonymizer_module.MAX_CHUNK_SIZE)
        protected_spans = build_protected_spans(effective_text, PROTECTED_PATTERNS)

        logger.warning("=== Step 2/6: NER Pass 1 (Entitaetserkennung) ===")
        pass1_entities = await detect_entities(
            effective_text, config, chunk_timings=result.pass1_chunk_timings,
        )
        pass1_count = len(pass1_entities)
        result.pass1_candidates = [
            {"text": e.text, "category": e.category, "context": e.context}
            for e in pass1_entities
        ]

        # Span resolution: per chunk, one occurrence per word-boundary match. Each Pass 1
        # entity is searched in every chunk because the LLM's flat return lost chunk provenance.
        raw_occurrences: list[EntityOccurrence] = []
        resolved_entity_keys: set[tuple[str, str]] = set()
        for chunk in chunks:
            for entity in pass1_entities:
                new_occs = resolve_occurrences_in_chunk(entity, chunk.text, chunk.start)
                if new_occs:
                    resolved_entity_keys.add((entity.text.lower(), entity.category))
                raw_occurrences.extend(new_occs)

        # Orphan Pass 1 entities: the LLM returned a text that word-boundary matched nowhere
        # in the effective document (hallucination or casing drift). Log for visibility;
        # the entity is silently dropped below because no occurrence was emitted.
        orphan_entities = [
            e for e in pass1_entities
            if (e.text.lower(), e.category) not in resolved_entity_keys
        ]
        if orphan_entities:
            orphan_preview = ", ".join(
                f"{e.text!r}({e.category})" for e in orphan_entities[:5]
            )
            logger.warning(
                "--- %d Pass 1 entity(ies) resolved to zero occurrences (dropped): %s%s ---",
                len(orphan_entities), orphan_preview,
                " ..." if len(orphan_entities) > 5 else "",
            )

        # Longest-match policy, then protected-span filter (the BUG-017 fix point).
        occurrences = apply_longest_match_policy(raw_occurrences)
        occurrences = filter_protected(occurrences, protected_spans)

        logger.warning(
            "=== Step 3/6: NER Pass 2 (Verifikation, %d Kandidaten -> %d Occurrences nach Filter) ===",
            pass1_count, len(occurrences),
        )
        verified_occurrences = await verify_occurrences(
            chunks=chunks,
            occurrences=occurrences,
            config=config,
            chunk_timings=result.pass2_chunk_timings,
            protected_spans=protected_spans,
        )

        # BUG-021: reconcile any proper-substring overlaps introduced by Pass 2
        # CORRECT rewrites or MISSED additions. This is the first point in the
        # pipeline where every producer's occurrences are visible in a single
        # list, so it is the right place to enforce INV-21 before
        # pseudonymize_by_span. Filter is idempotent; re-running it on Pass 1
        # survivors that were already filtered is a no-op.
        verified_occurrences = apply_longest_match_policy(verified_occurrences)
        # Ein bestätigter Name in wiederkehrenden Sprecherzeilen darf nicht
        # durch abschnittsweise Verwerfung teilweise im Klartext stehen bleiben.
        verified_occurrences = ergaenze_sprecher(effective_text, verified_occurrences)

        # Surface verified unique entities as the legacy `entities` list for compatibility
        # with verbose output (findings.json uses unique entity values).
        seen_for_report: set[tuple[str, str]] = set()
        verified_unique: list[DetectedEntity] = []
        for occ in verified_occurrences:
            key = (occ.entity.text.lower(), occ.entity.category)
            if key in seen_for_report:
                continue
            seen_for_report.add(key)
            verified_unique.append(occ.entity)

        result.entities_found = len(verified_unique)
        result.entities = [
            {"text": e.text, "category": e.category, "context": e.context}
            for e in verified_unique
        ]
        step_ner.duration_seconds = round(time.time() - t0, 2)
        cat_counts: dict[str, int] = {}
        for e in verified_unique:
            cat_counts[e.category] = cat_counts.get(e.category, 0) + 1
        cat_summary = ", ".join(f"{v}× {k}" for k, v in sorted(cat_counts.items()))
        step_ner.detail = (
            f"Pass 1: {pass1_count} Kandidaten, "
            f"Pass 2: {len(verified_unique)} Werte / {len(verified_occurrences)} Occurrences "
            f"({cat_summary})"
        )
        step_ner.status = StepStatus.DONE

        # Step 2b/2c: Klasse 1 and Klasse 2 transformers — unchanged, value-level on deduped entities.
        step_k1.status = StepStatus.RUNNING
        t0 = time.time()
        k1_entities = [
            {"text": e.text, "category": e.category, "context": e.context}
            for e in verified_unique
        ]
        logger.warning("=== Step 4/6: Klasse-1-Generalisierung ===")
        klasse1_descriptions = await compute_klasse1_descriptions(k1_entities, config)
        result.klasse1_descriptions = klasse1_descriptions
        step_k1.duration_seconds = round(time.time() - t0, 2)
        step_k1.detail = f"{len(klasse1_descriptions)} Beschreibungen generiert"
        step_k1.status = StepStatus.DONE

        step_k2.status = StepStatus.RUNNING
        t0 = time.time()
        k2_entities = [
            {"text": e.text, "category": e.category, "context": e.context}
            for e in verified_unique
        ]
        logger.warning("=== Step 5/6: Klasse-2-Generalisierung (strukturiert) ===")
        klasse2_descriptions = await compute_klasse2_descriptions(k2_entities, config)
        result.klasse2_descriptions = klasse2_descriptions
        step_k2.duration_seconds = round(time.time() - t0, 2)
        step_k2.detail = f"{len(klasse2_descriptions)} Beschreibungen generiert"
        step_k2.status = StepStatus.DONE

        # Step 3: Pseudonymization — span-addressed.
        step_pseudo.status = StepStatus.RUNNING
        t0 = time.time()
        logger.warning(
            "=== Step 6/6: Pseudonymisierung (span-addressed, %d occurrences) ===",
            len(verified_occurrences),
        )
        pseudonymized = pseudonymize_by_span(
            effective_text,
            verified_occurrences,
            store,
            config,
            klasse1_descriptions=klasse1_descriptions,
            klasse2_descriptions=klasse2_descriptions,
        )
        result.entities_pseudonymized = len(verified_unique)
        result.pseudonymized_text = pseudonymized
        result.pre_generalization_text = pseudonymized

        # Occurrence variant tracking (unchanged; records block-level casings/sources).
        for block in extracted.blocks:
            for entity in verified_unique:
                for match in re.finditer(re.escape(entity.text), block.text, re.IGNORECASE):
                    store.save_occurrence(
                        entity.text, match.group(), extracted.filename, block.source,
                    )

        step_pseudo.duration_seconds = round(time.time() - t0, 2)
        step_pseudo.detail = f"{len(verified_occurrences)} Ersetzungen (span-addressed)"
        step_pseudo.status = StepStatus.DONE

        # INV-22: final_text is the span-addressed pseudonymized text.
        # No further LLM pass touches or mints [...] tokens.
        result.final_text = pseudonymized

        result.status = JobStatus.SUCCESS
        result.mappings = store.get_all_mappings()

    except Exception as e:
        logger.exception(f"Processing failed for {filename}")
        result.status = JobStatus.FAILED
        result.warnings.append(f"Processing error: {e}")
        # Mark remaining steps as failed
        for step in result.steps:
            if step.status in (StepStatus.PENDING, StepStatus.RUNNING):
                step.status = StepStatus.FAILED
    finally:
        store.close()
        result.processing_time_seconds = round(time.time() - pipeline_start, 2)

    return result
