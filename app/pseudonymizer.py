"""Entity recognition and pseudonymization engine using local LLM."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from app.category_config import CategoryConfig, get_config, remap_category
from app.klasse1_transformer import KLASSE1_GENERALIZE_CATEGORIES
from app.klasse2_transformer import KLASSE2_HANDLED_CATEGORIES
from app.label_normalization import strict_strip_surrounding_brackets
from app.llm_client import chat_json
from app.mapping_store import MappingStore

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 2000  # characters per LLM call
MAX_CHUNK_SIZE = DEFAULT_CHUNK_SIZE


def set_chunk_size(size: int) -> None:
    """Override the chunk size at runtime."""
    global MAX_CHUNK_SIZE
    MAX_CHUNK_SIZE = size

SUPPRESS_MARKER = "[KONTAKTDATEN ENTFERNT]"


@dataclass
class DetectedEntity:
    text: str
    category: str
    context: str


@dataclass
class EntityOccurrence:
    """One located occurrence of a DetectedEntity in the effective document text.

    start and end are offsets into the effective document text (the full_text
    the pipeline is currently operating on, which may be clipped by --max-chunks).
    end is exclusive.
    """
    entity: DetectedEntity
    start: int
    end: int


@dataclass
class Chunk:
    """A chunk of the effective document text, with its start offset in that text.

    end is an exclusive computed boundary.
    """
    text: str
    start: int

    @property
    def end(self) -> int:
        return self.start + len(self.text)


async def _erkennungsantwort(nachrichten):
    """Eindeutige Listenform akzeptieren, fehlerhafte Antworten einmal wiederholen."""
    for versuch in range(2):
        try:
            result = await chat_json(nachrichten)
            if isinstance(result, list):
                result = {"entities": result}
            if not isinstance(result, dict) or not isinstance(result.get("entities"), list):
                raise ValueError("Entitätserkennung liefert keine gültige Kandidatenliste")
            for ent in result["entities"]:
                if (not isinstance(ent, dict) or not isinstance(ent.get("text"), str)
                        or not ent["text"].strip() or not isinstance(ent.get("category"), str)):
                    raise ValueError("Ungültiger Eintrag in der Kandidatenliste")
            return result
        except Exception:
            if versuch:
                raise
            logger.warning("Erkennungsantwort ungültig oder nicht verfügbar; Abschnitt wird einmal wiederholt.")


async def detect_entities(
    text: str,
    config: CategoryConfig | None = None,
    chunk_timings: list[dict] | None = None,
) -> list[DetectedEntity]:
    """Run NER on text using the local LLM. Chunks long texts."""
    if config is None:
        config = get_config()

    ner_prompt = config.build_ner_prompt()
    chunks = chunk_text(text, MAX_CHUNK_SIZE)
    all_entities: list[DetectedEntity] = []
    seen: set[tuple[str, str]] = set()

    valid_categories = {c.name for c in config.ner_categories()}

    chunks_ok = 0
    chunks_repaired = 0
    chunks_failed = 0

    for i, chunk in enumerate(chunks):
        t0 = time.time()
        entities_in_chunk = 0
        try:
            result = await _erkennungsantwort([
                {"role": "system", "content": ner_prompt},
                {"role": "user", "content": f"Analysiere folgenden Text:\n\n{chunk.text}"},
            ])
            for ent in result.get("entities", []):
                category = ent.get("category", "")
                category = remap_category(category, valid_categories)
                if category is None:
                    logger.warning(f"LLM returned unknown category '{ent.get('category', '')}', skipping")
                    continue
                entity_text = ent["text"]
                if not is_word_boundary_match(text, entity_text):
                    logger.debug(f"Rejected partial-word match: '{entity_text}'")
                    continue
                key = (entity_text.lower(), category)
                if key not in seen:
                    seen.add(key)
                    all_entities.append(DetectedEntity(
                        text=entity_text,
                        category=category,
                        context=ent.get("context", ""),
                    ))
                    entities_in_chunk += 1
            if result.get("_repaired"):
                chunks_repaired += 1
            else:
                chunks_ok += 1
            _preview = f"{chunk.text[:50]}...{chunk.text[-50:]}" if len(chunk.text) > 110 else chunk.text
            logger.warning(
                "--- NER chunk %d/%d OK (%d entities): %s ---",
                i + 1, len(chunks), entities_in_chunk, _preview,
            )
        except Exception as e:
            chunks_failed += 1
            logger.error("--- NER chunk %d/%d FAILED: %s ---", i + 1, len(chunks), e)
        if chunk_timings is not None:
            chunk_timings.append({
                "chunk": i + 1,
                "chars": len(chunk.text),
                "duration": round(time.time() - t0, 2),
                "entities": entities_in_chunk,
            })

    logger.warning(
        "=== NER summary: %d chunks, %d OK, %d repaired, %d failed, %d entities ===",
        len(chunks), chunks_ok, chunks_repaired, chunks_failed, len(all_entities),
    )
    if chunks_failed:
        raise RuntimeError(
            f"Entitätserkennung unvollständig: {chunks_failed} von {len(chunks)} "
            "Textabschnitten konnten auch nach Wiederholung nicht geprüft werden. "
            "Es wird kein freigabefähiges Ergebnis ausgegeben."
        )
    return all_entities


def pseudonymize_by_span(
    text: str,
    occurrences: list[EntityOccurrence],
    store: MappingStore,
    config: CategoryConfig | None = None,
    klasse1_descriptions: dict[str, str] | None = None,
    klasse2_descriptions: dict[str, str] | None = None,
) -> str:
    """Replace each occurrence's span with the pseudonym or description for its entity.

    Substitutions are applied in reverse order of start offset so earlier spans remain
    valid as the string mutates after them. One mapping-store decision per unique
    (text, category) is cached across occurrences so each entity value receives one pseudonym.
    """
    # BUG-021 sink guard: enforce INV-21 at the consumer boundary. If upstream
    # filtering is correct this is a no-op. If any producer path slipped an
    # engulfed span through, drop it here so the splice loop never applies a
    # proper-substring overlap (which would splice the inner span into the
    # already-emitted outer token and corrupt the output).
    # Local import avoids a circular dependency with app.span_resolver.
    from app.span_resolver import apply_longest_match_policy

    filtered = apply_longest_match_policy(occurrences)
    if len(filtered) != len(occurrences):
        dropped = len(occurrences) - len(filtered)
        logger.warning(
            "pseudonymize_by_span sink guard (BUG-021): dropped %d occurrence(s) "
            "whose span was a proper substring of another surviving span. "
            "INV-21 was violated by the caller; upstream filtering should run "
            "apply_longest_match_policy before reaching this function.",
            dropped,
        )
        occurrences = filtered

    if config is None:
        config = get_config()
    if klasse1_descriptions is None:
        klasse1_descriptions = {}
    if klasse2_descriptions is None:
        klasse2_descriptions = {}

    # One replacement token per unique (text.lower(), category); cache across occurrences.
    replacement_cache: dict[tuple[str, str], str | None] = {}

    def _replacement_for(entity: DetectedEntity) -> str | None:
        """Return the token (including brackets or SUPPRESS_MARKER) or None for 'flag' (skip)."""
        key = (entity.text.lower(), entity.category)
        if key in replacement_cache:
            return replacement_cache[key]

        cat_def = config.get(entity.category)
        action = cat_def.action if cat_def else "pseudonymize"

        if action == "suppress":
            store.save_mapping(entity.text, SUPPRESS_MARKER, entity.category, entity.context)
            replacement_cache[key] = SUPPRESS_MARKER
            return SUPPRESS_MARKER

        if action == "flag":
            replacement_cache[key] = None
            return None

        key_lower = key[0]
        existing = store.get_pseudonym(entity.text)
        if existing:
            pseudonym = existing
        elif (
            entity.category in KLASSE1_GENERALIZE_CATEGORIES
            and key_lower in klasse1_descriptions
        ):
            pseudonym = klasse1_descriptions[key_lower]
            store.save_mapping(entity.text, pseudonym, entity.category, entity.context)
        elif (
            entity.category in KLASSE2_HANDLED_CATEGORIES
            and key_lower in klasse2_descriptions
        ):
            pseudonym = klasse2_descriptions[key_lower]
            store.save_mapping(entity.text, pseudonym, entity.category, entity.context)
        else:
            raw_prefix = (cat_def.pseudonym_prefix if cat_def else None) or entity.category
            prefix = raw_prefix.upper()
            pseudonym = store.get_next_pseudonym(prefix)
            store.save_mapping(entity.text, pseudonym, entity.category, entity.context)

        stripped, well_formed = strict_strip_surrounding_brackets(pseudonym)
        if stripped != pseudonym:
            logger.warning(
                "pseudonymize_by_span sink guard (BUG-019): pseudonym for "
                "entity %r (category %s) was already bracketed (%r). "
                "Emitting without re-wrapping. Upstream producer should "
                "strip per INV-20.",
                entity.text, entity.category, pseudonym,
            )
            token = f"[{stripped}]" if well_formed else pseudonym
        else:
            token = f"[{pseudonym}]"

        replacement_cache[key] = token
        return token

    # Decide each span's replacement (None means: skip this occurrence).
    # Same-span duplicates (two distinct entities resolving to the same (start, end) —
    # possible when apply_longest_match_policy preserves equal-span occurrences from
    # different (text, category) pairs) are resolved with a "first occurrence wins"
    # policy: only the first surviving occurrence at a given (start, end) contributes
    # a substitution. Applying both would silently corrupt output because the second
    # would operate on already-substituted text.
    span_subs: list[tuple[int, int, str]] = []
    claimed_spans: set[tuple[int, int]] = set()
    for occ in occurrences:
        span_key = (occ.start, occ.end)
        if span_key in claimed_spans:
            logger.warning(
                "pseudonymize_by_span: dropping duplicate occurrence at span %s (entity %r, category %s); "
                "first occurrence at this span wins.",
                span_key, occ.entity.text, occ.entity.category,
            )
            continue
        token = _replacement_for(occ.entity)
        if token is None:
            continue
        claimed_spans.add(span_key)
        span_subs.append((occ.start, occ.end, token))

    # Apply in reverse order of start so earlier spans stay valid.
    span_subs.sort(key=lambda t: t[0], reverse=True)
    result = text
    for start, end, token in span_subs:
        result = result[:start] + token + result[end:]
    return result


def is_word_boundary_match(text: str, entity_text: str) -> bool:
    """Check whether entity_text appears at word boundaries in text."""
    pattern = re.compile(r"(?<!\w)" + re.escape(entity_text) + r"(?!\w)", re.IGNORECASE)
    return bool(pattern.search(text))


def chunk_text(text: str, max_size: int) -> list[Chunk]:
    """Split text into chunks respecting semantic boundaries; each chunk carries its start offset.

    Strategy unchanged (cascading fallback):
    1. Split on paragraph boundaries (\\n\\n)
    2. If a segment still exceeds max_size, split on line boundaries (\\n)
    3. If a single line still exceeds max_size, split on sentence boundaries (. )
    4. Hard cut at max_size as a last resort
    """
    if not text:
        return []
    if len(text) <= max_size:
        return [Chunk(text=text, start=0)]

    string_chunks = _chunk_text_strings(text, max_size)

    result: list[Chunk] = []
    cursor = 0
    for piece in string_chunks:
        # Locate each piece at or after `cursor` in the original text so offsets are accurate
        # even when the chunker produces separator variants that differ slightly from the source.
        idx = text.find(piece, cursor)
        if idx < 0:
            # Invariant violation: the chunker is supposed to produce only substrings
            # of the original text. Log loudly so this is visible in ops, then align at
            # cursor to keep offsets monotonic and avoid a cascading failure.
            logger.error(
                "chunk_text invariant violation: piece (%d chars) not found in text at or after cursor=%d; "
                "falling back to cursor alignment — chunk offsets for this piece will not round-trip.",
                len(piece), cursor,
            )
            idx = cursor
        result.append(Chunk(text=piece, start=idx))
        cursor = idx + len(piece)
    return result


def _chunk_text_strings(text: str, max_size: int) -> list[str]:
    """Return the raw string chunks (internal implementation of the cascading strategy)."""
    raw_chunks = _split_and_merge(text.split("\n\n"), "\n\n", max_size)

    result: list[str] = []
    for chunk in raw_chunks:
        if len(chunk) <= max_size:
            result.append(chunk)
        else:
            sub = _split_and_merge(chunk.split("\n"), "\n", max_size)
            for part in sub:
                if len(part) <= max_size:
                    result.append(part)
                else:
                    result.extend(_hard_split(part, max_size))
    return result


def _split_and_merge(segments: list[str], separator: str, max_size: int) -> list[str]:
    """Merge segments back together up to max_size, joining with separator."""
    chunks = []
    current = ""
    sep_len = len(separator)

    for seg in segments:
        needed = sep_len + len(seg) if current else len(seg)
        if len(current) + needed > max_size and current:
            chunks.append(current)
            current = seg
        else:
            current = current + separator + seg if current else seg

    if current:
        chunks.append(current)

    return chunks


def _hard_split(text: str, max_size: int) -> list[str]:
    """Split an oversized segment at sentence boundaries, then hard-cut."""
    sentences = re.split(r"(?<=\. )", text)
    merged = _split_and_merge(sentences, "", max_size)

    result = []
    for chunk in merged:
        while len(chunk) > max_size:
            result.append(chunk[:max_size])
            chunk = chunk[max_size:]
        if chunk:
            result.append(chunk)

    return result
