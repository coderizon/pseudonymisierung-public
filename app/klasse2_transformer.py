"""Klasse 2 structured generalization for trade secret categories.

Two-tier approach:
  Tier 1 (taxonomy-backed): LLM classifies entity text to a taxonomy code,
      then deterministic lookup produces the generalized label.
      Categories: FUNCTION_TITLE (ISCO-08), INDUSTRY (NACE Rev. 2),
      PROCESS_ARTIFACT (APQC PCF), BUSINESS_METRIC (banding tables).

  Tier 2 (structured LLM): LLM produces descriptions directly following
      category-specific transformation rules.
      Categories: PRODUCT, TECHNOLOGY, ORG_PARTNER, TEMPORAL,
      STRATEGY, NEGATIVE_KNOWHOW.

  Skipped: CONTRACT (token pseudonymization), QUASI_ID (flag-only).
"""

from __future__ import annotations

import logging
import re

from app.category_config import CategoryConfig, get_config
from app.label_normalization import strict_strip_surrounding_brackets
from app.llm_client import chat_json
from app.taxonomy_loader import (
    get_taxonomy,
    get_banding,
    get_percentage_rounder,
    Taxonomy,
)


def _normalize_description(label: str, *, context: str) -> str:
    """Strict-strip a description label, logging a warning on malformed shapes.

    `context` is a short free-form string embedded in the warning so the log
    points at the producer (e.g. "taxonomy label for INDUSTRY/62.01" or
    "banding label for metric 'Umsatz 1 Mio'").
    """
    stripped, ok = strict_strip_surrounding_brackets(label)
    if not ok:
        logger.warning(
            "BUG-019: malformed bracket shape %r in %s; keeping unchanged.",
            label, context,
        )
    return stripped

logger = logging.getLogger(__name__)

# Categories where LLM classifies → taxonomy/banding does the generalization
TAXONOMY_CATEGORIES = {"FUNCTION_TITLE", "INDUSTRY", "PROCESS_ARTIFACT"}
BANDING_CATEGORY = "BUSINESS_METRIC"

# Categories where LLM produces the generalized description directly
LLM_CATEGORIES = {
    "PRODUCT", "TECHNOLOGY", "ORG_PARTNER",
    "TEMPORAL", "STRATEGY", "NEGATIVE_KNOWHOW",
}

# Categories handled elsewhere (pseudonymizer token / flag)
SKIP_CATEGORIES = {"CONTRACT", "QUASI_ID"}

# All Klasse 2 categories this module handles
KLASSE2_HANDLED_CATEGORIES = TAXONOMY_CATEGORIES | {BANDING_CATEGORY} | LLM_CATEGORIES

# Map category → taxonomy loader name
_TAXONOMY_MAP = {
    "FUNCTION_TITLE": "ISCO-08",
    "INDUSTRY": "NACE Rev. 2",
    "PROCESS_ARTIFACT": "APQC PCF",
}

# Map category → target generalization level (from categories.yaml)
_DEFAULT_LEVELS = {
    "FUNCTION_TITLE": 2,
    "INDUSTRY": 1,
    "PROCESS_ARTIFACT": 2,  # APQC process group level
}

FALLBACK_DESCRIPTIONS: dict[str, str] = {
    "FUNCTION_TITLE": "Fachkraft",
    "INDUSTRY": "Wirtschaftssektor",
    "PROCESS_ARTIFACT": "Geschaeftsprozess",
    "BUSINESS_METRIC": "Kennzahl",
    "PRODUCT": "Produkt/Loesung",
    "TECHNOLOGY": "Technologieplattform",
    "ORG_PARTNER": "externer Partner",
    "TEMPORAL": "Zeitangabe",
    "STRATEGY": "strategische Initiative",
    "NEGATIVE_KNOWHOW": "eingestelltes Vorhaben",
}

# Euro-marker regex used by _apply_banding_result's override (Tier 1b hygiene,
# fix (i)). Matches `EUR`, `€`, or `Euro` case-insensitively. Word-boundary
# anchors apply only to the letter alternatives; the `€` symbol is already
# non-word and stands alone.
# See docs/engineering/specs/tier-1b-banding-hygiene.md.
_EURO_MARKER_RE = re.compile(r"(?:\bEUR\b|€|\bEuro\b)", re.IGNORECASE)


def _build_taxonomy_options(taxonomy: Taxonomy, level: int) -> str:
    """Format taxonomy entries at the target level as prompt options."""
    entries = taxonomy.at_level(level)
    lines = []
    for entry in sorted(entries, key=lambda e: e.code):
        label = entry.label_de or entry.label_en
        lines.append(f"  {entry.code}: {label}")
    return "\n".join(lines)


def _build_classification_prompt(
    taxonomy_entities: list[dict],
    config: CategoryConfig,
) -> str:
    """Build prompt for Tier 1: taxonomy classification.

    The LLM receives the valid codes at the target level for each taxonomy
    and must pick the best match for each entity.
    """
    sections: list[str] = []

    # Group entities by category
    by_category: dict[str, list[dict]] = {}
    for e in taxonomy_entities:
        by_category.setdefault(e["category"], []).append(e)

    for category, entities in by_category.items():
        tax_name = _TAXONOMY_MAP[category]
        taxonomy = get_taxonomy(tax_name)
        level = _DEFAULT_LEVELS[category]
        cat_def = config.get(category)
        level_name = (
            taxonomy.level_names[level - 1]
            if level <= len(taxonomy.level_names)
            else f"Level {level}"
        )

        options = _build_taxonomy_options(taxonomy, level)
        sections.append(
            f"### {category} -> {tax_name} {level_name}\n"
            f"Gueltige Codes:\n{options}"
        )

    lines = [
        "Du bist ein Experte fuer die Klassifikation von Geschaeftsentitaeten "
        "anhand internationaler Taxonomien.",
        "",
        "Deine Aufgabe: Ordne jede Entitaet dem passendsten Code aus der "
        "jeweiligen Taxonomie zu. Waehle den BESTEN Treffer aus den angegebenen "
        "gueltigen Codes.",
        "",
        "\n\n".join(sections),
        "",
        "WICHTIG: Deine Antwort wird direkt an einen JSON-Parser uebergeben.",
        "Gib ausschliesslich ein einziges JSON-Objekt zurueck, NICHTS anderes.",
        "Keine Markdown-Formatierung, keine Kommentare, keine Erklaerungen,",
        "keine Notizen, kein Text ausserhalb des JSON.",
        "",
        "Antwortformat:",
        "{",
        '  "classifications": [',
        '    {"original": "exakter Text", "category": "CATEGORY", "code": "bester Code"}',
        "  ]",
        "}",
        "",
        "Regeln:",
        "- Waehle den naechstliegenden Code aus den angegebenen Optionen",
        "- Bei Unsicherheit: den allgemeineren Code waehlen",
        "- Jeden Eintrag genau einmal klassifizieren",
    ]
    return "\n".join(lines)


def _build_banding_prompt() -> str:
    """Build prompt for BUSINESS_METRIC extraction."""
    return (
        "Du bist ein Experte fuer die Extraktion von Geschaeftskennzahlen.\n\n"
        "Deine Aufgabe: Extrahiere aus jedem BUSINESS_METRIC-Eintrag den "
        "numerischen Wert und den Kennzahlentyp.\n\n"
        "Kennzahlentypen:\n"
        '- "headcount": Mitarbeiterzahlen, FTE, Teamgroesse, Personalstaerke\n'
        '- "revenue": NUR Jahresumsatz oder Jahresbudget eines UNTERNEHMENS\n'
        '- "monetary": Einzelbetraege, Einsparungen, Kosten, Ausgaben, '
        "Projektbudgets (NICHT Unternehmensumsatz)\n"
        '- "percentage": Prozentwerte, Margen, Quoten, Anteile\n'
        '- "other": Sonstige Zahlen (Projekte, Vorhersagen, Stueckzahlen)\n\n'
        "WICHTIG: revenue ist NUR fuer den Gesamtumsatz/Jahresbudget eines "
        "Unternehmens. Einzelposten wie Einsparungen, Anschaffungskosten, "
        "Projektkosten sind monetary, NICHT revenue.\n\n"
        "WICHTIG: Deine Antwort wird direkt an einen JSON-Parser uebergeben.\n"
        "Gib ausschliesslich ein einziges JSON-Objekt zurueck, NICHTS anderes.\n"
        "Keine Markdown-Formatierung, keine Kommentare, keine Erklaerungen,\n"
        "keine Notizen, kein Text ausserhalb des JSON.\n\n"
        "Antwortformat:\n"
        "{\n"
        '  "metrics": [\n'
        '    {"original": "exakter Text", "value": 100.0, '
        '"metric_type": "headcount|revenue|monetary|percentage|other", '
        '"unit": "FTE|EUR|%|..."}\n'
        "  ]\n"
        "}\n\n"
        "Regeln:\n"
        "- value muss eine Zahl sein (keine Strings)\n"
        "- Bei Abkuerzungen wie 720K -> 720000, 4.2B -> 4200000000\n"
        "- Bei ca./circa/~ Angaben: den geschaetzten Wert nehmen\n"
        "- Jeden Eintrag genau einmal extrahieren"
    )


def _build_llm_generalization_prompt(config: CategoryConfig) -> str:
    """Build prompt for Tier 2: structured LLM generalization."""
    lines = [
        "Du bist ein Experte fuer die Abstraktion vertraulicher "
        "Geschaeftsinformationen in Geschaeftsdokumenten.",
        "",
        "Deine Aufgabe: Generalisiere jede Entitaet gemaess den "
        "kategoriespezifischen Transformationsregeln.",
        "",
        "Transformationsregeln:",
    ]

    for cat_name in sorted(LLM_CATEGORIES):
        cat_def = config.get(cat_name)
        if cat_def:
            lines.append(f"- {cat_name}: {cat_def.transformation.strip()}")

    lines.extend([
        "",
        "WICHTIG: Deine Antwort wird direkt an einen JSON-Parser uebergeben.",
        "Gib ausschliesslich ein einziges JSON-Objekt zurueck, NICHTS anderes.",
        "Keine Markdown-Formatierung, keine Kommentare, keine Erklaerungen,",
        "keine Notizen, kein Text ausserhalb des JSON.",
        "",
        "Antwortformat:",
        "{",
        '  "descriptions": [',
        '    {"original": "exakter Text", "description": "generalisierte Beschreibung"}',
        "  ]",
        "}",
        "",
        "Regeln:",
        "- Beschreibungen OHNE eckige Klammern (die werden spaeter hinzugefuegt)",
        "- Kurz und praegnant (2-5 Woerter bevorzugt)",
        "- Deutsch als Ausgabesprache",
        "- Jeden Eintrag genau einmal generalisieren",
        "- Bei STRATEGY und NEGATIVE_KNOWHOW: hoechste Schutzstufe, "
        "nur Metabeschreibung erhalten",
        "- Bei TEMPORAL: konkrete Jahre durch relative Angabe ersetzen "
        "(z.B. 2018 -> 'mehrere Jahre zuvor', 2025 -> 'aktuelles Jahr'), "
        "Zeitraeume als Dauer belassen (z.B. 'drei Monate' -> 'mehrmonatiger Zeitraum')",
    ])
    return "\n".join(lines)


def _build_entity_list(entities: list[dict]) -> str:
    """Format entities as a numbered list for the LLM user message."""
    lines = []
    for i, e in enumerate(entities, 1):
        ctx = e.get("context", "").strip()
        ctx_part = f' (Kontext: "{ctx}")' if ctx else ""
        lines.append(f'{i}. [{e["category"]}] "{e["text"]}"{ctx_part}')
    return "\n".join(lines)


def _apply_taxonomy_result(
    classifications: list[dict],
    config: CategoryConfig,
) -> dict[str, str]:
    """Convert LLM taxonomy classifications to generalized labels via lookup."""
    descriptions: dict[str, str] = {}

    for item in classifications:
        original = item.get("original", "")
        category = item.get("category", "")
        code = str(item.get("code", ""))

        if not original or not category or not code:
            continue

        tax_name = _TAXONOMY_MAP.get(category)
        if not tax_name:
            continue

        taxonomy = get_taxonomy(tax_name)
        target_level = _DEFAULT_LEVELS.get(category, 1)

        # Try direct lookup at target level first
        entry = taxonomy.lookup(code)
        if entry and entry.level == target_level:
            label = entry.label_de or entry.label_en
            descriptions[original.lower()] = _normalize_description(
                label, context=f"taxonomy label for {category}/{code}"
            )
        elif entry:
            # Walk up to target level
            generalized = taxonomy.generalize(code, target_level)
            if generalized:
                label = generalized.label_de or generalized.label_en
                descriptions[original.lower()] = _normalize_description(
                    label,
                    context=f"generalized taxonomy label for {category}/{code}",
                )
            else:
                logger.warning(
                    f"Could not generalize {code!r} to level {target_level} "
                    f"in {tax_name}, using fallback"
                )
                descriptions[original.lower()] = FALLBACK_DESCRIPTIONS.get(
                    category, category
                )
        else:
            logger.warning(
                f"Code {code!r} not found in {tax_name}, using fallback"
            )
            descriptions[original.lower()] = FALLBACK_DESCRIPTIONS.get(
                category, category
            )

    return descriptions


def _apply_banding_result(metrics: list[dict]) -> dict[str, str]:
    """Convert extracted metric values to banded labels via lookup."""
    descriptions: dict[str, str] = {}
    rounder = get_percentage_rounder()

    for item in metrics:
        original = item.get("original", "")
        value = item.get("value")
        metric_type = item.get("metric_type", "other")

        if not original or value is None:
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            descriptions[original.lower()] = FALLBACK_DESCRIPTIONS["BUSINESS_METRIC"]
            continue

        # Fix (i): if the LLM picked the catch-all metric_type="other" but the
        # entity text carries a Euro marker, flip to "monetary". The override
        # fires only on the 'other' -> 'monetary' transition; other metric_type
        # values (monetary, revenue, headcount, percentage) are respected so
        # the LLM keeps its revenue-vs-monetary and other judgments.
        if metric_type == "other" and _EURO_MARKER_RE.search(original):
            logger.debug(
                "Tier-1b fix (i): metric_type 'other' -> 'monetary' for %r "
                "(euro marker detected).",
                original,
            )
            metric_type = "monetary"

        if metric_type == "headcount":
            band = get_banding("dunbar_headcount").classify(value)
            if band:
                descriptions[original.lower()] = _normalize_description(
                    band.label_de,
                    context=f"headcount banding label for metric {original!r}",
                )
            else:
                descriptions[original.lower()] = f"Personalstaerke (ca. {int(value)})"

        elif metric_type == "revenue":
            band = get_banding("eu_sme_revenue").classify(value)
            if band:
                descriptions[original.lower()] = _normalize_description(
                    band.label_de,
                    context=f"revenue banding label for metric {original!r}",
                )
            else:
                descriptions[original.lower()] = "Finanzkennzahl"

        elif metric_type == "monetary":
            descriptions[original.lower()] = _normalize_description(
                _generalize_monetary(value),
                context=f"monetary generalisation for metric {original!r}",
            )

        elif metric_type == "percentage":
            descriptions[original.lower()] = _normalize_description(
                rounder.round(value),
                context=f"percentage rounding for metric {original!r}",
            )

        else:
            # Generic numeric: round to order of magnitude
            descriptions[original.lower()] = _normalize_description(
                _generalize_generic_number(value),
                context=f"generic numeric generalisation for metric {original!r}",
            )

    return descriptions


def _generalize_monetary(value: float) -> str:
    """Generalize a monetary amount into an order-of-magnitude band."""
    abs_value = abs(value)
    if abs_value < 10:
        return "einstelliger Betrag"
    elif abs_value < 100:
        return "zweistelliger Betrag"
    elif abs_value < 1_000:
        return "dreistelliger Betrag"
    elif abs_value < 10_000:
        return "vierstelliger Betrag"
    elif abs_value < 100_000:
        return "fuenfstelliger Betrag"
    elif abs_value < 1_000_000:
        return "sechsstelliger Betrag"
    elif abs_value < 10_000_000:
        return "niedriger Millionenbetrag"
    elif abs_value < 100_000_000:
        return "zweistelliger Millionenbetrag"
    else:
        return "hoher Millionen-/Milliardenbetrag"


def _generalize_generic_number(value: float) -> str:
    """Generalize a number without a specific banding table."""
    if value < 10:
        return "wenige"
    elif value < 50:
        return "mehrere Dutzend"
    elif value < 100:
        return "knapp hundert"
    elif value < 500:
        return "mehrere hundert"
    elif value < 1000:
        return "knapp tausend"
    else:
        magnitude = 10 ** (len(str(int(value))) - 1)
        rounded = int(round(value / magnitude) * magnitude)
        return f"ca. {rounded:,}".replace(",", ".")


def _dedup_entities(entities: list[dict]) -> list[dict]:
    """Deduplicate entities by lowercased text, keeping first occurrence."""
    seen: set[str] = set()
    unique: list[dict] = []
    for e in entities:
        key = e["text"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


async def compute_klasse2_descriptions(
    entities: list[dict],
    config: CategoryConfig | None = None,
) -> dict[str, str]:
    """Compute structured generalizations for Klasse 2 entities.

    Args:
        entities: List of dicts with keys "text", "category", "context".
        config: Category config (loaded from default if None).

    Returns:
        Mapping from original entity text (lowercased) to generalized
        description string. Descriptions do NOT include brackets.
    """
    if config is None:
        config = get_config()

    # Filter to Klasse 2 categories this module handles
    k2_entities = [
        e for e in entities
        if e["category"] in KLASSE2_HANDLED_CATEGORIES
    ]

    if not k2_entities:
        logger.warning("--- Klasse 2: keine Entitaeten zu generalisieren ---")
        return {}

    # Split by transformation strategy
    taxonomy_entities = _dedup_entities(
        [e for e in k2_entities if e["category"] in TAXONOMY_CATEGORIES]
    )
    banding_entities = _dedup_entities(
        [e for e in k2_entities if e["category"] == BANDING_CATEGORY]
    )
    llm_entities = _dedup_entities(
        [e for e in k2_entities if e["category"] in LLM_CATEGORIES]
    )

    descriptions: dict[str, str] = {}

    # Tier 1a: Taxonomy classification + deterministic lookup
    if taxonomy_entities:
        logger.warning(
            "--- Klasse 2 Tier 1a: Taxonomie-Klassifikation (%d Entitaeten) ---",
            len(taxonomy_entities),
        )
        descriptions.update(
            await _classify_with_taxonomy(taxonomy_entities, config)
        )

    # Tier 1b: Banding extraction + deterministic lookup
    if banding_entities:
        logger.warning(
            "--- Klasse 2 Tier 1b: Metrik-Banding (%d Entitaeten) ---",
            len(banding_entities),
        )
        descriptions.update(
            await _classify_with_banding(banding_entities)
        )

    # Tier 2: Structured LLM generalization
    if llm_entities:
        logger.warning(
            "--- Klasse 2 Tier 2: LLM-Generalisierung (%d Entitaeten) ---",
            len(llm_entities),
        )
        descriptions.update(
            await _generalize_with_llm(llm_entities, config)
        )

    # Ensure every input entity got a description (fallback if missed)
    for e in _dedup_entities(k2_entities):
        key = e["text"].lower()
        if key not in descriptions:
            logger.warning(
                f"No description for '{e['text']}' ({e['category']}), "
                "using fallback"
            )
            descriptions[key] = FALLBACK_DESCRIPTIONS.get(
                e["category"], e["category"]
            )

    return descriptions


async def _classify_with_taxonomy(
    entities: list[dict],
    config: CategoryConfig,
) -> dict[str, str]:
    """Tier 1a: LLM classifies to taxonomy code, then deterministic lookup."""
    system_prompt = _build_classification_prompt(entities, config)
    user_message = (
        "Klassifiziere folgende Entitaeten:\n\n"
        + _build_entity_list(entities)
    )

    try:
        result = await chat_json([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ])
        return _apply_taxonomy_result(
            result.get("classifications", []), config
        )
    except Exception as exc:
        logger.error(f"Taxonomy classification failed: {exc}")
        return {
            e["text"].lower(): FALLBACK_DESCRIPTIONS.get(
                e["category"], e["category"]
            )
            for e in entities
        }


async def _classify_with_banding(entities: list[dict]) -> dict[str, str]:
    """Tier 1b: LLM extracts numeric values, then deterministic banding."""
    system_prompt = _build_banding_prompt()
    user_message = (
        "Extrahiere die Kennzahlen:\n\n"
        + _build_entity_list(entities)
    )

    try:
        result = await chat_json([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ])
        return _apply_banding_result(result.get("metrics", []))
    except Exception as exc:
        logger.error(f"Metric extraction failed: {exc}")
        return {
            e["text"].lower(): FALLBACK_DESCRIPTIONS["BUSINESS_METRIC"]
            for e in entities
        }


_REPAIR_SYSTEM_PROMPT = (
    "Du bist ein Korrektur-Assistent. Der vorige Generalisierungs-Aufruf hat "
    "eine fehlerhaft geklammerte Beschreibung zurueckgegeben. Liefere die "
    "Beschreibung erneut, diesmal OHNE umgebende eckige Klammern.\n\n"
    "Antwortformat:\n"
    '{"description": "bare Beschreibung ohne Klammern"}'
)


async def _retry_description_without_brackets(
    entity: dict,
    malformed: str,
) -> str | None:
    """Ask the LLM to re-emit a description without surrounding brackets.

    Returns the repaired description (already strict-stripped if canonical)
    or None if the call fails or the repaired description is still malformed.
    """
    user_message = (
        f'Entitaet: "{entity["text"]}" (Kategorie: {entity["category"]})\n'
        f'Fehlerhafte vorherige Antwort: "{malformed}"\n'
        f"Bitte erneut liefern, ohne umgebende eckige Klammern."
    )
    try:
        result = await chat_json([
            {"role": "system", "content": _REPAIR_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ])
    except Exception as exc:
        logger.error(
            "BUG-019 repair call failed for entity %r (%s): %s",
            entity["text"], entity["category"], exc,
        )
        return None

    repaired = result.get("description", "")
    if not repaired:
        logger.warning(
            "BUG-019 repair call returned empty description for entity %r (%s).",
            entity["text"], entity["category"],
        )
        return None

    stripped, ok = strict_strip_surrounding_brackets(repaired)
    if not ok:
        logger.warning(
            "BUG-019 repair call returned still-malformed description %r for "
            "entity %r (%s); caller should fall back.",
            repaired, entity["text"], entity["category"],
        )
        return None
    return stripped


async def _generalize_with_llm(
    entities: list[dict],
    config: CategoryConfig,
) -> dict[str, str]:
    """Tier 2: LLM produces descriptions following category-specific rules.

    If the batch call fails (e.g. malformed JSON), retries entities
    individually so a single problematic entity does not lose the batch
    (BUG-013). Afterwards, enforces INV-20 on every returned description
    via strict-strip + per-entity self-repair (BUG-019). On double repair
    failure the entity falls through to FALLBACK_DESCRIPTIONS.
    """
    system_prompt = _build_llm_generalization_prompt(config)

    descriptions = await _try_llm_batch(system_prompt, entities)
    if descriptions is None:
        # Batch failed: retry each entity individually (BUG-013).
        logger.warning(
            "Batch LLM generalization failed, retrying %d entities individually",
            len(entities),
        )
        descriptions = {}
        for entity in entities:
            single = await _try_llm_batch(system_prompt, [entity])
            if single:
                descriptions.update(single)
            else:
                descriptions[entity["text"].lower()] = FALLBACK_DESCRIPTIONS.get(
                    entity["category"], entity["category"]
                )

    # BUG-019: enforce INV-20 on every description returned by the LLM.
    entities_by_key = {e["text"].lower(): e for e in entities}
    normalized: dict[str, str] = {}
    for key, description in descriptions.items():
        stripped, ok = strict_strip_surrounding_brackets(description)
        if ok:
            normalized[key] = stripped
            continue

        entity = entities_by_key.get(key)
        if entity is None:
            logger.warning(
                "BUG-019: malformed bracket shape %r for unknown key %r; "
                "keeping unchanged.",
                description, key,
            )
            normalized[key] = description
            continue

        repaired = await _retry_description_without_brackets(entity, description)
        if repaired is not None:
            normalized[key] = repaired
        else:
            fallback = FALLBACK_DESCRIPTIONS.get(
                entity["category"], entity["category"]
            )
            logger.warning(
                "BUG-019: could not repair malformed description %r for entity "
                "%r (%s); falling back to %r.",
                description, entity["text"], entity["category"], fallback,
            )
            normalized[key] = fallback

    return normalized


async def _try_llm_batch(
    system_prompt: str,
    entities: list[dict],
) -> dict[str, str] | None:
    """Attempt a single LLM generalization call. Returns None on failure."""
    user_message = (
        "Generalisiere folgende Entitaeten:\n\n"
        + _build_entity_list(entities)
    )
    try:
        result = await chat_json([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ])
        descriptions: dict[str, str] = {}
        for item in result.get("descriptions", []):
            original = item.get("original", "")
            description = item.get("description", "")
            if original and description:
                descriptions[original.lower()] = description
        return descriptions
    except Exception as exc:
        logger.error(f"LLM generalization failed: {exc}")
        return None
