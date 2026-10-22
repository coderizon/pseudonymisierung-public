"""Klasse 1 type-preserving generalization for ORG_CLIENT, ORG_UNIT, LOCATION.

Instead of simple pseudonym tokens (ORG_CLIENT_A, LOCATION_B), these categories
get semantic descriptions that preserve the entity type while removing identity:
  - ORG_CLIENT: "mittelstaendisches IT-Unternehmen", "europaeischer Industriekonzern"
  - ORG_UNIT: "Steuerungseinheit", "Fachbereich Digitalisierung", "Leitungsgremium"
  - LOCATION: "Standort Sueddeutschland", "europaeischer Standort"

Uses a single LLM call per batch to generate all descriptions at once.
"""

from __future__ import annotations

import logging

from app.category_config import CategoryConfig, get_config
from app.label_normalization import strict_strip_surrounding_brackets
from app.llm_client import chat_json

logger = logging.getLogger(__name__)

KLASSE1_GENERALIZE_CATEGORIES = {"ORG_CLIENT", "ORG_UNIT", "LOCATION"}


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
    """Ask the LLM to re-emit a Klasse 1 description without surrounding brackets."""
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

SYSTEM_PROMPT = """\
Du bist ein Experte fuer datenschutzkonforme Generalisierung von \
Organisationsdaten in Geschaeftsdokumenten.

Deine Aufgabe: Ersetze identifizierende Entitaeten durch typ-erhaltende \
Beschreibungen. Die Beschreibung muss den analytischen Rahmen bewahren, \
aber die konkrete Identitaet entfernen.

Transformationsregeln:

**ORG_CLIENT** (Mandantenname):
- Typ-erhaltende Ersetzung mit Branche + Groesse + Rechtsformklasse
- Beispiele:
  "AeroTech Solutions GmbH" -> "mittelstaendisches IT-Unternehmen"
  "Stadtwerke Freiburg" -> "kommunaler Energieversorger"
  "DataServices AG" -> "IT-Dienstleistungsunternehmen"
  "Deutsche Bahn AG" -> "grosser Infrastrukturkonzern"
- Rechtsform wird generalisiert (GmbH/AG/Inc -> weggelassen oder \
"Kapitalgesellschaft")
- Branchenzuordnung auf NACE-Sektorebene (z.B. "IT-Unternehmen", \
"Industriekonzern", "Finanzdienstleister")

**ORG_UNIT** (Abteilungen, Gremien, Teams):
- Funktionserhaltende Ersetzung: die organisatorische Funktion bleibt \
erkennbar, der spezifische Name wird abstrahiert
- Beispiele:
  "Abteilung Digital Innovation" -> "Fachbereich Digitalisierung"
  "Capability Center" -> "zentrale Kompetenzeinheit"
  "Steering Committee Q3" -> "Steuerungsgremium"
  "AG Organisationsdesign" -> "Arbeitsgruppe Organisationsentwicklung"
  "Dezernat III" -> "Leitungsebene"
  "Amt fuer Buergerservice" -> "Fachabteilung"
  "Hauptamt" -> "Zentralverwaltung"
- Funktionstypen: Steuerungseinheit, Fachbereich, Leitungsgremium, \
Arbeitsgruppe, Kompetenzeinheit, Stabsstelle, Zentralverwaltung

**LOCATION** (Standorte, Orte):
- Generalisierung auf regionale Ebene
- Deutsche Staedte -> Region: "Standort Sueddeutschland", \
"Standort Norddeutschland", "Standort Westdeutschland", \
"Standort Ostdeutschland", "Standort Mitteldeutschland"
- Oesterreich/Schweiz -> "Standort DACH-Region"
- Europaeische Staedte -> "europaeischer Standort"
- Aussereuropaeisch -> "internationaler Standort"
- Regionale Zuordnungen:
  Sueddeutschland: Bayern, Baden-Wuerttemberg
  Norddeutschland: Schleswig-Holstein, Hamburg, Bremen, Niedersachsen, \
Mecklenburg-Vorpommern
  Westdeutschland: NRW, Rheinland-Pfalz, Saarland, Hessen
  Ostdeutschland: Sachsen, Thueringen, Sachsen-Anhalt, Brandenburg, Berlin
  Mitteldeutschland: Alternative fuer Thueringen/Sachsen/Sachsen-Anhalt-Grenzbereich
- Bei Standortbezeichnungen mit Funktion (z.B. "Werk Sindelfingen"): \
Funktion erhalten -> "Produktionsstandort Sueddeutschland"

WICHTIG:
- Beschreibungen OHNE eckige Klammern zurueckgeben (die werden spaeter hinzugefuegt)
- Kurz und praegnant (2-4 Woerter bevorzugt)
- Konsistent: gleiche Entitaet = gleiche Beschreibung
- Deutsch als Ausgabesprache
- Wenn der Kontext nicht ausreicht fuer eine praezise Einordnung, \
eine allgemeinere aber korrekte Beschreibung waehlen

WICHTIG: Deine Antwort wird direkt an einen JSON-Parser uebergeben.
Gib ausschliesslich ein einziges JSON-Objekt zurueck, NICHTS anderes.
Keine Markdown-Formatierung, keine Kommentare, keine Erklaerungen,
keine Notizen, kein Text ausserhalb des JSON.

Antwortformat:
{
  "descriptions": [
    {"original": "exakter Originaltext", "description": "typ-erhaltende Beschreibung"}
  ]
}"""


async def compute_klasse1_descriptions(
    entities: list[dict],
    config: CategoryConfig | None = None,
) -> dict[str, str]:
    """Compute type-preserving descriptions for Klasse 1 entities.

    Args:
        entities: List of dicts with keys "text", "category", "context".
        config: Category config (loaded from default if None).

    Returns:
        Mapping from original entity text (lowercased) to description string.
        The descriptions do NOT include brackets.
    """
    if config is None:
        config = get_config()

    # Filter to only the categories that need generalization
    to_generalize = [
        e for e in entities
        if e["category"] in KLASSE1_GENERALIZE_CATEGORIES
    ]

    if not to_generalize:
        logger.warning("--- Klasse 1: keine Entitaeten zu generalisieren ---")
        return {}

    # Deduplicate by lowercased text (same entity may appear multiple times)
    seen: set[str] = set()
    unique: list[dict] = []
    for e in to_generalize:
        key = e["text"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(e)

    # Build the user message listing all entities with context
    lines = ["Generalisiere folgende Entitaeten:\n"]
    for i, e in enumerate(unique, 1):
        ctx = e.get("context", "").strip()
        ctx_part = f' (Kontext: "{ctx}")' if ctx else ""
        lines.append(f'{i}. [{e["category"]}] "{e["text"]}"{ctx_part}')

    user_message = "\n".join(lines)
    logger.warning("--- Klasse 1: %d eindeutige Entitaeten, LLM-Aufruf ---", len(unique))

    try:
        result = await chat_json([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ])

        descriptions: dict[str, str] = {}
        entities_by_key = {e["text"].lower(): e for e in unique}
        for item in result.get("descriptions", []):
            original = item.get("original", "")
            description = item.get("description", "")
            if not original or not description:
                continue

            key = original.lower()
            stripped, ok = strict_strip_surrounding_brackets(description)
            if ok:
                descriptions[key] = stripped
                continue

            entity = entities_by_key.get(key)
            if entity is None:
                logger.warning(
                    "BUG-019: malformed bracket shape %r for unknown key %r; "
                    "keeping unchanged.",
                    description, key,
                )
                descriptions[key] = description
                continue

            repaired = await _retry_description_without_brackets(entity, description)
            if repaired is not None:
                descriptions[key] = repaired
            else:
                fallback = _fallback_description(entity["category"])
                logger.warning(
                    "BUG-019: could not repair malformed description %r for entity "
                    "%r (%s); falling back to %r.",
                    description, entity["text"], entity["category"], fallback,
                )
                descriptions[key] = fallback

        # Verify all entities got a description; fall back to category label
        for e in unique:
            key = e["text"].lower()
            if key not in descriptions:
                logger.warning(
                    f"LLM did not return description for '{e['text']}' "
                    f"({e['category']}), using fallback"
                )
                descriptions[key] = _fallback_description(e["category"])

        return descriptions

    except Exception as exc:
        logger.error(f"Klasse 1 transformation failed: {exc}")
        # Fall back to generic descriptions per category
        return {
            e["text"].lower(): _fallback_description(e["category"])
            for e in unique
        }


def _fallback_description(category: str) -> str:
    """Return a safe generic description when LLM inference fails."""
    fallbacks = {
        "ORG_CLIENT": "Unternehmen",
        "ORG_UNIT": "Organisationseinheit",
        "LOCATION": "Standort",
    }
    return fallbacks.get(category, category)
