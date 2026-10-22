"""Category configuration loader and prompt builder.

Loads the 17-category taxonomy from categories.yaml and generates
NER/generalization prompts dynamically, preventing prompt/config drift.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CategoryDef:
    name: str
    class_: int
    active: bool
    action: str  # pseudonymize, suppress, generalize, flag
    protection_level: str
    description: str
    examples: dict[str, list[str]]
    negative_examples: dict[str, list[str]] = field(default_factory=dict)
    transformation: str = ""
    reference_standard: str = ""
    generalization_level: int | None = None
    pseudonym_prefix: str = ""


@dataclass
class CategoryConfig:
    profile: str
    exclusions: list[str]
    categories: dict[str, CategoryDef]

    def all_categories(self) -> list[CategoryDef]:
        return [c for c in self.categories.values() if c.active]

    def pii_categories(self) -> list[CategoryDef]:
        return [c for c in self.categories.values() if c.active and c.class_ == 1]

    def get(self, name: str) -> CategoryDef | None:
        return self.categories.get(name)

    def ner_categories(self) -> list[CategoryDef]:
        """Categories that participate in NER detection (excludes flag-only)."""
        return [c for c in self.categories.values() if c.active and c.action != "flag"]

    def build_ner_prompt(self) -> str:
        lines = [
            "Du bist ein Experte fuer die Erkennung personenbezogener Daten und "
            "Geschaeftsgeheimnisse in Geschaeftsdokumenten (Deutsch und Englisch).",
            "",
            "Deine Aufgabe: Identifiziere ALLE schuetzenswerten Entitaeten im gegebenen "
            "Text und gib sie als JSON zurueck.",
            "",
            "Es gibt zwei Klassen:",
            "- Klasse 1 (PII): Personenbezogene Daten, immer pseudonymisieren",
            "- Klasse 2 (Geschaeftsgeheimnisse): Kontextabhaengig generalisieren",
            "",
            "Kategorien:",
        ]

        for cat in self.ner_categories():
            class_label = "Klasse 1, PII" if cat.class_ == 1 else "Klasse 2, Geschaeftsgeheimnis"
            all_examples = []
            for lang_examples in cat.examples.values():
                all_examples.extend(lang_examples)
            example_str = ", ".join(f'"{e}"' for e in all_examples[:5])
            negatives_clause = _format_negatives_clause(cat)
            lines.append(
                f"- {cat.name} [{class_label}]: {cat.description}. "
                f"Beispiele: {example_str}{negatives_clause}"
            )

        lines.extend([
            "",
            "NICHT pseudonymisieren (Ausschlussliste):",
        ])
        for excl in self.exclusions:
            lines.append(f"- {excl}")

        lines.extend([
            "",
            "WICHTIG: Deine Antwort wird direkt an einen JSON-Parser uebergeben.",
            "Gib ausschliesslich ein einziges JSON-Objekt zurueck, NICHTS anderes.",
            "Keine Markdown-Formatierung, keine Kommentare, keine Erklaerungen,",
            "keine Notizen, kein Text ausserhalb des JSON.",
            "",
            "Antwortformat:",
            "{",
            '  "entities": [',
            '    {"text": "exakter Text aus dem Dokument", "category": "PERSON", "context": "kurzer Kontextsatz"}',
            "  ]",
            "}",
            "",
            "Regeln:",
            "- Extrahiere den EXAKTEN Text wie er im Dokument vorkommt",
            "- Erfasse ALLE Vorkommen, auch wenn sie sich wiederholen",
            "- Bei Unsicherheit: lieber einschliessen als auslassen (Recall > Precision)",
            "- Keine allgemeinen Begriffe pseudonymisieren (siehe Ausschlussliste)",
            "- Keine fiktiven Entitaeten erfinden, nur was tatsaechlich im Text steht",
            "- Dokument kann Deutsch, Englisch oder gemischt sein",
        ])

        return "\n".join(lines)

    def build_verification_prompt(self) -> str:
        """Build the Pass 2 verification prompt for entity candidates."""
        lines = [
            "Du bist ein Experte fuer die Verifikation erkannter Entitaeten in "
            "Geschaeftsdokumenten (Deutsch und Englisch).",
            "",
            "Du erhaeltst einen Textabschnitt und eine Liste von Kandidaten-Entitaeten, "
            "die in Pass 1 erkannt wurden. Deine Aufgabe:",
            "",
            "1. Pruefe jeden Kandidaten im Kontext des Textabschnitts.",
            "2. Entscheide fuer jeden: keep (bestaetigt), reject (Falsch-Positiv), "
            "oder correct (Text oder Kategorie korrigieren).",
            "3. Finde zusaetzlich FEHLENDE Entitaeten, die Pass 1 uebersehen hat.",
            "",
            "Pruefkriterien:",
            "- Ist der extrahierte Text ein vollstaendiges Wort oder Name, "
            "NICHT ein Teilwort innerhalb eines laengeren Worts?",
            "- Passt die zugewiesene Kategorie zum Kontext?",
            "- Steht der Text auf der Ausschlussliste (allgemeine Begriffe)?",
            "- Bei Unsicherheit: Status 'keep' waehlen (Recall > Precision, INV-07).",
            "",
            "Kategorien:",
        ]

        for cat in self.ner_categories():
            class_label = "Klasse 1, PII" if cat.class_ == 1 else "Klasse 2, Geschaeftsgeheimnis"
            negatives_clause = _format_negatives_clause(cat)
            lines.append(f"- {cat.name} [{class_label}]: {cat.description}{negatives_clause}")

        lines.extend([
            "",
            "NICHT pseudonymisieren (Ausschlussliste):",
        ])
        for excl in self.exclusions:
            lines.append(f"- {excl}")

        lines.extend([
            "",
            "WICHTIG: Deine Antwort wird direkt an einen JSON-Parser uebergeben.",
            "Gib ausschliesslich ein einziges JSON-Objekt zurueck, NICHTS anderes.",
            "Keine Markdown-Formatierung, keine Kommentare, keine Erklaerungen,",
            "keine Notizen, kein Text ausserhalb des JSON.",
            "",
            "Antwortformat:",
            "{",
            '  "verified": [',
            '    {"text": "...", "category": "...", "status": "keep|reject|correct", '
            '"corrected_text": "...", "corrected_category": "...", "reason": "..."}',
            "  ],",
            '  "missed": [',
            '    {"text": "exakter Text aus dem Dokument", "category": "...", "context": "Kontextsatz"}',
            "  ]",
            "}",
            "",
            "Regeln:",
            "- 'verified' muss JEDEN Kandidaten enthalten (mit status keep, reject oder correct)",
            "- Bei 'correct': corrected_text und/oder corrected_category angeben",
            "- Bei 'keep' oder 'reject': corrected_text und corrected_category weglassen",
            "- 'missed': nur Entitaeten, die tatsaechlich im Text stehen und nicht in der Kandidatenliste sind",
            "- Keine fiktiven Entitaeten erfinden",
            "- reason ist optional, nur bei reject oder correct nuetzlich",
        ])

        return "\n".join(lines)


def _format_negatives_clause(cat: CategoryDef) -> str:
    """Render a ``. NICHT <CAT>: "x", "y"`` suffix for inline prompt inclusion.

    Used by both build_ner_prompt and build_verification_prompt so the two
    rendering sites cannot drift. Returns an empty string when the category
    has no negative_examples, so callers can append unconditionally without
    a guard.

    Both languages flow into the same rendered list, mirroring the handling
    of positive examples today. No truncation: negative_examples lists are
    short by design (each entry is a distinct shape class), and showing them
    all is the behaviour this slice intends.
    """
    if not cat.negative_examples:
        return ""
    all_negatives: list[str] = []
    for lang_negatives in cat.negative_examples.values():
        all_negatives.extend(lang_negatives)
    if not all_negatives:
        return ""
    negatives_str = ", ".join(f'"{n}"' for n in all_negatives)
    return f". NICHT {cat.name}: {negatives_str}"


def _parse_category(name: str, data: dict[str, Any]) -> CategoryDef:
    return CategoryDef(
        name=name,
        class_=data["class"],
        active=data.get("active", True),
        action=data.get("action", "pseudonymize"),
        protection_level=data.get("protection_level", "medium"),
        description=data.get("description", ""),
        examples=data.get("examples", {}),
        negative_examples=data.get("negative_examples", {}),
        transformation=data.get("transformation", ""),
        reference_standard=data.get("reference_standard", ""),
        generalization_level=data.get("generalization_level"),
        pseudonym_prefix=data.get("pseudonym_prefix", name),
    )


def load(path: str | Path | None = None) -> CategoryConfig:
    if path is None:
        path = os.environ.get(
            "CATEGORIES_PATH",
            str(Path(__file__).parent / "categories.yaml"),
        )
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    categories = {}
    for name, cat_data in raw.get("categories", {}).items():
        categories[name] = _parse_category(name, cat_data)

    return CategoryConfig(
        profile=raw.get("profile", "standard"),
        exclusions=raw.get("exclusions", []),
        categories=categories,
    )


_config: CategoryConfig | None = None


def get_config() -> CategoryConfig:
    global _config
    if _config is None:
        _config = load()
    return _config


# Map common LLM-invented categories back to valid ones.
# The model sometimes reclassifies entities into categories that don't exist
# in categories.yaml (e.g., CONTACT -> EMAIL). This table catches those.
CATEGORY_SYNONYMS: dict[str, str] = {
    "EMAIL": "CONTACT",
    "E_MAIL": "CONTACT",
    "PHONE": "CONTACT",
    "TELEPHONE": "CONTACT",
    "PHONE_NUMBER": "CONTACT",
    "ADDRESS": "CONTACT",
    "NAME": "PERSON",
    "COMPANY": "ORG_PARTNER",
    "ORGANIZATION": "ORG_PARTNER",
    "ORGANISATION": "ORG_PARTNER",
    "DATE": "TEMPORAL",
    "DATUM": "TEMPORAL",
    "CITY": "LOCATION",
    "COUNTRY": "LOCATION",
    "REGION": "LOCATION",
    "NEGATIVE_KNOWLEDGE": "NEGATIVE_KNOWHOW",
}


def remap_category(category: str, valid_categories: set[str]) -> str | None:
    """Remap an LLM-invented category to a valid one, or return None if unmappable."""
    import logging
    if category in valid_categories:
        return category
    remapped = CATEGORY_SYNONYMS.get(category.upper())
    if remapped and remapped in valid_categories:
        logging.getLogger(__name__).info(f"Remapped category '{category}' -> '{remapped}'")
        return remapped
    return None
