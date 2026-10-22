"""Taxonomy and banding table loader for generalization hierarchies.

Loads reference data from config/taxonomies/ YAML files and provides
lookup APIs for deterministic generalization of Klasse 2 entities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.config import TAXONOMIES_DIR


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TaxonomyEntry:
    code: str
    level: int
    label_de: str
    label_en: str
    children: list[TaxonomyEntry] = field(default_factory=list)
    parent_code: str | None = None


@dataclass
class Taxonomy:
    name: str
    source: str
    levels: int
    level_names: list[str]
    entries: list[TaxonomyEntry]
    _by_code: dict[str, TaxonomyEntry] = field(default_factory=dict, repr=False)

    def lookup(self, code: str) -> TaxonomyEntry | None:
        """Return the entry for *code*, or ``None``."""
        return self._by_code.get(code)

    def at_level(self, level: int) -> list[TaxonomyEntry]:
        """Return all entries at the given hierarchy level."""
        return [e for e in self._by_code.values() if e.level == level]

    def parent_of(self, code: str) -> TaxonomyEntry | None:
        """Return the parent entry, or ``None`` for top-level entries."""
        entry = self._by_code.get(code)
        if entry is None or entry.parent_code is None:
            return None
        return self._by_code.get(entry.parent_code)

    def generalize(self, code: str, target_level: int) -> TaxonomyEntry | None:
        """Walk up the hierarchy until *target_level* is reached."""
        entry = self._by_code.get(code)
        while entry is not None and entry.level > target_level:
            entry = self._by_code.get(entry.parent_code) if entry.parent_code else None
        if entry is not None and entry.level == target_level:
            return entry
        return None


@dataclass
class BandEntry:
    min_value: float | None
    max_value: float | None
    label_de: str
    label_en: str


@dataclass
class BandingTable:
    name: str
    metric_type: str
    bands: list[BandEntry]

    def classify(self, value: float) -> BandEntry | None:
        """Return the band that *value* falls into, or ``None``."""
        for band in self.bands:
            above_min = band.min_value is None or value >= band.min_value
            below_max = band.max_value is None or value < band.max_value
            if above_min and below_max:
                return band
        return None


@dataclass
class PercentageRounder:
    """Handles 5er-Rundung: rounding to nearest step with qualitative bands."""

    name: str
    rounding_step: int
    qualitative_bands: list[BandEntry]

    def round(self, value: float) -> str:
        """Return a generalized string for the percentage *value*.

        Qualitative bands take precedence over numeric rounding.
        """
        for band in self.qualitative_bands:
            above_min = band.min_value is None or value >= band.min_value
            below_max = band.max_value is None or value < band.max_value
            if above_min and below_max:
                return band.label_de
        step = self.rounding_step
        rounded = int(step * round(value / step))
        return f"ca. {rounded}%"


# ---------------------------------------------------------------------------
# YAML parsing helpers
# ---------------------------------------------------------------------------

def _parse_entries(
    raw_entries: list[dict[str, Any]],
    parent_code: str | None = None,
) -> list[TaxonomyEntry]:
    """Recursively parse nested YAML entries into TaxonomyEntry trees."""
    result: list[TaxonomyEntry] = []
    for item in raw_entries:
        entry = TaxonomyEntry(
            code=str(item["code"]),
            level=item["level"],
            label_de=item.get("label_de", ""),
            label_en=item.get("label_en", ""),
            parent_code=parent_code,
        )
        if "children" in item:
            entry.children = _parse_entries(item["children"], parent_code=entry.code)
        result.append(entry)
    return result


def _flatten(entries: list[TaxonomyEntry]) -> dict[str, TaxonomyEntry]:
    """Build a flat code -> entry index from a nested tree."""
    index: dict[str, TaxonomyEntry] = {}
    for entry in entries:
        index[entry.code] = entry
        index.update(_flatten(entry.children))
    return index


def _parse_bands(raw_bands: list[dict[str, Any]]) -> list[BandEntry]:
    return [
        BandEntry(
            min_value=b.get("min"),
            max_value=b.get("max"),
            label_de=b.get("label_de", ""),
            label_en=b.get("label_en", ""),
        )
        for b in raw_bands
    ]


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

TAXONOMY_FILES: dict[str, str] = {
    "NACE Rev. 2": "nace_rev2.yaml",
    "ISCO-08": "isco08.yaml",
    "APQC PCF": "apqc_pcf.yaml",
}

BANDING_FILES: dict[str, str] = {
    "dunbar_headcount": "dunbar_headcount.yaml",
    "eu_sme_revenue": "eu_sme_revenue.yaml",
}

PERCENTAGE_FILE = "percentage_rounding.yaml"


def load_taxonomy(name: str, base_dir: str | Path | None = None) -> Taxonomy:
    """Load a hierarchical taxonomy by its canonical name."""
    filename = TAXONOMY_FILES.get(name)
    if filename is None:
        raise KeyError(f"Unknown taxonomy: {name!r}. Known: {list(TAXONOMY_FILES)}")
    path = Path(base_dir or TAXONOMIES_DIR) / filename
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    meta = raw.get("metadata", {})
    entries = _parse_entries(raw.get("entries", []))
    by_code = _flatten(entries)
    return Taxonomy(
        name=meta.get("name", name),
        source=meta.get("source", ""),
        levels=meta.get("levels", 0),
        level_names=meta.get("level_names", []),
        entries=entries,
        _by_code=by_code,
    )


def load_banding(name: str, base_dir: str | Path | None = None) -> BandingTable:
    """Load a banding table by its canonical name."""
    filename = BANDING_FILES.get(name)
    if filename is None:
        raise KeyError(f"Unknown banding table: {name!r}. Known: {list(BANDING_FILES)}")
    path = Path(base_dir or TAXONOMIES_DIR) / filename
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    meta = raw.get("metadata", {})
    return BandingTable(
        name=meta.get("name", name),
        metric_type=meta.get("metric_type", ""),
        bands=_parse_bands(raw.get("bands", [])),
    )


def load_percentage_rounder(base_dir: str | Path | None = None) -> PercentageRounder:
    """Load the percentage rounding configuration."""
    path = Path(base_dir or TAXONOMIES_DIR) / PERCENTAGE_FILE
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    meta = raw.get("metadata", {})
    return PercentageRounder(
        name=meta.get("name", "5er-Rundung"),
        rounding_step=raw.get("rounding_step", 5),
        qualitative_bands=_parse_bands(raw.get("qualitative_bands", [])),
    )


# ---------------------------------------------------------------------------
# Cached singletons
# ---------------------------------------------------------------------------

_taxonomy_cache: dict[str, Taxonomy] = {}
_banding_cache: dict[str, BandingTable] = {}
_percentage_rounder: PercentageRounder | None = None


def get_taxonomy(name: str) -> Taxonomy:
    """Return a cached Taxonomy instance."""
    if name not in _taxonomy_cache:
        _taxonomy_cache[name] = load_taxonomy(name)
    return _taxonomy_cache[name]


def get_banding(name: str) -> BandingTable:
    """Return a cached BandingTable instance."""
    if name not in _banding_cache:
        _banding_cache[name] = load_banding(name)
    return _banding_cache[name]


def get_percentage_rounder() -> PercentageRounder:
    """Return a cached PercentageRounder instance."""
    global _percentage_rounder
    if _percentage_rounder is None:
        _percentage_rounder = load_percentage_rounder()
    return _percentage_rounder
