"""Tests for taxonomy_loader: hierarchical taxonomies, banding tables, percentage rounding."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.taxonomy_loader import (
    BandingTable,
    PercentageRounder,
    Taxonomy,
    load_banding,
    load_percentage_rounder,
    load_taxonomy,
)

# Use the real config/taxonomies/ data files (committed reference data).
TAXONOMIES_DIR = Path(__file__).resolve().parent.parent / "config" / "taxonomies"


# ---------------------------------------------------------------------------
# NACE Rev. 2
# ---------------------------------------------------------------------------

class TestNACE:
    @pytest.fixture(scope="class")
    def nace(self) -> Taxonomy:
        return load_taxonomy("NACE Rev. 2", base_dir=TAXONOMIES_DIR)

    def test_loads(self, nace: Taxonomy) -> None:
        assert nace.name == "NACE Rev. 2"
        assert nace.levels == 4

    def test_level_1_has_21_sections(self, nace: Taxonomy) -> None:
        sections = nace.at_level(1)
        assert len(sections) == 21

    def test_lookup_section_j(self, nace: Taxonomy) -> None:
        entry = nace.lookup("J")
        assert entry is not None
        assert entry.label_de == "Information und Kommunikation"
        assert entry.level == 1

    def test_lookup_division_62(self, nace: Taxonomy) -> None:
        entry = nace.lookup("62")
        assert entry is not None
        assert entry.level == 2
        assert entry.parent_code == "J"

    def test_lookup_class_62_01(self, nace: Taxonomy) -> None:
        entry = nace.lookup("62.01")
        assert entry is not None
        assert entry.level == 4
        assert "Programmierung" in entry.label_de

    def test_parent_of(self, nace: Taxonomy) -> None:
        parent = nace.parent_of("62")
        assert parent is not None
        assert parent.code == "J"

    def test_generalize_to_level_1(self, nace: Taxonomy) -> None:
        result = nace.generalize("62.01", target_level=1)
        assert result is not None
        assert result.code == "J"

    def test_generalize_to_level_2(self, nace: Taxonomy) -> None:
        result = nace.generalize("62.01", target_level=2)
        assert result is not None
        assert result.code == "62"

    def test_generalize_unknown_code(self, nace: Taxonomy) -> None:
        assert nace.generalize("ZZ.99", target_level=1) is None

    def test_top_level_has_no_parent(self, nace: Taxonomy) -> None:
        assert nace.parent_of("A") is None


# ---------------------------------------------------------------------------
# ISCO-08
# ---------------------------------------------------------------------------

class TestISCO:
    @pytest.fixture(scope="class")
    def isco(self) -> Taxonomy:
        return load_taxonomy("ISCO-08", base_dir=TAXONOMIES_DIR)

    def test_loads(self, isco: Taxonomy) -> None:
        assert isco.name == "ISCO-08"
        assert isco.levels == 4

    def test_level_1_has_10_major_groups(self, isco: Taxonomy) -> None:
        majors = isco.at_level(1)
        assert len(majors) == 10

    def test_level_2_sub_major_groups(self, isco: Taxonomy) -> None:
        sub_majors = isco.at_level(2)
        assert len(sub_majors) == 43

    def test_lookup_major_group_1(self, isco: Taxonomy) -> None:
        entry = isco.lookup("1")
        assert entry is not None
        assert "Fuehrungskraefte" in entry.label_de

    def test_lookup_sub_major_33(self, isco: Taxonomy) -> None:
        entry = isco.lookup("33")
        assert entry is not None
        assert entry.level == 2
        assert entry.parent_code == "3"

    def test_generalize_to_level_2(self, isco: Taxonomy) -> None:
        # 3322 (Handelsvertreter) -> 33 (sub-major)
        result = isco.generalize("3322", target_level=2)
        assert result is not None
        assert result.code == "33"

    def test_generalize_to_level_1(self, isco: Taxonomy) -> None:
        result = isco.generalize("3322", target_level=1)
        assert result is not None
        assert result.code == "3"


# ---------------------------------------------------------------------------
# APQC PCF
# ---------------------------------------------------------------------------

class TestAPQC:
    @pytest.fixture(scope="class")
    def apqc(self) -> Taxonomy:
        return load_taxonomy("APQC PCF", base_dir=TAXONOMIES_DIR)

    def test_loads(self, apqc: Taxonomy) -> None:
        assert apqc.name == "APQC PCF"
        assert apqc.levels == 2

    def test_level_1_has_13_categories(self, apqc: Taxonomy) -> None:
        cats = apqc.at_level(1)
        assert len(cats) == 13

    def test_lookup_category_3(self, apqc: Taxonomy) -> None:
        entry = apqc.lookup("3.0")
        assert entry is not None
        assert "vermarkten" in entry.label_de.lower() or "Market" in entry.label_en

    def test_lookup_process_group(self, apqc: Taxonomy) -> None:
        entry = apqc.lookup("3.2")
        assert entry is not None
        assert entry.level == 2
        assert entry.parent_code == "3.0"


# ---------------------------------------------------------------------------
# Dunbar headcount banding
# ---------------------------------------------------------------------------

class TestDunbar:
    @pytest.fixture(scope="class")
    def dunbar(self) -> BandingTable:
        return load_banding("dunbar_headcount", base_dir=TAXONOMIES_DIR)

    def test_loads(self, dunbar: BandingTable) -> None:
        assert dunbar.name == "Dunbar/OD Banding"
        assert len(dunbar.bands) == 5

    def test_classify_small_team(self, dunbar: BandingTable) -> None:
        band = dunbar.classify(12)
        assert band is not None
        assert "Kleines Team" in band.label_de

    def test_classify_medium_team(self, dunbar: BandingTable) -> None:
        band = dunbar.classify(47)
        assert band is not None
        assert "Mittleres Team" in band.label_de

    def test_classify_midsize(self, dunbar: BandingTable) -> None:
        band = dunbar.classify(117)
        assert band is not None
        assert "Mittelstaendisch" in band.label_de

    def test_classify_large(self, dunbar: BandingTable) -> None:
        band = dunbar.classify(340)
        assert band is not None
        assert "Grosses Unternehmen" in band.label_de

    def test_classify_enterprise(self, dunbar: BandingTable) -> None:
        band = dunbar.classify(2400)
        assert band is not None
        assert "Konzernbereich" in band.label_de

    def test_boundary_15_goes_to_medium(self, dunbar: BandingTable) -> None:
        """min inclusive: 15 is in the 15-50 band."""
        band = dunbar.classify(15)
        assert band is not None
        assert "Mittleres Team" in band.label_de

    def test_boundary_14_stays_small(self, dunbar: BandingTable) -> None:
        band = dunbar.classify(14)
        assert band is not None
        assert "Kleines Team" in band.label_de


# ---------------------------------------------------------------------------
# EU-SME revenue banding
# ---------------------------------------------------------------------------

class TestEUSME:
    @pytest.fixture(scope="class")
    def sme(self) -> BandingTable:
        return load_banding("eu_sme_revenue", base_dir=TAXONOMIES_DIR)

    def test_loads(self, sme: BandingTable) -> None:
        assert "SME" in sme.name or "HGB" in sme.name
        assert len(sme.bands) == 4

    def test_classify_micro(self, sme: BandingTable) -> None:
        band = sme.classify(500_000)
        assert band is not None
        assert "Kleinstunternehmen" in band.label_de

    def test_classify_small(self, sme: BandingTable) -> None:
        band = sme.classify(8_700_000)
        assert band is not None
        assert "Kleinunternehmen" in band.label_de

    def test_classify_medium(self, sme: BandingTable) -> None:
        band = sme.classify(34_000_000)
        assert band is not None
        assert "Mittleres Unternehmen" in band.label_de

    def test_classify_large(self, sme: BandingTable) -> None:
        band = sme.classify(4_200_000_000)
        assert band is not None
        assert "Grossunternehmen" in band.label_de

    def test_boundary_2m(self, sme: BandingTable) -> None:
        """2M is in the small band (min inclusive)."""
        band = sme.classify(2_000_000)
        assert band is not None
        assert "Kleinunternehmen" in band.label_de


# ---------------------------------------------------------------------------
# Percentage rounding (5er-Rundung)
# ---------------------------------------------------------------------------

class TestPercentageRounding:
    @pytest.fixture(scope="class")
    def rounder(self) -> PercentageRounder:
        return load_percentage_rounder(base_dir=TAXONOMIES_DIR)

    def test_loads(self, rounder: PercentageRounder) -> None:
        assert rounder.rounding_step == 5

    def test_round_14_7(self, rounder: PercentageRounder) -> None:
        assert rounder.round(14.7) == "ca. 15%"

    def test_round_22(self, rounder: PercentageRounder) -> None:
        assert rounder.round(22) == "ca. 20%"

    def test_round_37_5(self, rounder: PercentageRounder) -> None:
        # 37.5 rounds to 40 (banker's rounding: 37.5/5 = 7.5 -> 8 * 5 = 40)
        result = rounder.round(37.5)
        assert result == "ca. 40%"

    def test_low_single_digit(self, rounder: PercentageRounder) -> None:
        assert rounder.round(3.2) == "niedrig einstellig"

    def test_high_above_80(self, rounder: PercentageRounder) -> None:
        assert rounder.round(87) == "hoch (>80%)"

    def test_exactly_80_is_high(self, rounder: PercentageRounder) -> None:
        """80 is in the high band (min inclusive)."""
        assert rounder.round(80) == "hoch (>80%)"

    def test_79_is_rounded(self, rounder: PercentageRounder) -> None:
        assert rounder.round(79) == "ca. 80%"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestErrors:
    def test_unknown_taxonomy_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown taxonomy"):
            load_taxonomy("NONEXISTENT", base_dir=TAXONOMIES_DIR)

    def test_unknown_banding_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown banding table"):
            load_banding("nonexistent", base_dir=TAXONOMIES_DIR)
