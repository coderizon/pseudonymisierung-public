"""Tier 1b banding hygiene tests.

Covers the two surgical fixes in app/klasse2_transformer.py described in
docs/engineering/specs/tier-1b-banding-hygiene.md:

- Fix (i): euro-marker metric_type override in _apply_banding_result.
- Fix (ii): finer-grained low-end bands in _generalize_monetary.

Also pins the current-correct behaviour on the 2026-04-09 BUG-018 symptom
inputs so a future regression reopens BUG-018 via test failure rather than
via production recurrence.
"""

from app.klasse2_transformer import (
    _apply_banding_result,
    _generalize_monetary,
)


class TestEuroMarkerOverride:
    """Fix (i): metric_type='other' combined with an EUR / € / Euro marker flips to monetary."""

    def test_eur_marker_flips_other_to_monetary(self):
        result = _apply_banding_result([
            {"original": "500 EUR", "value": 500, "metric_type": "other"},
        ])
        # After override, value=500 routes through _generalize_monetary:
        # 100 <= 500 < 1_000 -> "dreistelliger Betrag".
        assert result["500 eur"] == "dreistelliger Betrag"

    def test_euro_symbol_flips_other_to_monetary(self):
        result = _apply_banding_result([
            {"original": "10€", "value": 10, "metric_type": "other"},
        ])
        # After override + fix (ii): value=10 is in the zweistelliger band.
        assert result["10€"] == "zweistelliger Betrag"

    def test_euro_word_flips_other_to_monetary(self):
        result = _apply_banding_result([
            {"original": "85 Euro pro Vorgang", "value": 85, "metric_type": "other"},
        ])
        assert result["85 euro pro vorgang"] == "zweistelliger Betrag"

    def test_lowercase_eur_marker_also_flips(self):
        # Override regex must be case-insensitive.
        result = _apply_banding_result([
            {"original": "50 eur Einsparung", "value": 50, "metric_type": "other"},
        ])
        assert result["50 eur einsparung"] == "zweistelliger Betrag"

    def test_no_euro_marker_stays_in_generic_path(self):
        # No euro marker -> override does not fire. Value=42 with metric_type='other'
        # routes through _generalize_generic_number: value<50 -> "mehrere Dutzend".
        result = _apply_banding_result([
            {"original": "42 Projekte", "value": 42, "metric_type": "other"},
        ])
        assert result["42 projekte"] == "mehrere Dutzend"

    def test_monetary_already_classified_untouched(self):
        # LLM already classified monetary -> override must not fire (non-interference).
        result = _apply_banding_result([
            {"original": "500 EUR", "value": 500, "metric_type": "monetary"},
        ])
        assert result["500 eur"] == "dreistelliger Betrag"

    def test_revenue_already_classified_untouched(self):
        # LLM said revenue -> override must not fire even though the euro marker is present.
        # 2_000_000 <= 5_000_000 < 10_000_000 -> Kleinunternehmen band on EU-SME.
        result = _apply_banding_result([
            {"original": "5 Mio EUR Umsatz", "value": 5_000_000, "metric_type": "revenue"},
        ])
        assert "Kleinunternehmen" in result["5 mio eur umsatz"]

    def test_eur_as_substring_does_not_flip(self):
        # "Neuronen" contains "EUR" as a substring (positions 1-3: e-u-r) but
        # with no word boundaries around it; the word-boundary anchor in the
        # override regex must prevent a false-positive override on this shape.
        result = _apply_banding_result([
            {"original": "7 Neuronen", "value": 7, "metric_type": "other"},
        ])
        # Override should not fire; value=7 routes to _generalize_generic_number
        # (value<10 -> "wenige").
        assert result["7 neuronen"] == "wenige"


class TestFinerGrainedMonetaryBands:
    """Fix (ii): _generalize_monetary resolves einstelliger / zweistelliger / dreistelliger below 1_000."""

    def test_under_10_is_einstelliger(self):
        assert _generalize_monetary(0) == "einstelliger Betrag"
        assert _generalize_monetary(1) == "einstelliger Betrag"
        assert _generalize_monetary(9) == "einstelliger Betrag"
        assert _generalize_monetary(9.99) == "einstelliger Betrag"

    def test_ten_to_99_is_zweistelliger(self):
        assert _generalize_monetary(10) == "zweistelliger Betrag"
        assert _generalize_monetary(50) == "zweistelliger Betrag"
        assert _generalize_monetary(99) == "zweistelliger Betrag"
        assert _generalize_monetary(99.99) == "zweistelliger Betrag"

    def test_hundred_to_999_is_dreistelliger(self):
        assert _generalize_monetary(100) == "dreistelliger Betrag"
        assert _generalize_monetary(500) == "dreistelliger Betrag"
        assert _generalize_monetary(999) == "dreistelliger Betrag"

    def test_1000_crosses_into_vierstelliger(self):
        assert _generalize_monetary(1_000) == "vierstelliger Betrag"
        assert _generalize_monetary(1_500) == "vierstelliger Betrag"

    def test_negative_values_band_by_magnitude(self):
        # _generalize_monetary uses abs(value) internally.
        assert _generalize_monetary(-50) == "zweistelliger Betrag"
        assert _generalize_monetary(-1_000_000) == "niedriger Millionenbetrag"


class TestBug018RegressionPin:
    """Pin the 2026-04-22 observed-correct behaviour on the 2026-04-09 BUG-018 symptom inputs.

    BUG-018 did not reproduce on gemma4:26b across crafted and NGO-transcript
    inputs on 2026-04-22; German-thousands-dot parsing is currently intact.
    These tests encode that correct behaviour so a future regression in the
    value-parsing or banding paths reopens BUG-018 via test failure rather
    than via production recurrence. See docs/tasks/bugs.md 'Stale / Monitoring'.
    """

    def test_500k_eur_bands_sechsstelliger(self):
        # Canonical BUG-018 input: "500.000 EUR Einsparung".
        result = _apply_banding_result([
            {"original": "500.000 EUR Einsparung", "value": 500_000, "metric_type": "monetary"},
        ])
        assert result["500.000 eur einsparung"] == "sechsstelliger Betrag"

    def test_1_25m_eur_bands_niedriger_millionenbetrag(self):
        result = _apply_banding_result([
            {"original": "1.250.000 EUR Budget", "value": 1_250_000, "metric_type": "monetary"},
        ])
        assert result["1.250.000 eur budget"] == "niedriger Millionenbetrag"

    def test_20k_eur_bands_fuenfstelliger(self):
        # From the NGO transcript 2026-04-22 run.
        result = _apply_banding_result([
            {"original": "20.000 Euro", "value": 20_000, "metric_type": "monetary"},
        ])
        assert result["20.000 euro"] == "fuenfstelliger Betrag"

    def test_generalize_monetary_pins_bug018_magnitudes(self):
        assert _generalize_monetary(500_000) == "sechsstelliger Betrag"
        assert _generalize_monetary(1_250_000) == "niedriger Millionenbetrag"
        assert _generalize_monetary(20_000) == "fuenfstelliger Betrag"
