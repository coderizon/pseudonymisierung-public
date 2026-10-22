"""Tests for BUG-020 / BUG-024: BUSINESS_METRIC scope tightening.

Covers the new per-category negative_examples schema field in categories.yaml,
its rendering in both prompt builders, and the BUSINESS_METRIC-specific content
update (tightened description, extended positive examples, populated negatives).
Scope-invariant shape check guards the full yaml against hand-edit drift.

Spec: docs/engineering/specs/bug-020-business-metric-scope.md
"""

from __future__ import annotations

from app.category_config import (
    CategoryConfig,
    CategoryDef,
    _parse_category,
    get_config,
)


class TestSchemaExtension:
    """The CategoryDef dataclass and _parse_category carry the new field."""

    def test_parse_category_reads_negative_examples_when_present(self):
        data = {
            "class": 2,
            "active": True,
            "action": "generalize",
            "protection_level": "medium",
            "description": "Irrelevant for this test.",
            "examples": {"de": ["eins"], "en": ["one"]},
            "transformation": "",
            "negative_examples": {
                "de": ["negativ eins", "negativ zwei"],
                "en": ["negative one"],
            },
        }
        cat = _parse_category("TEST_CAT", data)
        assert cat.negative_examples == {
            "de": ["negativ eins", "negativ zwei"],
            "en": ["negative one"],
        }

    def test_parse_category_defaults_to_empty_dict_when_negatives_absent(self):
        data = {
            "class": 1,
            "active": True,
            "action": "pseudonymize",
            "protection_level": "high",
            "description": "Irrelevant for this test.",
            "examples": {"de": ["eins"]},
            "transformation": "",
        }
        cat = _parse_category("TEST_CAT", data)
        assert cat.negative_examples == {}

    def test_category_def_has_negative_examples_field(self):
        cat = CategoryDef(
            name="X",
            class_=1,
            active=True,
            action="pseudonymize",
            protection_level="high",
            description="",
            examples={},
            transformation="",
        )
        assert cat.negative_examples == {}


def _build_minimal_config(with_negatives: bool) -> CategoryConfig:
    """Construct a small two-category CategoryConfig for rendering tests.

    One category (NEGCAT) either carries negative_examples or not, depending
    on the flag. The other (BARECAT) never does. Both are active Klasse 2
    generalize categories so they appear in ner_categories() output.
    """
    negcat = CategoryDef(
        name="NEGCAT",
        class_=2,
        active=True,
        action="generalize",
        protection_level="medium",
        description="Negcat description.",
        examples={"de": ["positiv eins"], "en": ["positive one"]},
        negative_examples=(
            {"de": ["negativ eins", "negativ zwei"], "en": ["negative one"]}
            if with_negatives
            else {}
        ),
        transformation="",
    )
    barecat = CategoryDef(
        name="BARECAT",
        class_=2,
        active=True,
        action="generalize",
        protection_level="medium",
        description="Barecat description.",
        examples={"de": ["positiv drei"], "en": ["positive three"]},
        transformation="",
    )
    return CategoryConfig(
        profile="test",
        exclusions=[],
        categories={"NEGCAT": negcat, "BARECAT": barecat},
    )


class TestNerPromptNegativesRendering:
    """build_ner_prompt renders NICHT <CAT>: ... inline when negatives are populated."""

    def test_ner_prompt_renders_nicht_clause_when_negatives_present(self):
        config = _build_minimal_config(with_negatives=True)
        prompt = config.build_ner_prompt()
        assert "NICHT NEGCAT:" in prompt
        assert '"negativ eins"' in prompt
        assert '"negativ zwei"' in prompt
        assert '"negative one"' in prompt

    def test_ner_prompt_omits_nicht_clause_when_negatives_absent(self):
        config = _build_minimal_config(with_negatives=False)
        prompt = config.build_ner_prompt()
        assert "NICHT NEGCAT:" not in prompt
        assert "NICHT BARECAT:" not in prompt

    def test_ner_prompt_barecat_never_has_nicht_clause(self):
        """BARECAT has no negatives in either config; it must never carry a NICHT section."""
        for with_neg in (True, False):
            config = _build_minimal_config(with_negatives=with_neg)
            prompt = config.build_ner_prompt()
            assert "NICHT BARECAT:" not in prompt


class TestVerificationPromptNegativesRendering:
    """build_verification_prompt renders NICHT <CAT>: ... inline when negatives are populated."""

    def test_verification_prompt_renders_nicht_clause_when_negatives_present(self):
        config = _build_minimal_config(with_negatives=True)
        prompt = config.build_verification_prompt()
        assert "NICHT NEGCAT:" in prompt
        assert '"negativ eins"' in prompt
        assert '"negativ zwei"' in prompt
        assert '"negative one"' in prompt

    def test_verification_prompt_omits_nicht_clause_when_negatives_absent(self):
        config = _build_minimal_config(with_negatives=False)
        prompt = config.build_verification_prompt()
        assert "NICHT NEGCAT:" not in prompt
        assert "NICHT BARECAT:" not in prompt

    def test_verification_prompt_barecat_never_has_nicht_clause(self):
        for with_neg in (True, False):
            config = _build_minimal_config(with_negatives=with_neg)
            prompt = config.build_verification_prompt()
            assert "NICHT BARECAT:" not in prompt


class TestScopeInvariant:
    """Schema-level invariant: every category's negative_examples shape is valid.

    Cheap guard against a future hand-edit to categories.yaml introducing
    a malformed shape (for example a flat list instead of a language-keyed
    map of string lists), which would otherwise surface only at prompt-
    render time as garbled output.
    """

    def test_every_category_has_valid_negative_examples_shape(self):
        config = get_config()
        for name, cat in config.categories.items():
            ne = cat.negative_examples
            assert isinstance(ne, dict), (
                f"{name}.negative_examples must be a dict, got {type(ne).__name__}"
            )
            for lang, entries in ne.items():
                assert isinstance(lang, str), (
                    f"{name}.negative_examples key must be a string, got {type(lang).__name__}"
                )
                assert isinstance(entries, list), (
                    f"{name}.negative_examples[{lang!r}] must be a list, "
                    f"got {type(entries).__name__}"
                )
                for i, entry in enumerate(entries):
                    assert isinstance(entry, str), (
                        f"{name}.negative_examples[{lang!r}][{i}] must be a string, "
                        f"got {type(entry).__name__}"
                    )


class TestBusinessMetricContentUpdate:
    """Assert the BUSINESS_METRIC entry in the real categories.yaml matches the spec."""

    def test_description_has_numeric_value_anchor(self):
        config = get_config()
        bm = config.get("BUSINESS_METRIC")
        assert bm is not None
        assert "mit konkretem Zahlenwert" in bm.description

    def test_positive_examples_include_standalone_percentages(self):
        config = get_config()
        bm = config.get("BUSINESS_METRIC")
        de = bm.examples.get("de", [])
        assert "50%" in de
        assert "60% Overhead" in de

    def test_negative_examples_de_has_seven_shape_hinted_entries(self):
        config = get_config()
        bm = config.get("BUSINESS_METRIC")
        de = bm.negative_examples.get("de", [])
        assert len(de) == 7
        # Spot-check for shape-hint labels (parentheses with shape class).
        assert any("(Programmbezeichnung ohne Zahlenwert)" in e for e in de)
        assert any("(Personenbeschreibung ohne Zahlenwert)" in e for e in de)
        assert any("(generische Speichergröße)" in e for e in de)
        assert any("(generischer Netzwerkdurchsatz)" in e for e in de)
        assert any("(IT-Troubleshooting-Zahl)" in e for e in de)

    def test_negative_examples_en_has_seven_shape_hinted_entries(self):
        # Symmetric with de: seven en entries across the same five shape classes.
        config = get_config()
        bm = config.get("BUSINESS_METRIC")
        en = bm.negative_examples.get("en", [])
        assert len(en) == 7
        assert any("(generic storage size)" in e for e in en)
        assert any("(program label with no numeric value)" in e for e in en)

    def test_ner_prompt_from_real_config_carries_business_metric_content(self):
        config = get_config()
        prompt = config.build_ner_prompt()
        assert "mit konkretem Zahlenwert" in prompt
        assert '"50%"' in prompt
        assert '"60% Overhead"' in prompt
        assert "NICHT BUSINESS_METRIC:" in prompt
        assert "Stadthaushalt-Programm" in prompt

    def test_verification_prompt_from_real_config_carries_business_metric_negatives(self):
        config = get_config()
        prompt = config.build_verification_prompt()
        assert "NICHT BUSINESS_METRIC:" in prompt
        assert "Stadthaushalt-Programm" in prompt
        assert "1,5 GB Logdaten" in prompt

    def test_person_category_has_no_negatives(self):
        """PERSON is unchanged by this slice and must still have empty negatives."""
        config = get_config()
        person = config.get("PERSON")
        assert person is not None
        assert person.negative_examples == {}

    def test_ner_prompt_does_not_render_nicht_for_unchanged_categories(self):
        config = get_config()
        prompt = config.build_ner_prompt()
        # Pick a few categories that this slice does NOT touch; none of them
        # should carry a NICHT clause in the rendered prompt.
        for unchanged in ("PERSON", "CONTACT", "ORG_CLIENT", "ORG_PARTNER"):
            assert f"NICHT {unchanged}:" not in prompt
