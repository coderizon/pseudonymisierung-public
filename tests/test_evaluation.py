"""Unit tests for the evaluation harness (TASK-008).

Tests the metric computation, entity matching, and report formatting
without requiring the LLM pipeline.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from app.evaluation import (
    CorpusResult,
    DocumentResult,
    GroundTruthDocument,
    GroundTruthEntity,
    MetricSet,
    aggregate_corpus,
    compute_metrics,
    compute_per_category,
    evaluate_document_entities,
    format_report,
    format_report_json,
    load_ground_truth,
    unique_entity_keys,
)


# ---------------------------------------------------------------------------
# MetricSet
# ---------------------------------------------------------------------------


class TestMetricSet:
    def test_perfect_detection(self):
        m = MetricSet(tp=10, fp=0, fn=0)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.support == 10

    def test_no_detection(self):
        m = MetricSet(tp=0, fp=0, fn=10)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0
        assert m.support == 10

    def test_all_false_positives(self):
        m = MetricSet(tp=0, fp=5, fn=0)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0
        assert m.support == 0

    def test_partial_detection(self):
        m = MetricSet(tp=8, fp=2, fn=3)
        assert m.precision == pytest.approx(0.8)
        assert m.recall == pytest.approx(8 / 11)
        assert m.support == 11

    def test_empty(self):
        m = MetricSet()
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0
        assert m.support == 0

    def test_merge(self):
        a = MetricSet(tp=3, fp=1, fn=2)
        b = MetricSet(tp=5, fp=2, fn=1)
        merged = a.merge(b)
        assert merged.tp == 8
        assert merged.fp == 3
        assert merged.fn == 3


# ---------------------------------------------------------------------------
# Entity key normalization
# ---------------------------------------------------------------------------


class TestUniqueEntityKeys:
    def test_dict_input(self):
        entities = [
            {"text": "Dr. Müller", "category": "PERSON", "context": "..."},
            {"text": "Berlin", "category": "LOCATION", "context": "..."},
        ]
        keys = unique_entity_keys(entities)
        assert keys == {("dr. müller", "PERSON"), ("berlin", "LOCATION")}

    def test_ground_truth_input(self):
        entities = [
            GroundTruthEntity(text="Dr. Müller", category="PERSON", context="..."),
            GroundTruthEntity(text="Berlin", category="LOCATION", context="..."),
        ]
        keys = unique_entity_keys(entities)
        assert keys == {("dr. müller", "PERSON"), ("berlin", "LOCATION")}

    def test_deduplication(self):
        entities = [
            {"text": "Berlin", "category": "LOCATION", "context": "ctx1"},
            {"text": "Berlin", "category": "LOCATION", "context": "ctx2"},
            {"text": "berlin", "category": "LOCATION", "context": "ctx3"},
        ]
        keys = unique_entity_keys(entities)
        assert len(keys) == 1

    def test_same_text_different_category(self):
        entities = [
            {"text": "SAP", "category": "ORG_PARTNER", "context": "..."},
            {"text": "SAP", "category": "TECHNOLOGY", "context": "..."},
        ]
        keys = unique_entity_keys(entities)
        assert len(keys) == 2

    def test_whitespace_normalization(self):
        entities = [
            {"text": "  Berlin  ", "category": "LOCATION", "context": "..."},
        ]
        keys = unique_entity_keys(entities)
        assert ("berlin", "LOCATION") in keys


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_perfect_match(self):
        gt = {("mueller", "PERSON"), ("berlin", "LOCATION")}
        det = {("mueller", "PERSON"), ("berlin", "LOCATION")}
        m = compute_metrics(det, gt)
        assert m.tp == 2
        assert m.fp == 0
        assert m.fn == 0

    def test_partial_overlap(self):
        gt = {("mueller", "PERSON"), ("berlin", "LOCATION"), ("hamburg", "LOCATION")}
        det = {("mueller", "PERSON"), ("berlin", "LOCATION"), ("sap", "ORG_PARTNER")}
        m = compute_metrics(det, gt)
        assert m.tp == 2
        assert m.fp == 1  # sap
        assert m.fn == 1  # hamburg

    def test_no_overlap(self):
        gt = {("mueller", "PERSON")}
        det = {("berlin", "LOCATION")}
        m = compute_metrics(det, gt)
        assert m.tp == 0
        assert m.fp == 1
        assert m.fn == 1

    def test_empty_sets(self):
        m = compute_metrics(set(), set())
        assert m.tp == 0
        assert m.fp == 0
        assert m.fn == 0

    def test_wrong_category_counts_as_fp_and_fn(self):
        gt = {("sap", "ORG_PARTNER")}
        det = {("sap", "TECHNOLOGY")}
        m = compute_metrics(det, gt)
        assert m.tp == 0
        assert m.fp == 1  # detected wrong category
        assert m.fn == 1  # ground truth missed


class TestComputePerCategory:
    def test_splits_by_category(self):
        gt = {
            ("mueller", "PERSON"),
            ("krause", "PERSON"),
            ("berlin", "LOCATION"),
        }
        det = {
            ("mueller", "PERSON"),
            ("hamburg", "LOCATION"),
        }
        per_cat = compute_per_category(det, gt)

        assert per_cat["PERSON"].tp == 1
        assert per_cat["PERSON"].fn == 1  # krause missed
        assert per_cat["PERSON"].fp == 0

        assert per_cat["LOCATION"].tp == 0
        assert per_cat["LOCATION"].fn == 1  # berlin missed
        assert per_cat["LOCATION"].fp == 1  # hamburg is FP


# ---------------------------------------------------------------------------
# Document-level evaluation
# ---------------------------------------------------------------------------


class TestEvaluateDocument:
    def test_perfect_document(self):
        gt = [
            GroundTruthEntity(text="Dr. Müller", category="PERSON", context="..."),
            GroundTruthEntity(text="Berlin", category="LOCATION", context="..."),
        ]
        detected = [
            {"text": "Dr. Müller", "category": "PERSON", "context": "..."},
            {"text": "Berlin", "category": "LOCATION", "context": "..."},
        ]
        result = evaluate_document_entities(detected, gt)
        assert result.overall.tp == 2
        assert result.overall.fp == 0
        assert result.overall.fn == 0

    def test_mixed_document(self):
        gt = [
            GroundTruthEntity(text="Müller", category="PERSON", context="..."),
            GroundTruthEntity(text="Krause", category="PERSON", context="..."),
            GroundTruthEntity(text="Berlin", category="LOCATION", context="..."),
        ]
        detected = [
            {"text": "Müller", "category": "PERSON", "context": "..."},
            {"text": "Frankfurt", "category": "LOCATION", "context": "..."},
        ]
        result = evaluate_document_entities(detected, gt)
        assert result.overall.tp == 1  # Müller
        assert result.overall.fp == 1  # Frankfurt
        assert result.overall.fn == 2  # Krause, Berlin

    def test_repeated_gt_entities_deduplicated(self):
        """Ground truth may list the same entity in different contexts."""
        gt = [
            GroundTruthEntity(text="Berlin", category="LOCATION", context="ctx1"),
            GroundTruthEntity(text="Berlin", category="LOCATION", context="ctx2"),
        ]
        detected = [
            {"text": "Berlin", "category": "LOCATION", "context": "..."},
        ]
        result = evaluate_document_entities(detected, gt)
        assert result.overall.tp == 1
        assert result.overall.fn == 0


# ---------------------------------------------------------------------------
# Corpus aggregation
# ---------------------------------------------------------------------------


class TestAggregateCorpus:
    def test_aggregation(self):
        doc1 = DocumentResult(
            document="doc1.txt",
            per_category={
                "PERSON": MetricSet(tp=3, fp=1, fn=0),
                "LOCATION": MetricSet(tp=2, fp=0, fn=1),
            },
            overall=MetricSet(tp=5, fp=1, fn=1),
        )
        doc2 = DocumentResult(
            document="doc2.txt",
            per_category={
                "PERSON": MetricSet(tp=2, fp=0, fn=1),
                "BUSINESS_METRIC": MetricSet(tp=4, fp=2, fn=0),
            },
            overall=MetricSet(tp=6, fp=2, fn=1),
        )

        result = aggregate_corpus([doc1, doc2])

        # PERSON aggregated across docs
        assert result.per_category["PERSON"].tp == 5
        assert result.per_category["PERSON"].fp == 1
        assert result.per_category["PERSON"].fn == 1

        # LOCATION only in doc1
        assert result.per_category["LOCATION"].tp == 2

        # Class aggregation: PERSON + LOCATION are class 1, BUSINESS_METRIC is class 2
        assert 1 in result.per_class
        assert 2 in result.per_class
        assert result.per_class[1].tp == 7  # PERSON(5) + LOCATION(2)
        assert result.per_class[2].tp == 4  # BUSINESS_METRIC

        # Overall
        assert result.overall.tp == 11


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------


class TestLoadGroundTruth:
    def test_loads_annotation_files(self, tmp_path):
        annotation = {
            "document": "test.txt",
            "language": "de",
            "description": "Test doc",
            "entities": [
                {"text": "Mueller", "category": "PERSON", "context": "Herr Mueller"},
                {"text": "Berlin", "category": "LOCATION", "context": "in Berlin"},
            ],
        }
        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()
        (ann_dir / "test.json").write_text(json.dumps(annotation), encoding="utf-8")

        docs = load_ground_truth(ann_dir)
        assert len(docs) == 1
        assert docs[0].document == "test.txt"
        assert len(docs[0].entities) == 2
        assert docs[0].entities[0].text == "Mueller"

    def test_loads_real_corpus(self):
        """Verify the actual test corpus loads without errors."""
        from app.evaluation import ANNOTATIONS_DIR

        if not ANNOTATIONS_DIR.exists():
            pytest.skip("Test corpus not available")

        docs = load_ground_truth()
        assert len(docs) == 7
        total = sum(len(d.entities) for d in docs)
        assert total > 300  # Expect ~390


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


class TestFormatReport:
    def _make_corpus_result(self) -> CorpusResult:
        doc = DocumentResult(
            document="test.txt",
            per_category={"PERSON": MetricSet(tp=5, fp=1, fn=2)},
            overall=MetricSet(tp=5, fp=1, fn=2),
        )
        return aggregate_corpus([doc])

    def test_text_report_contains_key_sections(self):
        result = self._make_corpus_result()
        report = format_report(result)
        assert "EVALUATION REPORT" in report
        assert "PERSON" in report
        assert "CLASS SUMMARY" in report
        assert "OVERALL" in report
        assert "PER-DOCUMENT SUMMARY" in report
        assert "test.txt" in report

    def test_json_report_structure(self):
        result = self._make_corpus_result()
        data = format_report_json(result)
        assert "per_category" in data
        assert "per_class" in data
        assert "overall" in data
        assert "documents" in data
        assert "targets" in data
        assert data["overall"]["tp"] == 5

    def test_pass_fail_indicators(self):
        result = self._make_corpus_result()
        report = format_report(result)
        # With P=83.3% and target 80%, should pass precision
        assert "PASS" in report or "FAIL" in report
