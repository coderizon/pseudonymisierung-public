"""Evaluation harness: measure detection quality against ground-truth annotations.

Computes precision, recall, and F1 per category, per class, and overall
by comparing detected entities against annotated ground truth from the
synthetic test corpus (TASK-005).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.category_config import get_config

logger = logging.getLogger(__name__)

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "test_corpus"
ANNOTATIONS_DIR = CORPUS_DIR / "annotations"
DOCUMENTS_DIR = CORPUS_DIR / "documents"


@dataclass
class MetricSet:
    """Precision/recall/F1 for a single evaluation slice."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def support(self) -> int:
        """Number of ground-truth entities in this slice."""
        return self.tp + self.fn

    def merge(self, other: MetricSet) -> MetricSet:
        """Combine two metric sets (micro-averaging)."""
        return MetricSet(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
        )


@dataclass
class DocumentResult:
    """Evaluation result for a single document."""

    document: str
    per_category: dict[str, MetricSet] = field(default_factory=dict)
    overall: MetricSet = field(default_factory=MetricSet)


@dataclass
class CorpusResult:
    """Evaluation result for the full corpus."""

    documents: list[DocumentResult] = field(default_factory=list)
    per_category: dict[str, MetricSet] = field(default_factory=dict)
    per_class: dict[int, MetricSet] = field(default_factory=dict)
    overall: MetricSet = field(default_factory=MetricSet)


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------


@dataclass
class GroundTruthEntity:
    text: str
    category: str
    context: str


@dataclass
class GroundTruthDocument:
    document: str
    language: str
    description: str
    entities: list[GroundTruthEntity]


def load_ground_truth(annotations_dir: Path | None = None) -> list[GroundTruthDocument]:
    """Load all ground-truth annotation files from disk."""
    if annotations_dir is None:
        annotations_dir = ANNOTATIONS_DIR

    docs: list[GroundTruthDocument] = []
    for path in sorted(annotations_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        entities = [
            GroundTruthEntity(
                text=e["text"],
                category=e["category"],
                context=e.get("context", ""),
            )
            for e in data.get("entities", [])
        ]
        docs.append(GroundTruthDocument(
            document=data["document"],
            language=data.get("language", "de"),
            description=data.get("description", ""),
            entities=entities,
        ))
    return docs


# ---------------------------------------------------------------------------
# Entity normalization and matching
# ---------------------------------------------------------------------------


def _normalize_key(text: str, category: str) -> tuple[str, str]:
    """Normalize an entity to a comparable key: (lowercased stripped text, category)."""
    return (text.strip().lower(), category)


def unique_entity_keys(
    entities: list[dict | GroundTruthEntity],
) -> set[tuple[str, str]]:
    """Extract unique (text, category) keys from a list of entities.

    Accepts both dicts (pipeline output) and GroundTruthEntity objects.
    """
    keys: set[tuple[str, str]] = set()
    for e in entities:
        if isinstance(e, GroundTruthEntity):
            keys.add(_normalize_key(e.text, e.category))
        else:
            keys.add(_normalize_key(e["text"], e["category"]))
    return keys


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_metrics(
    detected_keys: set[tuple[str, str]],
    ground_truth_keys: set[tuple[str, str]],
) -> MetricSet:
    """Compute TP/FP/FN from two sets of (text, category) keys."""
    tp = len(detected_keys & ground_truth_keys)
    fp = len(detected_keys - ground_truth_keys)
    fn = len(ground_truth_keys - detected_keys)
    return MetricSet(tp=tp, fp=fp, fn=fn)


def compute_per_category(
    detected_keys: set[tuple[str, str]],
    ground_truth_keys: set[tuple[str, str]],
    categories: list[str] | None = None,
) -> dict[str, MetricSet]:
    """Compute metrics per category.

    If *categories* is None, derives the list from the union of both sets.
    """
    if categories is None:
        categories = sorted({k[1] for k in detected_keys | ground_truth_keys})

    result: dict[str, MetricSet] = {}
    for cat in categories:
        det = {k for k in detected_keys if k[1] == cat}
        gt = {k for k in ground_truth_keys if k[1] == cat}
        result[cat] = compute_metrics(det, gt)
    return result


def evaluate_document_entities(
    detected: list[dict],
    ground_truth: list[GroundTruthEntity],
) -> DocumentResult:
    """Compare detected entities against ground truth for one document."""
    det_keys = unique_entity_keys(detected)
    gt_keys = unique_entity_keys(ground_truth)

    per_cat = compute_per_category(det_keys, gt_keys)
    overall = compute_metrics(det_keys, gt_keys)

    return DocumentResult(
        document="",
        per_category=per_cat,
        overall=overall,
    )


def aggregate_corpus(doc_results: list[DocumentResult]) -> CorpusResult:
    """Aggregate per-document results into corpus-level metrics."""
    config = get_config()
    category_class_map: dict[str, int] = {}
    for cat_def in config.all_categories():
        category_class_map[cat_def.name] = cat_def.class_

    corpus = CorpusResult(documents=doc_results)

    # Merge per-category across documents (micro-averaging)
    all_categories: set[str] = set()
    for doc in doc_results:
        all_categories.update(doc.per_category.keys())

    for cat in sorted(all_categories):
        merged = MetricSet()
        for doc in doc_results:
            if cat in doc.per_category:
                merged = merged.merge(doc.per_category[cat])
        corpus.per_category[cat] = merged

    # Aggregate per class
    for cat, metrics in corpus.per_category.items():
        cls = category_class_map.get(cat, 0)
        if cls not in corpus.per_class:
            corpus.per_class[cls] = MetricSet()
        corpus.per_class[cls] = corpus.per_class[cls].merge(metrics)

    # Overall
    overall = MetricSet()
    for metrics in corpus.per_category.values():
        overall = overall.merge(metrics)
    corpus.overall = overall

    return corpus


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

# Target thresholds from test corpus README
TARGETS = {
    "recall_class1": 0.90,
    "recall_class2": 0.85,
    "precision": 0.80,
}


def format_pct(value: float) -> str:
    """Format a float as a percentage string."""
    return f"{value * 100:.1f}%"


def _pass_fail(value: float, target: float) -> str:
    return "PASS" if value >= target else "FAIL"


def format_report(result: CorpusResult) -> str:
    """Format corpus evaluation results as a readable text report."""
    config = get_config()
    category_class_map: dict[str, int] = {}
    for cat_def in config.all_categories():
        category_class_map[cat_def.name] = cat_def.class_

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("EVALUATION REPORT")
    lines.append("=" * 78)
    lines.append("")

    # Per-category table
    header = f"{'Category':<22} {'Cls':>3} {'TP':>4} {'FP':>4} {'FN':>4}  {'Prec':>6} {'Recall':>6} {'F1':>6}  {'Supp':>4}"
    lines.append(header)
    lines.append("-" * len(header))

    for cat in sorted(result.per_category.keys()):
        m = result.per_category[cat]
        cls = category_class_map.get(cat, 0)
        lines.append(
            f"{cat:<22} {cls:>3} {m.tp:>4} {m.fp:>4} {m.fn:>4}"
            f"  {format_pct(m.precision):>6} {format_pct(m.recall):>6} {format_pct(m.f1):>6}"
            f"  {m.support:>4}"
        )

    lines.append("-" * len(header))
    lines.append("")

    # Per-class summary
    lines.append("CLASS SUMMARY")
    lines.append("-" * 50)
    for cls in sorted(result.per_class.keys()):
        m = result.per_class[cls]
        label = "PII" if cls == 1 else "Business Secrets" if cls == 2 else f"Class {cls}"
        recall_target = TARGETS["recall_class1"] if cls == 1 else TARGETS["recall_class2"]
        lines.append(
            f"  Klasse {cls} ({label}):"
            f"  P={format_pct(m.precision)}"
            f"  R={format_pct(m.recall)}"
            f"  F1={format_pct(m.f1)}"
            f"  (Support: {m.support})"
        )
        lines.append(
            f"    Precision vs target {format_pct(TARGETS['precision'])}: {_pass_fail(m.precision, TARGETS['precision'])}"
        )
        lines.append(
            f"    Recall vs target {format_pct(recall_target)}: {_pass_fail(m.recall, recall_target)}"
        )
    lines.append("")

    # Overall
    m = result.overall
    lines.append("OVERALL")
    lines.append("-" * 50)
    lines.append(
        f"  P={format_pct(m.precision)}  R={format_pct(m.recall)}  F1={format_pct(m.f1)}"
        f"  (TP={m.tp}, FP={m.fp}, FN={m.fn}, Support={m.support})"
    )
    lines.append("")

    # Per-document summary
    lines.append("PER-DOCUMENT SUMMARY")
    lines.append("-" * 50)
    for doc in result.documents:
        m = doc.overall
        lines.append(
            f"  {doc.document:<45}"
            f"  P={format_pct(m.precision):>6} R={format_pct(m.recall):>6} F1={format_pct(m.f1):>6}"
        )
    lines.append("")
    lines.append("=" * 78)

    return "\n".join(lines)


def format_report_json(result: CorpusResult) -> dict:
    """Format corpus evaluation results as a JSON-serializable dict."""
    config = get_config()
    category_class_map: dict[str, int] = {}
    for cat_def in config.all_categories():
        category_class_map[cat_def.name] = cat_def.class_

    def _metrics_dict(m: MetricSet) -> dict:
        return {
            "tp": m.tp,
            "fp": m.fp,
            "fn": m.fn,
            "precision": round(m.precision, 4),
            "recall": round(m.recall, 4),
            "f1": round(m.f1, 4),
            "support": m.support,
        }

    return {
        "per_category": {
            cat: {**_metrics_dict(m), "class": category_class_map.get(cat, 0)}
            for cat, m in sorted(result.per_category.items())
        },
        "per_class": {
            str(cls): _metrics_dict(m) for cls, m in sorted(result.per_class.items())
        },
        "overall": _metrics_dict(result.overall),
        "documents": [
            {
                "document": doc.document,
                "metrics": _metrics_dict(doc.overall),
            }
            for doc in result.documents
        ],
        "targets": TARGETS,
    }
