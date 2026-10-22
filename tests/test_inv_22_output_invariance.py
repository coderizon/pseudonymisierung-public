"""INV-22 invariance: after pseudonymize_by_span, no stage modifies or adds [...] tokens.

The set of bracketed tokens in `result.final_text` equals the set that
`pseudonymize_by_span` emitted. The `enable_generalization` parameter and the
`generalization_enabled` result field are gone.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import fields

import pytest
from unittest.mock import patch

from app.orchestrator import ProcessingResult, process_document
from app.pseudonymizer import DetectedEntity


_BRACKET_TOKEN_RE = re.compile(r"\[[^\[\]]+\]")


def _tokens_in(text: str) -> set[str]:
    return set(_BRACKET_TOKEN_RE.findall(text))


@pytest.mark.asyncio
async def test_final_text_equals_pseudonymized_text():
    """After retirement, the orchestrator sets final_text to the pseudonymized
    text verbatim. No post-pseudonymization LLM pass runs."""

    async def _fake_warm_up():
        return None

    async def _fake_detect(_text, _config, chunk_timings=None):
        return [DetectedEntity(text="Microsoft", category="ORG_PARTNER", context="")]

    async def _fake_verify(**kwargs):
        return list(kwargs["occurrences"])

    async def _fake_k1(_entities, _config):
        return {}

    async def _fake_k2(_entities, _config):
        return {"microsoft": "Software-Gigant"}

    text = "Microsoft ist ein grosser Anbieter.\n"
    data = text.encode("utf-8")

    with (
        patch("app.orchestrator.detect_entities", new=_fake_detect),
        patch("app.orchestrator.verify_occurrences", new=_fake_verify),
        patch("app.orchestrator.compute_klasse1_descriptions", new=_fake_k1),
        patch("app.orchestrator.compute_klasse2_descriptions", new=_fake_k2),
        patch("app.orchestrator.warm_up", new=_fake_warm_up),
    ):
        result = await process_document(
            data=data,
            filename="inv22.txt",
            project_id="inv22-test",
        )

    assert result.final_text == result.pseudonymized_text
    assert _tokens_in(result.final_text) <= _tokens_in(result.pseudonymized_text)


def test_processing_result_has_no_generalization_enabled_field():
    """After retirement, ProcessingResult no longer declares a
    generalization_enabled field. Inspected at the dataclass level so the
    test does not depend on runtime defaults."""
    field_names = {f.name for f in fields(ProcessingResult)}
    assert "generalization_enabled" not in field_names, (
        "ProcessingResult still declares a generalization_enabled field; "
        "the retirement did not fully remove the field."
    )


def test_process_document_signature_has_no_enable_generalization():
    """After retirement, the enable_generalization kwarg is gone from
    process_document's signature."""
    sig = inspect.signature(process_document)
    assert "enable_generalization" not in sig.parameters, (
        "process_document still accepts an enable_generalization kwarg; "
        "the retirement did not fully remove the parameter."
    )
