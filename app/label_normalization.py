"""Label shape normalisation.

This module centralises the strict bracket-stripping rule required by INV-20.
The `[...]` form is the pipeline's output convention at replacement-emission
time in `app.pseudonymizer.pseudonymize_by_span`. Every description or
pseudonym flowing into that emission step must be a bare label. This helper
is the single point where that shape is enforced for inputs that may already
carry brackets (LLM output, defensive checks on deterministic sources).
"""

from __future__ import annotations


def strict_strip_surrounding_brackets(label: str) -> tuple[str, bool]:
    """Strip exactly one surrounding `[...]` pair iff the label is well-formed.

    A label is well-formed when it either has no brackets at all, or it starts
    with `[`, ends with `]`, and contains no other `[` or `]` characters. In
    both cases the returned boolean is True.

    On any other shape (asymmetric brackets, nested brackets, multiple bracket
    pairs) the original label is returned unchanged and the returned boolean
    is False. The caller is expected to treat the False signal as a recovery
    trigger, not to silently normalise.

    Returns:
        A tuple `(stripped, well_formed)`.
    """
    if "[" not in label and "]" not in label:
        return label, True

    if (
        label.startswith("[")
        and label.endswith("]")
        and label.count("[") == 1
        and label.count("]") == 1
    ):
        return label[1:-1], True

    return label, False
