"""pytest integration: assert that your model is *confident*, not just correct.

Registered automatically as a pytest plugin on install, so no conftest wiring is
needed::

    def test_faq_answers_are_grounded(semantic_entropy):
        result = semantic_entropy("What is our refund window?", my_sampler)
        assert_confident(result, threshold=0.4)

An accuracy assertion tells you the model got it right *this time*. A semantic
entropy assertion tells you it was not guessing — which is what stops the same
test passing by luck next week.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from .entailment import EntailmentModel, LexicalEntailment
from .score import DEFAULT_N_SAMPLES, score, score_samples
from .types import EntropyResult

__all__ = [
    "assert_confident",
    "assert_uncertain",
    "semantic_entropy",
    "entropy_of",
]


def assert_confident(
    result: EntropyResult, *, threshold: float = 0.5, normalized: bool = True
) -> None:
    """Fail if semantic entropy is at or above ``threshold``.

    The failure message embeds the full cluster breakdown, so a red test tells
    you *which* meanings the model was torn between — not merely that a number
    was too big.
    """
    value = result.score_for(normalized=normalized)
    if value >= threshold:
        raise AssertionError(
            f"semantic entropy {value:.4f} >= threshold {threshold:.4f} "
            f"({result.n_clusters} distinct meanings in {result.n_samples} samples)\n"
            + result.explain(threshold=threshold)
        )


def assert_uncertain(
    result: EntropyResult, *, threshold: float = 0.5, normalized: bool = True
) -> None:
    """Fail if the model was *confident* — for testing that the gate fires.

    Use it on prompts your system is supposed to refuse: unanswerable questions,
    out-of-scope requests, facts outside the knowledge cut-off.
    """
    value = result.score_for(normalized=normalized)
    if value < threshold:
        raise AssertionError(
            f"expected uncertainty but semantic entropy was {value:.4f} < {threshold:.4f}\n"
            + result.explain(threshold=threshold)
        )


def entropy_of(
    prompt: str,
    sampler: Any = None,
    *,
    samples: Optional[Any] = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    entailment: Optional[EntailmentModel] = None,
    **kwargs: Any,
) -> EntropyResult:
    """Score inside a test, defaulting to the offline lexical backend.

    Tests must not silently download a 70 MB NLI checkpoint on a cold CI runner,
    so the default here is deterministic and dependency-free. Pass
    ``entailment=`` explicitly when you do want the real model.
    """
    backend = entailment if entailment is not None else LexicalEntailment()
    if samples is not None:
        return score_samples(prompt, samples, entailment=backend, **kwargs)
    if sampler is None:
        raise ValueError("entropy_of needs either a sampler or samples=")
    return score(prompt, sampler, n_samples=n_samples, entailment=backend, **kwargs)


@pytest.fixture
def semantic_entropy():
    """Fixture returning :func:`entropy_of`."""
    return entropy_of


@pytest.fixture
def lexical_entailment():
    """A fresh offline entailment backend for a test."""
    return LexicalEntailment()
