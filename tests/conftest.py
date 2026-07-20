"""Shared canned fixtures — the whole suite runs offline, with no model downloads."""

from __future__ import annotations

import pytest

from semantic_entropy_gate import CannedEntailment, LexicalEntailment
from semantic_entropy_gate.types import EntailmentLabel

# A confident model: six phrasings, one meaning.
#
# These stay resolvable by the *lexical* backend, which sees only the two answers
# and never the question. A real NLI model conditioned on the question would also
# merge "The Eiffel Tower is in Paris." here; the heuristic would not, so the
# fixtures avoid leaning on that.
CONFIDENT_SAMPLES = [
    "Paris.",
    "It is Paris.",
    "Paris, France.",
    "In Paris.",
    "Paris",
    "The city is Paris.",
]

# A confabulating model: six mutually exclusive numeric guesses.
CONFABULATING_SAMPLES = [
    "About 610 Kelvin.",
    "Roughly 503 Kelvin.",
    "It boils at approximately 337 Kelvin.",
    "Around 610 K.",
    "Approximately 575 Kelvin.",
    "It is about 400 Kelvin.",
]

# Two competing meanings, unevenly split (4 vs 2) -> the WARN band.
SPLIT_SAMPLES = [
    "Alexander Fleming.",
    "It was Alexander Fleming.",
    "Fleming, Alexander.",
    "Alexander Fleming, the bacteriologist.",
    "Howard Florey.",
    "It was Howard Florey.",
]


@pytest.fixture
def lexical():
    return LexicalEntailment()


@pytest.fixture
def canned():
    """Entailment oracle that answers from an explicit table.

    Everything not listed is NEUTRAL, i.e. a separate cluster — so a test that
    forgets a pair fails loudly rather than silently merging meanings.
    """

    def make(table=None, **kwargs):
        return CannedEntailment(table or {}, **kwargs)

    return make


@pytest.fixture
def all_equivalent():
    """An oracle that says every pair is equivalent -> exactly one cluster."""
    return CannedEntailment(default=EntailmentLabel.ENTAILMENT)


@pytest.fixture
def none_equivalent():
    """An oracle that says nothing matches -> one cluster per sample."""
    return CannedEntailment(default=EntailmentLabel.CONTRADICTION)


@pytest.fixture
def confident_samples():
    return list(CONFIDENT_SAMPLES)


@pytest.fixture
def confabulating_samples():
    return list(CONFABULATING_SAMPLES)


@pytest.fixture
def split_samples():
    return list(SPLIT_SAMPLES)
