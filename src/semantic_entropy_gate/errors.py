"""Exception hierarchy for semantic-entropy-gate.

Every error the library raises derives from :class:`SemanticEntropyError`, so a
host application can wrap the whole gate in a single ``except`` and fall back to
its own policy without accidentally swallowing unrelated bugs.
"""

from __future__ import annotations


class SemanticEntropyError(Exception):
    """Base class for every error raised by semantic-entropy-gate."""


class SamplingError(SemanticEntropyError):
    """The sampler failed, or returned an unusable set of generations."""


class EntailmentBackendError(SemanticEntropyError):
    """An entailment backend could not be constructed or produced no verdict."""


class CalibrationError(SemanticEntropyError):
    """The labelled dev set is unusable (empty, single-class, malformed)."""


class GateBlockedError(SemanticEntropyError):
    """Raised by :meth:`Gate.run` when the policy is ``BLOCK`` and ``raise_on_block``.

    The offending :class:`~semantic_entropy_gate.types.GateDecision` is attached as
    ``.decision`` so callers can log the full uncertainty trace.
    """

    def __init__(self, message: str, decision: object = None) -> None:
        super().__init__(message)
        self.decision = decision
