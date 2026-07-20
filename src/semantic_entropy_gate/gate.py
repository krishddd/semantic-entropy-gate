"""Middleware: gate any agent / LLM call on its own semantic uncertainty.

The gate sits *in front of* a world-altering action. It samples the model's
answer N times, measures semantic entropy, and only then decides whether the
action may run:

    entropy < warn_threshold        -> ALLOW   (execute silently)
    warn <= entropy < defer         -> WARN    (execute, attach an uncertainty note)
    defer <= entropy < block        -> DEFER   (do not execute; forage / ask a human)
    entropy >= block_threshold      -> BLOCK   (refuse)

This is the software realisation of the *information-seeking loop* in the
research note: when the epistemic (ambiguity) term of expected free energy
dominates the pragmatic term, the only rational policies are the ones that
*acquire information* rather than the ones that act. DEFER is that phase
transition, expressed as a return value your orchestrator can act on.

Every decision carries the full :class:`~semantic_entropy_gate.types.EntropyResult`
with it, so "the gate stopped my agent" is always answerable with evidence.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from .entailment import EntailmentModel, auto_entailment
from .errors import GateBlockedError
from .sampling import Sampler
from .score import DEFAULT_N_SAMPLES, score, score_samples
from .types import EntropyResult, GateAction, GateDecision

__all__ = ["Gate", "gate", "DEFAULT_THRESHOLD"]

DEFAULT_THRESHOLD = 0.55
"""Uncalibrated default on the *normalised* entropy scale.

It is a starting point, not a claim. Run
:func:`semantic_entropy_gate.calibrate.calibrate` on ~100 labelled prompts from
your own task and replace it — the correct threshold is task-dependent and the
library will tell you the AUROC so you know whether the signal is even usable.
"""


class Gate:
    """A configured uncertainty gate.

    Parameters
    ----------
    sampler:
        How to draw N generations (see :mod:`semantic_entropy_gate.sampling`).
        Optional: :meth:`check_samples` and :meth:`run` with ``samples=`` let you
        gate on generations you already have.
    threshold:
        The DEFER boundary on normalised semantic entropy. The single number you
        get out of calibration.
    warn_threshold:
        Below ``threshold``; entropy above it is executed but annotated. Defaults
        to ``0.6 * threshold``.
    block_threshold:
        Above ``threshold``; entropy above it is refused outright. ``None``
        disables BLOCK (everything uncertain merely defers).
    on_defer / on_block:
        Callables ``(prompt, EntropyResult) -> Any`` producing the fallback value
        — ask a clarifying question, run a retrieval step, return an abstention.
        Default: ``None`` answer, with the reason on the decision.
    raise_on_block:
        Raise :class:`~semantic_entropy_gate.errors.GateBlockedError` instead of
        returning a decision. Useful in a pipeline that already has error
        handling.
    history:
        Retain every decision for later reporting (bounded by ``max_history``).
    """

    def __init__(
        self,
        sampler: Optional[Sampler] = None,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        warn_threshold: Optional[float] = None,
        block_threshold: Optional[float] = None,
        n_samples: int = DEFAULT_N_SAMPLES,
        entailment: Optional[EntailmentModel] = None,
        judge: Optional[Callable[[str], str]] = None,
        strict: bool = True,
        normalized: bool = True,
        on_defer: Optional[Callable[[str, EntropyResult], Any]] = None,
        on_block: Optional[Callable[[str, EntropyResult], Any]] = None,
        raise_on_block: bool = False,
        record_history: bool = True,
        max_history: int = 1000,
    ) -> None:
        if not 0.0 <= threshold:
            raise ValueError("threshold must be non-negative")
        if warn_threshold is None:
            warn_threshold = 0.6 * threshold
        if warn_threshold > threshold:
            raise ValueError("warn_threshold must be <= threshold")
        if block_threshold is not None and block_threshold < threshold:
            raise ValueError("block_threshold must be >= threshold")

        self.sampler = sampler
        self.threshold = threshold
        self.warn_threshold = warn_threshold
        self.block_threshold = block_threshold
        self.n_samples = n_samples
        self.entailment = entailment if entailment is not None else auto_entailment(judge=judge)
        self.strict = strict
        self.normalized = normalized
        self.on_defer = on_defer
        self.on_block = on_block
        self.raise_on_block = raise_on_block
        self.record_history = record_history
        self.max_history = max_history
        self.history: List[GateDecision] = []

    # -------------------------------------------------------------- measuring

    def measure(self, prompt: str, **kwargs: Any) -> EntropyResult:
        """Sample and score ``prompt`` without deciding anything."""
        if self.sampler is None:
            raise ValueError("Gate has no sampler; use check_samples(prompt, samples) instead")
        return score(
            prompt,
            self.sampler,
            n_samples=kwargs.pop("n_samples", self.n_samples),
            entailment=self.entailment,
            strict=self.strict,
            **kwargs,
        )

    def check(self, prompt: str, **kwargs: Any) -> GateDecision:
        """Sample, score and classify ``prompt`` into a :class:`GateDecision`."""
        return self.decide(self.measure(prompt, **kwargs))

    def check_samples(self, prompt: str, samples: Sequence[Any], **kwargs: Any) -> GateDecision:
        """Decide from generations you already hold — no model call."""
        result = score_samples(
            prompt, samples, entailment=self.entailment, strict=self.strict, **kwargs
        )
        return self.decide(result)

    def decide(self, result: EntropyResult) -> GateDecision:
        """Apply the threshold ladder to an already-computed result."""
        value = result.score_for(normalized=self.normalized)
        if self.block_threshold is not None and value >= self.block_threshold:
            action = GateAction.BLOCK
            reason = f"semantic entropy {value:.3f} >= block threshold {self.block_threshold:.3f}"
        elif value >= self.threshold:
            action = GateAction.DEFER
            reason = f"semantic entropy {value:.3f} >= defer threshold {self.threshold:.3f}"
        elif value >= self.warn_threshold:
            action = GateAction.WARN
            reason = f"semantic entropy {value:.3f} >= warn threshold {self.warn_threshold:.3f}"
        else:
            action = GateAction.ALLOW
            reason = f"semantic entropy {value:.3f} < warn threshold {self.warn_threshold:.3f}"

        warning = None
        if action is GateAction.WARN:
            warning = (
                f"Model uncertainty elevated: {result.n_clusters} distinct meanings across "
                f"{result.n_samples} samples (agreement {result.agreement:.0%}). "
                f"Consensus answer may be unreliable."
            )
        decision = GateDecision(
            action=action,
            result=result,
            threshold=self.threshold,
            reason=reason,
            warning=warning,
            metadata={
                "warn_threshold": self.warn_threshold,
                "block_threshold": self.block_threshold,
                "normalized": self.normalized,
                "score": value,
            },
        )
        self._record(decision)
        return decision

    # --------------------------------------------------------------- gating

    def run(
        self,
        prompt: str,
        action: Optional[Callable[..., Any]] = None,
        /,
        *args: Any,
        samples: Optional[Sequence[Any]] = None,
        **kwargs: Any,
    ) -> GateDecision:
        """Gate an action behind the uncertainty check.

        ``action`` is invoked **only** if the decision is ALLOW or WARN, and
        receives ``*args, **kwargs`` plus the keyword ``entropy_result`` if its
        signature accepts one. The returned decision always carries the full
        trace, whether or not the action ran.

        ``prompt`` and ``action`` are positional-only so that a wrapped function
        taking its own ``prompt=`` keyword does not collide with them. The name
        ``samples`` is still reserved by this method.

        >>> decision = gate.run("Refund order 41?", issue_refund, order_id=41)
        >>> if decision.executed:
        ...     print(decision.answer)
        ... else:
        ...     print(decision.explain())
        """
        decision = (
            self.check_samples(prompt, samples) if samples is not None else self.check(prompt)
        )

        if decision.action is GateAction.BLOCK:
            if self.raise_on_block:
                raise GateBlockedError(decision.reason, decision)
            decision.answer = self.on_block(prompt, decision.result) if self.on_block else None
            return decision

        if decision.action is GateAction.DEFER:
            decision.answer = self.on_defer(prompt, decision.result) if self.on_defer else None
            return decision

        if action is not None:
            decision.answer = _invoke(action, args, kwargs, decision.result)
            decision.executed = True
        else:
            decision.answer = decision.result.consensus_answer
            decision.executed = True
        return decision

    def wrap(self, fn: Callable[..., Any], *, prompt_arg: int = 0) -> Callable[..., GateDecision]:
        """Wrap a callable so every invocation is gated. Returns a decision.

        ``prompt_arg`` selects which positional argument carries the prompt (or
        pass the prompt as the keyword ``prompt``).
        """

        def wrapper(*args: Any, **kwargs: Any) -> GateDecision:
            prompt = kwargs.get("prompt")
            if prompt is None:
                if len(args) <= prompt_arg:
                    raise ValueError(
                        f"gated call needs a prompt at positional index {prompt_arg} "
                        "or as the keyword 'prompt'"
                    )
                prompt = args[prompt_arg]
            return self.run(str(prompt), fn, *args, **kwargs)

        wrapper.__name__ = getattr(fn, "__name__", "gated")
        wrapper.__doc__ = getattr(fn, "__doc__", None)
        wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
        wrapper.gate = self  # type: ignore[attr-defined]
        return wrapper

    def guard(self, fn: Callable[..., Any]) -> Callable[..., GateDecision]:
        """Decorator form of :meth:`wrap`.

        >>> @gate.guard
        ... def answer(prompt: str) -> str:
        ...     return llm(prompt)
        """
        return self.wrap(fn)

    # -------------------------------------------------------------- reporting

    def _record(self, decision: GateDecision) -> None:
        if not self.record_history:
            return
        self.history.append(decision)
        if len(self.history) > self.max_history:
            del self.history[: len(self.history) - self.max_history]

    def stats(self) -> Dict[str, Any]:
        """Aggregate counters over the gate's decision history."""
        counts: Dict[str, int] = {a.value: 0 for a in GateAction}
        for decision in self.history:
            counts[decision.action.value] += 1
        total = len(self.history) or 1
        entropies = [d.score for d in self.history]
        return {
            "total": len(self.history),
            "counts": counts,
            "deferral_rate": (counts["defer"] + counts["block"]) / total,
            "mean_entropy": sum(entropies) / total if entropies else 0.0,
            "threshold": self.threshold,
            "warn_threshold": self.warn_threshold,
            "block_threshold": self.block_threshold,
            "entailment_backend": self.entailment.name,
        }


def gate(
    sampler: Optional[Sampler] = None,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    **kwargs: Any,
) -> Gate:
    """Convenience constructor: ``g = gate(my_sampler, threshold=0.5)``."""
    return Gate(sampler, threshold=threshold, **kwargs)


def _invoke(
    action: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Dict[str, Any],
    result: EntropyResult,
) -> Any:
    """Call ``action``, passing ``entropy_result`` only if it wants it."""
    import inspect

    try:
        sig = inspect.signature(action)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return action(*args, **kwargs)
    params = sig.parameters
    accepts = "entropy_result" in params or any(p.kind is p.VAR_KEYWORD for p in params.values())
    if accepts:
        return action(*args, entropy_result=result, **kwargs)
    return action(*args, **kwargs)
