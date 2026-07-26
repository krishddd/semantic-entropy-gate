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

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence

from .active_inference import Policy, default_policies, rank_policies
from .entailment import EntailmentModel, auto_entailment
from .errors import GateBlockedError
from .probes import SemanticEntropyProbe
from .refusal import RefusalDetector
from .safety import (
    DEFAULT_LIMITS,
    MIN_SAMPLES_FOR_ENTROPY,
    Limits,
    validate_threshold,
)
from .sampling import Sampler
from .score import DEFAULT_N_SAMPLES, score, score_samples
from .types import EntropyResult, Estimator, GateAction, GateDecision

__all__ = ["Gate", "gate", "DEFAULT_THRESHOLD", "REFUSAL_POLICIES", "MODEL_ACCESS"]

MODEL_ACCESS = ("api_only", "local_same_model", "local_proxy_model")
"""How much of the model under test the deployment can actually see.

``api_only``          — a black-box API (GPT/Claude/Gemini). No hidden states, so
                        no Semantic Entropy Probe; only the full N-sample path.
``local_same_model``  — you host *the same model* you are gating, so its hidden
                        states are the ones the probe was trained on. The only
                        configuration in which a probe is valid.
``local_proxy_model`` — you host a *different* model as a stand-in. Its activation
                        geometry is not the probe's; a probe here scores confident
                        nonsense, so the gate refuses it.
"""

REFUSAL_POLICIES = ("defer", "block", "allow")
"""What to do when every generation declined to answer.

``"defer"`` (default) hands control back — the agent should retrieve, ask, or
escalate, which is precisely what an honest "I don't know" calls for.
``"block"`` refuses outright. ``"allow"`` restores the pre-0.3.0 behaviour of
reading the (legitimately low) entropy as permission; the abstention still
travels on the decision as a warning.
"""

_WARNED_BACKENDS: set = set()
"""Backends already warned about, so a server building a gate per request does
not fill its logs with the same line."""

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
        fail_closed: bool = True,
        require_reliable: bool = True,
        limits: Limits = DEFAULT_LIMITS,
        min_samples: int = MIN_SAMPLES_FOR_ENTROPY,
        refusal_policy: str = "defer",
        refusal_detector: Optional[RefusalDetector] = None,
        require_production_backend: bool = False,
        policies: Optional[Sequence[Policy]] = None,
        route_policies: bool = False,
        probe: Optional[SemanticEntropyProbe] = None,
        model_access: str = "api_only",
        model_family: Optional[str] = None,
        probe_floor: float = 0.15,
        probe_ceiling: float = 0.85,
        probe_fast_allow: bool = False,
    ) -> None:
        config_warnings = validate_threshold(threshold, normalized=normalized)
        if warn_threshold is None:
            warn_threshold = 0.6 * threshold
        else:
            config_warnings += validate_threshold(
                warn_threshold, normalized=normalized, name="warn_threshold"
            )
        # Ordering is a hard invariant: warn above defer, or block below defer,
        # is a misconfiguration that would issue inverted decisions for the gate's
        # whole lifetime, so it raises rather than warns.
        if warn_threshold > threshold:
            raise ValueError(
                f"warn_threshold ({warn_threshold:.4f}) must be <= threshold "
                f"({threshold:.4f}): a WARN band above the DEFER band is inverted."
            )
        if block_threshold is not None:
            config_warnings += validate_threshold(
                block_threshold, normalized=normalized, name="block_threshold"
            )
            if block_threshold < threshold:
                raise ValueError(
                    f"block_threshold ({block_threshold:.4f}) must be >= threshold "
                    f"({threshold:.4f}): a BLOCK band below the DEFER band is inverted."
                )
        # Equality is legal but collapses a tier: warn == threshold disables WARN
        # (ALLOW runs up to the defer line); block == threshold disables DEFER
        # (uncertain calls BLOCK outright). That can be deliberate, but a gate that
        # advertises four tiers and silently issues three is exactly the kind of
        # unstated claim this library refuses to make, so say it out loud.
        if warn_threshold == threshold:
            config_warnings.append(
                f"warn_threshold equals threshold ({threshold:.4f}): the WARN tier is "
                "zero-width and will never fire (calls go ALLOW -> DEFER directly)."
            )
        if block_threshold is not None and block_threshold == threshold:
            config_warnings.append(
                f"block_threshold equals threshold ({threshold:.4f}): the DEFER tier is "
                "zero-width and will never fire (uncertain calls BLOCK outright)."
            )
        for message in config_warnings:
            warnings.warn(message, UserWarning, stacklevel=2)

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
        if refusal_policy not in REFUSAL_POLICIES:
            raise ValueError(
                f"refusal_policy must be one of {REFUSAL_POLICIES}, got {refusal_policy!r}"
            )
        self.fail_closed = fail_closed
        self.require_reliable = require_reliable
        self.limits = limits
        self.min_samples = min_samples
        self.refusal_policy = refusal_policy
        self.refusal_detector = refusal_detector
        # DEFER means "acquire information before acting" — but *which* action
        # reduces this uncertainty is a policy question, not a threshold question.
        # When given a policy set (or route_policies=True for the default act/
        # forage pair), the gate ranks them by expected free energy and stamps the
        # winner onto every DEFER decision, so the on_defer handler can route to
        # retrieval / clarification / escalation instead of one generic bucket.
        if policies is not None:
            self.policies: Optional[List[Policy]] = list(policies)
        elif route_policies:
            self.policies = default_policies()
        else:
            self.policies = None

        # The score is only as trustworthy as the oracle underneath it. Shipping
        # the stdlib heuristic in front of an irreversible action is a mistake
        # that looks exactly like a working guardrail, so say so out loud.
        self.backend_tier = getattr(self.entailment, "tier", "production")
        if self.backend_tier != "production":
            message = (
                f"entailment backend {self.entailment.name!r} is tier "
                f"{self.backend_tier!r}: it compares words, not meanings, and is "
                "intended for tests and a first look rather than production "
                'traffic. Install a real NLI model (pip install "semantic-entropy-'
                'gate[hf]") or pass judge=<chat callable>.'
            )
            if require_production_backend:
                raise ValueError(message)
            config_warnings = list(config_warnings) + [message]
            if self.entailment.name not in _WARNED_BACKENDS:
                _WARNED_BACKENDS.add(self.entailment.name)
                warnings.warn(message, UserWarning, stacklevel=2)
        self.require_production_backend = require_production_backend

        # --- Semantic Entropy Probe capability guard (D-3) ---------------------
        # A probe predicts entropy from a single hidden state at ~1/10th the cost,
        # but only for the model whose activation geometry it was trained on.
        # Every way this can be misconfigured produces a *silently* invalid score,
        # so each one is a hard error at construction, not a runtime surprise.
        if model_access not in MODEL_ACCESS:
            raise ValueError(f"model_access must be one of {MODEL_ACCESS}, got {model_access!r}")
        self.model_access = model_access
        self.model_family = model_family
        self.probe = probe
        self.probe_floor = probe_floor
        self.probe_ceiling = probe_ceiling
        self.probe_fast_allow = probe_fast_allow
        if probe is not None:
            if model_access != "local_same_model":
                raise ValueError(
                    "A Semantic Entropy Probe reads output_hidden_states, which exist "
                    "only for a locally-hosted model. model_access="
                    f"{model_access!r} cannot supply them: an api_only deployment has no "
                    "hidden states, and a local_proxy_model's activations are a "
                    "different geometry than the probe was trained on, so its scores "
                    "would be confident nonsense. Set model_access='local_same_model' "
                    "only if you host the same model you are gating; otherwise drop "
                    "probe= and use the full N-sample path."
                )
            if not probe.is_fitted:
                raise ValueError("probe is not fitted; call .fit(...) before gating on it")
            if not (0.0 <= probe_floor < probe_ceiling <= 1.0):
                raise ValueError(
                    f"probe band invalid: need 0 <= probe_floor ({probe_floor}) < "
                    f"probe_ceiling ({probe_ceiling}) <= 1"
                )
            if (
                model_family is not None
                and probe.model_family is not None
                and probe.model_family != model_family
            ):
                raise ValueError(
                    f"probe was trained on model_family {probe.model_family!r} but this "
                    f"gate is for {model_family!r}: applying a probe to another model's "
                    "activations produces invalid scores. Retrain the probe on this "
                    "model, or gate the model it was trained on."
                )
            # These append AFTER the config_warnings emit-loop above, so they must
            # warn themselves to reach the operator rather than sit silently on the
            # gate. A probe pointed at the wrong (or an unvalidated) model is the
            # exact silent-failure D-3 is about, so it must be audible.
            if probe.model_family is None:
                message = (
                    "probe has no recorded model_family, so the gate cannot verify it "
                    "matches the model being gated; a probe from the wrong model scores "
                    "silently. Set probe.model_family (and the gate's model_family) to "
                    "make the check enforceable."
                )
                config_warnings.append(message)
                warnings.warn(message, UserWarning, stacklevel=2)
            if probe.train_auroc is None:
                message = (
                    "probe reports no validation AUROC (train_auroc is None): it may be "
                    "unvalidated. A probe fast-path is only as trustworthy as the probe."
                )
                config_warnings.append(message)
                warnings.warn(message, UserWarning, stacklevel=2)

        self.config_warnings = config_warnings
        self.history: List[GateDecision] = []

    # -------------------------------------------------------------- measuring

    def measure(
        self, prompt: str, *, hidden_state: Optional[Sequence[float]] = None, **kwargs: Any
    ) -> EntropyResult:
        """Sample and score ``prompt`` without deciding anything.

        If a ``hidden_state`` is supplied and this gate has a probe, the fast/slow
        cascade runs first: when the probe is confident (below the floor or above
        the ceiling) its cheap estimate is returned directly; only the ambiguous
        middle band pays for the full N-sample measurement. Average latency
        collapses toward one forward pass while the hard cases still get the real
        pipeline.

        Propagates exceptions. Use :meth:`check` if you want the failure turned
        into a fail-closed decision instead of an exception.
        """
        if hidden_state is not None and self.probe is not None:
            screened = self._probe_screen(prompt, hidden_state)
            if screened is not None:
                return screened
        if self.sampler is None:
            raise ValueError("Gate has no sampler; use check_samples(prompt, samples) instead")
        return score(
            prompt,
            self.sampler,
            n_samples=kwargs.pop("n_samples", self.n_samples),
            entailment=self.entailment,
            strict=self.strict,
            limits=kwargs.pop("limits", self.limits),
            min_samples=kwargs.pop("min_samples", self.min_samples),
            refusal_detector=kwargs.pop("refusal_detector", self.refusal_detector),
            **kwargs,
        )

    def _probe_screen(self, prompt: str, hidden_state: Sequence[float]) -> Optional[EntropyResult]:
        """Return a probe estimate when the probe is confident, else ``None``.

        A high-confidence *high*-entropy reading always short-circuits (it can only
        make the gate more cautious). A high-confidence *low*-entropy reading only
        short-circuits when ``probe_fast_allow`` is set, because fast-allowing an
        irreversible action on a single-pass estimate is a weaker measurement than
        the N-sample one it replaces — opt-in, in keeping with "a measurement that
        did not happen is not evidence of confidence".
        """
        p = self.probe.predict_proba(hidden_state)
        if p >= self.probe_ceiling:
            return self._probe_result(prompt, p, band="high")
        if p <= self.probe_floor and self.probe_fast_allow:
            return self._probe_result(prompt, p, band="low")
        return None

    def _probe_result(self, prompt: str, p: float, *, band: str) -> EntropyResult:
        """Build the minimal EntropyResult carrying a probe estimate."""
        note = (
            f"Semantic Entropy Probe estimate (P(high entropy)={p:.3f}) from a single "
            "hidden state, not an N-sample measurement; the probe was confident enough "
            f"to short-circuit ({band} band). Trained on model_family="
            f"{self.probe.model_family!r}."
        )
        return EntropyResult(
            prompt=prompt,
            samples=[],
            clusters=[],
            entropy=p,
            normalized_entropy=p,
            naive_entropy=0.0,
            estimator=Estimator.PROBE,
            entailment_backend=f"sep-probe(model_family={self.probe.model_family})",
            metadata={
                "probe": True,
                "probe_proba": p,
                "probe_band": band,
                "probe_floor": self.probe_floor,
                "probe_ceiling": self.probe_ceiling,
                "model_family": self.probe.model_family,
                "probe_train_auroc": self.probe.train_auroc,
            },
            warnings=[note],
            reliable=True,
        )

    def check(self, prompt: str, **kwargs: Any) -> GateDecision:
        """Sample, score and classify ``prompt`` into a :class:`GateDecision`.

        With ``fail_closed=True`` (the default) a sampler or entailment-backend
        failure does not raise: it returns a DEFER/BLOCK decision carrying the
        error. The alternative — an exception propagating into a caller whose
        ``except`` clause falls back to "just run the action" — is how guardrails
        get bypassed in production.
        """
        try:
            return self.decide(self.measure(prompt, **kwargs))
        except Exception as exc:  # noqa: BLE001 - deliberately broad: fail closed
            if not self.fail_closed:
                raise
            return self._failure_decision(prompt, exc)

    def check_samples(self, prompt: str, samples: Sequence[Any], **kwargs: Any) -> GateDecision:
        """Decide from generations you already hold — no model call."""
        try:
            result = score_samples(
                prompt,
                samples,
                entailment=self.entailment,
                strict=self.strict,
                limits=kwargs.pop("limits", self.limits),
                min_samples=kwargs.pop("min_samples", self.min_samples),
                refusal_detector=kwargs.pop("refusal_detector", self.refusal_detector),
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - deliberately broad: fail closed
            if not self.fail_closed:
                raise
            return self._failure_decision(prompt, exc)
        return self.decide(result)

    def _failure_decision(self, prompt: str, exc: BaseException) -> GateDecision:
        """Turn a measurement failure into the most conservative decision available.

        BLOCK when the gate has a block threshold configured (the operator has
        said some things must never run unverified); otherwise DEFER. Never
        ALLOW: an uncertainty check that did not run has not shown the model was
        certain.
        """
        action = GateAction.BLOCK if self.block_threshold is not None else GateAction.DEFER
        result = EntropyResult(
            prompt=prompt,
            samples=[],
            clusters=[],
            entropy=float("inf"),
            normalized_entropy=1.0,
            naive_entropy=0.0,
            estimator=Estimator.DISCRETE,
            entailment_backend=self.entailment.name,
            metadata={"error": f"{type(exc).__name__}: {exc}"},
            warnings=[
                f"uncertainty measurement failed ({type(exc).__name__}: {exc}); "
                "treating the call as maximally uncertain"
            ],
            reliable=False,
        )
        decision = GateDecision(
            action=action,
            result=result,
            threshold=self.threshold,
            reason=f"measurement failed and the gate is fail-closed: {type(exc).__name__}: {exc}",
            metadata={"failed": True, "error_type": type(exc).__name__, "score": 1.0},
        )
        self._record(decision)
        return decision

    def decide(self, result: EntropyResult) -> GateDecision:
        """Apply the threshold ladder to an already-computed result.

        An **unreliable** measurement short-circuits the ladder: a low score
        produced by a pipeline that could not detect disagreement is not
        evidence of agreement, so it is never allowed to open the gate. Set
        ``require_reliable=False`` to opt out, which you should only do if you
        have another check downstream.
        """
        value = result.score_for(normalized=self.normalized)

        if self.require_reliable and not result.reliable:
            action = GateAction.BLOCK if self.block_threshold is not None else GateAction.DEFER
            why = result.warnings[0] if result.warnings else "measurement marked unreliable"
            decision = GateDecision(
                action=action,
                result=result,
                threshold=self.threshold,
                reason=(
                    f"unreliable measurement (entropy {value:.3f} is not evidence of "
                    f"confidence): {why}"
                ),
                metadata={
                    "warn_threshold": self.warn_threshold,
                    "block_threshold": self.block_threshold,
                    "normalized": self.normalized,
                    "score": value,
                    "unreliable": True,
                },
            )
            self._record(decision)
            return decision

        if result.abstained and self.refusal_policy != "allow":
            # The measurement is sound and the entropy is genuinely low — the
            # model is consistent. It is consistently declining to answer, and a
            # non-answer is not authorisation to act.
            action = GateAction.BLOCK if self.refusal_policy == "block" else GateAction.DEFER
            decision = GateDecision(
                action=action,
                result=result,
                threshold=self.threshold,
                reason=(
                    f"model declined to answer in all {result.n_samples} generations; "
                    f"semantic entropy {value:.3f} reflects a consistent NON-ANSWER, "
                    "not a confident one"
                ),
                metadata={
                    "warn_threshold": self.warn_threshold,
                    "block_threshold": self.block_threshold,
                    "normalized": self.normalized,
                    "score": value,
                    "abstained": True,
                    "refusal_rate": result.refusal_rate,
                },
            )
            self._record(decision)
            return decision

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

        # Caveats accumulate rather than overwrite each other: an allowed call
        # that was *both* borderline and a non-answer must carry both facts, or
        # the one that got dropped is the one the reader needed.
        notes: List[str] = []
        if action is GateAction.WARN:
            notes.append(
                f"Model uncertainty elevated: {result.n_clusters} distinct meanings across "
                f"{result.n_samples} samples (agreement {result.agreement:.0%}). "
                f"Consensus answer may be unreliable."
            )
        if result.abstained:
            notes.append(
                f"The model declined to answer in all {result.n_samples} generations "
                f"(refusal_policy={self.refusal_policy!r}); the low score reflects a "
                "consistent non-answer."
            )
        if action.allowed and result.warnings:
            notes.extend(result.warnings)
        warning = " ".join(notes) if notes else None
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
                "unreliable": not result.reliable,
            },
        )
        if action is GateAction.DEFER:
            self._route_policies(decision)
        self._record(decision)
        return decision

    def _route_policies(self, decision: GateDecision) -> None:
        """Stamp the expected-free-energy-minimising next action onto a DEFER.

        No-op unless the gate was configured with a policy set. Ranking is over
        the *result*, so the recommendation reflects how much uncertainty is
        actually left to resolve: foraging wins when ambiguity is high, and a
        policy that gains nothing epistemically loses. The full ranking is kept on
        the decision so the choice is auditable, not just asserted.
        """
        if not self.policies:
            return
        ranked = rank_policies(self.policies, decision.result, normalized=self.normalized)
        if not ranked:
            return
        decision.recommended_policy = ranked[0].policy.name
        decision.metadata["policy_ranking"] = [e.to_dict() for e in ranked]

    # --------------------------------------------------------------- gating

    def run(
        self,
        prompt: str,
        action: Optional[Callable[..., Any]] = None,
        /,
        *args: Any,
        samples: Optional[Sequence[Any]] = None,
        hidden_state: Optional[Sequence[float]] = None,
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
            self.check_samples(prompt, samples)
            if samples is not None
            else self.check(prompt, hidden_state=hidden_state)
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

    def resolve(
        self,
        prompt: str,
        forage: Callable[[str, GateDecision], Optional[str]],
        *,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> GateDecision:
        """Run a **bounded** information-seeking loop around a DEFER.

        The gate itself is single-shot: :meth:`check` measures once and returns.
        The natural next move on DEFER — acquire context, then measure again — is
        a loop, and a loop with no cap is a production hazard: a structurally
        ambiguous prompt ("what is the best programming language?") stays high-
        entropy forever, a forager that doesn't actually reduce uncertainty spins,
        and an oracle jittering near the threshold oscillates. Each turn costs N
        generations and O(N^2) NLI calls, so an unbounded loop is unbounded spend.

        ``resolve`` is that loop with the cap built in, so it cannot be forgotten.
        ``forage(prompt, decision) -> Optional[str]`` acquires information and
        returns a **revised prompt** (e.g. the question with retrieved context
        appended) to try again, or ``None`` to stop foraging and accept the
        current decision. After ``max_retries`` unsuccessful attempts the loop
        exits with the last DEFER decision, stamped ``max_retries_exceeded`` so the
        caller can escalate to a human rather than hang.

        ALLOW / WARN / BLOCK all terminate immediately — foraging only makes sense
        while the gate is asking for more information.
        """
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")

        decision = self.check(prompt, **kwargs)
        attempts = 0
        while decision.action is GateAction.DEFER and attempts < max_retries:
            attempts += 1
            revised = forage(prompt, decision)
            if revised is None:
                # The forager has nothing more to try; accept the current DEFER
                # rather than loop on an unchanged prompt.
                break
            prompt = revised
            decision = self.check(prompt, **kwargs)

        decision.metadata["defer_attempts"] = attempts
        if decision.action is GateAction.DEFER and attempts >= max_retries > 0:
            decision.metadata["max_retries_exceeded"] = True
            decision.reason += (
                f" (still deferring after {attempts} forage attempt(s); escalate to a human)"
            )
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

    def relabel_sample(self, k: int = 30, *, seed: int = 0) -> List[Dict[str, Any]]:
        """A deterministic random sample of recent decisions, ready for `sem-gate label`.

        The honest response to drift is re-labelling a sample of *current*
        traffic — and "a sample" is where honesty quietly leaks: hand-picking
        the prompts that look easy (or hard) biases the recalibration toward
        the answer someone wants. This makes the sample **protocol instead of
        trust**: seeded, drawn uniformly from the gate's own history, and
        reproducible by anyone holding the same history and seed. Write the
        rows to JSONL and label them:

            write_jsonl("relabel.jsonl", gate.relabel_sample(30))
            # sem-gate label --input relabel.jsonl --out dev2.jsonl ...

        Generations are included, so labelling is offline and the scores are
        re-derivable. Failed and unreliable measurements are excluded — they
        carry no answer to judge.
        """
        import random as _random

        eligible = [
            d
            for d in self.history
            if d.result.reliable and not d.metadata.get("failed") and d.result.samples
        ]
        rng = _random.Random(seed)
        chosen = eligible if len(eligible) <= k else rng.sample(eligible, k)
        return [
            {
                "prompt": d.result.prompt,
                "samples": [s.text for s in d.result.samples],
                "labeled_answer": d.result.consensus_answer,
            }
            for d in chosen
        ]

    def check_drift(self, calibration: Any, *, alpha: float = 0.01, min_live: int = 20):
        """Does live traffic still look like the dev set this gate was calibrated on?

        The one assumption `sem-gate doctor` cannot verify at deploy time —
        that the dev set represents live traffic — becomes measurable the moment
        the gate has history: compare the entropy scores of real decisions
        against the calibration's ``dev_scores`` with a two-sample KS test.

        ``calibration`` is a :class:`~semantic_entropy_gate.types.CalibrationResult`
        (or any object with ``dev_scores``), typically the one `doctor` returned
        or `Report.load(...).calibration`. Raises
        :class:`~semantic_entropy_gate.errors.CalibrationError` when there is
        not enough live history to say anything defensible — a verdict from five
        decisions would be noise wearing a formula.

        Failed and unreliable measurements are excluded: they score a synthetic
        1.0 that says nothing about the traffic distribution.
        """
        from .validation import detect_drift

        dev_scores = getattr(calibration, "dev_scores", None) or calibration
        live = [d.score for d in self.history if d.result.reliable and not d.metadata.get("failed")]
        return detect_drift(live, list(dev_scores), alpha=alpha, min_live=min_live)

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
            "backend_tier": self.backend_tier,
            "abstentions": sum(1 for d in self.history if d.result.abstained),
            "unreliable": sum(1 for d in self.history if not d.result.reliable),
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
