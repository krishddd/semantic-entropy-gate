"""Core data types for semantic-entropy-gate.

The pipeline is a straight line and each stage leaves an auditable artefact
behind:

    prompt -> [Sample, ...] -> [EntailmentJudgement, ...] -> [SemanticCluster, ...] -> EntropyResult

Every dataclass here is JSON-serialisable via :meth:`to_dict`, so a decision made
in production can be replayed, diffed and explained months later. That is the
whole point of the library: the number is useless if you cannot show a developer
*why* the model was judged uncertain.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class EntailmentLabel(str, Enum):
    """Standard three-way NLI verdict (MNLI label set)."""

    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"
    ENTAILMENT = "entailment"

    @property
    def score(self) -> int:
        """Numeric encoding used by the clustering algorithm (0/1/2)."""
        return {"contradiction": 0, "neutral": 1, "entailment": 2}[self.value]


@dataclass
class Sample:
    """One generation drawn from the model under test.

    ``logprob`` is the **length-normalised** mean token log-probability,
    ``(1/T) * sum_t log p(t_t | t_<t)``. It is optional: black-box APIs that hide
    token log-probabilities simply leave it ``None``, which routes the estimator
    to the discrete (count-based) formulation.
    """

    text: str
    logprob: Optional[float] = None
    tokens: Optional[int] = None
    raw: Optional[Any] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "logprob": self.logprob, "tokens": self.tokens}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sample":
        return cls(
            text=data["text"],
            logprob=data.get("logprob"),
            tokens=data.get("tokens"),
        )


@dataclass
class EntailmentJudgement:
    """A single directional NLI call: does ``premise`` entail ``hypothesis``?"""

    premise_index: int
    hypothesis_index: int
    label: EntailmentLabel
    confidence: Optional[float] = None
    backend: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "premise_index": self.premise_index,
            "hypothesis_index": self.hypothesis_index,
            "label": self.label.value,
            "confidence": self.confidence,
            "backend": self.backend,
        }


@dataclass
class SemanticCluster:
    """A semantic equivalence class: generations that bidirectionally entail.

    ``probability`` is the aggregated (and renormalised) mass of the cluster —
    Rao-Blackwellised from the member log-probabilities when available, otherwise
    the empirical frequency ``|C| / N``.
    """

    id: int
    member_indices: List[int]
    members: List[str]
    probability: float = 0.0
    representative: str = ""

    @property
    def size(self) -> int:
        return len(self.member_indices)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "member_indices": list(self.member_indices),
            "members": list(self.members),
            "probability": self.probability,
            "representative": self.representative,
            "size": self.size,
        }


class Estimator(str, Enum):
    """Which entropy estimator produced the score."""

    RAO_BLACKWELL = "rao_blackwell"
    """Length-normalised sequence likelihoods aggregated per cluster (white-box)."""

    DISCRETE = "discrete"
    """Empirical cluster frequencies only — needs no token log-probabilities."""


@dataclass
class EntropyResult:
    """The full, auditable outcome of scoring one prompt.

    Attributes
    ----------
    entropy:
        Semantic entropy in **nats**, over the semantic clusters. ``0`` means every
        sample meant the same thing; ``log(n_samples)`` is the maximum (every
        sample meant something different).
    normalized_entropy:
        ``entropy / log(n_samples)``, in ``[0, 1]``. Use this for thresholds that
        should survive a change in sample count.
    naive_entropy:
        Entropy over *raw strings* rather than meanings. Reported purely for
        contrast: the gap ``naive_entropy - entropy`` is exactly the lexical
        (paraphrase) uncertainty that token-level methods mistake for
        hallucination.
    """

    prompt: str
    samples: List[Sample]
    clusters: List[SemanticCluster]
    entropy: float
    normalized_entropy: float
    naive_entropy: float
    estimator: Estimator
    judgements: List[EntailmentJudgement] = field(default_factory=list)
    entailment_backend: str = ""
    cluster_assignments: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    refusal_rate: float = 0.0
    """Share of generations that declined to answer, in ``[0, 1]``.

    A separate axis from :attr:`reliable`: a unanimous refusal is a *successful*
    measurement of a *non-answer*. See :mod:`semantic_entropy_gate.refusal`.
    """

    abstained: bool = False
    """Every generation declined to answer.

    Semantic entropy is legitimately ~0 here — the model is consistent — but it
    is consistent about **not knowing**. Reading that zero as permission is how a
    gate ends up authorising an irreversible action on the strength of ten
    repetitions of "I don't know".
    """

    reliable: bool = True
    """Whether this measurement is capable of detecting disagreement at all.

    ``False`` means the pipeline broke in a way that *manufactures* a low score:
    too few generations, byte-identical generations, empty output, a poisoned
    log-probability. A low entropy from an unreliable measurement is not
    evidence of confidence, and :class:`~semantic_entropy_gate.gate.Gate`
    refuses to treat it as such.
    """

    # ------------------------------------------------------------------ views

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    @property
    def max_entropy(self) -> float:
        """Entropy of a uniform distribution over ``n_samples`` distinct meanings."""
        return math.log(self.n_samples) if self.n_samples > 1 else 0.0

    @property
    def lexical_entropy(self) -> float:
        """Uncertainty attributable to *phrasing* rather than *meaning* (nats).

        Never negative: semantic clustering can only merge outcomes, so the
        semantic entropy is bounded above by the naive string entropy.
        """
        return max(0.0, self.naive_entropy - self.entropy)

    @property
    def majority_cluster(self) -> Optional[SemanticCluster]:
        """The highest-probability semantic cluster — the model's modal answer."""
        if not self.clusters:
            return None
        return max(self.clusters, key=lambda c: (c.probability, c.size))

    @property
    def consensus_answer(self) -> Optional[str]:
        cluster = self.majority_cluster
        return cluster.representative if cluster else None

    @property
    def agreement(self) -> float:
        """Share of probability mass sitting in the majority cluster, in ``[0, 1]``."""
        cluster = self.majority_cluster
        return cluster.probability if cluster else 0.0

    def is_confabulation(self, threshold: float, *, normalized: bool = True) -> bool:
        """Flag the output as a likely confabulation at ``threshold``.

        ``normalized=True`` compares against :attr:`normalized_entropy` (the
        sample-count-independent score); pass ``False`` to threshold raw nats.

        **Fails closed.** A non-finite score (``NaN`` from a poisoned
        log-probability) would make every ``>=`` comparison ``False`` and read as
        confidence; here it returns ``True`` instead. An *unreliable* measurement
        (see :attr:`reliable`) also returns ``True``: if the pipeline could not
        actually look for disagreement, it has not shown there is none.
        """
        value = self.normalized_entropy if normalized else self.entropy
        if not math.isfinite(value):
            return True
        if not self.reliable:
            return True
        return value >= threshold

    def score_for(self, *, normalized: bool = True) -> float:
        return self.normalized_entropy if normalized else self.entropy

    # ----------------------------------------------------------- transparency

    def cluster_table(self) -> List[Tuple[int, int, float, str]]:
        """``(cluster_id, size, probability, representative)`` rows, most likely first."""
        rows = [
            (c.id, c.size, c.probability, c.representative)
            for c in sorted(self.clusters, key=lambda c: -c.probability)
        ]
        return rows

    def explain(self, *, threshold: Optional[float] = None, width: int = 78) -> str:
        """Render the human-readable audit trail for this decision.

        This is what you show a developer (or paste into an incident ticket) when
        they ask *"why did the gate stop my agent?"*.
        """
        from .safety import sanitize_text

        def clean(text: str, budget: int) -> str:
            """Untrusted model output, made safe to print and clipped to width."""
            flat = " ".join(sanitize_text(text).split())
            return flat if len(flat) <= budget else flat[: budget - 3] + "..."

        bar = "=" * width
        rule = "-" * width
        lines = [bar, "SEMANTIC ENTROPY REPORT", bar]
        lines.append(f"Prompt:    {clean(self.prompt, width - 10)}")
        lines.append(
            f"Samples:   {self.n_samples}   Clusters: {self.n_clusters}   "
            f"Estimator: {self.estimator.value}"
        )
        lines.append(f"Entailment backend: {self.entailment_backend or 'n/a'}")
        lines.append(rule)
        lines.append(
            f"Semantic entropy:   {self.entropy:.4f} nats "
            f"(normalized {self.normalized_entropy:.3f}, max {self.max_entropy:.4f})"
        )
        lines.append(
            f"Naive string entropy: {self.naive_entropy:.4f} nats  "
            f"-> lexical-only component {self.lexical_entropy:.4f}"
        )
        lines.append(f"Majority-cluster agreement: {self.agreement:.1%}")
        if self.refusal_rate:
            lines.append(
                f"Declined to answer: {self.refusal_rate:.0%} of generations"
                + ("  <- ALL of them" if self.abstained else "")
            )

        if self.abstained:
            lines.append(rule)
            lines.append("NON-ANSWER: the model consistently declined to answer.")
            lines.append("  Low entropy here means 'reliably no answer', not 'reliably correct'.")

        if not self.reliable or self.warnings:
            lines.append(rule)
            header = (
                "MEASUREMENT NOT RELIABLE - this score is not evidence of confidence"
                if not self.reliable
                else "MEASUREMENT NOTES"
            )
            lines.append(header)
            for warning in self.warnings:
                for index, chunk in enumerate(_wrap(sanitize_text(warning), width - 4)):
                    lines.append(("  - " if index == 0 else "    ") + chunk)

        lines.append(rule)
        lines.append("SEMANTIC CLUSTERS (meaning groups the model produced)")
        for cid, size, prob, rep in self.cluster_table():
            filled = max(0, min(20, int(round(prob * 20))))
            meter = "#" * filled + "." * (20 - filled)
            lines.append(f"  [{cid}] p={prob:6.3f} |{meter}| n={size:<3d} {clean(rep, width - 34)}")
        if self.n_clusters > 1:
            lines.append(rule)
            lines.append("DISAGREEMENT: the model asserted mutually exclusive answers.")
            for cid, _size, prob, rep in self.cluster_table()[:4]:
                lines.append(f"  cluster {cid} ({prob:.0%}): {clean(rep, width - 22)}")
        if threshold is not None:
            flagged = self.is_confabulation(threshold)
            if not self.reliable:
                verdict = "TREATED AS UNCERTAIN (measurement unreliable)"
            elif self.abstained:
                verdict = "NON-ANSWER (model declined; low entropy is not permission)"
            elif flagged:
                verdict = "CONFABULATION SUSPECTED"
            else:
                verdict = "within confidence budget"
            lines.append(rule)
            lines.append(
                f"Threshold {threshold:.3f} (normalized) -> "
                f"{self.normalized_entropy:.3f} : {verdict}"
            )
        lines.append(bar)
        return "\n".join(lines)

    # ------------------------------------------------------------ (de)serial.

    def to_dict(self, *, include_judgements: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "prompt": self.prompt,
            "samples": [s.to_dict() for s in self.samples],
            "clusters": [c.to_dict() for c in self.clusters],
            "cluster_assignments": list(self.cluster_assignments),
            "entropy": self.entropy,
            "normalized_entropy": self.normalized_entropy,
            "naive_entropy": self.naive_entropy,
            "lexical_entropy": self.lexical_entropy,
            "max_entropy": self.max_entropy,
            "agreement": self.agreement,
            "consensus_answer": self.consensus_answer,
            "estimator": self.estimator.value,
            "entailment_backend": self.entailment_backend,
            "n_samples": self.n_samples,
            "n_clusters": self.n_clusters,
            "reliable": self.reliable,
            "refusal_rate": self.refusal_rate,
            "abstained": self.abstained,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }
        if include_judgements:
            data["judgements"] = [j.to_dict() for j in self.judgements]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntropyResult":
        clusters = [
            SemanticCluster(
                id=c["id"],
                member_indices=list(c["member_indices"]),
                members=list(c["members"]),
                probability=c.get("probability", 0.0),
                representative=c.get("representative", ""),
            )
            for c in data.get("clusters", [])
        ]
        judgements = [
            EntailmentJudgement(
                premise_index=j["premise_index"],
                hypothesis_index=j["hypothesis_index"],
                label=EntailmentLabel(j["label"]),
                confidence=j.get("confidence"),
                backend=j.get("backend", ""),
            )
            for j in data.get("judgements", [])
        ]
        return cls(
            prompt=data["prompt"],
            samples=[Sample.from_dict(s) for s in data.get("samples", [])],
            clusters=clusters,
            entropy=data["entropy"],
            normalized_entropy=data["normalized_entropy"],
            naive_entropy=data.get("naive_entropy", 0.0),
            estimator=Estimator(data.get("estimator", "discrete")),
            judgements=judgements,
            entailment_backend=data.get("entailment_backend", ""),
            cluster_assignments=list(data.get("cluster_assignments", [])),
            metadata=dict(data.get("metadata", {})),
            warnings=list(data.get("warnings", [])),
            refusal_rate=float(data.get("refusal_rate", 0.0)),
            abstained=bool(data.get("abstained", False)),
            # Absent in v0.1.0 reports. Default to reliable so old artefacts keep
            # deserialising, but a missing flag is recorded rather than assumed.
            reliable=bool(data.get("reliable", True)),
        )


class GateAction(str, Enum):
    """What the middleware decided to do with the call."""

    ALLOW = "allow"
    """Entropy below the warn threshold — proceed silently."""

    WARN = "warn"
    """Elevated entropy — proceed, but attach an uncertainty warning."""

    DEFER = "defer"
    """Above the defer threshold — hand control back (ask a human, search, retry)."""

    BLOCK = "block"
    """Above the block threshold — refuse to execute the world-altering action."""

    @property
    def allowed(self) -> bool:
        return self in (GateAction.ALLOW, GateAction.WARN)


@dataclass
class GateDecision:
    """The middleware verdict for one gated call, with its full justification."""

    action: GateAction
    result: EntropyResult
    threshold: float
    reason: str
    answer: Optional[Any] = None
    executed: bool = False
    warning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    recommended_policy: Optional[str] = None
    """On a DEFER, the name of the expected-free-energy-minimising next action.

    ``None`` unless the gate was given a policy set (see
    :mod:`semantic_entropy_gate.active_inference`). A generic DEFER says "do not
    act"; this says *which* information-seeking move — retrieve, clarify,
    re-sample, escalate — the ambiguity actually calls for. The full ranking is on
    ``metadata['policy_ranking']``.
    """

    @property
    def allowed(self) -> bool:
        return self.action.allowed

    @property
    def score(self) -> float:
        return self.result.normalized_entropy

    def explain(self) -> str:
        header = f"GATE DECISION: {self.action.value.upper()}  ({self.reason})"
        body = self.result.explain(threshold=self.threshold)
        return f"{header}\n{body}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "threshold": self.threshold,
            "reason": self.reason,
            "executed": self.executed,
            "warning": self.warning,
            "answer": self.answer if _json_safe(self.answer) else repr(self.answer),
            "recommended_policy": self.recommended_policy,
            "result": self.result.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass
class ThresholdPoint:
    """One operating point on the ROC / threshold sweep."""

    threshold: float
    tpr: float
    fpr: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    youden_j: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReliabilityBin:
    """One bin of a reliability diagram: predicted vs. observed hallucination rate.

    A threshold gate reads the entropy score *as if* it were a probability — "0.6
    means roughly a 60% chance this answer is wrong". ``mean_predicted`` is what
    the score claimed on average inside this bin; ``fraction_positive`` is what
    actually happened. Their gap is the miscalibration ECE sums up.
    """

    lower: float
    upper: float
    count: int
    mean_predicted: float
    fraction_positive: float

    @property
    def gap(self) -> float:
        """``|predicted - observed|`` for this bin — the local calibration error."""
        return abs(self.mean_predicted - self.fraction_positive)

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "gap": self.gap}


@dataclass
class CalibrationResult:
    """Output of :func:`semantic_entropy_gate.calibrate.calibrate`.

    ``threshold`` is the recommended decision boundary on the *normalised*
    semantic entropy; ``auroc`` reports how separable the labelled dev set was in
    the first place. An AUROC near 0.5 means semantic entropy carries no signal
    for your task and you should not deploy the gate on it.
    """

    threshold: float
    auroc: float
    auroc_lower: float
    auroc_upper: float
    auprc: float
    criterion: str
    n_samples: int
    n_positive: int
    n_negative: int
    operating_point: ThresholdPoint
    curve: List[ThresholdPoint] = field(default_factory=list)
    base_rate: float = 0.0
    confidence: float = 0.95
    dev_scores: List[float] = field(default_factory=list)
    """The entropy scores this calibration was fitted on. Kept so runtime drift
    detection (:meth:`~semantic_entropy_gate.gate.Gate.check_drift`) has its
    reference distribution — without it, "does live traffic still look like the
    dev set?" is unanswerable."""

    caveats: List[str] = field(default_factory=list)
    """Reasons this threshold is less trustworthy than its decimals suggest."""

    ece: Optional[float] = None
    """Expected Calibration Error of the raw entropy score read as a probability.

    AUROC asks "does a higher score rank hallucinations above correct answers?" —
    a *ranking* question. ECE asks the *threshold* question: "when the score says
    0.6, is the answer wrong ~60% of the time?" A gate compares the score to a
    fixed number, so it implicitly trusts the score as a probability; ECE measures
    how far that trust is misplaced. ``None`` when calibration ran without labels
    binned finely enough to estimate it. See :data:`ECE_DEPLOY_MAX`.
    """

    reliability: List["ReliabilityBin"] = field(default_factory=list)
    """The per-bin reliability diagram behind :attr:`ece`, for plotting/inspection."""

    @property
    def calibrated(self) -> Optional[bool]:
        """Whether the score is usable as a probability (ECE within budget).

        ``None`` when ECE was not estimated. A gate can separate well (high AUROC)
        yet be badly calibrated (high ECE): the scores rank correctly but cluster
        in a band the threshold never reaches. Both must hold before a fixed
        threshold behaves as intended.
        """
        if self.ece is None:
            return None
        from .calibrate import ECE_DEPLOY_MAX

        return self.ece <= ECE_DEPLOY_MAX

    @property
    def separates(self) -> bool:
        """Does this dev set *establish* a signal, not merely suggest one?

        True only when the whole confidence interval sits above chance. A point
        estimate of 0.8 with an interval of [0.45, 0.95] has not shown anything;
        reporting it as if it had is the overconfidence this library detects,
        turned on the library itself.
        """
        return self.auroc_lower > 0.5

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def trustworthy(self) -> bool:
        """No caveats, better-than-chance separation, and calibrated if measured."""
        return (
            not self.caveats
            and self.auroc >= 0.65
            and self.separates
            and self.calibrated is not False
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold": self.threshold,
            "auroc": self.auroc,
            "auroc_lower": self.auroc_lower,
            "auroc_upper": self.auroc_upper,
            "auroc_ci": [self.auroc_lower, self.auroc_upper],
            "confidence": self.confidence,
            "separates": self.separates,
            "auprc": self.auprc,
            "criterion": self.criterion,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "base_rate": self.base_rate,
            "ece": self.ece,
            "calibrated": self.calibrated,
            "reliability": [b.to_dict() for b in self.reliability],
            "caveats": list(self.caveats),
            "dev_scores": [round(v, 6) for v in self.dev_scores],
            "trustworthy": self.trustworthy,
            "operating_point": self.operating_point.to_dict(),
            "curve": [p.to_dict() for p in self.curve],
            "metadata": dict(self.metadata),
        }

    def explain(self, width: int = 78) -> str:
        bar = "=" * width
        op = self.operating_point
        quality = "strong" if self.auroc >= 0.8 else "moderate" if self.auroc >= 0.65 else "weak"
        return "\n".join(
            [
                bar,
                "SEMANTIC ENTROPY CALIBRATION",
                bar,
                f"Dev set:      {self.n_samples} prompts "
                f"({self.n_positive} hallucinated / {self.n_negative} correct, "
                f"base rate {self.base_rate:.1%})",
                f"AUROC:        {self.auroc:.4f}  "
                f"[{self.auroc_lower:.3f}, {self.auroc_upper:.3f}] "
                f"{int(self.confidence * 100)}% CI  ({quality} separation)",
                (
                    "              the interval clears chance: the signal is real"
                    if self.separates
                    else "              the interval INCLUDES 0.5: separation is NOT established"
                ),
                f"AUPRC:        {self.auprc:.4f}",
                (
                    f"ECE:          {self.ece:.4f}  "
                    + (
                        "(score reads as a probability)"
                        if self.calibrated
                        else "(score is a RANKING, not a probability at this threshold)"
                    )
                    if self.ece is not None
                    else "ECE:          n/a"
                ),
                "-" * width,
                f"Criterion:    {self.criterion}",
                f"Threshold:    {self.threshold:.4f}  (normalized semantic entropy)",
                f"  TPR (catch rate)  {op.tpr:.3f}      FPR (false alarms) {op.fpr:.3f}",
                f"  precision {op.precision:.3f}  recall {op.recall:.3f}  "
                f"F1 {op.f1:.3f}  accuracy {op.accuracy:.3f}",
            ]
            + (
                ["-" * width, "CAVEATS"] + [f"  - {c}" for c in self.caveats]
                if self.caveats
                else []
            )
            + [bar]
        )


def _wrap(text: str, width: int) -> List[str]:
    """Minimal greedy word-wrap (no textwrap import for one call site)."""
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= width:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _json_safe(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, type(None), list, dict, tuple))


def normalize_texts(samples: Sequence[Any]) -> List[Sample]:
    """Coerce a heterogeneous sampler return value into ``list[Sample]``.

    Accepts plain strings, ``(text, logprob)`` pairs, mappings with a ``text``
    key, and :class:`Sample` instances — so a user's sampler can be as simple as
    ``lambda p, n: [call_model(p) for _ in range(n)]``.
    """
    out: List[Sample] = []
    for item in samples:
        if isinstance(item, Sample):
            out.append(item)
        elif isinstance(item, str):
            out.append(Sample(text=item))
        elif isinstance(item, dict):
            out.append(Sample.from_dict(item))
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            text, logprob = item
            out.append(Sample(text=str(text), logprob=None if logprob is None else float(logprob)))
        else:
            out.append(Sample(text=str(item)))
    return out
