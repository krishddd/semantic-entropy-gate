"""Preflight: is this gate actually fit to protect a production agent?

Installing the package is not the same as deploying it correctly, and every way
of deploying it *incorrectly* produces a reassuringly low entropy (see
``docs/THREAT_MODEL.md``). So the failure you are most likely to ship is a gate
that looks like it is working.

This module answers the question directly, by *running* the checks rather than
describing them::

    sem-gate doctor --sampler myproject.llm:sample

    [PASS] entailment backend   cross-encoder:cross-encoder/nli-deberta-v3-xsmall (production)
    [PASS] backend loads        3 test pairs classified in 0.41s
    [PASS] sampler callable     myproject.llm:sample
    [FAIL] sampler diversity    10 draws produced 1 distinct string
           -> Your sampler looks deterministic (temperature 0, or a cache in
              front of it). Semantic entropy over identical samples is 0
              regardless of correctness: the gate would allow everything.
    [WARN] threshold            using the uncalibrated default (0.55)
           -> Run `sem-gate calibrate` on ~100 labelled prompts from your task.

    NOT READY: 1 failure, 1 warning

The sampler-diversity check is the important one. It is the only check here that
cannot be done by inspection — it requires actually drawing samples and looking
at them — and a deterministic sampler is the single most common way this library
gets deployed as a no-op.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .entailment import EntailmentModel, auto_entailment
from .refusal import DEFAULT_REFUSAL_DETECTOR, RefusalDetector
from .sampling import resolve_sampler

__all__ = ["Check", "PreflightReport", "preflight", "PASS", "WARN", "FAIL", "SKIP"]

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"

_ORDER = {FAIL: 0, WARN: 1, SKIP: 2, PASS: 3}


@dataclass
class Check:
    """One preflight result."""

    name: str
    status: str
    detail: str = ""
    remedy: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "remedy": self.remedy,
        }


@dataclass
class PreflightReport:
    """All checks, plus the single question a deployer cares about."""

    checks: List[Check] = field(default_factory=list)
    calibration: Optional[Any] = None
    """The fitted :class:`~semantic_entropy_gate.types.CalibrationResult` when a
    labelled dev set was supplied — so `doctor` hands you the threshold to
    deploy in the same breath as the verdict."""

    @property
    def failures(self) -> List[Check]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def warnings(self) -> List[Check]:
        return [c for c in self.checks if c.status == WARN]

    @property
    def ready(self) -> bool:
        """No failures. Warnings are survivable; failures are not."""
        return not self.failures

    def add(self, name: str, status: str, detail: str = "", remedy: str = "") -> Check:
        check = Check(name=name, status=status, detail=detail, remedy=remedy)
        self.checks.append(check)
        return check

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "n_failures": len(self.failures),
            "n_warnings": len(self.warnings),
            "checks": [c.to_dict() for c in self.checks],
            "calibration": self.calibration.to_dict() if self.calibration else None,
        }

    def render(self, width: int = 78) -> str:
        lines = ["=" * width, "SEMANTIC ENTROPY GATE - PREFLIGHT", "=" * width]
        for check in self.checks:
            lines.append(f"[{check.status:4s}] {check.name:<22s} {check.detail}")
            if check.remedy:
                for i, chunk in enumerate(_wrap(check.remedy, width - 11)):
                    lines.append(("       -> " if i == 0 else "          ") + chunk)
        lines.append("-" * width)
        if self.ready and not self.warnings:
            lines.append("READY: this gate is fit to protect a production agent.")
        elif self.ready:
            lines.append(
                f"READY WITH WARNINGS: {len(self.warnings)} warning(s). "
                "The gate will work, but read them."
            )
        else:
            lines.append(
                f"NOT READY: {len(self.failures)} failure(s), {len(self.warnings)} warning(s). "
                "Fix the failures before gating anything irreversible."
            )
        lines.append("=" * width)
        return "\n".join(lines)


def preflight(
    *,
    sampler: Optional[Callable[..., Any]] = None,
    entailment: Optional[EntailmentModel] = None,
    judge: Optional[Callable[[str], str]] = None,
    threshold: Optional[float] = None,
    calibrated: bool = False,
    n_samples: int = 10,
    probe_prompt: str = "In one sentence, what is the capital of France?",
    refusal_detector: Optional[RefusalDetector] = None,
    require_production_backend: bool = True,
    dev_set: Optional[Sequence[Any]] = None,
    confidence: float = 0.95,
) -> PreflightReport:
    """Run every deployment check and report whether this setup is safe to ship.

    Parameters
    ----------
    sampler:
        Your sampler. If given it is **actually called** — that is the only way
        to detect a deterministic one. Costs ``n_samples`` generations.
    entailment:
        The backend you intend to deploy. Defaults to :func:`auto_entailment`.
    threshold / calibrated:
        The threshold you intend to use, and whether it came from
        :func:`~semantic_entropy_gate.calibrate.calibrate` rather than the
        uncalibrated default.
    require_production_backend:
        Treat a ``triage``-tier backend as a failure rather than a warning.
    dev_set:
        Labelled prompts from **your task** (:class:`~semantic_entropy_gate.dataset.DatasetRow`,
        or anything with ``.prompt`` / ``.samples`` / ``.label``). This unlocks
        the one check none of the others can substitute for: whether semantic
        entropy actually separates hallucinations on your task, reported as an
        AUROC **with a confidence interval**. Rows carrying generations are
        scored offline; prompt-only rows need ``sampler``. Without it, the
        report carries an explicit SKIP naming this as the biggest remaining
        unknown — a wall of PASSes must not imply a validation that never ran.
    """
    report = PreflightReport()
    backend = _check_backend(report, entailment, judge, require_production_backend)
    _check_backend_loads(report, backend)
    _check_sampler(report, sampler, n_samples, probe_prompt)
    _check_threshold(report, threshold, calibrated)
    _check_refusal_detector(report, refusal_detector)
    report.calibration = _check_task_separation(
        report, dev_set, backend, sampler, n_samples, confidence
    )
    report.checks.sort(key=lambda c: _ORDER[c.status])
    return report


def _check_task_separation(
    report: PreflightReport,
    dev_set: Optional[Sequence[Any]],
    backend: EntailmentModel,
    sampler: Optional[Callable[..., Any]],
    n_samples: int,
    confidence: float,
) -> Optional[Any]:
    """The only check that answers the question that actually matters.

    Everything else here verifies that the *machinery* works. This one asks
    whether semantic entropy separates hallucinations **on your task** — and
    that cannot be answered by inspection, by a clever heuristic, or by trusting
    the paper's numbers. It needs prompts from your own domain, labelled with
    whether the model was actually right.

    So when no labelled data is supplied the check reports SKIP with the biggest
    remaining unknown stated plainly, rather than letting a wall of PASSes imply
    a validation that never happened.
    """
    from .calibrate import calibrate
    from .errors import CalibrationError
    from .score import score, score_samples

    if not dev_set:
        report.add(
            "task separation",
            SKIP,
            "no labelled dev set supplied",
            "THIS IS THE BIGGEST REMAINING UNKNOWN. Every other check above "
            "verifies that the machinery runs; none of them can tell you whether "
            "semantic entropy actually separates hallucinations on YOUR task. "
            "Only labelled data answers that. Build a dev set with "
            "`sem-gate label`, then re-run with --dev-set. ~100 prompts is "
            "usually enough; 30 is the minimum worth reporting.",
        )
        return None

    results, labels = [], []
    for row in dev_set:
        label = getattr(row, "label", None)
        if label is None:
            continue
        try:
            if getattr(row, "has_samples", False):
                result = score_samples(row.prompt, row.samples, entailment=backend)
            elif sampler is not None:
                result = score(row.prompt, sampler, n_samples=n_samples, entailment=backend)
            else:
                report.add(
                    "task separation",
                    FAIL,
                    "dev set has prompts but no generations, and no sampler was given",
                    "Either include a 'samples' field per row, or pass --sampler so "
                    "the prompts can be sampled.",
                )
                return None
        except Exception as exc:  # noqa: BLE001
            report.add("task separation", FAIL, f"scoring failed: {type(exc).__name__}: {exc}")
            return None
        results.append(result)
        labels.append(int(label))

    if len(labels) < 2 or len(set(labels)) < 2:
        report.add(
            "task separation",
            FAIL,
            f"{len(labels)} usable labelled row(s), {len(set(labels))} distinct label(s)",
            "Separation cannot be measured without both classes: prompts the model "
            "answered correctly (label 0) AND prompts where it hallucinated (label 1).",
        )
        return None

    try:
        calibration = calibrate(results, labels, confidence=confidence)
    except CalibrationError as exc:
        report.add("task separation", FAIL, str(exc))
        return None

    detail = (
        f"AUROC {calibration.auroc:.3f} "
        f"[{calibration.auroc_lower:.3f}, {calibration.auroc_upper:.3f}] "
        f"on {calibration.n_samples} labelled prompts"
    )

    if calibration.auroc < 0.5 and calibration.auroc_upper < 0.5:
        report.add(
            "task separation",
            FAIL,
            detail + " - ANTI-CORRELATED",
            "Semantic entropy is pointing the wrong way: high entropy is predicting "
            "CORRECT answers on this dev set. Check that label 1 means 'the model "
            "hallucinated' and not the reverse, and that samples line up with prompts.",
        )
    elif not calibration.separates:
        needed = _needed_for(calibration)
        report.add(
            "task separation",
            FAIL,
            detail,
            f"The confidence interval includes 0.5, so this dev set does not "
            f"establish that semantic entropy separates hallucinations on your task. "
            f"{needed} Do not deploy on this evidence: an unvalidated gate that "
            "defers traffic costs you money and buys nothing measurable.",
        )
    elif calibration.auroc < 0.65:
        report.add(
            "task separation",
            WARN,
            detail,
            "Real but weak separation. Usable as a soft signal (warn / defer); do "
            "not hard-block on it. Pair the gate with retrieval grounding.",
        )
    elif calibration.caveats:
        report.add(
            "task separation",
            WARN,
            detail,
            "Separation established, but: " + " ".join(calibration.caveats),
        )
    else:
        report.add(
            "task separation",
            PASS,
            detail,
            "" if calibration.auroc >= 0.8 else "Moderate but solid separation.",
        )

    report.add(
        "suggested threshold",
        PASS if calibration.trustworthy else WARN,
        f"{calibration.threshold:.4f} ({calibration.criterion}, "
        f"TPR {calibration.operating_point.tpr:.2f} / "
        f"FPR {calibration.operating_point.fpr:.2f})",
        ""
        if calibration.trustworthy
        else "Fitted on evidence that carries caveats - treat as provisional.",
    )
    return calibration


def _needed_for(calibration: Any) -> str:
    from .calibrate import required_dev_set_size

    needed = required_dev_set_size(
        calibration.auroc,
        confidence=calibration.confidence,
        positive_rate=calibration.base_rate or 0.5,
    )
    if needed is None:
        return (
            "At the observed effect size no realistic dev set would settle it, "
            "which is itself the answer: the signal is not there for this task."
        )
    if needed <= calibration.n_samples:
        return "Label more prompts, or check that your labels are correct."
    return f"About {needed} labelled prompts would settle it at this effect size."


# ------------------------------------------------------------------- checks


def _check_backend(
    report: PreflightReport,
    entailment: Optional[EntailmentModel],
    judge: Optional[Callable[[str], str]],
    require_production: bool,
) -> EntailmentModel:
    backend = entailment if entailment is not None else auto_entailment(judge=judge, quiet=True)
    tier = getattr(backend, "tier", "production")
    if tier == "production":
        report.add("entailment backend", PASS, f"{backend.name} ({tier})")
    else:
        report.add(
            "entailment backend",
            FAIL if require_production else WARN,
            f"{backend.name} ({tier})",
            "This backend compares words, not meanings. It is built for tests and a "
            "first look, not for gating live traffic. Install a real NLI model with "
            '`pip install "semantic-entropy-gate[hf]"`, or pass '
            "judge=<your chat callable> to use an LLM as the entailment oracle.",
        )
    return backend


def _check_backend_loads(report: PreflightReport, backend: EntailmentModel) -> None:
    """Actually classify a few pairs — a lazily-loaded model may fail on first use."""
    pairs = [
        ("Paris.", "The capital is Paris."),
        ("Paris.", "Lyon."),
        ("It costs 30 dollars.", "It costs 90 dollars."),
    ]
    started = time.time()
    try:
        for premise, hypothesis in pairs:
            backend.classify(premise, hypothesis, context="What is the capital of France?")
    except Exception as exc:  # noqa: BLE001 - the point is to catch anything
        report.add(
            "backend loads",
            FAIL,
            f"{type(exc).__name__}: {exc}",
            "The entailment backend could not classify a trivial pair. With a "
            "cross-encoder this usually means the checkpoint failed to download or "
            "torch is broken; with an LLM judge, that the API call failed.",
        )
        return
    elapsed = time.time() - started
    report.add("backend loads", PASS, f"{len(pairs)} test pairs classified in {elapsed:.2f}s")
    if elapsed > 5.0:
        report.add(
            "backend latency",
            WARN,
            f"{elapsed / len(pairs):.2f}s per pair",
            "Clustering makes up to N*(N-1) calls per prompt. At this latency a "
            "10-sample check would take minutes. Consider a smaller checkpoint, a "
            "GPU, or a SemanticEntropyProbe for the hot path.",
        )


def _check_sampler(
    report: PreflightReport,
    sampler: Optional[Callable[..., Any]],
    n_samples: int,
    probe_prompt: str,
) -> None:
    if sampler is None:
        report.add(
            "sampler",
            SKIP,
            "not supplied",
            "Pass --sampler module:function to check the single most common "
            "deployment mistake: a sampler that returns identical generations.",
        )
        return

    try:
        draw = resolve_sampler(sampler)
    except Exception as exc:  # noqa: BLE001
        report.add("sampler callable", FAIL, f"{type(exc).__name__}: {exc}")
        return
    report.add("sampler callable", PASS, getattr(sampler, "__name__", repr(sampler)))

    try:
        samples = draw(probe_prompt, n_samples)
    except Exception as exc:  # noqa: BLE001
        report.add(
            "sampler runs",
            FAIL,
            f"{type(exc).__name__}: {exc}",
            "The sampler raised when called. In production this would become a "
            "fail-closed DEFER on every request, which is safe but useless.",
        )
        return

    returned = len(samples)
    if returned < n_samples:
        report.add(
            "sampler count",
            FAIL if returned < max(2, n_samples // 2) else WARN,
            f"returned {returned} of {n_samples} requested",
            "Fewer generations than requested biases the score toward 'confident'. "
            "Check that your client honours the n / num_return_sequences parameter "
            "and is not rate-limiting.",
        )
    else:
        report.add("sampler count", PASS, f"returned {returned} of {n_samples}")

    distinct = {s.text.strip() for s in samples}
    if len(distinct) == 1:
        report.add(
            "sampler diversity",
            FAIL,
            f"{returned} draws produced 1 distinct string",
            "Your sampler looks deterministic (temperature 0, a cache in front of "
            "it, or a fixed seed). Semantic entropy over identical samples is 0 "
            "regardless of correctness, so the gate would allow everything. Set "
            "temperature to ~1.0 and make sure each draw is independent.",
        )
    elif len(distinct) < max(2, returned // 3):
        report.add(
            "sampler diversity",
            WARN,
            f"{returned} draws produced only {len(distinct)} distinct strings",
            "Low diversity suppresses the signal. If this is a genuinely easy "
            "prompt it is fine; if it holds for hard prompts too, raise the "
            "temperature.",
        )
    else:
        report.add(
            "sampler diversity", PASS, f"{returned} draws produced {len(distinct)} distinct strings"
        )

    with_logprobs = sum(1 for s in samples if s.logprob is not None)
    if with_logprobs == returned:
        report.add("log-probabilities", PASS, "available (Rao-Blackwellised estimator)")
    elif with_logprobs == 0:
        report.add(
            "log-probabilities",
            PASS,
            "absent (discrete estimator)",
            "Fine - the black-box estimator is used. Returning length-normalised "
            "mean token log-probabilities would give a more precise score at the "
            "same sample count.",
        )
    else:
        report.add(
            "log-probabilities",
            WARN,
            f"only {with_logprobs} of {returned} samples carry one",
            "Partial log-probabilities are ignored (the discrete estimator is used) "
            "because mixing would weight scored generations against unscored ones.",
        )


def _check_threshold(report: PreflightReport, threshold: Optional[float], calibrated: bool) -> None:
    if threshold is None:
        report.add(
            "threshold",
            WARN,
            "not supplied",
            "Pass the threshold you intend to deploy so it can be sanity-checked.",
        )
        return
    if not 0.0 <= threshold <= 1.0:
        report.add(
            "threshold",
            FAIL,
            f"{threshold} is outside [0, 1]",
            "Normalised semantic entropy is bounded by 1.0; a threshold above it can never fire.",
        )
        return
    if calibrated:
        report.add("threshold", PASS, f"{threshold:.4f} (calibrated)")
    else:
        report.add(
            "threshold",
            WARN,
            f"{threshold:.4f} (uncalibrated default)",
            "The correct threshold is task-dependent. Label ~100 prompts from your "
            "own task and run `sem-gate calibrate` - it reports the AUROC, which "
            "tells you whether semantic entropy separates your hallucinations at all.",
        )


def _check_refusal_detector(report: PreflightReport, detector: Optional[RefusalDetector]) -> None:
    detector = detector or DEFAULT_REFUSAL_DETECTOR
    probes = ["I don't know.", "Unknown.", "I cannot answer that."]
    answers = ["Paris.", "It is 30 days from delivery."]
    missed = [t for t in probes if not detector.is_refusal(t)[0]]
    false_positives = [t for t in answers if detector.is_refusal(t)[0]]
    if false_positives:
        report.add(
            "refusal detector",
            FAIL,
            f"{detector.name} flagged a real answer: {false_positives[0]!r}",
            "A detector that flags answers will defer good traffic and get the gate switched off.",
        )
    elif missed:
        report.add(
            "refusal detector",
            WARN,
            f"{detector.name} missed {len(missed)} of {len(probes)} standard refusals",
            "Refusals it misses fall through to the ordinary entropy check, which "
            "often catches them anyway. Extend with extra_patterns= if your domain "
            "phrases abstention unusually.",
        )
    else:
        report.add("refusal detector", PASS, detector.name)


def _wrap(text: str, width: int) -> List[str]:
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


def check_gate(gate: Any, **kwargs: Any) -> PreflightReport:
    """Preflight an already-configured :class:`~semantic_entropy_gate.gate.Gate`."""
    return preflight(
        sampler=gate.sampler,
        entailment=gate.entailment,
        threshold=gate.threshold,
        refusal_detector=gate.refusal_detector,
        **kwargs,
    )
