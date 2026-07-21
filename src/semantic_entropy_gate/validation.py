"""Label quality and traffic drift: the two assumptions `doctor` used to take on faith.

After v0.5.0, `sem-gate doctor --dev-set` establishes whether semantic entropy
separates hallucinations on your task. That verdict rests on two assumptions it
could not check:

1. **the labels are right** — AUROC against wrong labels measures nothing;
2. **the dev prompts resemble live traffic** — a threshold calibrated on one
   distribution is deployed against another.

Neither can be verified against *truth* — ground truth is yours, and a library
that claimed otherwise would be lying. But both have a checkable shadow:

**Labels.** A mislabel cannot be proven, but it can be *ranked*. A row labelled
"correct" whose generations scatter across five meanings, or labelled
"hallucinated" while the model answers unanimously, is exactly where labelling
mistakes concentrate (the intuition behind confident-learning methods: sort by
disagreement between annotation and signal, re-review the top). And one class of
label error is exact, not statistical: the same prompt labelled both ways.

**Traffic.** Whether the dev set resembled live traffic is unknowable at
deploy time and *measurable afterwards*: the gate keeps its decision history, so
the distribution of live entropy scores can be compared against the dev set's
score distribution with a two-sample Kolmogorov–Smirnov test. When they diverge,
the calibrated threshold is being applied to traffic it was never fitted on —
the honest response is to re-label a sample of current traffic and recalibrate,
and :meth:`~semantic_entropy_gate.gate.Gate.check_drift` says so.

Both checks follow the house rule: they never claim more than they measured.
A suspect is a *review candidate*, not a verdict; drift is *distribution
change*, not proof the threshold is wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .errors import CalibrationError
from .types import EntropyResult

__all__ = [
    "LabelSuspect",
    "LabelAudit",
    "audit_labels",
    "ks_2sample",
    "DriftReport",
    "detect_drift",
]


# ============================================================ label quality


@dataclass
class LabelSuspect:
    """One row whose label disagrees with the entropy signal.

    Not an accusation — a *review candidate*. ``misfit`` is how far the score
    sits on the wrong side for its label (0 = perfectly consistent, 1 = maximal
    contradiction). Mislabels concentrate at the top of this ranking; so do the
    genuinely-hard rows, which is fine: both deserve a second look.
    """

    index: int
    prompt: str
    label: int
    score: float
    misfit: float
    why: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "prompt": self.prompt,
            "label": self.label,
            "score": round(self.score, 4),
            "misfit": round(self.misfit, 4),
            "why": self.why,
        }


@dataclass
class LabelAudit:
    """Everything checkable about a labelled dev set, short of the truth."""

    suspects: List[LabelSuspect] = field(default_factory=list)
    conflicts: List[Tuple[str, List[int]]] = field(default_factory=list)
    """Prompts labelled inconsistently: ``(prompt, [row indices])``. Unlike
    suspects these are not statistical — the same question cannot be both a
    correct answer and a hallucination, so at least one label is wrong."""

    n_rows: int = 0

    @property
    def clean(self) -> bool:
        return not self.conflicts and not self.suspects

    @property
    def suspect_rate(self) -> float:
        return len(self.suspects) / self.n_rows if self.n_rows else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "suspects": [s.to_dict() for s in self.suspects],
            "conflicts": [
                {"prompt": prompt, "indices": indices} for prompt, indices in self.conflicts
            ],
            "suspect_rate": self.suspect_rate,
            "clean": self.clean,
        }


def audit_labels(
    results: Sequence[Union[EntropyResult, float]],
    labels: Sequence[Union[int, bool]],
    *,
    prompts: Optional[Sequence[str]] = None,
    max_suspects: int = 10,
    misfit_threshold: float = 0.5,
    normalized: bool = True,
) -> LabelAudit:
    """Rank the labels most likely to be wrong, and find the ones provably inconsistent.

    ``misfit`` for a row is the score's distance onto the wrong side for its
    label: a "correct" row scores ``entropy`` (high entropy contradicts the
    label), a "hallucinated" row scores ``1 - entropy``. Rows above
    ``misfit_threshold`` become suspects, worst first, capped at
    ``max_suspects``.

    This deliberately does **not** fold into `reliable`/`caveats` machinery as a
    hard failure: a dev set with a few hard rows is healthy, and a checker that
    cried wolf on every difficult example would train people to ignore it. It
    flags, explains, and stops.
    """
    if len(results) != len(labels):
        raise CalibrationError(f"results/labels length mismatch: {len(results)} vs {len(labels)}")

    scores: List[float] = []
    for item in results:
        if isinstance(item, EntropyResult):
            scores.append(item.score_for(normalized=normalized))
        else:
            scores.append(float(item))

    row_prompts: List[str] = []
    for i, item in enumerate(results):
        if prompts is not None and i < len(prompts):
            row_prompts.append(str(prompts[i]))
        elif isinstance(item, EntropyResult):
            row_prompts.append(item.prompt)
        else:
            row_prompts.append(f"row {i}")

    audit = LabelAudit(n_rows=len(scores))

    # Exact inconsistencies: identical prompt, different labels.
    seen: Dict[str, List[int]] = {}
    for i, prompt in enumerate(row_prompts):
        key = " ".join(prompt.split()).lower()
        seen.setdefault(key, []).append(i)
    for _key, indices in seen.items():
        if len({int(bool(labels[i])) for i in indices}) > 1:
            audit.conflicts.append((row_prompts[indices[0]], indices))

    # Statistical suspects: label contradicts the entropy signal.
    candidates: List[LabelSuspect] = []
    for i, (score, label) in enumerate(zip(scores, labels)):
        y = int(bool(label))
        misfit = score if y == 0 else 1.0 - score
        if misfit < misfit_threshold:
            continue
        if y == 0:
            why = (
                f"labelled CORRECT but the model produced high disagreement "
                f"(entropy {score:.2f}); if the consensus answer really is right, "
                "this is a hard row — otherwise the label is wrong"
            )
        else:
            why = (
                f"labelled HALLUCINATED but the model answered near-unanimously "
                f"(entropy {score:.2f}); consistent-but-wrong is possible (semantic "
                "entropy cannot see it), or the label is wrong"
            )
        candidates.append(
            LabelSuspect(
                index=i,
                prompt=row_prompts[i],
                label=y,
                score=score,
                misfit=misfit,
                why=why,
            )
        )
    candidates.sort(key=lambda s: -s.misfit)
    audit.suspects = candidates[:max_suspects]
    return audit


# ================================================================== drift


def ks_2sample(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Two-sample Kolmogorov–Smirnov test: ``(statistic, p_value)``.

    Pure stdlib. The statistic is the maximum distance between the two empirical
    CDFs; the p-value uses the standard asymptotic Kolmogorov distribution with
    the small-sample correction ``lambda = (sqrt(ne) + 0.12 + 0.11/sqrt(ne)) * D``
    (Numerical Recipes). Accurate enough for a monitoring verdict; this is a
    smoke alarm, not a journal submission, and it is labelled as such where the
    verdict is rendered.
    """
    if not a or not b:
        raise CalibrationError("ks_2sample needs two non-empty samples")
    xs = sorted(float(v) for v in a)
    ys = sorted(float(v) for v in b)
    n1, n2 = len(xs), len(ys)

    # Ties need care: the CDF gap may only be measured *between* distinct
    # values, so on equal values both pointers advance past the whole tie group
    # before the distance is taken. Entropy scores tie constantly (a confident
    # prompt is exactly 0.0), and the naive interleaving would read two
    # identical samples as maximally different.
    i = j = 0
    d = 0.0
    while i < n1 and j < n2:
        value = min(xs[i], ys[j])
        while i < n1 and xs[i] == value:
            i += 1
        while j < n2 and ys[j] == value:
            j += 1
        d = max(d, abs(i / n1 - j / n2))

    ne = n1 * n2 / (n1 + n2)
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * d
    # Q_KS(lam) = 2 * sum_{k>=1} (-1)^(k-1) exp(-2 k^2 lam^2). The alternating
    # series only converges for lam away from 0; at small lam it oscillates and
    # a truncated sum lands near 0 — a spurious "significant" verdict from two
    # identical samples. Numerical Recipes' probks answers non-convergence with
    # 1.0 (no evidence of difference), and so does this.
    p = 0.0
    previous_term = float("inf")
    converged = False
    for k in range(1, 101):
        term = 2.0 * ((-1.0) ** (k - 1)) * math.exp(-2.0 * (k * lam) ** 2)
        p += term
        if abs(term) < 1e-10 or abs(term) < 1e-6 * previous_term:
            converged = True
            break
        previous_term = abs(term)
    if not converged:
        p = 1.0
    return d, min(1.0, max(0.0, p))


@dataclass
class DriftReport:
    """Live traffic vs the dev set the threshold was calibrated on."""

    statistic: float
    p_value: float
    n_live: int
    n_dev: int
    drifted: bool
    alpha: float
    live_mean: float
    dev_mean: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 6),
            "n_live": self.n_live,
            "n_dev": self.n_dev,
            "drifted": self.drifted,
            "alpha": self.alpha,
            "live_mean": round(self.live_mean, 4),
            "dev_mean": round(self.dev_mean, 4),
        }

    def explain(self, width: int = 78) -> str:
        bar = "=" * width
        direction = (
            "live entropy runs HIGHER than the dev set (more uncertain traffic: "
            "expect more deferrals than calibration predicted)"
            if self.live_mean > self.dev_mean
            else "live entropy runs LOWER than the dev set (easier traffic, or a "
            "degraded sampler quietly suppressing disagreement - check `doctor`)"
        )
        lines = [
            bar,
            "TRAFFIC DRIFT CHECK (live gate scores vs calibration dev set)",
            bar,
            f"KS statistic: {self.statistic:.4f}   p-value: {self.p_value:.4g}   "
            f"(n_live={self.n_live}, n_dev={self.n_dev})",
            f"mean entropy: live {self.live_mean:.3f} vs dev {self.dev_mean:.3f}",
            "-" * width,
        ]
        if self.drifted:
            lines += [
                f"DRIFT DETECTED (p < {self.alpha}): the traffic this gate is judging no",
                "longer looks like the dev set its threshold was calibrated on. The",
                "threshold is not automatically wrong, but the evidence behind it no",
                f"longer applies. {direction}",
                "",
                "Remedy: `sem-gate label` a sample of RECENT prompts, re-run",
                "`sem-gate doctor --dev-set`, and redeploy the refitted threshold.",
            ]
        else:
            lines.append(
                f"no significant drift (p >= {self.alpha}): the calibration evidence "
                "still describes current traffic."
            )
        lines.append(bar)
        return "\n".join(lines)


def detect_drift(
    live_scores: Sequence[float],
    dev_scores: Sequence[float],
    *,
    alpha: float = 0.01,
    min_live: int = 20,
) -> DriftReport:
    """Compare live gate scores against the calibration dev set's scores.

    ``alpha`` is deliberately conservative (0.01, not 0.05): a drift monitor
    that pages on noise gets unplugged, exactly like a gate that defers on
    noise. ``min_live`` refuses to render a verdict on fewer than 20 live
    decisions — a KS test on 5 points is a coin flip wearing a formula.
    """
    if len(live_scores) < min_live:
        raise CalibrationError(
            f"only {len(live_scores)} live decision(s); drift detection needs at least "
            f"{min_live} to say anything defensible. Keep serving and check again."
        )
    statistic, p = ks_2sample(live_scores, dev_scores)
    return DriftReport(
        statistic=statistic,
        p_value=p,
        n_live=len(live_scores),
        n_dev=len(dev_scores),
        drifted=p < alpha,
        alpha=alpha,
        live_mean=sum(live_scores) / len(live_scores),
        dev_mean=sum(dev_scores) / len(dev_scores),
    )
