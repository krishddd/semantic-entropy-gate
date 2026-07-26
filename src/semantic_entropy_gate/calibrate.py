"""Threshold calibration on a labelled dev set, with AUROC.

A guardrail with an uncalibrated threshold is theatre. Give this module ~100
prompts from *your* task, each labelled ``1`` if the model's answer was wrong /
hallucinated and ``0`` if it was correct, and it will:

1. report **AUROC** — how well semantic entropy separates the two classes at all
   (0.5 = no signal, do not deploy; the Nature paper reports ~0.75-0.85 on
   free-form QA);
2. report **AUPRC** and the base rate, which is what actually matters when
   hallucinations are rare;
3. sweep every candidate threshold and pick one under an explicit criterion —
   Youden's J, max F1, a target false-positive budget, or a target recall.

Everything is exact rank statistics on plain Python lists: no numpy, no sklearn.
AUROC uses the Mann-Whitney U identity with proper mid-rank tie handling, so a
dev set where many prompts share an entropy value (very common at small N) is
scored correctly rather than optimistically.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .errors import CalibrationError
from .types import CalibrationResult, EntropyResult, ReliabilityBin, ThresholdPoint

__all__ = [
    "calibrate",
    "auroc",
    "auroc_ci",
    "required_dev_set_size",
    "auprc",
    "roc_curve",
    "threshold_sweep",
    "reliability_curve",
    "expected_calibration_error",
    "PlattScaler",
    "fit_platt",
    "ECE_DEPLOY_MAX",
    "CRITERIA",
]

ECE_DEPLOY_MAX = 0.10
"""Deployment budget for Expected Calibration Error.

Above this, the entropy score does not behave as a probability: a fixed
threshold on it does not mean what its decimals suggest, and either the threshold
must be fitted directly (which :func:`calibrate` already does) or the score must
be :class:`PlattScaler`-mapped before it is read as ``P(hallucination)``. The
0.10 figure is the common reporting convention, not a law; tighten it for
higher-stakes gates.
"""

Scored = Union[EntropyResult, float, int]

SMALL_DEV_SET = 30
"""Below this many labelled prompts, a fitted threshold is a number rather than a
measurement. Calibration still runs, but says so."""

CRITERIA = ("youden", "f1", "target_fpr", "target_recall", "accuracy")
"""Selection rules understood by :func:`calibrate`."""


def _as_score(item: Scored, *, normalized: bool = True) -> float:
    if isinstance(item, EntropyResult):
        return item.score_for(normalized=normalized)
    return float(item)


def _validate_scores(scores: Sequence[float]) -> List[float]:
    """Reject non-finite scores instead of ranking them.

    ``NaN`` compares ``False`` against everything, so Python's sort silently
    produces an arbitrary order and the resulting AUROC is a plausible-looking
    number computed from nonsense. A calibration that is quietly wrong is worse
    than one that refuses to run, because its output is a deployed threshold.
    """
    bad = [i for i, s in enumerate(scores) if not math.isfinite(s)]
    if bad:
        preview = ", ".join(str(i) for i in bad[:5])
        raise CalibrationError(
            f"{len(bad)} score(s) are NaN or infinite (index {preview}"
            f"{'...' if len(bad) > 5 else ''}). Ranking them would silently corrupt "
            "the AUROC. Drop those rows, or fix the sampler that produced them."
        )
    return [float(s) for s in scores]


def calibrate(
    results: Sequence[Scored],
    labels: Sequence[Union[int, bool]],
    *,
    criterion: str = "youden",
    target_fpr: float = 0.1,
    target_recall: float = 0.8,
    normalized: bool = True,
    confidence: float = 0.95,
    metadata: Optional[Dict[str, Any]] = None,
) -> CalibrationResult:
    """Pick a decision threshold from a labelled dev set.

    Parameters
    ----------
    results:
        :class:`EntropyResult` objects (or raw scores) for each dev prompt.
    labels:
        ``1``/``True`` = the model **hallucinated** (the positive class we want to
        catch), ``0``/``False`` = the answer was correct.
    criterion:
        - ``"youden"``     maximise ``TPR - FPR`` (balanced; the default)
        - ``"f1"``         maximise F1 on the positive (hallucination) class
        - ``"accuracy"``   maximise plain accuracy
        - ``"target_fpr"`` highest recall subject to ``FPR <= target_fpr`` — use
          this when false alarms are expensive (a gate that defers constantly
          gets switched off)
        - ``"target_recall"`` lowest FPR subject to ``TPR >= target_recall`` — use
          this when a missed hallucination is expensive
    normalized:
        Threshold on normalised entropy (recommended) rather than raw nats.

    Raises
    ------
    CalibrationError
        Empty input, mismatched lengths, or a single-class dev set (AUROC is
        undefined and any threshold would be arbitrary).
    """
    if len(results) != len(labels):
        raise CalibrationError(f"results/labels length mismatch: {len(results)} vs {len(labels)}")
    if not results:
        raise CalibrationError("empty dev set")
    if criterion not in CRITERIA:
        raise CalibrationError(f"unknown criterion {criterion!r}; expected one of {CRITERIA}")

    scores = _validate_scores([_as_score(r, normalized=normalized) for r in results])
    ys = [1 if bool(y) else 0 for y in labels]
    n_pos = sum(ys)
    n_neg = len(ys) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise CalibrationError(
            "dev set contains a single class "
            f"({n_pos} hallucinated / {n_neg} correct); "
            "calibration needs both positive and negative examples"
        )

    curve = threshold_sweep(scores, ys)
    point = _select(curve, criterion, target_fpr=target_fpr, target_recall=target_recall)
    auc, lower, upper = auroc_ci(scores, ys, confidence=confidence)

    # ECE only means anything when the score lives on the probability scale it is
    # being read as. Raw-nats calibration is a valid ranking exercise but its
    # scores are unbounded, so skip the probability metric rather than report a
    # meaningless number.
    reliability = reliability_curve(scores, ys) if normalized else []
    ece = expected_calibration_error(scores, ys) if normalized else None

    # A threshold fitted to a handful of prompts is a number, not a measurement.
    # It will look authoritative in a report, so the report has to say otherwise.
    caveats: List[str] = []
    if lower <= 0.5:
        needed = required_dev_set_size(auc, confidence=confidence, positive_rate=n_pos / len(ys))
        extra = (
            f" About {needed} labelled prompts would settle it at this effect size."
            if needed
            else " At this effect size no realistic dev set would settle it, which is "
            "itself the answer: the signal is not there."
        )
        caveats.append(
            f"the {int(confidence * 100)}% confidence interval for AUROC "
            f"[{lower:.3f}, {upper:.3f}] includes 0.5, so this dev set does NOT "
            "establish that semantic entropy separates hallucinations on your task." + extra
        )
    if len(ys) < SMALL_DEV_SET:
        caveats.append(
            f"fitted on only {len(ys)} prompts (recommended >= {SMALL_DEV_SET}): this "
            "threshold and its AUROC are both high-variance, and a rerun on different "
            "prompts may move them substantially"
        )
    if min(n_pos, n_neg) < 5:
        caveats.append(
            f"the smaller class has {min(n_pos, n_neg)} example(s); metrics conditioned "
            "on it (precision, recall) are essentially unmeasured"
        )
    if len(set(scores)) < 3:
        caveats.append(
            f"the dev set contains only {len(set(scores))} distinct score(s), so the "
            "threshold sweep had almost nothing to choose between"
        )
    if ece is not None and ece > ECE_DEPLOY_MAX:
        caveats.append(
            f"the score is not a calibrated probability (ECE {ece:.3f} > "
            f"{ECE_DEPLOY_MAX:.2f}): AUROC {auc:.3f} says it *ranks* hallucinations "
            "well, but a fixed threshold reads it as a probability and that reading "
            "is off. The fitted threshold above still works (it is chosen on the "
            "ranking); only stop interpreting the raw score as 'X% likely wrong'. "
            "Use calibrate.fit_platt(...) if you need probabilities downstream."
        )

    return CalibrationResult(
        threshold=point.threshold,
        auroc=auc,
        auroc_lower=lower,
        auroc_upper=upper,
        confidence=confidence,
        auprc=auprc(scores, ys),
        criterion=criterion,
        n_samples=len(ys),
        n_positive=n_pos,
        n_negative=n_neg,
        operating_point=point,
        curve=curve,
        base_rate=n_pos / len(ys),
        dev_scores=list(scores),
        ece=ece,
        reliability=reliability,
        caveats=caveats,
        metadata={
            "normalized": normalized,
            "target_fpr": target_fpr,
            "target_recall": target_recall,
            **(metadata or {}),
        },
    )


def calibrate_from_pairs(
    pairs: Iterable[Tuple[Scored, Union[int, bool]]], **kwargs: Any
) -> CalibrationResult:
    """``calibrate`` over an iterable of ``(result, label)`` tuples."""
    items = list(pairs)
    return calibrate([p[0] for p in items], [p[1] for p in items], **kwargs)


# ------------------------------------------------------------------- metrics


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney U) identity.

    Ties get mid-ranks, which is the difference between an honest AUROC and an
    inflated one when many prompts share a score (unavoidable at N=10 samples,
    where normalised entropy takes only a handful of distinct values).
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise CalibrationError("AUROC is undefined for a single-class label set")
    scores = _validate_scores(scores)

    ranks = _mid_ranks(scores)
    rank_sum_pos = sum(r for r, y in zip(ranks, labels) if y == 1)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


Z_FOR_CONFIDENCE = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
"""Two-sided normal quantiles, so no scipy dependency is needed."""


def auroc_ci(
    scores: Sequence[float], labels: Sequence[int], *, confidence: float = 0.95
) -> Tuple[float, float, float]:
    """AUROC with a confidence interval: ``(auc, lower, upper)``.

    **Why this matters more than the point estimate.** An AUROC of 0.78 measured
    on 30 prompts and one measured on 300 are not the same claim, but they print
    identically. Reporting the point estimate alone is exactly the overconfidence
    this library exists to detect — applied to the library's own output.

    If the interval includes 0.5 you have not established that semantic entropy
    separates hallucinations on your task *at all*, however good the number
    looks. Deploying on that basis is guessing with extra steps.

    Uses the Hanley & McNeil (1982) closed form: with ``A`` the AUC,

        Q1 = A / (2 - A),  Q2 = 2A^2 / (1 + A)
        SE = sqrt( [A(1-A) + (n_pos-1)(Q1 - A^2) + (n_neg-1)(Q2 - A^2)] / (n_pos * n_neg) )

    It is the standard analytic estimator, needs no resampling (so it is exactly
    reproducible), and is mildly conservative — it assumes exponential score
    distributions, which real entropy scores are not. Treat the interval as
    indicative, not exact; the conclusion you should draw from it ("is the lower
    bound above 0.5?") is robust to that approximation.
    """
    if confidence not in Z_FOR_CONFIDENCE:
        raise CalibrationError(
            f"confidence must be one of {sorted(Z_FOR_CONFIDENCE)}, got {confidence}"
        )
    a = auroc(scores, labels)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    # Continuity correction for the degenerate ends. At an observed AUC of
    # exactly 1.0 (common on small dev sets) the Hanley-McNeil variance is 0 and
    # the interval collapses to [1.0, 1.0] — "perfect separation, established
    # with certainty, from 12 prompts". That is precisely the overconfidence
    # this library exists to flag, so the variance is computed as if half a
    # discordant pair had been observed: the least separation the data could
    # still be hiding.
    pairs = n_pos * n_neg
    a_var = min(max(a, 0.5 / pairs), 1.0 - 0.5 / pairs)

    q1 = a_var / (2.0 - a_var)
    q2 = (2.0 * a_var * a_var) / (1.0 + a_var)
    numerator = (
        a_var * (1.0 - a_var)
        + (n_pos - 1) * (q1 - a_var * a_var)
        + (n_neg - 1) * (q2 - a_var * a_var)
    )
    variance = max(0.0, numerator) / pairs
    se = math.sqrt(variance)
    z = Z_FOR_CONFIDENCE[confidence]
    return a, max(0.0, a - z * se), min(1.0, a + z * se)


def required_dev_set_size(
    observed_auroc: float,
    *,
    confidence: float = 0.95,
    positive_rate: float = 0.5,
    max_n: int = 5000,
) -> Optional[int]:
    """Roughly how many labelled prompts would establish this AUROC as real.

    Answers the question a deployer actually asks when told "your interval
    includes 0.5": *how many more labels do I need?* Returns the smallest total
    dev-set size whose lower confidence bound clears 0.5 at the observed effect
    size, or ``None`` if no size within ``max_n`` would (i.e. the effect is too
    small to be worth chasing — the signal is not there).
    """
    if observed_auroc <= 0.5:
        return None
    a = observed_auroc
    q1 = a / (2.0 - a)
    q2 = (2.0 * a * a) / (1.0 + a)
    z = Z_FOR_CONFIDENCE[confidence]

    for n in range(10, max_n + 1, 2):
        n_pos = max(1, int(round(n * positive_rate)))
        n_neg = max(1, n - n_pos)
        numerator = a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)
        se = math.sqrt(max(0.0, numerator) / (n_pos * n_neg))
        if a - z * se > 0.5:
            return n
    return None


def _mid_ranks(values: Sequence[float]) -> List[float]:
    """1-based ranks, averaged within groups of equal value."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def auprc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Average precision: ``sum_k (R_k - R_{k-1}) * P_k`` over descending score.

    The step-wise (non-interpolated) estimator, which does not optimistically
    bridge across gaps in the curve.
    """
    n_pos = sum(labels)
    if n_pos == 0:
        raise CalibrationError("AUPRC is undefined with no positive examples")
    scores = _validate_scores(scores)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = 0
    fp = 0
    prev_recall = 0.0
    total = 0.0
    i = 0
    while i < len(order):
        j = i
        # Consume all examples sharing this score at once: a threshold cannot
        # separate tied scores, so they must be counted as one operating point.
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            if labels[order[k]] == 1:
                tp += 1
            else:
                fp += 1
        recall = tp / n_pos
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        total += (recall - prev_recall) * precision
        prev_recall = recall
        i = j + 1
    return total


def reliability_curve(
    scores: Sequence[float], labels: Sequence[int], *, n_bins: int = 10
) -> List[ReliabilityBin]:
    """Bin scores into ``n_bins`` equal-width buckets over ``[0, 1]``.

    Each bucket reports how many predictions fell in it, the mean predicted value
    (the score, read as ``P(hallucination)``), and the observed hallucination rate
    (fraction of positives). A perfectly calibrated score has ``mean_predicted ==
    fraction_positive`` in every non-empty bin. Empty bins are dropped rather than
    reported as a spurious ``0 == 0`` agreement.

    Scores are expected in ``[0, 1]`` (normalised entropy). Anything outside is
    clamped into the end bins, because a probability estimate cannot live outside
    the unit interval — and a raw-nats score does not belong here at all.
    """
    if n_bins < 1:
        raise CalibrationError("n_bins must be >= 1")
    if len(scores) != len(labels):
        raise CalibrationError(f"scores/labels length mismatch: {len(scores)} vs {len(labels)}")
    scores = _validate_scores(scores)
    ys = [1 if bool(y) else 0 for y in labels]

    counts = [0] * n_bins
    sum_pred = [0.0] * n_bins
    sum_pos = [0] * n_bins
    for s, y in zip(scores, ys):
        clamped = min(1.0, max(0.0, s))
        # The top edge (1.0) belongs to the last bin, not a phantom (n_bins+1)th.
        idx = min(n_bins - 1, int(clamped * n_bins))
        counts[idx] += 1
        sum_pred[idx] += clamped
        sum_pos[idx] += y

    bins: List[ReliabilityBin] = []
    for i in range(n_bins):
        if counts[i] == 0:
            continue
        bins.append(
            ReliabilityBin(
                lower=i / n_bins,
                upper=(i + 1) / n_bins,
                count=counts[i],
                mean_predicted=sum_pred[i] / counts[i],
                fraction_positive=sum_pos[i] / counts[i],
            )
        )
    return bins


def expected_calibration_error(
    scores: Sequence[float], labels: Sequence[int], *, n_bins: int = 10
) -> float:
    """Count-weighted mean gap between predicted and observed hallucination rate.

    ``ECE = sum_b (n_b / N) * |mean_predicted_b - fraction_positive_b|``.

    This is the metric a *threshold* gate actually needs, and the one AUROC is
    silent on: an AUROC of 0.87 whose scores all sit between 0.3 and 0.5 never
    reaches a defer threshold of 0.7, so the gate is always ALLOW despite
    "excellent" separation. ECE catches that; AUROC cannot.
    """
    bins = reliability_curve(scores, labels, n_bins=n_bins)
    total = len(scores)
    if total == 0:
        raise CalibrationError("cannot compute ECE on an empty dev set")
    return sum(b.count * b.gap for b in bins) / total


class PlattScaler:
    """A one-dimensional logistic map ``score -> P(hallucination)``.

    When ECE is poor, the entropy score ranks well but is not itself a
    probability. Platt scaling fits ``sigmoid(a * score + b)`` on the dev set so
    the *output* can be read as a probability — turning a ranking signal into a
    calibrated one without disturbing the ordering (``a`` is constrained positive
    only implicitly by the fit; a genuinely anti-correlated score will produce a
    negative ``a``, which :func:`calibrate` already flags upstream).

    Pure-Python gradient descent, matching the dependency-free house style; a
    two-parameter fit on a few hundred points converges in milliseconds.
    """

    def __init__(self, a: float = 1.0, b: float = 0.0) -> None:
        self.a = a
        self.b = b

    def __call__(self, score: float) -> float:
        return _sigmoid(self.a * float(score) + self.b)

    def transform(self, scores: Sequence[float]) -> List[float]:
        return [self(s) for s in scores]

    def to_dict(self) -> Dict[str, float]:
        return {"a": self.a, "b": self.b}


def fit_platt(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    learning_rate: float = 0.5,
    epochs: int = 500,
) -> PlattScaler:
    """Fit a :class:`PlattScaler` mapping raw scores to calibrated probabilities."""
    xs = _validate_scores(scores)
    ys = [1.0 if bool(y) else 0.0 for y in labels]
    if len(xs) != len(ys):
        raise CalibrationError(f"scores/labels length mismatch: {len(xs)} vs {len(ys)}")
    if not xs:
        raise CalibrationError("no data to fit Platt scaling")
    if len(set(int(y) for y in ys)) < 2:
        raise CalibrationError("Platt scaling needs both classes present")

    a, b = 1.0, 0.0
    n = len(xs)
    for _ in range(epochs):
        grad_a = grad_b = 0.0
        for x, y in zip(xs, ys):
            p = _sigmoid(a * x + b)
            error = p - y
            grad_a += error * x
            grad_b += error
        a -= learning_rate * grad_a / n
        b -= learning_rate * grad_b / n
    return PlattScaler(a, b)


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def threshold_sweep(scores: Sequence[float], labels: Sequence[int]) -> List[ThresholdPoint]:
    """Every distinct operating point, ordered by ascending threshold.

    A prompt is flagged when ``score >= threshold``. Candidate thresholds are the
    distinct observed scores plus a sentinel above the maximum (flag nothing), so
    the sweep spans the full ROC from (1,1) to (0,0).
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise CalibrationError("threshold sweep needs both classes present")
    scores = _validate_scores(scores)

    candidates = sorted(set(scores))
    step = 1e-9 if len(candidates) < 2 else (candidates[-1] - candidates[0]) / 1000 or 1e-9
    candidates.append(candidates[-1] + max(step, 1e-9))

    points: List[ThresholdPoint] = []
    for threshold in candidates:
        tp = fp = fn = tn = 0
        for score, y in zip(scores, labels):
            flagged = score >= threshold
            if y == 1 and flagged:
                tp += 1
            elif y == 1:
                fn += 1
            elif flagged:
                fp += 1
            else:
                tn += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tpr
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        accuracy = (tp + tn) / len(labels)
        points.append(
            ThresholdPoint(
                threshold=threshold,
                tpr=tpr,
                fpr=fpr,
                precision=precision,
                recall=recall,
                f1=f1,
                accuracy=accuracy,
                youden_j=tpr - fpr,
            )
        )
    return points


def roc_curve(scores: Sequence[float], labels: Sequence[int]) -> List[Tuple[float, float]]:
    """``[(fpr, tpr), ...]`` ordered by ascending FPR — plot-ready."""
    points = threshold_sweep(scores, labels)
    return sorted(((p.fpr, p.tpr) for p in points))


def _select(
    curve: Sequence[ThresholdPoint],
    criterion: str,
    *,
    target_fpr: float,
    target_recall: float,
) -> ThresholdPoint:
    if criterion == "youden":
        return max(curve, key=lambda p: (p.youden_j, -p.threshold))
    if criterion == "f1":
        return max(curve, key=lambda p: (p.f1, -p.threshold))
    if criterion == "accuracy":
        return max(curve, key=lambda p: (p.accuracy, -p.threshold))
    if criterion == "target_fpr":
        feasible = [p for p in curve if p.fpr <= target_fpr]
        if not feasible:
            # No threshold meets the budget; return the least-bad one rather than
            # silently pretending the constraint was satisfiable.
            return min(curve, key=lambda p: (p.fpr, -p.tpr))
        return max(feasible, key=lambda p: (p.tpr, -p.threshold))
    if criterion == "target_recall":
        feasible = [p for p in curve if p.tpr >= target_recall]
        if not feasible:
            return max(curve, key=lambda p: (p.tpr, -p.fpr))
        return min(feasible, key=lambda p: (p.fpr, p.threshold))
    raise CalibrationError(f"unknown criterion {criterion!r}")  # pragma: no cover


def apply_threshold(
    results: Sequence[EntropyResult],
    threshold: float,
    *,
    normalized: bool = True,
) -> List[bool]:
    """Vectorised ``is_confabulation`` over a batch."""
    return [r.is_confabulation(threshold, normalized=normalized) for r in results]


def sweep_callable(
    scorer: Callable[[str], EntropyResult], prompts: Sequence[str]
) -> List[EntropyResult]:
    """Score a list of prompts with any callable scorer (helper for dev-set builds)."""
    return [scorer(p) for p in prompts]
