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
from .types import CalibrationResult, EntropyResult, ThresholdPoint

__all__ = [
    "calibrate",
    "auroc",
    "auprc",
    "roc_curve",
    "threshold_sweep",
    "CRITERIA",
]

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

    # A threshold fitted to a handful of prompts is a number, not a measurement.
    # It will look authoritative in a report, so the report has to say otherwise.
    caveats: List[str] = []
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

    return CalibrationResult(
        threshold=point.threshold,
        auroc=auroc(scores, ys),
        auprc=auprc(scores, ys),
        criterion=criterion,
        n_samples=len(ys),
        n_positive=n_pos,
        n_negative=n_neg,
        operating_point=point,
        curve=curve,
        base_rate=n_pos / len(ys),
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
