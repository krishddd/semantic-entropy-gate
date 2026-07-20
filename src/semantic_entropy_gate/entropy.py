"""Entropy estimators over semantic clusters.

Two estimators, chosen by what your model exposes:

**Rao-Blackwellised (white-box).** When the sampler returns length-normalised
sequence log-probabilities, the mass of a cluster is the log-sum-exp of its
members' likelihoods, renormalised over clusters::

    log p(C_k) = logsumexp_{s in C_k} (1/T_s) * sum_t log p(t_t | t_<t)
    p(C_k)     = exp(log p(C_k)) / sum_j exp(log p(C_j))
    SE         = -sum_k p(C_k) log p(C_k)

Length normalisation matters: without it, a longer answer is exponentially
penalised for being long rather than for being wrong.

**Discrete (black-box).** With text-only APIs, mass is the empirical frequency
``|C_k| / N``. This plugin estimator is known to *under*-estimate the true
semantic entropy at small N, because rare-but-valid meanings simply are not
sampled; :func:`chao1_alphabet_size` and :func:`miller_madow_correction` quantify
and partially correct that bias, and both are surfaced in the report so nobody
reads a small-N score as gospel.

All entropies are in **nats** (natural log).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from .types import Estimator, Sample, SemanticCluster

__all__ = [
    "logsumexp",
    "shannon_entropy",
    "cluster_probabilities",
    "semantic_entropy",
    "naive_string_entropy",
    "predictive_entropy",
    "chao1_alphabet_size",
    "miller_madow_correction",
]


def logsumexp(values: Sequence[float]) -> float:
    """Numerically stable ``log(sum(exp(v)))``."""
    if not values:
        return float("-inf")
    finite = [v for v in values if v != float("-inf")]
    if not finite:
        return float("-inf")
    peak = max(finite)
    return peak + math.log(sum(math.exp(v - peak) for v in finite))


def shannon_entropy(probabilities: Sequence[float]) -> float:
    """``-sum p log p`` in nats, ignoring zero-mass outcomes (``0 log 0 = 0``)."""
    total = 0.0
    for p in probabilities:
        if p > 0.0:
            total -= p * math.log(p)
    return max(0.0, total)


def cluster_probabilities(
    clusters: Sequence[SemanticCluster],
    samples: Sequence[Sample],
    *,
    estimator: Optional[Estimator] = None,
) -> Tuple[List[float], Estimator]:
    """Compute ``p(C_k)`` for every cluster.

    ``estimator=None`` auto-selects: Rao-Blackwell when *every* sample carries a
    log-probability, discrete otherwise. Mixing is deliberately not allowed —
    a partially-scored sample set would silently weight the scored generations
    against unscored ones.
    """
    n = len(samples)
    if n == 0:
        return [], Estimator.DISCRETE

    have_logprobs = all(s.logprob is not None for s in samples)
    if estimator is None:
        estimator = Estimator.RAO_BLACKWELL if have_logprobs else Estimator.DISCRETE
    if estimator is Estimator.RAO_BLACKWELL and not have_logprobs:
        estimator = Estimator.DISCRETE

    if estimator is Estimator.DISCRETE:
        return [len(c.member_indices) / n for c in clusters], Estimator.DISCRETE

    cluster_logs = [
        logsumexp([samples[i].logprob for i in c.member_indices])  # type: ignore[misc]
        for c in clusters
    ]
    normalizer = logsumexp(cluster_logs)
    if normalizer == float("-inf"):  # pragma: no cover - all-zero likelihoods
        return [len(c.member_indices) / n for c in clusters], Estimator.DISCRETE
    return [math.exp(lp - normalizer) for lp in cluster_logs], Estimator.RAO_BLACKWELL


def semantic_entropy(
    clusters: Sequence[SemanticCluster],
    samples: Sequence[Sample],
    *,
    estimator: Optional[Estimator] = None,
) -> Tuple[float, List[float], Estimator]:
    """Return ``(entropy_nats, cluster_probabilities, estimator_used)``."""
    probs, used = cluster_probabilities(clusters, samples, estimator=estimator)
    return shannon_entropy(probs), probs, used


def naive_string_entropy(samples: Sequence[Sample]) -> float:
    """Entropy over exact output strings — the *lexical* baseline.

    Reported alongside the semantic score purely so the difference is visible.
    Ten different phrasings of one fact score high here and ~0 semantically; that
    gap is the false-positive rate you avoid by clustering.
    """
    if not samples:
        return 0.0
    counts = Counter(s.text.strip() for s in samples)
    n = len(samples)
    return shannon_entropy([c / n for c in counts.values()])


def predictive_entropy(samples: Sequence[Sample]) -> Optional[float]:
    """Monte-Carlo sequence-level predictive entropy, ``-(1/N) sum log p(s)``.

    The classic length-normalised-likelihood uncertainty baseline. ``None`` when
    log-probabilities are unavailable. Kept for comparison in reports: it is the
    number semantic entropy is meant to beat.
    """
    logprobs = [s.logprob for s in samples if s.logprob is not None]
    if not logprobs or len(logprobs) != len(samples):
        return None
    return -sum(logprobs) / len(logprobs)


def chao1_alphabet_size(clusters: Sequence[SemanticCluster]) -> float:
    """Chao1 lower bound on the number of *distinct meanings* the model can emit.

    ``S_chao1 = S_obs + f1^2 / (2 * f2)`` where ``f1``/``f2`` are the counts of
    clusters seen exactly once / twice. When the estimate materially exceeds the
    observed cluster count, your sample size is too small and the discrete
    entropy is biased low.
    """
    observed = len(clusters)
    if observed == 0:
        return 0.0
    singletons = sum(1 for c in clusters if len(c.member_indices) == 1)
    doubletons = sum(1 for c in clusters if len(c.member_indices) == 2)
    if doubletons == 0:
        # Bias-corrected form for the f2 == 0 case.
        return observed + singletons * (singletons - 1) / 2.0
    return observed + (singletons**2) / (2.0 * doubletons)


def miller_madow_correction(entropy: float, n_clusters: int, n_samples: int) -> float:
    """Miller-Madow bias-corrected entropy: ``H + (K - 1) / (2N)``.

    A first-order correction for the plugin estimator's systematic downward bias
    at small ``N``. Reported as ``metadata['entropy_bias_corrected']``; the gate
    itself thresholds the uncorrected value so that the calibrated threshold and
    the runtime score are always the same quantity.
    """
    if n_samples <= 0:
        return entropy
    return entropy + (n_clusters - 1) / (2.0 * n_samples)


def entropy_diagnostics(
    clusters: Sequence[SemanticCluster],
    samples: Sequence[Sample],
    entropy: float,
) -> Dict[str, float]:
    """Small-sample diagnostics attached to every result's ``metadata``."""
    n = len(samples)
    k = len(clusters)
    chao1 = chao1_alphabet_size(clusters)
    diagnostics: Dict[str, float] = {
        "entropy_bias_corrected": miller_madow_correction(entropy, k, n),
        "chao1_alphabet_size": chao1,
        "singleton_clusters": float(sum(1 for c in clusters if len(c.member_indices) == 1)),
        "coverage": float(k) / chao1 if chao1 > 0 else 1.0,
    }
    pe = predictive_entropy(samples)
    if pe is not None:
        diagnostics["predictive_entropy"] = pe
    return diagnostics
