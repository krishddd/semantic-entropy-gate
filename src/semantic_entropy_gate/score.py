"""The primary surface: :func:`score` — one prompt in, one auditable result out.

    result = score("Who invented the telephone?", my_sampler)
    print(result.normalized_entropy)   # 0.0 == unanimous meaning, 1.0 == chaos
    print(result.explain())            # the full why-trace

Pipeline: sample N -> cluster by bidirectional entailment -> entropy over
clusters. Everything each stage decided is preserved on the result.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Union

from .clustering import cluster as cluster_samples
from .entailment import EntailmentModel, auto_entailment
from .entropy import entropy_diagnostics, naive_string_entropy, semantic_entropy
from .errors import SamplingError
from .sampling import Sampler, resolve_sampler
from .types import EntropyResult, Estimator, Sample, normalize_texts

__all__ = ["score", "score_samples", "score_batch", "DEFAULT_N_SAMPLES"]

DEFAULT_N_SAMPLES = 10
"""Farquhar et al. use 10 generations; 5 is a usable budget cut, below 5 the
discrete estimator's downward bias dominates."""


def score(
    prompt: str,
    sampler: Sampler,
    *,
    n_samples: int = DEFAULT_N_SAMPLES,
    entailment: Optional[EntailmentModel] = None,
    strict: bool = True,
    estimator: Optional[Estimator] = None,
    context: Optional[str] = None,
    judge: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> EntropyResult:
    """Score one prompt for semantic entropy.

    Parameters
    ----------
    prompt:
        The question put to the model. Also used as NLI context unless
        ``context`` overrides it.
    sampler:
        Any callable shape accepted by
        :func:`semantic_entropy_gate.sampling.resolve_sampler`. **Must sample at
        non-zero temperature** — greedy decoding produces identical generations
        and therefore a meaningless entropy of 0.
    n_samples:
        Number of generations. Cost is linear in N for sampling and up to
        quadratic in N for NLI calls.
    entailment:
        The equivalence oracle. Defaults to :func:`auto_entailment` (local
        cross-encoder if installed, else ``judge``, else lexical heuristic).
    strict:
        Strict (both directions must entail) vs relaxed clustering.
    estimator:
        Force ``Estimator.RAO_BLACKWELL`` or ``Estimator.DISCRETE``. ``None``
        auto-selects on whether every sample carries a log-probability.

    Raises
    ------
    SamplingError
        If the sampler yields nothing, or ``n_samples < 1``.
    """
    if n_samples < 1:
        raise SamplingError(f"n_samples must be >= 1, got {n_samples}")

    started = time.time()
    draw = resolve_sampler(sampler)
    samples = draw(prompt, n_samples)
    elapsed_sampling = time.time() - started

    meta = dict(metadata or {})
    meta["sampling_seconds"] = round(elapsed_sampling, 4)
    meta["requested_samples"] = n_samples
    return score_samples(
        prompt,
        samples,
        entailment=entailment,
        strict=strict,
        estimator=estimator,
        context=context,
        judge=judge,
        metadata=meta,
    )


def score_samples(
    prompt: str,
    samples: Union[Sequence[Sample], Sequence[str], Sequence[Any]],
    *,
    entailment: Optional[EntailmentModel] = None,
    strict: bool = True,
    estimator: Optional[Estimator] = None,
    context: Optional[str] = None,
    judge: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> EntropyResult:
    """Score generations you already have — no sampler, no model call.

    This is the offline path: batch-score a logged dataset, re-score with a
    different entailment backend, or reproduce a historical decision exactly.
    """
    normalized: List[Sample] = normalize_texts(list(samples))
    if not normalized:
        raise SamplingError("no samples to score")

    if entailment is None:
        entailment = auto_entailment(judge=judge)

    nli_context = prompt if context is None else context
    started = time.time()
    outcome = cluster_samples(normalized, entailment, context=nli_context, strict=strict)
    elapsed_clustering = time.time() - started

    entropy, probabilities, used = semantic_entropy(
        outcome.clusters, normalized, estimator=estimator
    )
    for semantic_cluster, probability in zip(outcome.clusters, probabilities):
        semantic_cluster.probability = probability

    naive = naive_string_entropy(normalized)
    max_entropy = _log(len(normalized))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    meta = dict(metadata or {})
    meta["clustering_seconds"] = round(elapsed_clustering, 4)
    meta["entailment_calls"] = len(outcome.judgements)
    meta["strict_entailment"] = strict
    meta.update(entropy_diagnostics(outcome.clusters, normalized, entropy))

    return EntropyResult(
        prompt=prompt,
        samples=normalized,
        clusters=outcome.clusters,
        entropy=entropy,
        normalized_entropy=min(1.0, normalized_entropy),
        naive_entropy=naive,
        estimator=used,
        judgements=outcome.judgements,
        entailment_backend=entailment.name,
        cluster_assignments=outcome.assignments,
        metadata=meta,
    )


def score_batch(
    prompts: Sequence[str],
    sampler: Sampler,
    *,
    n_samples: int = DEFAULT_N_SAMPLES,
    entailment: Optional[EntailmentModel] = None,
    strict: bool = True,
    estimator: Optional[Estimator] = None,
    judge: Optional[Any] = None,
    on_result: Optional[Any] = None,
) -> List[EntropyResult]:
    """Score many prompts, reusing one entailment backend (and therefore its cache).

    ``on_result(index, result)`` is invoked after each prompt so a caller can show
    progress without waiting for the whole batch.
    """
    if entailment is None:
        entailment = auto_entailment(judge=judge)
    results: List[EntropyResult] = []
    for index, prompt in enumerate(prompts):
        result = score(
            prompt,
            sampler,
            n_samples=n_samples,
            entailment=entailment,
            strict=strict,
            estimator=estimator,
        )
        results.append(result)
        if on_result is not None:
            on_result(index, result)
    return results


def _log(n: int) -> float:
    import math

    return math.log(n) if n > 1 else 0.0
