"""The primary surface: :func:`score` — one prompt in, one auditable result out.

    result = score("Who invented the telephone?", my_sampler)
    print(result.normalized_entropy)   # 0.0 == unanimous meaning, 1.0 == chaos
    print(result.explain())            # the full why-trace

Pipeline: sample N -> cluster by bidirectional entailment -> entropy over
clusters. Everything each stage decided is preserved on the result.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Union

from .clustering import cluster as cluster_samples
from .entailment import EntailmentModel, auto_entailment
from .entropy import entropy_diagnostics, naive_string_entropy, semantic_entropy
from .errors import SamplingError
from .safety import (
    DEFAULT_LIMITS,
    MIN_SAMPLES_FOR_ENTROPY,
    Limits,
    check_samples,
    guard_entropy,
    has_unsafe_characters,
)
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
    limits: Limits = DEFAULT_LIMITS,
    min_samples: int = MIN_SAMPLES_FOR_ENTROPY,
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
    limits:
        Resource ceilings (sample count, text length, NLI budget). See
        :class:`~semantic_entropy_gate.safety.Limits`.
    min_samples:
        Below this the result is marked **unreliable** rather than confident: a
        sampler that returned one generation is indistinguishable from a certain
        model if you only look at the entropy.

    Raises
    ------
    SamplingError
        If the sampler yields nothing, or ``n_samples < 1``.

    Notes
    -----
    The number of generations the sampler actually returned is compared against
    ``n_samples``. A short return is the single most common silent failure in
    production (rate limits, partial API errors) and it always biases the score
    toward *confident*, so it is recorded and, when severe, marks the result
    unreliable.
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
        limits=limits,
        min_samples=min_samples,
        requested=n_samples,
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
    limits: Limits = DEFAULT_LIMITS,
    min_samples: int = MIN_SAMPLES_FOR_ENTROPY,
    requested: Optional[int] = None,
) -> EntropyResult:
    """Score generations you already have — no sampler, no model call.

    This is the offline path: batch-score a logged dataset, re-score with a
    different entailment backend, or reproduce a historical decision exactly.

    The sample set is validated first (see
    :func:`~semantic_entropy_gate.safety.check_samples`). Anything that would
    make a low entropy *meaningless* rather than *reassuring* — too few
    generations, byte-identical generations, empty output, poisoned
    log-probabilities — marks the result ``reliable=False`` and is listed in
    ``result.warnings``. Nothing is dropped silently.
    """
    raw: List[Sample] = normalize_texts(list(samples))
    integrity = check_samples(raw, limits=limits, min_samples=min_samples, requested=requested)
    normalized: List[Sample] = integrity.samples

    if entailment is None:
        entailment = auto_entailment(judge=judge)

    nli_context = prompt if context is None else context
    if len(nli_context) > limits.max_prompt_chars:
        nli_context = nli_context[: limits.max_prompt_chars]
        integrity.add(
            f"prompt truncated to {limits.max_prompt_chars} characters for entailment context"
        )

    # Bound the O(N^2) NLI cost before doing any of it. Refusing up front beats
    # discovering the bill afterwards.
    n = len(normalized)
    worst_case_calls = n * (n - 1)
    if worst_case_calls > limits.max_entailment_calls:
        raise SamplingError(
            f"{n} generations would need up to {worst_case_calls} entailment calls, "
            f"above the limit of {limits.max_entailment_calls}. Lower n_samples or raise "
            "Limits(max_entailment_calls=...) deliberately."
        )

    started = time.time()
    outcome = cluster_samples(normalized, entailment, context=nli_context, strict=strict)
    elapsed_clustering = time.time() - started

    entropy, probabilities, used = semantic_entropy(
        outcome.clusters, normalized, estimator=estimator
    )
    entropy, entropy_warning = guard_entropy(entropy)
    if entropy_warning:
        integrity.add(entropy_warning, fatal=True)
    for semantic_cluster, probability in zip(outcome.clusters, probabilities):
        semantic_cluster.probability = probability

    naive = naive_string_entropy(normalized)
    max_entropy = _log(len(normalized))
    if max_entropy > 0 and math.isfinite(entropy):
        normalized_entropy = min(1.0, entropy / max_entropy)
    elif not math.isfinite(entropy):
        # Unknown uncertainty reads as maximal uncertainty, never as confidence.
        normalized_entropy = 1.0
    else:
        normalized_entropy = 0.0

    tampered = sum(1 for s in normalized if has_unsafe_characters(s.text))
    if tampered:
        integrity.add(
            f"{tampered} generation(s) contain terminal control characters or bidirectional "
            "overrides; they are stripped from every rendered report, but their presence "
            "suggests the output is trying to alter what a reviewer sees"
        )

    if not strict:
        integrity.add(
            "relaxed entailment clustering is in use: it merges more answers and therefore "
            "reports lower entropy than the strict rule the threshold was calibrated on"
        )

    meta = dict(metadata or {})
    meta["clustering_seconds"] = round(elapsed_clustering, 4)
    meta["entailment_calls"] = len(outcome.judgements)
    meta["strict_entailment"] = strict
    meta["integrity"] = integrity.to_dict()
    meta.update(entropy_diagnostics(outcome.clusters, normalized, entropy))

    return EntropyResult(
        prompt=prompt,
        samples=normalized,
        clusters=outcome.clusters,
        entropy=entropy,
        normalized_entropy=normalized_entropy,
        naive_entropy=naive,
        estimator=used,
        judgements=outcome.judgements,
        entailment_backend=entailment.name,
        cluster_assignments=outcome.assignments,
        metadata=meta,
        warnings=list(integrity.warnings),
        reliable=integrity.reliable,
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
    return math.log(n) if n > 1 else 0.0
