"""Greedy bidirectional-entailment clustering into semantic equivalence classes.

The algorithm is the one from Farquhar et al. (Nature 2024) and the reference
``jlko/semantic_uncertainty`` implementation:

    for each generation s_i:
        if s_i is unassigned:
            open a new cluster C with s_i as its nucleus
            for each later unassigned s_j:
                if bidirectional_entailment(s_i, s_j):   # both directions
                    add s_j to C

It is greedy (each candidate is compared to the cluster *nucleus*, not to every
member), which keeps the NLI budget at O(n^2) worst case and O(n) when the model
is confident — and confident is the common case, so this is cheap in practice.

Greediness means clustering is order-sensitive; :func:`cluster` therefore records
every entailment judgement it made so a reviewer can reconstruct exactly why two
answers landed in the same or different buckets.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .entailment import EntailmentModel
from .types import EntailmentJudgement, Sample, SemanticCluster

__all__ = ["cluster", "ClusterOutcome"]


class ClusterOutcome(tuple):
    """``(clusters, assignments, judgements)`` with named access."""

    __slots__ = ()

    def __new__(
        cls,
        clusters: List[SemanticCluster],
        assignments: List[int],
        judgements: List[EntailmentJudgement],
    ) -> "ClusterOutcome":
        return super().__new__(cls, (clusters, assignments, judgements))

    @property
    def clusters(self) -> List[SemanticCluster]:
        return self[0]

    @property
    def assignments(self) -> List[int]:
        return self[1]

    @property
    def judgements(self) -> List[EntailmentJudgement]:
        return self[2]


def cluster(
    samples: Sequence[Sample],
    entailment: EntailmentModel,
    *,
    context: str = "",
    strict: bool = True,
    exact_match_shortcut: bool = True,
) -> ClusterOutcome:
    """Partition ``samples`` into semantic equivalence classes.

    Parameters
    ----------
    samples:
        Generations drawn for one prompt.
    entailment:
        The NLI oracle (see :mod:`semantic_entropy_gate.entailment`).
    context:
        The question. Prepended to both sides of every NLI call, because bare
        short answers ("Paris", "1876") are not comparable propositions on their
        own.
    strict:
        ``True`` requires entailment in both directions (paper default).
        ``False`` uses the relaxed rule (no contradiction and not mutually
        neutral), which merges timid-neutral paraphrases.
    exact_match_shortcut:
        Skip the NLI call for byte-identical generations. Pure saving, no
        behaviour change — identical strings are trivially equivalent.

    Returns
    -------
    ClusterOutcome
        ``clusters`` (probabilities not yet filled in — that is the estimator's
        job), ``assignments[i]`` giving the cluster id of sample ``i``, and the
        complete list of entailment judgements performed.
    """
    n = len(samples)
    assignments: List[int] = [-1] * n
    judgements: List[EntailmentJudgement] = []
    clusters: List[SemanticCluster] = []

    for i in range(n):
        if assignments[i] != -1:
            continue
        cluster_id = len(clusters)
        assignments[i] = cluster_id
        member_indices = [i]

        for j in range(i + 1, n):
            if assignments[j] != -1:
                continue
            if exact_match_shortcut and samples[i].text.strip() == samples[j].text.strip():
                assignments[j] = cluster_id
                member_indices.append(j)
                continue
            equivalent, forward, backward = entailment.bidirectional(
                samples[i].text, samples[j].text, context=context, strict=strict
            )
            judgements.append(_judgement(i, j, forward, entailment))
            judgements.append(_judgement(j, i, backward, entailment))
            if equivalent:
                assignments[j] = cluster_id
                member_indices.append(j)

        clusters.append(
            SemanticCluster(
                id=cluster_id,
                member_indices=member_indices,
                members=[samples[k].text for k in member_indices],
                representative=_representative(samples, member_indices),
            )
        )

    return ClusterOutcome(clusters, assignments, judgements)


def _judgement(
    premise_index: int,
    hypothesis_index: int,
    verdict: Tuple[object, Optional[float]],
    entailment: EntailmentModel,
) -> EntailmentJudgement:
    label, confidence = verdict
    return EntailmentJudgement(
        premise_index=premise_index,
        hypothesis_index=hypothesis_index,
        label=label,  # type: ignore[arg-type]
        confidence=confidence,
        backend=entailment.name,
    )


def _representative(samples: Sequence[Sample], member_indices: Sequence[int]) -> str:
    """Pick the cluster's spokesperson.

    The most likely member if log-probabilities are available, otherwise the
    median-length one — a proxy for "typically phrased", avoiding both the terse
    fragment and the rambling outlier.
    """
    if not member_indices:
        return ""
    with_lp = [i for i in member_indices if samples[i].logprob is not None]
    if with_lp:
        best = max(with_lp, key=lambda i: samples[i].logprob)  # type: ignore[arg-type]
        return samples[best].text
    ordered = sorted(member_indices, key=lambda i: len(samples[i].text))
    return samples[ordered[len(ordered) // 2]].text
