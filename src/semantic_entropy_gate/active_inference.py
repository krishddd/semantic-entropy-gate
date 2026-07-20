"""Expected-free-energy bridge: turning entropy into a *policy*, not just a number.

The research note this library was built from frames uncertainty the way active
inference does. A policy ``u`` is scored by its expected free energy

    G(u) = pragmatic_cost(u) + ambiguity(u) - information_gain(u)

where the **ambiguity** term is exactly the expected entropy of observations
given hidden states — which is what semantic entropy measures for a language
model. When ambiguity dominates the pragmatic term, acting is irrational and the
free-energy-minimising policies are the ones that *gather information*: search,
retrieve, ask a clarifying question. That is the "information-seeking loop", and
it is the principled version of the DEFER branch in
:class:`~semantic_entropy_gate.gate.Gate`.

This module keeps that machinery deliberately small and dependency-free: it is a
policy *ranker* you can drop in front of a tool call, not a full POMDP planner.
If you already run `pymdp`, feed :func:`ambiguity_from_entropy` straight into the
likelihood precision of your ``A`` matrix instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .types import EntropyResult

__all__ = [
    "Policy",
    "PolicyEvaluation",
    "ambiguity_from_entropy",
    "expected_free_energy",
    "rank_policies",
    "should_forage",
]


@dataclass
class Policy:
    """A candidate action the agent could take next.

    Parameters
    ----------
    name:
        Identifier, e.g. ``"execute_refund"`` or ``"search_web"``.
    pragmatic_value:
        How much this policy advances the goal, in ``[0, 1]``. Executing the
        user's request scores high; a web search scores ~0.
    information_gain:
        Expected reduction in semantic entropy if this policy runs, in
        ``[0, 1]``. Retrieval and clarifying questions score high; a
        world-altering tool call gains nothing epistemically.
    epistemic:
        Marks the policy as information-seeking (used by :func:`should_forage`).
    irreversible:
        World-altering. Ambiguity is weighted harder for these, because a wrong
        irreversible action cannot be walked back.
    """

    name: str
    pragmatic_value: float = 0.0
    information_gain: float = 0.0
    epistemic: bool = False
    irreversible: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyEvaluation:
    """Scored policy: lower ``efe`` is preferred."""

    policy: Policy
    efe: float
    pragmatic_term: float
    ambiguity_term: float
    information_term: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.name,
            "efe": self.efe,
            "pragmatic_term": self.pragmatic_term,
            "ambiguity_term": self.ambiguity_term,
            "information_term": self.information_term,
            "epistemic": self.policy.epistemic,
            "irreversible": self.policy.irreversible,
        }


def ambiguity_from_entropy(
    result: EntropyResult, *, risk_weight: float = 1.0, normalized: bool = True
) -> float:
    """Map a semantic-entropy result onto the EFE ambiguity term.

    Returns a non-negative scalar. With ``normalized=True`` the value lives in
    ``[0, risk_weight]``, which makes it directly comparable to a pragmatic value
    expressed in ``[0, 1]`` — the comparison that decides act-vs-forage.
    """
    value = result.score_for(normalized=normalized)
    return max(0.0, risk_weight * value)


def expected_free_energy(
    policy: Policy,
    result: EntropyResult,
    *,
    risk_weight: float = 1.0,
    irreversible_penalty: float = 2.0,
    normalized: bool = True,
) -> PolicyEvaluation:
    """Score one policy under the current uncertainty.

    ``G = -pragmatic_value + weight * ambiguity - information_gain``

    Irreversible policies multiply the ambiguity term by
    ``irreversible_penalty``: the same uncertainty should stop a database write
    long before it stops drafting a sentence.
    """
    weight = risk_weight * (irreversible_penalty if policy.irreversible else 1.0)
    ambiguity = ambiguity_from_entropy(result, risk_weight=weight, normalized=normalized)
    # An epistemic policy's information gain is only worth what there is left to
    # learn: searching when the model is already certain buys nothing.
    available = ambiguity_from_entropy(result, risk_weight=1.0, normalized=normalized)
    information = policy.information_gain * available
    pragmatic = -policy.pragmatic_value
    # Foraging does not itself commit to an uncertain observation, so it is not
    # charged the ambiguity of the *answer* it is trying to resolve.
    if policy.epistemic:
        ambiguity = 0.0
    efe = pragmatic + ambiguity - information
    return PolicyEvaluation(
        policy=policy,
        efe=efe,
        pragmatic_term=pragmatic,
        ambiguity_term=ambiguity,
        information_term=-information,
    )


def rank_policies(
    policies: Sequence[Policy],
    result: EntropyResult,
    *,
    risk_weight: float = 1.0,
    irreversible_penalty: float = 2.0,
    normalized: bool = True,
) -> List[PolicyEvaluation]:
    """Rank candidate policies by expected free energy, best (lowest) first."""
    evaluations = [
        expected_free_energy(
            p,
            result,
            risk_weight=risk_weight,
            irreversible_penalty=irreversible_penalty,
            normalized=normalized,
        )
        for p in policies
    ]
    return sorted(evaluations, key=lambda e: (e.efe, e.policy.name))


def should_forage(
    result: EntropyResult,
    *,
    threshold: float = 0.55,
    normalized: bool = True,
) -> bool:
    """The phase-transition test: has ambiguity overtaken pragmatic value?

    A thin, honest wrapper around the threshold comparison — kept as a named
    function because the *reason* matters at the call site: this is not "the
    score is high", it is "no acting policy currently minimises expected free
    energy".
    """
    return result.score_for(normalized=normalized) >= threshold


def softmax_policy(
    evaluations: Sequence[PolicyEvaluation], *, precision: float = 4.0
) -> Dict[str, float]:
    """Posterior over policies, ``q(u) = softmax(-precision * G(u))``.

    ``precision`` is the inverse-temperature (``gamma`` in the active-inference
    literature): high precision means near-deterministic selection of the best
    policy, low precision means exploratory hedging.
    """
    if not evaluations:
        return {}
    scores = [-precision * e.efe for e in evaluations]
    peak = max(scores)
    exps = [math.exp(s - peak) for s in scores]
    total = sum(exps)
    return {e.policy.name: v / total for e, v in zip(evaluations, exps)}


def explain_ranking(evaluations: Sequence[PolicyEvaluation], width: int = 78) -> str:
    """Human-readable justification for the selected policy."""
    if not evaluations:
        return "no policies to rank"
    bar = "=" * width
    lines = [bar, "EXPECTED FREE ENERGY RANKING (lower is preferred)", bar]
    lines.append(f"{'policy':<24}{'G':>9}{'pragmatic':>12}{'ambiguity':>12}{'info gain':>12}")
    lines.append("-" * width)
    for evaluation in evaluations:
        lines.append(
            f"{evaluation.policy.name[:23]:<24}"
            f"{evaluation.efe:>9.3f}"
            f"{evaluation.pragmatic_term:>12.3f}"
            f"{evaluation.ambiguity_term:>12.3f}"
            f"{evaluation.information_term:>12.3f}"
        )
    best = evaluations[0]
    lines.append("-" * width)
    phase = "FORAGING" if best.policy.epistemic else "EXECUTION"
    lines.append(f"selected: {best.policy.name}  -> phase: {phase}")
    lines.append(bar)
    return "\n".join(lines)


def default_policies(
    *, action_name: str = "execute", forage_name: str = "gather_information"
) -> List[Policy]:
    """The minimal two-policy world: act, or go find out.

    Enough to reproduce the act/forage phase transition without asking the user
    to specify a generative model.
    """
    return [
        Policy(name=action_name, pragmatic_value=1.0, information_gain=0.0, irreversible=True),
        Policy(name=forage_name, pragmatic_value=0.0, information_gain=0.9, epistemic=True),
    ]


def decide(
    result: EntropyResult,
    policies: Optional[Sequence[Policy]] = None,
    **kwargs: Any,
) -> PolicyEvaluation:
    """Rank the default (or given) policies and return the winner."""
    ranked = rank_policies(list(policies or default_policies()), result, **kwargs)
    return ranked[0]
