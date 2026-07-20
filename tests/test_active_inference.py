"""Expected-free-energy policy ranking: act, or go find out."""

import pytest

from semantic_entropy_gate import LexicalEntailment, score_samples
from semantic_entropy_gate.active_inference import (
    Policy,
    ambiguity_from_entropy,
    decide,
    default_policies,
    expected_free_energy,
    explain_ranking,
    rank_policies,
    should_forage,
    softmax_policy,
)


@pytest.fixture
def certain(confident_samples):
    return score_samples("q", confident_samples, entailment=LexicalEntailment())


@pytest.fixture
def uncertain(confabulating_samples):
    return score_samples("q", confabulating_samples, entailment=LexicalEntailment())


def test_ambiguity_tracks_normalized_entropy(certain, uncertain):
    assert ambiguity_from_entropy(certain) == 0.0
    assert ambiguity_from_entropy(uncertain) > 0.8


def test_ambiguity_is_scaled_by_the_risk_weight(uncertain):
    base = ambiguity_from_entropy(uncertain)
    assert ambiguity_from_entropy(uncertain, risk_weight=2.0) == pytest.approx(2 * base)


def test_ambiguity_is_never_negative(certain):
    assert ambiguity_from_entropy(certain, risk_weight=5.0) >= 0.0


def test_confident_agent_chooses_to_act(certain):
    best = decide(certain)
    assert best.policy.name == "execute"
    assert best.policy.epistemic is False


def test_uncertain_agent_chooses_to_forage(uncertain):
    best = decide(uncertain)
    assert best.policy.name == "gather_information"
    assert best.policy.epistemic is True


def test_irreversible_policies_are_penalised_harder(uncertain):
    risky = Policy(name="write_db", pragmatic_value=1.0, irreversible=True)
    safe = Policy(name="draft_text", pragmatic_value=1.0, irreversible=False)
    risky_efe = expected_free_energy(risky, uncertain).efe
    safe_efe = expected_free_energy(safe, uncertain).efe
    assert risky_efe > safe_efe


def test_epistemic_policies_are_not_charged_ambiguity(uncertain):
    forage = Policy(name="search", information_gain=1.0, epistemic=True)
    evaluation = expected_free_energy(forage, uncertain)
    assert evaluation.ambiguity_term == 0.0
    assert evaluation.information_term < 0.0  # information gain lowers G


def test_information_gain_is_worthless_when_already_certain(certain):
    forage = Policy(name="search", information_gain=1.0, epistemic=True)
    assert expected_free_energy(forage, certain).information_term == 0.0


def test_ranking_is_sorted_by_expected_free_energy(uncertain):
    ranked = rank_policies(default_policies(), uncertain)
    assert [e.efe for e in ranked] == sorted(e.efe for e in ranked)


def test_ranking_is_deterministic_on_ties(certain):
    policies = [Policy(name="b"), Policy(name="a")]
    ranked = rank_policies(policies, certain)
    assert [e.policy.name for e in ranked] == ["a", "b"]


def test_should_forage_matches_the_threshold(certain, uncertain):
    assert should_forage(certain, threshold=0.55) is False
    assert should_forage(uncertain, threshold=0.55) is True


def test_softmax_policy_is_a_distribution(uncertain):
    ranked = rank_policies(default_policies(), uncertain)
    posterior = softmax_policy(ranked)
    assert sum(posterior.values()) == pytest.approx(1.0)
    assert posterior["gather_information"] > posterior["execute"]


def test_softmax_precision_sharpens_the_posterior(uncertain):
    ranked = rank_policies(default_policies(), uncertain)
    loose = softmax_policy(ranked, precision=0.5)["gather_information"]
    tight = softmax_policy(ranked, precision=20.0)["gather_information"]
    assert tight > loose


def test_softmax_of_nothing_is_empty():
    assert softmax_policy([]) == {}


def test_explain_ranking_names_the_phase(uncertain, certain):
    assert "FORAGING" in explain_ranking(rank_policies(default_policies(), uncertain))
    assert "EXECUTION" in explain_ranking(rank_policies(default_policies(), certain))


def test_explain_ranking_handles_no_policies():
    assert explain_ranking([]) == "no policies to rank"


def test_evaluation_serialises(uncertain):
    data = expected_free_energy(default_policies()[0], uncertain).to_dict()
    assert data["policy"] == "execute"
    assert data["irreversible"] is True
    assert "ambiguity_term" in data


def test_custom_policy_names_are_respected():
    policies = default_policies(action_name="refund", forage_name="ask_human")
    assert [p.name for p in policies] == ["refund", "ask_human"]
