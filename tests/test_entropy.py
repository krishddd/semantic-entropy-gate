"""Entropy maths: the numbers must be right before anything else matters."""

import math

import pytest

from semantic_entropy_gate.entropy import (
    chao1_alphabet_size,
    cluster_probabilities,
    entropy_diagnostics,
    logsumexp,
    miller_madow_correction,
    naive_string_entropy,
    predictive_entropy,
    semantic_entropy,
    shannon_entropy,
)
from semantic_entropy_gate.types import Estimator, Sample, SemanticCluster


def make_clusters(sizes):
    clusters = []
    index = 0
    for cid, size in enumerate(sizes):
        indices = list(range(index, index + size))
        clusters.append(
            SemanticCluster(id=cid, member_indices=indices, members=[f"s{i}" for i in indices])
        )
        index += size
    return clusters


def test_logsumexp_matches_naive_computation():
    values = [-1.0, -2.5, -0.3]
    expected = math.log(sum(math.exp(v) for v in values))
    assert logsumexp(values) == pytest.approx(expected)


def test_logsumexp_is_stable_for_very_negative_values():
    values = [-1000.0, -1001.0]
    # Naive exp() would underflow to 0 and produce -inf.
    assert logsumexp(values) == pytest.approx(-1000.0 + math.log(1 + math.exp(-1.0)))


def test_logsumexp_of_empty_is_negative_infinity():
    assert logsumexp([]) == float("-inf")


def test_shannon_entropy_of_certainty_is_zero():
    assert shannon_entropy([1.0]) == 0.0
    assert shannon_entropy([1.0, 0.0, 0.0]) == 0.0


def test_shannon_entropy_of_uniform_is_log_k():
    assert shannon_entropy([0.25] * 4) == pytest.approx(math.log(4))


def test_shannon_entropy_ignores_zero_mass_outcomes():
    assert shannon_entropy([0.5, 0.5, 0.0]) == pytest.approx(math.log(2))


def test_discrete_probabilities_are_cluster_frequencies():
    clusters = make_clusters([3, 1])
    samples = [Sample(text=f"s{i}") for i in range(4)]
    probs, estimator = cluster_probabilities(clusters, samples)
    assert estimator is Estimator.DISCRETE
    assert probs == pytest.approx([0.75, 0.25])


def test_rao_blackwell_selected_when_every_sample_has_a_logprob():
    clusters = make_clusters([2, 1])
    samples = [
        Sample(text="a", logprob=-0.1),
        Sample(text="b", logprob=-0.1),
        Sample(text="c", logprob=-2.0),
    ]
    probs, estimator = cluster_probabilities(clusters, samples)
    assert estimator is Estimator.RAO_BLACKWELL
    assert sum(probs) == pytest.approx(1.0)
    # The two high-likelihood members should dominate the low-likelihood singleton.
    assert probs[0] > probs[1]


def test_rao_blackwell_falls_back_when_a_logprob_is_missing():
    clusters = make_clusters([1, 1])
    samples = [Sample(text="a", logprob=-0.1), Sample(text="b")]
    probs, estimator = cluster_probabilities(clusters, samples, estimator=Estimator.RAO_BLACKWELL)
    assert estimator is Estimator.DISCRETE
    assert probs == pytest.approx([0.5, 0.5])


def test_rao_blackwell_aggregates_by_logsumexp_not_by_mean():
    # Two equally-likely samples in one cluster must outweigh one identical
    # sample alone: mass is summed within a cluster, not averaged.
    clusters = make_clusters([2, 1])
    samples = [Sample(text="a", logprob=-1.0)] * 3
    probs, _ = cluster_probabilities(clusters, samples)
    assert probs[0] == pytest.approx(2 / 3)
    assert probs[1] == pytest.approx(1 / 3)


def test_semantic_entropy_is_zero_for_a_single_cluster():
    clusters = make_clusters([5])
    samples = [Sample(text="x") for _ in range(5)]
    entropy, probs, _ = semantic_entropy(clusters, samples)
    assert entropy == 0.0
    assert probs == [1.0]


def test_semantic_entropy_is_maximal_when_every_sample_differs():
    clusters = make_clusters([1, 1, 1, 1])
    samples = [Sample(text=f"s{i}") for i in range(4)]
    entropy, _, _ = semantic_entropy(clusters, samples)
    assert entropy == pytest.approx(math.log(4))


def test_naive_string_entropy_counts_exact_strings():
    samples = [Sample(text="Paris"), Sample(text="Paris"), Sample(text="Lyon")]
    assert naive_string_entropy(samples) == pytest.approx(shannon_entropy([2 / 3, 1 / 3]))


def test_naive_entropy_exceeds_semantic_entropy_for_paraphrases():
    # Six distinct strings, one meaning: the whole point of the method.
    texts = ["Paris", "in Paris", "Paris, France", "It is Paris", "Paris!", "The city is Paris"]
    samples = [Sample(text=t) for t in texts]
    clusters = make_clusters([6])
    entropy, _, _ = semantic_entropy(clusters, samples)
    assert entropy == 0.0
    assert naive_string_entropy(samples) == pytest.approx(math.log(6))


def test_predictive_entropy_is_none_without_complete_logprobs():
    assert predictive_entropy([Sample(text="a"), Sample(text="b", logprob=-1.0)]) is None


def test_predictive_entropy_is_negative_mean_logprob():
    samples = [Sample(text="a", logprob=-1.0), Sample(text="b", logprob=-3.0)]
    assert predictive_entropy(samples) == pytest.approx(2.0)


def test_chao1_exceeds_observed_count_when_singletons_dominate():
    clusters = make_clusters([1, 1, 1, 2])  # 3 singletons, 1 doubleton
    assert chao1_alphabet_size(clusters) == pytest.approx(4 + 9 / 2)


def test_chao1_handles_zero_doubletons():
    clusters = make_clusters([1, 1])
    assert chao1_alphabet_size(clusters) == pytest.approx(2 + 1.0)


def test_chao1_of_no_clusters_is_zero():
    assert chao1_alphabet_size([]) == 0.0


def test_miller_madow_correction_increases_entropy():
    assert miller_madow_correction(1.0, n_clusters=4, n_samples=10) == pytest.approx(1.15)


def test_miller_madow_is_a_noop_for_a_single_cluster():
    assert miller_madow_correction(0.0, n_clusters=1, n_samples=10) == 0.0


def test_diagnostics_report_coverage_and_bias():
    clusters = make_clusters([1, 1, 2])
    samples = [Sample(text=f"s{i}") for i in range(4)]
    entropy, _, _ = semantic_entropy(clusters, samples)
    diagnostics = entropy_diagnostics(clusters, samples, entropy)
    assert diagnostics["singleton_clusters"] == 2.0
    assert diagnostics["chao1_alphabet_size"] >= len(clusters)
    assert 0.0 < diagnostics["coverage"] <= 1.0
    assert diagnostics["entropy_bias_corrected"] > entropy
    assert "predictive_entropy" not in diagnostics


def test_empty_sample_set_is_handled():
    probs, estimator = cluster_probabilities([], [])
    assert probs == []
    assert estimator is Estimator.DISCRETE
    assert naive_string_entropy([]) == 0.0
