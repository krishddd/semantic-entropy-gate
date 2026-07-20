"""Greedy bidirectional-entailment clustering."""

from semantic_entropy_gate.clustering import cluster
from semantic_entropy_gate.entailment import CannedEntailment, LexicalEntailment
from semantic_entropy_gate.types import EntailmentLabel, Sample


def samples(*texts, logprobs=None):
    if logprobs is None:
        return [Sample(text=t) for t in texts]
    return [Sample(text=t, logprob=lp) for t, lp in zip(texts, logprobs)]


def test_everything_equivalent_yields_one_cluster(all_equivalent):
    outcome = cluster(samples("a", "b", "c"), all_equivalent)
    assert len(outcome.clusters) == 1
    assert outcome.assignments == [0, 0, 0]
    assert outcome.clusters[0].size == 3


def test_nothing_equivalent_yields_one_cluster_per_sample(none_equivalent):
    outcome = cluster(samples("a", "b", "c"), none_equivalent)
    assert len(outcome.clusters) == 3
    assert outcome.assignments == [0, 1, 2]


def test_identical_strings_skip_the_nli_call():
    oracle = CannedEntailment(default=EntailmentLabel.CONTRADICTION)
    outcome = cluster(samples("Paris", "Paris", "Lyon"), oracle)
    assert outcome.assignments[0] == outcome.assignments[1]
    # Only the Paris/Lyon pair was ever sent to the oracle.
    assert all("Paris" in pair and "Lyon" in pair for pair in oracle.calls)


def test_exact_match_shortcut_can_be_disabled():
    oracle = CannedEntailment(default=EntailmentLabel.CONTRADICTION)
    outcome = cluster(samples("Paris", "Paris"), oracle, exact_match_shortcut=False)
    # Identical strings still short-circuit inside CannedEntailment itself.
    assert len(outcome.clusters) == 1
    assert oracle.calls


def test_two_meanings_are_separated_by_the_lexical_backend(split_samples):
    outcome = cluster(
        [Sample(text=t) for t in split_samples],
        LexicalEntailment(),
        context="Who discovered penicillin?",
    )
    fleming = {i for i, s in enumerate(split_samples) if "Fleming" in s}
    florey = {i for i, s in enumerate(split_samples) if "Florey" in s}
    fleming_clusters = {outcome.assignments[i] for i in fleming}
    florey_clusters = {outcome.assignments[i] for i in florey}
    assert not (fleming_clusters & florey_clusters), "Fleming and Florey must not merge"


def test_confabulating_numbers_land_in_distinct_clusters(confabulating_samples):
    outcome = cluster([Sample(text=t) for t in confabulating_samples], LexicalEntailment())
    # "About 610 Kelvin." and "Around 610 K." agree; the rest disagree.
    assert len(outcome.clusters) >= 4


def test_confident_paraphrases_collapse_to_one_cluster(confident_samples):
    outcome = cluster(
        [Sample(text=t) for t in confident_samples],
        LexicalEntailment(),
        context="In which city is the Eiffel Tower?",
    )
    assert len(outcome.clusters) == 1


def test_every_judgement_is_recorded_in_both_directions(none_equivalent):
    outcome = cluster(samples("a", "b"), none_equivalent)
    assert len(outcome.judgements) == 2
    directions = {(j.premise_index, j.hypothesis_index) for j in outcome.judgements}
    assert directions == {(0, 1), (1, 0)}
    assert all(j.backend == "canned" for j in outcome.judgements)


def test_assignments_index_align_with_cluster_membership(none_equivalent):
    outcome = cluster(samples("a", "b", "c"), none_equivalent)
    for index, cluster_id in enumerate(outcome.assignments):
        assert index in outcome.clusters[cluster_id].member_indices


def test_representative_is_the_most_likely_member_when_logprobs_exist(all_equivalent):
    outcome = cluster(
        samples("unlikely", "likely", "middling", logprobs=[-5.0, -0.1, -2.0]), all_equivalent
    )
    assert outcome.clusters[0].representative == "likely"


def test_representative_is_the_median_length_member_without_logprobs(all_equivalent):
    outcome = cluster(samples("a", "abc", "abcdefgh"), all_equivalent)
    assert outcome.clusters[0].representative == "abc"


def test_single_sample_produces_a_single_cluster_and_no_nli_calls():
    oracle = CannedEntailment()
    outcome = cluster(samples("only"), oracle)
    assert len(outcome.clusters) == 1
    assert outcome.judgements == []
    assert oracle.calls == []


def test_empty_input_is_handled(none_equivalent):
    outcome = cluster([], none_equivalent)
    assert outcome.clusters == []
    assert outcome.assignments == []


def test_relaxed_mode_merges_what_strict_mode_separates():
    table = {("A", "B"): EntailmentLabel.ENTAILMENT, ("B", "A"): EntailmentLabel.NEUTRAL}
    strict = cluster(samples("A", "B"), CannedEntailment(table), strict=True)
    relaxed = cluster(samples("A", "B"), CannedEntailment(table), strict=False)
    assert len(strict.clusters) == 2
    assert len(relaxed.clusters) == 1


def test_cluster_outcome_tuple_unpacking_still_works(none_equivalent):
    clusters, assignments, judgements = cluster(samples("a", "b"), none_equivalent)
    assert len(clusters) == 2
    assert len(assignments) == 2
    assert len(judgements) == 2
