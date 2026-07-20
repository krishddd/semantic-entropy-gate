"""The scoring pipeline and sampler normalisation."""

import math

import pytest

from semantic_entropy_gate import LexicalEntailment, Sample, score, score_batch, score_samples
from semantic_entropy_gate.errors import SamplingError
from semantic_entropy_gate.sampling import from_texts, resolve_sampler
from semantic_entropy_gate.types import Estimator

# ------------------------------------------------------------------ samplers


def test_batch_sampler_shape():
    call = resolve_sampler(lambda prompt, n: [f"{prompt}-{i}" for i in range(n)])
    assert [s.text for s in call("q", 3)] == ["q-0", "q-1", "q-2"]


def test_single_shot_sampler_is_called_n_times():
    calls = []

    def sampler(prompt):
        calls.append(prompt)
        return "answer"

    result = resolve_sampler(sampler)("q", 4)
    assert len(result) == 4
    assert calls == ["q"] * 4


def test_tuple_sampler_carries_logprobs():
    call = resolve_sampler(lambda p, n: [("a", -0.5), ("b", -1.5)])
    out = call("q", 2)
    assert out[0].logprob == -0.5
    assert out[1].text == "b"


def test_dict_and_sample_shapes_are_accepted():
    call = resolve_sampler(lambda p, n: [{"text": "a", "logprob": -0.2}, Sample(text="b")])
    out = call("q", 2)
    assert out[0].logprob == -0.2
    assert out[1].logprob is None


def test_sampler_returning_a_bare_string_is_rejected():
    with pytest.raises(SamplingError, match="single string"):
        resolve_sampler(lambda p, n: "just one")("q", 2)


def test_sampler_returning_none_is_rejected():
    with pytest.raises(SamplingError, match="None"):
        resolve_sampler(lambda p, n: None)("q", 2)


def test_sampler_returning_nothing_is_rejected():
    with pytest.raises(SamplingError, match="no generations"):
        resolve_sampler(lambda p, n: [])("q", 2)


def test_non_callable_sampler_is_rejected():
    with pytest.raises(SamplingError, match="callable"):
        resolve_sampler("nope")


def test_fixed_sampler_truncates_to_n():
    sampler = from_texts(["a", "b", "c"])
    assert len(sampler("q", 2)) == 2
    assert len(sampler("q", 99)) == 3


# --------------------------------------------------------------------- score


def test_confident_model_scores_zero_entropy(confident_samples):
    result = score(
        "In which city is the Eiffel Tower located?",
        from_texts(confident_samples),
        n_samples=6,
        entailment=LexicalEntailment(),
    )
    assert result.n_clusters == 1
    assert result.entropy == 0.0
    assert result.normalized_entropy == 0.0
    assert result.agreement == 1.0
    assert result.is_confabulation(0.55) is False


def test_confabulating_model_scores_high_entropy(confabulating_samples):
    result = score(
        "What is the boiling point of astatine?",
        from_texts(confabulating_samples),
        n_samples=6,
        entailment=LexicalEntailment(),
    )
    assert result.n_clusters >= 4
    assert result.normalized_entropy > 0.8
    assert result.is_confabulation(0.55) is True


def test_paraphrase_variance_is_attributed_to_lexical_not_semantic(confident_samples):
    result = score_samples(
        "In which city is the Eiffel Tower located?",
        confident_samples,
        entailment=LexicalEntailment(),
    )
    assert result.entropy == 0.0
    assert result.naive_entropy == pytest.approx(math.log(6))
    assert result.lexical_entropy == pytest.approx(math.log(6))


def test_lexical_entropy_is_never_negative(all_equivalent):
    result = score_samples("q", ["same", "same", "same"], entailment=all_equivalent)
    assert result.lexical_entropy >= 0.0


def test_normalized_entropy_is_bounded(none_equivalent):
    result = score_samples("q", [f"answer {i}" for i in range(5)], entailment=none_equivalent)
    assert result.normalized_entropy == pytest.approx(1.0)
    assert result.entropy == pytest.approx(math.log(5))


def test_single_sample_has_zero_max_entropy(none_equivalent):
    result = score_samples("q", ["only"], entailment=none_equivalent)
    assert result.max_entropy == 0.0
    assert result.normalized_entropy == 0.0


def test_estimator_is_rao_blackwell_with_logprobs(all_equivalent):
    result = score_samples("q", [("a", -0.1), ("b", -0.2)], entailment=all_equivalent)
    assert result.estimator is Estimator.RAO_BLACKWELL
    assert "predictive_entropy" in result.metadata


def test_estimator_is_discrete_without_logprobs(all_equivalent):
    result = score_samples("q", ["a", "b"], entailment=all_equivalent)
    assert result.estimator is Estimator.DISCRETE


def test_forced_estimator_is_respected(all_equivalent):
    result = score_samples(
        "q", [("a", -0.1), ("b", -5.0)], entailment=all_equivalent, estimator=Estimator.DISCRETE
    )
    assert result.estimator is Estimator.DISCRETE


def test_metadata_records_provenance(confident_samples):
    result = score("q", from_texts(confident_samples), n_samples=6, entailment=LexicalEntailment())
    assert result.entailment_backend == "lexical-heuristic"
    assert result.metadata["requested_samples"] == 6
    assert result.metadata["strict_entailment"] is True
    assert "clustering_seconds" in result.metadata
    assert "sampling_seconds" in result.metadata
    assert "chao1_alphabet_size" in result.metadata


def test_prompt_is_used_as_nli_context_by_default():
    seen = []

    class Spy(LexicalEntailment):
        def classify(self, premise, hypothesis, *, context=""):
            seen.append(context)
            return super().classify(premise, hypothesis, context=context)

    score_samples("the question?", ["a", "b"], entailment=Spy())
    assert seen and all(c == "the question?" for c in seen)


def test_context_override_is_used_instead_of_the_prompt():
    seen = []

    class Spy(LexicalEntailment):
        def classify(self, premise, hypothesis, *, context=""):
            seen.append(context)
            return super().classify(premise, hypothesis, context=context)

    score_samples("the question?", ["a", "b"], entailment=Spy(), context="override")
    assert seen and all(c == "override" for c in seen)


def test_zero_samples_requested_is_rejected():
    with pytest.raises(SamplingError, match=">= 1"):
        score("q", from_texts(["a"]), n_samples=0)


def test_scoring_no_samples_is_rejected():
    with pytest.raises(SamplingError, match="no samples"):
        score_samples("q", [])


def test_score_batch_shares_one_backend():
    backend = LexicalEntailment()
    results = score_batch(["q1", "q2"], from_texts(["a", "a"]), n_samples=2, entailment=backend)
    assert len(results) == 2
    assert all(r.entailment_backend == backend.name for r in results)


def test_score_batch_invokes_the_progress_callback():
    seen = []
    score_batch(
        ["q1", "q2"],
        from_texts(["a"]),
        n_samples=1,
        entailment=LexicalEntailment(),
        on_result=lambda i, r: seen.append(i),
    )
    assert seen == [0, 1]


def test_result_round_trips_through_json(confabulating_samples):
    from semantic_entropy_gate.types import EntropyResult

    original = score_samples("q", confabulating_samples, entailment=LexicalEntailment())
    restored = EntropyResult.from_dict(original.to_dict())
    assert restored.entropy == pytest.approx(original.entropy)
    assert restored.n_clusters == original.n_clusters
    assert restored.cluster_assignments == original.cluster_assignments
    assert len(restored.judgements) == len(original.judgements)
    assert restored.estimator is original.estimator


def test_explain_contains_the_evidence(confabulating_samples):
    result = score_samples("q", confabulating_samples, entailment=LexicalEntailment())
    text = result.explain(threshold=0.55)
    assert "SEMANTIC ENTROPY REPORT" in text
    assert "SEMANTIC CLUSTERS" in text
    assert "CONFABULATION SUSPECTED" in text
    assert "lexical-heuristic" in text


def test_explain_without_a_threshold_omits_the_verdict(confident_samples):
    text = score_samples("q", confident_samples, entailment=LexicalEntailment()).explain()
    assert "CONFABULATION SUSPECTED" not in text
    assert "within confidence budget" not in text


def test_cluster_table_is_ordered_by_probability(split_samples):
    result = score_samples(
        "Who discovered penicillin?", split_samples, entailment=LexicalEntailment()
    )
    probabilities = [row[2] for row in result.cluster_table()]
    assert probabilities == sorted(probabilities, reverse=True)


def test_consensus_answer_comes_from_the_majority_cluster(split_samples):
    result = score_samples(
        "Who discovered penicillin?", split_samples, entailment=LexicalEntailment()
    )
    assert "Fleming" in result.consensus_answer
    assert result.agreement > 0.5
