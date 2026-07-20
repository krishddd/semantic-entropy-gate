"""Entailment backends: the equivalence oracle the whole method rests on."""

import warnings

import pytest

from semantic_entropy_gate.entailment import (
    CachedEntailment,
    CannedEntailment,
    CrossEncoderEntailment,
    LexicalEntailment,
    LLMJudgeEntailment,
    _build_label_map,
    _with_context,
    auto_entailment,
)
from semantic_entropy_gate.errors import EntailmentBackendError
from semantic_entropy_gate.types import EntailmentLabel

# --------------------------------------------------------------------- lexical


def test_conflicting_numbers_are_a_contradiction(lexical):
    label, _ = lexical.classify("It boils at 610 Kelvin.", "It boils at 337 Kelvin.")
    assert label is EntailmentLabel.CONTRADICTION


def test_matching_numbers_are_not_a_contradiction(lexical):
    label, _ = lexical.classify("It boils at 610 Kelvin.", "610 Kelvin.")
    assert label is EntailmentLabel.ENTAILMENT


def test_number_words_are_normalised_to_digits(lexical):
    label, _ = lexical.classify("There were three survivors.", "There were 7 survivors.")
    assert label is EntailmentLabel.CONTRADICTION


def test_negation_polarity_mismatch_is_a_contradiction(lexical):
    label, _ = lexical.classify(
        "The treaty was ratified by France.", "The treaty was not ratified by France."
    )
    assert label is EntailmentLabel.CONTRADICTION


def test_paraphrase_entails_in_both_directions(lexical):
    assert lexical.classify("Paris.", "It is in Paris.")[0] is EntailmentLabel.ENTAILMENT
    assert lexical.classify("It is in Paris.", "Paris.")[0] is EntailmentLabel.ENTAILMENT


def test_unrelated_answers_are_neutral(lexical):
    label, _ = lexical.classify("Alexander Fleming.", "Guido van Rossum.")
    assert label is EntailmentLabel.NEUTRAL


def test_empty_strings_do_not_crash(lexical):
    assert lexical.classify("", "")[0] is EntailmentLabel.ENTAILMENT
    assert lexical.classify("", "something")[0] is EntailmentLabel.NEUTRAL


def test_bidirectional_strict_requires_both_directions():
    # Canned: A entails B but not the reverse.
    oracle = CannedEntailment(
        {("A", "B"): EntailmentLabel.ENTAILMENT, ("B", "A"): EntailmentLabel.NEUTRAL}
    )
    equivalent, _, _ = oracle.bidirectional("A", "B", strict=True)
    assert equivalent is False


def test_bidirectional_relaxed_accepts_a_neutral_pairing():
    oracle = CannedEntailment(
        {("A", "B"): EntailmentLabel.ENTAILMENT, ("B", "A"): EntailmentLabel.NEUTRAL}
    )
    equivalent, _, _ = oracle.bidirectional("A", "B", strict=False)
    assert equivalent is True


def test_bidirectional_relaxed_still_rejects_mutual_neutrality():
    oracle = CannedEntailment(default=EntailmentLabel.NEUTRAL)
    equivalent, _, _ = oracle.bidirectional("A", "B", strict=False)
    assert equivalent is False


def test_bidirectional_relaxed_rejects_any_contradiction():
    oracle = CannedEntailment(
        {("A", "B"): EntailmentLabel.ENTAILMENT, ("B", "A"): EntailmentLabel.CONTRADICTION}
    )
    equivalent, _, _ = oracle.bidirectional("A", "B", strict=False)
    assert equivalent is False


# ----------------------------------------------------------------- llm judge


def test_llm_judge_parses_each_verdict():
    replies = iter(["entailment", "  Contradiction.  ", "neutral"])
    judge = LLMJudgeEntailment(lambda _prompt: next(replies))
    assert judge.classify("a", "b")[0] is EntailmentLabel.ENTAILMENT
    assert judge.classify("a", "b")[0] is EntailmentLabel.CONTRADICTION
    assert judge.classify("a", "b")[0] is EntailmentLabel.NEUTRAL


def test_llm_judge_prefers_contradiction_when_both_words_appear():
    judge = LLMJudgeEntailment(lambda _p: "This is a contradiction, not entailment.")
    assert judge.classify("a", "b")[0] is EntailmentLabel.CONTRADICTION


def test_llm_judge_falls_back_to_neutral_on_garbage():
    # Neutral keeps answers in separate clusters -> reports MORE uncertainty,
    # which is the safe direction for a guardrail.
    judge = LLMJudgeEntailment(lambda _p: "I'm not sure what you mean")
    assert judge.classify("a", "b")[0] is EntailmentLabel.NEUTRAL


def test_llm_judge_strict_parse_raises_on_garbage():
    judge = LLMJudgeEntailment(lambda _p: "???", strict_parse=True)
    with pytest.raises(EntailmentBackendError):
        judge.classify("a", "b")


def test_llm_judge_wraps_backend_exceptions():
    def broken(_prompt):
        raise RuntimeError("429 rate limited")

    with pytest.raises(EntailmentBackendError, match="429"):
        LLMJudgeEntailment(broken).classify("a", "b")


def test_llm_judge_rejects_a_non_callable():
    with pytest.raises(EntailmentBackendError):
        LLMJudgeEntailment("not callable")


def test_llm_judge_prompt_includes_question_and_both_statements():
    captured = {}

    def judge(prompt):
        captured["prompt"] = prompt
        return "entailment"

    LLMJudgeEntailment(judge).classify("premise text", "hypothesis text", context="the question?")
    assert "the question?" in captured["prompt"]
    assert "premise text" in captured["prompt"]
    assert "hypothesis text" in captured["prompt"]


# -------------------------------------------------------------------- canned


def test_canned_returns_entailment_for_identical_strings():
    oracle = CannedEntailment(default=EntailmentLabel.CONTRADICTION)
    assert oracle.classify("same", "same")[0] is EntailmentLabel.ENTAILMENT


def test_canned_symmetric_lookup():
    oracle = CannedEntailment({("A", "B"): EntailmentLabel.ENTAILMENT}, symmetric=True)
    assert oracle.classify("B", "A")[0] is EntailmentLabel.ENTAILMENT


def test_canned_delegates_to_a_fallback_backend():
    oracle = CannedEntailment({}, fallback=LexicalEntailment())
    assert oracle.classify("610 Kelvin", "337 Kelvin")[0] is EntailmentLabel.CONTRADICTION


# -------------------------------------------------------------------- cached


def test_cache_avoids_repeat_calls():
    inner = CannedEntailment(default=EntailmentLabel.NEUTRAL)
    cached = CachedEntailment(inner)
    cached.classify("a", "b")
    cached.classify("a", "b")
    assert cached.stats == {"hits": 1, "misses": 1, "size": 1}
    assert len(inner.calls) == 1


def test_cache_distinguishes_direction_and_context():
    cached = CachedEntailment(CannedEntailment(default=EntailmentLabel.NEUTRAL))
    cached.classify("a", "b")
    cached.classify("b", "a")
    cached.classify("a", "b", context="different question")
    assert cached.stats["misses"] == 3


def test_cache_respects_maxsize():
    cached = CachedEntailment(CannedEntailment(default=EntailmentLabel.NEUTRAL), maxsize=1)
    cached.classify("a", "b")
    cached.classify("c", "d")
    assert cached.stats["size"] == 1


def test_cached_name_records_the_wrapped_backend():
    assert CachedEntailment(LexicalEntailment()).name == "cached(lexical-heuristic)"


# ---------------------------------------------------------------------- auto


def test_auto_entailment_warns_when_it_falls_back_to_the_heuristic():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        with pytest.warns(UserWarning, match="lexical heuristic"):
            backend = auto_entailment()
        assert "lexical-heuristic" in backend.name
    else:  # pragma: no cover - only on machines with the hf extra installed
        assert "cross-encoder" in auto_entailment().name


def test_auto_entailment_quiet_suppresses_the_warning():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            auto_entailment(quiet=True)
    else:  # pragma: no cover
        auto_entailment(quiet=True)


def test_auto_entailment_prefers_a_judge_over_the_heuristic():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        backend = auto_entailment(judge=lambda _p: "entailment", quiet=True)
        assert "llm-judge" in backend.name
    else:  # pragma: no cover
        pass


# ------------------------------------------------------------- cross-encoder


def test_cross_encoder_reports_a_helpful_error_without_the_extra():
    backend = CrossEncoderEntailment("some/model")
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        with pytest.raises(EntailmentBackendError, match=r"semantic-entropy-gate\[hf\]"):
            backend.classify("a", "b")
    else:  # pragma: no cover - the extra is installed on this machine
        pass


def test_cross_encoder_is_lazy_and_downloads_nothing_at_construction():
    backend = CrossEncoderEntailment("definitely/not-a-real-model")
    assert backend._model is None
    assert backend.name.startswith("cross-encoder:")


def test_label_map_handles_arbitrary_id2label_casing():
    mapping = _build_label_map({0: "CONTRADICTION", 1: "Neutral", 2: "entailment"})
    assert mapping == {
        0: EntailmentLabel.CONTRADICTION,
        1: EntailmentLabel.NEUTRAL,
        2: EntailmentLabel.ENTAILMENT,
    }


def test_context_is_prepended_to_the_nli_input():
    assert _with_context("Paris.", "Where is it?") == "Where is it? Paris."
    assert _with_context("Paris.", "  ") == "Paris."


def test_entailment_label_scores_match_the_mnli_encoding():
    assert EntailmentLabel.CONTRADICTION.score == 0
    assert EntailmentLabel.NEUTRAL.score == 1
    assert EntailmentLabel.ENTAILMENT.score == 2
