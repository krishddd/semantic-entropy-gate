"""Refusal detection (V7): telling "I know" apart from "I know that I don't know".

A unanimous refusal is a *reliable measurement of a non-answer*. Semantic entropy
is correctly ~0 — the model is consistent — but reading that zero as permission
lets a gate authorise an irreversible action on the strength of ten repetitions
of "I don't know".

The hardest requirement here is the **false-positive** direction: an answer that
merely opens with a hedge must still be treated as an answer, or operators will
switch the gate off.
"""

import pytest

from semantic_entropy_gate import (
    CallableRefusalDetector,
    Gate,
    GateAction,
    LexicalEntailment,
    LLMRefusalDetector,
    NullRefusalDetector,
    PatternRefusalDetector,
    build_report,
    detect_refusals,
    score,
    score_samples,
)
from semantic_entropy_gate.refusal import DEFAULT_REFUSAL_DETECTOR, describe
from semantic_entropy_gate.sampling import from_texts
from semantic_entropy_gate.types import EntropyResult, Sample

REFUSALS = [
    "I don't know.",
    "I do not know.",
    "I'm not sure.",
    "I am not certain.",
    "Unknown.",
    "N/A",
    "I cannot answer that.",
    "I can't determine that.",
    "Unable to determine.",
    "There is no information available on that.",
    "I don't have access to that information.",
    "I have no information about this.",
    "As an AI language model, I cannot verify that.",
    "I'm sorry, but I don't have that data.",
    "It is not possible to determine this.",
    "That cannot be determined from the available data.",
    "This is beyond my knowledge.",
    "Insufficient information.",
    "I would need more context.",
    "idk",
]

# Answers that a naive keyword matcher would wrongly flag. These matter more
# than the positives: a false positive defers a good answer and erodes trust.
ANSWERS = [
    "Paris.",
    "Alexander Fleming discovered penicillin.",
    "I'm not sure of the exact figure, but revenue was about $4.2 million in Q3.",
    "I don't know why the timeout fires, but the fix is to raise it to 30 seconds.",
    "I cannot answer in one word: the capital is Canberra, not Sydney.",
    "The record is unclear on the exact date, though most historians say 1834.",
    "There is no information in the header, but the body contains the user id.",
    "As an AI assistant I can tell you the boiling point is 100 degrees Celsius.",
    "8849 metres.",
    "It is 30 days from the delivery date, per section 4 of the policy.",
]


# ===========================================================================
# The detector itself
# ===========================================================================


@pytest.mark.parametrize("text", REFUSALS)
def test_refusals_are_detected(text):
    flagged, reason = DEFAULT_REFUSAL_DETECTOR.is_refusal(text)
    assert flagged is True, f"missed refusal: {text!r}"
    assert reason


@pytest.mark.parametrize("text", ANSWERS)
def test_substantive_answers_are_not_flagged(text):
    flagged, _ = DEFAULT_REFUSAL_DETECTOR.is_refusal(text)
    assert flagged is False, f"false positive on a real answer: {text!r}"


def test_a_hedge_followed_by_content_is_an_answer():
    # The distinguishing rule: what is left after the refusal phrase is removed.
    hedged = "I'm not sure, but I believe the capital is Canberra in the ACT."
    assert DEFAULT_REFUSAL_DETECTOR.is_refusal(hedged)[0] is False


def test_a_hedge_with_nothing_behind_it_is_a_refusal():
    assert DEFAULT_REFUSAL_DETECTOR.is_refusal("I'm not sure, sorry.")[0] is True


def test_stacked_refusal_phrases_do_not_count_as_content():
    assert (
        DEFAULT_REFUSAL_DETECTOR.is_refusal("I don't know, and I'm not sure I can determine that.")[
            0
        ]
        is True
    )


def test_empty_text_is_not_a_refusal():
    # Empty output is a broken measurement (safety.py), not a considered refusal.
    assert DEFAULT_REFUSAL_DETECTOR.is_refusal("")[0] is False
    assert DEFAULT_REFUSAL_DETECTOR.is_refusal("   ")[0] is False
    assert DEFAULT_REFUSAL_DETECTOR.is_refusal(None)[0] is False


def test_punctuation_and_case_do_not_matter():
    for variant in ("UNKNOWN", "unknown.", "  Unknown!  ", "'unknown'"):
        assert DEFAULT_REFUSAL_DETECTOR.is_refusal(variant)[0] is True


def test_min_content_words_is_tunable():
    strict = PatternRefusalDetector(min_content_words=10)
    # With a high bar, even a short real answer after a hedge reads as a refusal.
    assert strict.is_refusal("I'm not sure, but it is Canberra.")[0] is True


def test_extra_patterns_extend_the_detector():
    detector = PatternRefusalDetector(extra_patterns=[r"the record is silent"])
    assert detector.is_refusal("The record is silent.")[0] is True
    assert DEFAULT_REFUSAL_DETECTOR.is_refusal("The record is silent.")[0] is False


# ===========================================================================
# Scanning a sample set
# ===========================================================================


def test_scan_reports_a_unanimous_refusal():
    report = detect_refusals([Sample(text=t) for t in REFUSALS[:4]])
    assert report.unanimous is True
    assert report.mixed is False
    assert report.rate == 1.0
    assert report.n_refusals == 4


def test_scan_reports_a_mixed_set():
    samples = [Sample(text=t) for t in ["I don't know.", "Paris.", "Paris, France."]]
    report = detect_refusals(samples)
    assert report.mixed is True
    assert report.unanimous is False
    assert report.rate == pytest.approx(1 / 3)


def test_scan_of_pure_answers_is_empty():
    report = detect_refusals([Sample(text=t) for t in ANSWERS[:4]])
    assert report.n_refusals == 0
    assert report.rate == 0.0
    assert describe(report) is None


def test_describe_explains_the_distinction():
    report = detect_refusals([Sample(text=t) for t in REFUSALS[:3]])
    note = describe(report)
    assert "NOT KNOWING" in note
    assert "reliably correct" in note


def test_report_serialises():
    data = detect_refusals([Sample(text="I don't know."), Sample(text="Paris")]).to_dict()
    assert data["n_refusals"] == 1
    assert data["mixed"] is True
    assert data["detector"] == "pattern-refusal"


# ===========================================================================
# Integration with scoring
# ===========================================================================


def test_unanimous_refusal_is_a_reliable_measurement_of_a_non_answer():
    result = score_samples("Who was CEO?", REFUSALS[:4], entailment=LexicalEntailment())
    # The measurement worked fine — this is NOT an integrity failure.
    assert result.reliable is True
    # But the model did not answer.
    assert result.abstained is True
    assert result.refusal_rate == 1.0


def test_the_two_axes_stay_distinct():
    """`reliable` and `abstained` must never be conflated.

    A broken sampler and a cautious model are opposite problems with opposite
    remedies; one flag meaning both would leave a reviewer unable to tell them
    apart.
    """
    refused = score_samples("q", REFUSALS[:4], entailment=LexicalEntailment())
    broken = score_samples("q", ["same", "same", "same"], entailment=LexicalEntailment())

    assert (refused.reliable, refused.abstained) == (True, False) or refused.abstained
    assert refused.reliable is True and refused.abstained is True
    assert broken.reliable is False and broken.abstained is False


def test_partial_refusal_is_recorded_but_does_not_abstain():
    result = score_samples(
        "q",
        ["I don't know.", "Paris.", "Paris, France.", "It is Paris."],
        entailment=LexicalEntailment(),
    )
    assert result.abstained is False
    assert result.refusal_rate == pytest.approx(0.25)
    assert any("could not decide whether it knows" in w for w in result.warnings)


def test_refusal_metadata_is_attached():
    result = score_samples("q", REFUSALS[:3], entailment=LexicalEntailment())
    assert result.metadata["refusals"]["unanimous"] is True
    assert result.metadata["refusals"]["detector"] == "pattern-refusal"


def test_detection_can_be_disabled():
    result = score_samples(
        "q",
        REFUSALS[:4],
        entailment=LexicalEntailment(),
        refusal_detector=NullRefusalDetector(),
    )
    assert result.abstained is False
    assert result.refusal_rate == 0.0


def test_a_custom_detector_is_honoured():
    detector = CallableRefusalDetector(lambda t: "banana" in t.lower())
    result = score_samples(
        "q",
        ["banana", "banana!", "banana?"],
        entailment=LexicalEntailment(),
        refusal_detector=detector,
    )
    assert result.abstained is True


def test_a_crashing_custom_detector_does_not_block_traffic():
    def broken(_text):
        raise RuntimeError("detector exploded")

    result = score_samples(
        "q",
        ["Paris", "Paris."],
        entailment=LexicalEntailment(),
        refusal_detector=CallableRefusalDetector(broken),
    )
    assert result.abstained is False


def test_score_forwards_the_detector():
    result = score(
        "q",
        from_texts(REFUSALS[:4]),
        n_samples=4,
        entailment=LexicalEntailment(),
        refusal_detector=NullRefusalDetector(),
    )
    assert result.abstained is False


# ===========================================================================
# The gate: a non-answer is not authorisation
# ===========================================================================


def test_unanimous_refusal_does_not_open_the_gate():
    gate = Gate(None, threshold=0.55, entailment=LexicalEntailment())
    decision = gate.check_samples("Who was CEO in 2015?", REFUSALS[:4])
    assert decision.action is GateAction.DEFER
    assert decision.allowed is False
    assert "NON-ANSWER" in decision.reason
    assert decision.metadata["abstained"] is True


def test_refusal_never_runs_the_action():
    executed = []
    gate = Gate(None, threshold=0.55, entailment=LexicalEntailment())
    decision = gate.run("q", lambda: executed.append(True), samples=REFUSALS[:4])
    assert decision.executed is False
    assert executed == []


def test_refusal_policy_block():
    gate = Gate(
        None,
        threshold=0.55,
        block_threshold=0.9,
        entailment=LexicalEntailment(),
        refusal_policy="block",
    )
    assert gate.check_samples("q", REFUSALS[:4]).action is GateAction.BLOCK


# Refusals worded consistently enough to land in the same semantic cluster, so
# entropy is genuinely LOW — this is the case where only the abstention flag
# stands between a non-answer and an open gate.
COHERENT_REFUSALS = ["I don't know.", "I do not know.", "I don't know", "I don't know!"]


def test_low_entropy_refusals_are_the_dangerous_case():
    result = score_samples("q", COHERENT_REFUSALS, entailment=LexicalEntailment())
    assert result.normalized_entropy < 0.55  # would pass a threshold check
    assert result.abstained is True  # caught by the other axis


def test_low_entropy_refusals_still_do_not_open_the_gate():
    gate = Gate(None, threshold=0.55, entailment=LexicalEntailment())
    decision = gate.check_samples("q", COHERENT_REFUSALS)
    assert decision.allowed is False
    assert "NON-ANSWER" in decision.reason


def test_refusal_policy_allow_restores_the_old_behaviour():
    gate = Gate(None, threshold=0.55, entailment=LexicalEntailment(), refusal_policy="allow")
    decision = gate.check_samples("q", COHERENT_REFUSALS)
    # Entropy is legitimately low, so with the policy off the ladder allows it...
    assert decision.allowed is True
    assert decision.result.abstained is True
    # ...but the abstention still travels with the decision.
    assert decision.warning is not None
    assert "declined" in decision.warning


def test_varied_refusals_also_defer_on_entropy_grounds():
    """Defence in depth: refusals phrased differently look like disagreement too.

    Whether that happens depends on the entailment oracle — a real NLI model
    merges "I don't know" with "Unknown", the lexical heuristic does not. The
    abstention flag is what makes the outcome the same either way.
    """
    gate = Gate(None, threshold=0.55, entailment=LexicalEntailment(), refusal_policy="allow")
    assert gate.check_samples("q", REFUSALS[:4]).allowed is False


def test_the_defer_hook_fires_on_a_refusal():
    gate = Gate(
        None,
        threshold=0.55,
        entailment=LexicalEntailment(),
        on_defer=lambda prompt, result: f"escalated ({result.refusal_rate:.0%} declined)",
    )
    decision = gate.run("q", lambda: "should not run", samples=REFUSALS[:4])
    assert decision.answer == "escalated (100% declined)"


def test_an_unknown_refusal_policy_is_rejected():
    with pytest.raises(ValueError, match="refusal_policy"):
        Gate(None, entailment=LexicalEntailment(), refusal_policy="ignore")


def test_answers_still_pass_the_gate():
    """The honest case is untouched: real answers must still ALLOW."""
    gate = Gate(None, threshold=0.55, entailment=LexicalEntailment())
    decision = gate.check_samples("q", ["Paris.", "It is Paris.", "Paris, France.", "In Paris."])
    assert decision.action is GateAction.ALLOW
    assert decision.result.abstained is False


def test_gate_stats_count_abstentions():
    gate = Gate(None, threshold=0.55, entailment=LexicalEntailment())
    gate.check_samples("q1", REFUSALS[:4])
    gate.check_samples("q2", ["Paris.", "It is Paris.", "Paris, France."])
    stats = gate.stats()
    assert stats["abstentions"] == 1
    assert stats["unreliable"] == 0


# ===========================================================================
# LLM detector
# ===========================================================================


def test_llm_detector_reads_the_final_line():
    detector = LLMRefusalDetector(lambda _p: "Reasoning about it...\ndeclined")
    assert detector.is_refusal("I have no idea")[0] is True


def test_llm_detector_treats_answered_as_an_answer():
    detector = LLMRefusalDetector(lambda _p: "answered")
    assert detector.is_refusal("Paris")[0] is False


def test_llm_detector_fences_untrusted_input():
    captured = {}

    def judge(prompt):
        captured["prompt"] = prompt
        return "answered"

    LLMRefusalDetector(judge, question="Where?").is_refusal("Paris")
    assert "<<<Paris>>>" in captured["prompt"]
    assert "UNTRUSTED DATA" in captured["prompt"]


def test_a_failing_llm_detector_does_not_block_traffic():
    def broken(_prompt):
        raise RuntimeError("judge down")

    assert LLMRefusalDetector(broken).is_refusal("anything")[0] is False


def test_llm_detector_rejects_a_non_callable():
    with pytest.raises(ValueError, match="callable"):
        LLMRefusalDetector("not callable")


# ===========================================================================
# Surfacing
# ===========================================================================


def test_explain_names_the_non_answer():
    result = score_samples("q", REFUSALS[:4], entailment=LexicalEntailment())
    text = result.explain(threshold=0.55)
    assert "NON-ANSWER" in text
    assert "Declined to answer: 100%" in text
    assert "reliably no answer" in text


def test_report_counts_and_marks_abstentions():
    result = score_samples("q", REFUSALS[:4], entailment=LexicalEntailment())
    report = build_report([result], threshold=0.55)
    assert report.summary()["abstained"] == 1
    assert report.summary()["mean_refusal_rate"] == 1.0
    assert "non-answer" in report.to_markdown()


def test_serialisation_round_trips_the_refusal_fields():
    result = score_samples("q", REFUSALS[:4], entailment=LexicalEntailment())
    restored = EntropyResult.from_dict(result.to_dict())
    assert restored.abstained is True
    assert restored.refusal_rate == 1.0


def test_older_reports_without_refusal_fields_still_load():
    result = score_samples("q", ["a", "b"], entailment=LexicalEntailment())
    data = result.to_dict()
    del data["abstained"]
    del data["refusal_rate"]
    restored = EntropyResult.from_dict(data)
    assert restored.abstained is False
    assert restored.refusal_rate == 0.0
