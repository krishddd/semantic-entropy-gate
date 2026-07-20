"""Calibration: AUROC, AUPRC and threshold selection."""

import pytest

from semantic_entropy_gate.calibrate import (
    _mid_ranks,
    apply_threshold,
    auprc,
    auroc,
    calibrate,
    calibrate_from_pairs,
    roc_curve,
    threshold_sweep,
)
from semantic_entropy_gate.errors import CalibrationError


def test_perfect_separation_gives_auroc_one():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0


def test_inverted_separation_gives_auroc_zero():
    assert auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0


def test_all_scores_tied_gives_auroc_half():
    # Mid-rank tie handling: a detector that cannot distinguish anything must
    # score exactly chance, not 1.0.
    assert auroc([0.5] * 6, [0, 0, 0, 1, 1, 1]) == pytest.approx(0.5)


def test_partial_ties_are_handled_correctly():
    # One positive tied with one negative at 0.5.
    value = auroc([0.1, 0.5, 0.5, 0.9], [0, 0, 1, 1])
    assert value == pytest.approx(0.875)


def test_mid_ranks_average_within_tie_groups():
    assert _mid_ranks([1.0, 2.0, 2.0, 3.0]) == [1.0, 2.5, 2.5, 4.0]


def test_auroc_requires_both_classes():
    with pytest.raises(CalibrationError, match="single-class"):
        auroc([0.1, 0.2], [1, 1])


def test_auprc_is_one_for_perfect_ranking():
    assert auprc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)


def test_auprc_equals_base_rate_for_random_tied_scores():
    assert auprc([0.5] * 4, [0, 0, 1, 1]) == pytest.approx(0.5)


def test_auprc_requires_a_positive():
    with pytest.raises(CalibrationError, match="no positive"):
        auprc([0.1, 0.2], [0, 0])


def test_threshold_sweep_spans_the_full_roc():
    curve = threshold_sweep([0.1, 0.4, 0.6, 0.9], [0, 0, 1, 1])
    assert curve[0].tpr == 1.0 and curve[0].fpr == 1.0  # flag everything
    assert curve[-1].tpr == 0.0 and curve[-1].fpr == 0.0  # flag nothing


def test_threshold_sweep_metrics_are_consistent():
    curve = threshold_sweep([0.1, 0.4, 0.6, 0.9], [0, 0, 1, 1])
    for point in curve:
        assert 0.0 <= point.tpr <= 1.0
        assert 0.0 <= point.fpr <= 1.0
        assert point.youden_j == pytest.approx(point.tpr - point.fpr)
        assert point.recall == pytest.approx(point.tpr)


def test_roc_curve_is_sorted_by_fpr():
    points = roc_curve([0.1, 0.4, 0.6, 0.9], [0, 0, 1, 1])
    assert points == sorted(points)


def test_youden_finds_the_separating_threshold():
    cal = calibrate([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], criterion="youden")
    assert cal.auroc == 1.0
    assert 0.2 < cal.threshold <= 0.8
    assert cal.operating_point.tpr == 1.0
    assert cal.operating_point.fpr == 0.0


def test_target_fpr_respects_the_budget():
    scores = [0.1, 0.3, 0.5, 0.55, 0.7, 0.9]
    labels = [0, 0, 1, 0, 1, 1]
    cal = calibrate(scores, labels, criterion="target_fpr", target_fpr=0.0)
    assert cal.operating_point.fpr == 0.0
    assert cal.criterion == "target_fpr"


def test_target_fpr_returns_the_least_bad_point_when_infeasible():
    # Every threshold that catches anything also raises a false alarm.
    cal = calibrate([0.5, 0.5, 0.4], [1, 0, 0], criterion="target_fpr", target_fpr=-1.0)
    assert cal.operating_point.fpr >= 0.0


def test_target_recall_minimises_false_alarms_subject_to_recall():
    scores = [0.1, 0.3, 0.5, 0.7, 0.9]
    labels = [0, 0, 1, 1, 1]
    cal = calibrate(scores, labels, criterion="target_recall", target_recall=1.0)
    assert cal.operating_point.tpr >= 1.0


def test_target_recall_falls_back_when_infeasible():
    cal = calibrate([0.5, 0.4], [1, 0], criterion="target_recall", target_recall=2.0)
    assert cal.operating_point.tpr <= 1.0


def test_f1_and_accuracy_criteria_run():
    scores = [0.1, 0.2, 0.7, 0.8]
    labels = [0, 0, 1, 1]
    assert calibrate(scores, labels, criterion="f1").operating_point.f1 == pytest.approx(1.0)
    assert calibrate(scores, labels, criterion="accuracy").operating_point.accuracy == 1.0


def test_unknown_criterion_is_rejected():
    with pytest.raises(CalibrationError, match="unknown criterion"):
        calibrate([0.1, 0.9], [0, 1], criterion="vibes")


def test_length_mismatch_is_rejected():
    with pytest.raises(CalibrationError, match="mismatch"):
        calibrate([0.1, 0.9], [0])


def test_empty_dev_set_is_rejected():
    with pytest.raises(CalibrationError, match="empty"):
        calibrate([], [])


def test_single_class_dev_set_is_rejected():
    with pytest.raises(CalibrationError, match="single class"):
        calibrate([0.1, 0.9], [1, 1])


def test_calibration_reports_the_class_balance():
    cal = calibrate([0.1, 0.2, 0.8], [0, 0, 1])
    assert cal.n_samples == 3
    assert cal.n_positive == 1
    assert cal.n_negative == 2
    assert cal.base_rate == pytest.approx(1 / 3)


def test_calibration_serialises_and_explains():
    cal = calibrate([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    data = cal.to_dict()
    assert data["auroc"] == 1.0
    assert data["operating_point"]["tpr"] == 1.0
    assert len(data["curve"]) == len(cal.curve)
    text = cal.explain()
    assert "AUROC" in text and "strong separation" in text


def test_weak_separation_is_labelled_as_such():
    cal = calibrate([0.5, 0.5, 0.5, 0.5001], [0, 1, 0, 1])
    assert "weak" in cal.explain() or "moderate" in cal.explain()


def test_calibrate_accepts_entropy_results(confident_samples, confabulating_samples):
    from semantic_entropy_gate import LexicalEntailment, score_samples

    backend = LexicalEntailment()
    results = [
        score_samples("q1", confident_samples, entailment=backend),
        score_samples("q2", confabulating_samples, entailment=backend),
    ]
    cal = calibrate(results, [0, 1])
    assert cal.auroc == 1.0
    assert 0.0 < cal.threshold <= 1.0


def test_calibrate_from_pairs_matches_calibrate():
    pairs = [(0.1, 0), (0.2, 0), (0.8, 1), (0.9, 1)]
    assert calibrate_from_pairs(pairs).auroc == 1.0


def test_apply_threshold_flags_the_right_results(confident_samples, confabulating_samples):
    from semantic_entropy_gate import LexicalEntailment, score_samples

    backend = LexicalEntailment()
    results = [
        score_samples("q1", confident_samples, entailment=backend),
        score_samples("q2", confabulating_samples, entailment=backend),
    ]
    assert apply_threshold(results, 0.55) == [False, True]


def test_raw_nats_calibration_is_supported():
    from semantic_entropy_gate import LexicalEntailment, score_samples

    backend = LexicalEntailment()
    results = [
        score_samples("q1", ["a", "a", "a"], entailment=backend),
        score_samples("q2", ["1", "2", "3"], entailment=backend),
    ]
    cal = calibrate(results, [0, 1], normalized=False)
    assert cal.metadata["normalized"] is False
    assert cal.threshold > 0.0
