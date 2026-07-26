"""Calibration on the probability scale: ECE, reliability curve, Platt scaling.

AUROC answers a *ranking* question ("do higher scores sit above hallucinations?").
A threshold gate compares the score to a fixed number, so it implicitly trusts
the score as a *probability* — which is the question ECE answers. These tests pin
down the distinction the review (C-1) flagged as missing.
"""

import math

import pytest

from semantic_entropy_gate.calibrate import (
    ECE_DEPLOY_MAX,
    calibrate,
    expected_calibration_error,
    fit_platt,
    reliability_curve,
)
from semantic_entropy_gate.errors import CalibrationError


def _calibrated_dataset(per_bin=100):
    """Scores whose value equals the true positive rate in their bin -> ECE ~ 0."""
    scores, labels = [], []
    for center in [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]:
        positives = round(center * per_bin)
        for k in range(per_bin):
            scores.append(center)
            labels.append(1 if k < positives else 0)
    return scores, labels


def test_ece_is_zero_on_perfectly_calibrated_scores():
    scores, labels = _calibrated_dataset()
    assert expected_calibration_error(scores, labels) == pytest.approx(0.0, abs=1e-9)


def test_ece_is_large_when_scores_are_a_shifted_ranking():
    # Perfect ranking (AUROC = 1) but compressed into [0.1, 0.5]: never says "wrong"
    # with high probability, so as a probability it is badly wrong.
    scores, labels = _calibrated_dataset()
    compressed = [s * 0.4 + 0.1 for s in scores]
    assert expected_calibration_error(compressed, labels) > ECE_DEPLOY_MAX


def test_reliability_curve_drops_empty_bins_and_covers_the_top_edge():
    bins = reliability_curve([0.0, 0.5, 1.0], [0, 0, 1], n_bins=10)
    # Three occupied bins only; the 1.0 lands in the last bin, not a phantom 11th.
    assert [b.count for b in bins] == [1, 1, 1]
    assert max(b.upper for b in bins) == pytest.approx(1.0)
    assert all(0.0 <= b.lower < b.upper <= 1.0 for b in bins)


def test_reliability_bin_gap_matches_predicted_minus_observed():
    (bin_,) = reliability_curve([0.8, 0.8], [1, 0], n_bins=10)
    assert bin_.mean_predicted == pytest.approx(0.8)
    assert bin_.fraction_positive == pytest.approx(0.5)
    assert bin_.gap == pytest.approx(0.3)


def test_ece_weights_bins_by_count():
    # One lonely miscalibrated point must not outweigh a big well-behaved bin.
    scores = [0.5] * 99 + [0.5]
    labels = [1 if k < 50 else 0 for k in range(99)] + [0]
    ece = expected_calibration_error(scores, labels)
    assert 0.0 <= ece <= 0.02


def test_empty_input_is_rejected_not_silently_zero():
    with pytest.raises(CalibrationError):
        expected_calibration_error([], [])


def test_platt_scaling_lowers_ece_on_a_shifted_score():
    scores, labels = _calibrated_dataset()
    compressed = [s * 0.4 + 0.1 for s in scores]
    before = expected_calibration_error(compressed, labels)
    scaler = fit_platt(compressed, labels)
    after = expected_calibration_error(scaler.transform(compressed), labels)
    assert after < before
    # Output is a genuine probability in the unit interval.
    assert all(0.0 <= scaler(s) <= 1.0 for s in compressed)


def test_platt_needs_both_classes():
    with pytest.raises(CalibrationError):
        fit_platt([0.2, 0.8], [1, 1])


def test_calibrate_reports_ece_and_flags_uncalibrated_scores():
    scores, labels = _calibrated_dataset()
    compressed = [s * 0.4 + 0.1 for s in scores]
    result = calibrate(compressed, labels)
    assert result.ece is not None and result.ece > ECE_DEPLOY_MAX
    assert result.calibrated is False
    # Strong ranking (a linear shift preserves order) yet not trustworthy, because
    # the score is not a probability — exactly the AUROC/ECE divergence C-1 is
    # about: separation can be real while the raw score is unreadable as a chance.
    assert result.auroc > 0.75
    assert result.separates is True
    assert result.trustworthy is False
    assert any("ECE" in c for c in result.caveats)


def test_calibrate_marks_calibrated_scores_as_such():
    scores, labels = _calibrated_dataset()
    result = calibrate(scores, labels)
    assert result.calibrated is True
    assert result.ece == pytest.approx(0.0, abs=1e-9)
    assert "ece" in result.to_dict() and result.to_dict()["calibrated"] is True


def test_raw_nats_calibration_skips_ece_rather_than_reporting_nonsense():
    # Unbounded raw-nats scores are not on the probability scale; ECE is undefined
    # there and must be omitted, not computed from clamped garbage.
    result = calibrate([0.1, 0.9, 2.3, 3.1], [0, 0, 1, 1], normalized=False)
    assert result.ece is None
    assert result.calibrated is None
    assert result.reliability == []


def test_reliability_curve_rejects_nonfinite_scores():
    with pytest.raises(CalibrationError):
        reliability_curve([0.5, math.nan], [0, 1])
