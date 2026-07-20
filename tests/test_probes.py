"""Semantic Entropy Probes: the single-pass approximation."""

import random

import pytest

from semantic_entropy_gate import LexicalEntailment, score_samples
from semantic_entropy_gate.errors import CalibrationError
from semantic_entropy_gate.probes import (
    SemanticEntropyProbe,
    TokenPosition,
    _fit_standardizer,
    _sigmoid,
    binarize_entropy,
)


def synthetic_hidden_states(n=80, dimension=12, seed=0):
    """Two linearly separable clouds standing in for real residual streams."""
    rng = random.Random(seed)
    states, labels = [], []
    for i in range(n):
        label = i % 2
        centre = 1.0 if label else -1.0
        states.append([rng.gauss(centre, 0.6) for _ in range(dimension)])
        labels.append(label)
    return states, labels


def test_sigmoid_is_stable_at_both_extremes():
    assert _sigmoid(0.0) == 0.5
    assert _sigmoid(1000.0) == pytest.approx(1.0)
    assert _sigmoid(-1000.0) == pytest.approx(0.0)  # would overflow if written naively


def test_standardizer_centres_and_scales():
    mean, scale = _fit_standardizer([[0.0, 5.0], [2.0, 5.0]])
    assert mean == [1.0, 5.0]
    assert scale[0] == pytest.approx(1.0)
    assert scale[1] == 1.0  # zero variance must not divide by zero


def test_probe_learns_a_separable_signal():
    states, labels = synthetic_hidden_states()
    probe = SemanticEntropyProbe(epochs=300).fit(states, labels)
    assert probe.train_auroc > 0.9
    assert probe.n_train == len(labels)
    assert len(probe.weights) == 12


def test_probe_predicts_the_right_side():
    states, labels = synthetic_hidden_states()
    probe = SemanticEntropyProbe(epochs=300).fit(states, labels)
    assert probe.predict_proba([2.0] * 12) > 0.5
    assert probe.predict_proba([-2.0] * 12) < 0.5
    assert probe.predict([2.0] * 12) is True


def test_probe_generalises_to_held_out_data():
    train_states, train_labels = synthetic_hidden_states(seed=1)
    test_states, test_labels = synthetic_hidden_states(seed=2)
    probe = SemanticEntropyProbe(epochs=300).fit(train_states, train_labels)
    metrics = probe.evaluate(test_states, test_labels)
    assert metrics["auroc"] > 0.85
    assert metrics["accuracy"] > 0.8
    assert metrics["n"] == len(test_labels)


def test_probe_round_trips_through_json(tmp_path):
    states, labels = synthetic_hidden_states()
    probe = SemanticEntropyProbe(epochs=100).fit(states, labels)
    path = str(tmp_path / "probe.json")
    probe.save(path)
    restored = SemanticEntropyProbe.load(path)
    assert restored.predict_proba(states[0]) == pytest.approx(probe.predict_proba(states[0]))
    assert restored.train_auroc == pytest.approx(probe.train_auroc)
    assert restored.position == probe.position


def test_unfitted_probe_refuses_to_predict():
    with pytest.raises(CalibrationError, match="not fitted"):
        SemanticEntropyProbe().predict_proba([1.0, 2.0])


def test_wrong_dimensionality_is_rejected():
    states, labels = synthetic_hidden_states(dimension=4)
    probe = SemanticEntropyProbe(epochs=20).fit(states, labels)
    with pytest.raises(CalibrationError, match="4-dim"):
        probe.predict_proba([1.0, 2.0])


def test_ragged_training_data_is_rejected():
    with pytest.raises(CalibrationError, match="same dimensionality"):
        SemanticEntropyProbe().fit([[1.0, 2.0], [3.0]], [0, 1])


def test_mismatched_lengths_are_rejected():
    with pytest.raises(CalibrationError, match="mismatch"):
        SemanticEntropyProbe().fit([[1.0], [2.0]], [0])


def test_no_training_examples_is_rejected():
    with pytest.raises(CalibrationError, match="no training examples"):
        SemanticEntropyProbe().fit([], [])


def test_single_class_training_is_rejected():
    with pytest.raises(CalibrationError, match="single class"):
        SemanticEntropyProbe().fit([[1.0], [2.0]], [1, 1])


def test_binarize_uses_the_median_by_default(confident_samples, confabulating_samples):
    backend = LexicalEntailment()
    results = [
        score_samples("q1", confident_samples, entailment=backend),
        score_samples("q2", confabulating_samples, entailment=backend),
    ]
    labels, threshold = binarize_entropy(results)
    assert labels == [0, 1]
    assert 0.0 < threshold < 1.0


def test_binarize_respects_an_explicit_threshold(confident_samples, confabulating_samples):
    backend = LexicalEntailment()
    results = [
        score_samples("q1", confident_samples, entailment=backend),
        score_samples("q2", confabulating_samples, entailment=backend),
    ]
    labels, threshold = binarize_entropy(results, threshold=0.55)
    assert labels == [0, 1]
    assert threshold == 0.55


def test_binarize_rejects_an_empty_set():
    with pytest.raises(CalibrationError, match="no results"):
        binarize_entropy([])


def test_train_from_results_wires_the_whole_path(confident_samples, confabulating_samples):
    backend = LexicalEntailment()
    results = []
    states = []
    rng = random.Random(7)
    for i in range(20):
        uncertain = i % 2 == 1
        samples = confabulating_samples if uncertain else confident_samples
        results.append(score_samples(f"q{i}", samples, entailment=backend))
        centre = 1.0 if uncertain else -1.0
        states.append([rng.gauss(centre, 0.5) for _ in range(8)])

    probe = SemanticEntropyProbe.train_from_results(
        results, states, position=TokenPosition.TBG, layer=-4, epochs=200
    )
    assert probe.position == TokenPosition.TBG
    assert probe.layer == -4
    assert probe.entropy_threshold is not None
    assert probe.train_auroc > 0.85


def test_explain_reports_its_own_uncertainty():
    states, labels = synthetic_hidden_states()
    probe = SemanticEntropyProbe(epochs=200).fit(states, labels)
    text = probe.explain()
    assert "SEMANTIC ENTROPY PROBE" in text
    assert "Training AUROC" in text
    assert "Validate on" in text


def test_unfitted_probe_explains_that_it_is_unfitted():
    assert "not fitted" in SemanticEntropyProbe().explain()


def test_token_positions_are_named():
    assert TokenPosition.TBG == "tbg"
    assert TokenPosition.SLT == "slt"
