"""The probe fast/slow cascade (X-2) and its capability guard (D-3).

A Semantic Entropy Probe estimates entropy from one hidden state at ~1/10th the
cost, but only for the model whose activation geometry it was trained on. These
tests pin down both halves: the cascade must save the full pipeline for the
ambiguous middle band, and every way of pointing a probe at the wrong model must
fail loudly at construction rather than score silently.
"""

import warnings

import pytest

from semantic_entropy_gate.entailment import LexicalEntailment
from semantic_entropy_gate.gate import Gate
from semantic_entropy_gate.probes import SemanticEntropyProbe
from semantic_entropy_gate.sampling import from_texts
from semantic_entropy_gate.types import Estimator, GateAction, Sample


def _fitted_probe(model_family="llama-3-8b"):
    # 1-D hidden state: larger value -> higher predicted semantic entropy.
    hidden = [[x / 10.0] for x in range(20)]
    labels = [0] * 10 + [1] * 10
    return SemanticEntropyProbe(model_family=model_family, epochs=300).fit(hidden, labels)


def _sampler(_prompt, n):
    # A reliable, low-entropy full-path result for the middle-band fall-through.
    variants = ["the answer is 42", "the answer is 42.", "answer is 42", "yes 42"]
    return [Sample(variants[i % len(variants)]) for i in range(n)]


# ---------------------------------------------------------------- D-3 guard


def test_probe_rejected_without_local_same_model_access():
    probe = _fitted_probe()
    with pytest.raises(ValueError, match="hidden_states|model_access"):
        Gate(None, probe=probe, model_access="api_only", entailment=LexicalEntailment())
    with pytest.raises(ValueError, match="hidden_states|model_access"):
        Gate(
            None,
            probe=probe,
            model_access="local_proxy_model",
            entailment=LexicalEntailment(),
        )


def test_probe_family_mismatch_is_a_hard_error():
    probe = _fitted_probe(model_family="llama-3-8b")
    with pytest.raises(ValueError, match="trained on model_family"):
        Gate(
            None,
            probe=probe,
            model_access="local_same_model",
            model_family="gpt-4",
            entailment=LexicalEntailment(),
        )


def test_unfitted_probe_is_rejected():
    with pytest.raises(ValueError, match="not fitted"):
        Gate(
            None,
            probe=SemanticEntropyProbe(),
            model_access="local_same_model",
            entailment=LexicalEntailment(),
        )


def test_bad_model_access_value_is_rejected():
    with pytest.raises(ValueError, match="model_access"):
        Gate(None, model_access="whatever", entailment=LexicalEntailment())


def test_probe_without_family_warns_but_constructs():
    probe = _fitted_probe(model_family=None)
    with pytest.warns(UserWarning, match="model_family"):
        gate = Gate(
            None,
            probe=probe,
            model_access="local_same_model",
            entailment=LexicalEntailment(),
        )
    assert any("model_family" in w for w in gate.config_warnings)


def test_invalid_probe_band_is_rejected():
    probe = _fitted_probe()
    with pytest.raises(ValueError, match="probe band"):
        Gate(
            None,
            probe=probe,
            model_access="local_same_model",
            probe_floor=0.9,
            probe_ceiling=0.1,
            entailment=LexicalEntailment(),
        )


# ------------------------------------------------------------- the cascade


def _valid_gate(**kwargs):
    kwargs.setdefault("entailment", LexicalEntailment())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Gate(
            _sampler,
            threshold=0.5,
            n_samples=4,
            probe=_fitted_probe(),
            model_access="local_same_model",
            model_family="llama-3-8b",
            **kwargs,
        )


def test_high_confidence_probe_defers_without_sampling():
    gate = _valid_gate()
    decision = gate.check("q", hidden_state=[3.0])
    assert decision.action is GateAction.DEFER
    assert decision.result.estimator is Estimator.PROBE
    assert decision.result.n_samples == 0  # never paid for generations
    assert decision.result.metadata["probe_band"] == "high"


def test_low_confidence_probe_only_fast_allows_when_opted_in():
    # Default: a low probe reading must NOT fast-allow; it falls to the full path.
    gate = _valid_gate(probe_fast_allow=False)
    decision = gate.check("q", hidden_state=[-3.0])
    assert decision.result.estimator is not Estimator.PROBE  # full pipeline ran
    assert decision.result.n_samples == 4


def test_fast_allow_lets_a_confident_low_probe_short_circuit():
    gate = _valid_gate(probe_fast_allow=True)
    decision = gate.check("q", hidden_state=[-3.0])
    assert decision.action is GateAction.ALLOW
    assert decision.result.estimator is Estimator.PROBE
    assert decision.result.metadata["probe_band"] == "low"


def test_middle_band_falls_through_to_the_full_pipeline():
    gate = _valid_gate(probe_fast_allow=True)
    # A hidden state the probe is unsure about pays for the real measurement.
    decision = gate.check("q", hidden_state=[0.9])
    assert decision.result.estimator is not Estimator.PROBE
    assert decision.result.n_samples == 4


def test_no_hidden_state_means_no_probe_path():
    gate = _valid_gate(probe_fast_allow=True)
    decision = gate.check("q")  # no hidden state -> full path
    assert decision.result.estimator is not Estimator.PROBE


def test_gate_without_probe_ignores_hidden_state():
    gate = Gate(from_texts(["a", "b"]), threshold=0.5, entailment=LexicalEntailment())
    # Passing a hidden state to a probe-less gate is harmless: full path runs.
    decision = gate.check("q", hidden_state=[1.0])
    assert decision.result.estimator is not Estimator.PROBE


def test_model_family_round_trips_through_serialisation():
    probe = _fitted_probe(model_family="llama-3-8b")
    restored = SemanticEntropyProbe.from_dict(probe.to_dict())
    assert restored.model_family == "llama-3-8b"
