"""Preflight: catching a misconfigured deployment before it protects anything.

Installing the package is not the same as deploying it correctly, and every
incorrect deployment produces a reassuringly low entropy. These tests pin the
checks that tell a developer their gate is a no-op *before* it is in front of
production traffic.
"""

import pytest

from semantic_entropy_gate import (
    CrossEncoderEntailment,
    Gate,
    LexicalEntailment,
    LLMJudgeEntailment,
    PreflightReport,
    preflight,
)
from semantic_entropy_gate.preflight import FAIL, PASS, SKIP, WARN, Check, check_gate
from semantic_entropy_gate.refusal import CallableRefusalDetector


def status_of(report: PreflightReport, name: str) -> str:
    for check in report.checks:
        if check.name == name:
            return check.status
    raise AssertionError(f"no check named {name!r} in {[c.name for c in report.checks]}")


def good_sampler(prompt, n):
    return [f"answer variant {i}" for i in range(n)]


def deterministic_sampler(prompt, n):
    return ["the same answer every time"] * n


def short_sampler(prompt, n):
    return ["only one"]


# ===========================================================================
# Backend tier
# ===========================================================================


def test_triage_backend_fails_preflight_by_default():
    report = preflight(entailment=LexicalEntailment())
    assert status_of(report, "entailment backend") == FAIL
    assert report.ready is False
    remedy = report.failures[0].remedy
    assert "semantic-entropy-gate[hf]" in remedy
    assert "judge=" in remedy


def test_triage_backend_can_be_downgraded_to_a_warning():
    report = preflight(entailment=LexicalEntailment(), require_production_backend=False)
    assert status_of(report, "entailment backend") == WARN
    assert report.ready is True


def test_production_backend_passes():
    report = preflight(entailment=LLMJudgeEntailment(lambda _p: "neutral"))
    assert status_of(report, "entailment backend") == PASS


def test_backends_declare_a_tier():
    assert LexicalEntailment().tier == "triage"
    assert LLMJudgeEntailment(lambda _p: "neutral").tier == "production"
    assert CrossEncoderEntailment("some/model").tier == "production"


def test_a_backend_that_cannot_classify_fails():
    class Broken(LexicalEntailment):
        def classify(self, premise, hypothesis, *, context=""):
            raise RuntimeError("checkpoint missing")

    report = preflight(entailment=Broken(), require_production_backend=False)
    assert status_of(report, "backend loads") == FAIL
    assert "checkpoint missing" in report.failures[0].detail


def test_a_working_backend_reports_its_timing():
    report = preflight(entailment=LexicalEntailment(), require_production_backend=False)
    assert status_of(report, "backend loads") == PASS
    assert (
        "test pairs classified" in [c.detail for c in report.checks if c.name == "backend loads"][0]
    )


# ===========================================================================
# Sampler — the checks that need the sampler to actually run
# ===========================================================================


def test_a_deterministic_sampler_fails():
    """The single most common way this library is deployed as a no-op."""
    report = preflight(
        sampler=deterministic_sampler,
        entailment=LexicalEntailment(),
        require_production_backend=False,
    )
    assert status_of(report, "sampler diversity") == FAIL
    assert report.ready is False
    remedy = [c for c in report.checks if c.name == "sampler diversity"][0].remedy
    assert "temperature" in remedy
    assert "allow everything" in remedy


def test_a_diverse_sampler_passes():
    report = preflight(
        sampler=good_sampler, entailment=LexicalEntailment(), require_production_backend=False
    )
    assert status_of(report, "sampler diversity") == PASS
    assert status_of(report, "sampler count") == PASS


def test_a_short_sampler_return_is_caught():
    report = preflight(
        sampler=short_sampler,
        entailment=LexicalEntailment(),
        n_samples=10,
        require_production_backend=False,
    )
    assert status_of(report, "sampler count") == FAIL
    assert "1 of 10" in [c for c in report.checks if c.name == "sampler count"][0].detail


def test_a_mildly_short_return_only_warns():
    report = preflight(
        sampler=lambda p, n: [f"a{i}" for i in range(8)],
        entailment=LexicalEntailment(),
        n_samples=10,
        require_production_backend=False,
    )
    assert status_of(report, "sampler count") == WARN


def test_a_raising_sampler_fails():
    def broken(prompt, n):
        raise RuntimeError("api key missing")

    report = preflight(
        sampler=broken, entailment=LexicalEntailment(), require_production_backend=False
    )
    assert status_of(report, "sampler runs") == FAIL
    assert "api key missing" in report.failures[0].detail


def test_a_non_callable_sampler_fails():
    report = preflight(
        sampler="not callable", entailment=LexicalEntailment(), require_production_backend=False
    )
    assert status_of(report, "sampler callable") == FAIL


def test_no_sampler_is_skipped_not_failed():
    report = preflight(entailment=LexicalEntailment(), require_production_backend=False)
    assert status_of(report, "sampler") == SKIP
    assert report.ready is True


def test_logprobs_are_reported_when_present():
    report = preflight(
        sampler=lambda p, n: [(f"a{i}", -0.5) for i in range(n)],
        entailment=LexicalEntailment(),
        require_production_backend=False,
    )
    assert status_of(report, "log-probabilities") == PASS


def test_partial_logprobs_warn():
    def mixed(prompt, n):
        return [("a", -0.5)] + [f"b{i}" for i in range(n - 1)]

    report = preflight(
        sampler=mixed, entailment=LexicalEntailment(), require_production_backend=False
    )
    assert status_of(report, "log-probabilities") == WARN


# ===========================================================================
# Threshold
# ===========================================================================


def test_an_uncalibrated_threshold_warns():
    report = preflight(
        threshold=0.55, entailment=LexicalEntailment(), require_production_backend=False
    )
    assert status_of(report, "threshold") == WARN
    assert "calibrate" in [c for c in report.checks if c.name == "threshold"][0].remedy


def test_a_calibrated_threshold_passes():
    report = preflight(
        threshold=0.62,
        calibrated=True,
        entailment=LexicalEntailment(),
        require_production_backend=False,
    )
    assert status_of(report, "threshold") == PASS


def test_an_out_of_range_threshold_fails():
    report = preflight(
        threshold=5.0, entailment=LexicalEntailment(), require_production_backend=False
    )
    assert status_of(report, "threshold") == FAIL


def test_a_missing_threshold_warns():
    report = preflight(entailment=LexicalEntailment(), require_production_backend=False)
    assert status_of(report, "threshold") == WARN


# ===========================================================================
# Refusal detector
# ===========================================================================


def test_the_default_refusal_detector_passes():
    report = preflight(entailment=LexicalEntailment(), require_production_backend=False)
    assert status_of(report, "refusal detector") == PASS


def test_a_detector_that_flags_real_answers_fails():
    everything = CallableRefusalDetector(lambda _t: True)
    report = preflight(
        refusal_detector=everything,
        entailment=LexicalEntailment(),
        require_production_backend=False,
    )
    assert status_of(report, "refusal detector") == FAIL
    assert "defer good traffic" in report.failures[0].remedy


def test_a_detector_that_misses_refusals_warns():
    nothing = CallableRefusalDetector(lambda _t: False)
    report = preflight(
        refusal_detector=nothing,
        entailment=LexicalEntailment(),
        require_production_backend=False,
    )
    assert status_of(report, "refusal detector") == WARN


# ===========================================================================
# Report shape
# ===========================================================================


def test_failures_sort_to_the_top():
    report = preflight(sampler=deterministic_sampler, entailment=LexicalEntailment())
    assert report.checks[0].status == FAIL


def test_report_renders_remedies():
    text = preflight(entailment=LexicalEntailment()).render()
    assert "SEMANTIC ENTROPY GATE - PREFLIGHT" in text
    assert "NOT READY" in text
    assert "->" in text  # a remedy is shown


def test_a_clean_report_says_ready():
    report = preflight(
        sampler=good_sampler,
        entailment=LLMJudgeEntailment(lambda _p: "neutral"),
        threshold=0.6,
        calibrated=True,
    )
    assert report.ready is True
    assert report.warnings == []
    assert "READY: this gate is fit" in report.render()


def test_report_serialises():
    data = preflight(entailment=LexicalEntailment()).to_dict()
    assert data["ready"] is False
    assert data["n_failures"] >= 1
    assert all({"name", "status", "detail", "remedy"} <= set(c) for c in data["checks"])


def test_check_serialises():
    assert Check("n", PASS, "d", "r").to_dict() == {
        "name": "n",
        "status": PASS,
        "detail": "d",
        "remedy": "r",
    }


def test_check_gate_preflights_a_configured_gate():
    gate = Gate(good_sampler, threshold=0.55, entailment=LexicalEntailment())
    report = check_gate(gate, require_production_backend=False)
    assert status_of(report, "sampler diversity") == PASS
    assert status_of(report, "threshold") == WARN  # uncalibrated


# ===========================================================================
# Gate-level enforcement
# ===========================================================================


def test_gate_records_its_backend_tier():
    gate = Gate(None, entailment=LexicalEntailment())
    assert gate.backend_tier == "triage"
    assert gate.stats()["backend_tier"] == "triage"


def test_gate_can_refuse_a_triage_backend_outright():
    with pytest.raises(ValueError, match="tier 'triage'"):
        Gate(None, entailment=LexicalEntailment(), require_production_backend=True)


def test_a_production_backend_satisfies_the_strict_gate():
    gate = Gate(
        None,
        entailment=LLMJudgeEntailment(lambda _p: "neutral"),
        require_production_backend=True,
    )
    assert gate.backend_tier == "production"


def test_the_triage_warning_is_recorded_on_the_gate():
    gate = Gate(None, entailment=LexicalEntailment())
    assert any("tier 'triage'" in w for w in gate.config_warnings)
