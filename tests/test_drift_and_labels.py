"""Label quality and traffic drift — the two assumptions doctor took on faith.

Neither can be verified against truth, and these tests also pin that the
*wording* never claims otherwise: a suspect is a review candidate, drift is a
distribution change, and small samples refuse to render a verdict at all.
"""

import random

import pytest

from semantic_entropy_gate import (
    Gate,
    LexicalEntailment,
    LLMJudgeEntailment,
    audit_labels,
    calibrate,
    detect_drift,
    ks_2sample,
    preflight,
)
from semantic_entropy_gate.dataset import DatasetRow
from semantic_entropy_gate.errors import CalibrationError
from semantic_entropy_gate.preflight import FAIL, PASS, WARN
from semantic_entropy_gate.report import Report, build_report
from semantic_entropy_gate.types import Sample

PRODUCTION = LLMJudgeEntailment(lambda _p: "neutral")


# ===========================================================================
# audit_labels: suspects
# ===========================================================================


def test_a_backwards_label_tops_the_suspect_ranking():
    scores = [0.05, 0.10, 0.95, 0.90, 0.92]  # last three clearly "guessing"
    labels = [0, 0, 1, 1, 0]  # ...but the last is labelled CORRECT
    audit = audit_labels(scores, labels, prompts=[f"p{i}" for i in range(5)])
    assert audit.suspects
    assert audit.suspects[0].index == 4
    assert audit.suspects[0].label == 0
    assert "labelled CORRECT" in audit.suspects[0].why


def test_a_unanimous_row_labelled_hallucinated_is_a_suspect():
    scores = [0.05, 0.9, 0.02]
    labels = [0, 1, 1]  # index 2: near-zero entropy yet labelled hallucinated
    audit = audit_labels(scores, labels)
    suspect = next(s for s in audit.suspects if s.index == 2)
    assert "near-unanimously" in suspect.why
    # The wording must admit the alternative this library cannot see:
    assert "consistent-but-wrong" in suspect.why


def test_consistent_labels_produce_no_suspects():
    scores = [0.1, 0.15, 0.2, 0.85, 0.9, 0.95]
    labels = [0, 0, 0, 1, 1, 1]
    audit = audit_labels(scores, labels)
    assert audit.suspects == []
    assert audit.clean


def test_suspects_are_ranked_worst_first_and_capped():
    scores = [0.6, 0.7, 0.8, 0.9]
    labels = [0, 0, 0, 0]  # all four contradict their label, increasingly badly
    audit = audit_labels(scores, labels, max_suspects=2)
    assert len(audit.suspects) == 2
    assert audit.suspects[0].misfit >= audit.suspects[1].misfit
    assert audit.suspects[0].index == 3


def test_misfit_threshold_is_tunable():
    audit = audit_labels([0.4], [0], misfit_threshold=0.3)
    assert len(audit.suspects) == 1
    audit = audit_labels([0.4], [0], misfit_threshold=0.5)
    assert audit.suspects == []


def test_entropy_results_supply_their_own_prompts(confident_samples):
    from semantic_entropy_gate import score_samples

    result = score_samples("the question", confident_samples, entailment=LexicalEntailment())
    audit = audit_labels([result], [1])  # unanimous answer labelled hallucinated
    assert audit.suspects[0].prompt == "the question"


def test_mismatched_lengths_are_rejected():
    with pytest.raises(CalibrationError, match="mismatch"):
        audit_labels([0.1, 0.2], [0])


# ===========================================================================
# audit_labels: exact conflicts
# ===========================================================================


def test_the_same_prompt_labelled_both_ways_is_a_conflict():
    scores = [0.1, 0.1, 0.9]
    labels = [0, 1, 1]
    prompts = ["What is X?", "what is  x?", "other"]  # same question, spacing/case aside
    audit = audit_labels(scores, labels, prompts=prompts)
    assert len(audit.conflicts) == 1
    prompt, indices = audit.conflicts[0]
    assert indices == [0, 1]
    assert not audit.clean


def test_duplicates_with_agreeing_labels_are_not_conflicts():
    audit = audit_labels([0.1, 0.12], [0, 0], prompts=["same q", "same q"])
    assert audit.conflicts == []


def test_audit_serialises():
    data = audit_labels([0.9], [0], prompts=["p"]).to_dict()
    assert data["n_rows"] == 1
    assert data["suspects"][0]["misfit"] == 0.9
    assert data["clean"] is False


# ===========================================================================
# doctor: label consistency check
# ===========================================================================


def make_rows(n_correct, n_wrong):
    rows = []
    for i in range(n_correct):
        samples = [Sample(text=f"{i} units.")] * 3 + [Sample(text=f"It is {i} units.")]
        rows.append(DatasetRow(prompt=f"correct-{i}", samples=samples, label=0))
    rng = random.Random(0)
    for i in range(n_wrong):
        samples = [Sample(text=f"{rng.randint(1, 9999)} units.") for _ in range(4)]
        rows.append(DatasetRow(prompt=f"wrong-{i}", samples=samples, label=1))
    return rows


def check_named(report, name):
    for check in report.checks:
        if check.name == name:
            return check
    raise AssertionError(f"no check named {name!r}")


def test_doctor_passes_consistent_labels():
    report = preflight(entailment=PRODUCTION, dev_set=make_rows(20, 20))
    assert check_named(report, "label consistency").status == PASS


def test_doctor_fails_on_conflicting_duplicates():
    rows = make_rows(15, 15)
    dup = rows[0]
    rows.append(DatasetRow(prompt=dup.prompt, samples=dup.samples, label=1))
    report = preflight(entailment=PRODUCTION, dev_set=rows)
    check = check_named(report, "label consistency")
    assert check.status == FAIL
    assert "labelled both ways" in check.detail
    assert "at least one of each pair is wrong" in check.remedy
    assert report.ready is False


def test_doctor_warns_when_many_labels_contradict_the_signal():
    rows = make_rows(10, 10)
    # Flip a third of the labels: signal survives, but suspects abound.
    for row in rows[:3] + rows[10:13]:
        row.label = 1 - row.label
    report = preflight(entailment=PRODUCTION, dev_set=rows)
    check = check_named(report, "label consistency")
    assert check.status in (WARN, FAIL)
    if check.status == WARN:
        assert "Not proof of mislabelling" in check.remedy
        assert "sem-gate label --relabel" in check.remedy


# ===========================================================================
# ks_2sample
# ===========================================================================


def test_identical_samples_do_not_reject():
    a = [i / 100 for i in range(100)]
    d, p = ks_2sample(a, list(a))
    assert d == pytest.approx(0.0, abs=1e-9)
    assert p > 0.99


def test_disjoint_samples_reject_hard():
    a = [0.1 + i / 1000 for i in range(100)]
    b = [0.8 + i / 1000 for i in range(100)]
    d, p = ks_2sample(a, b)
    assert d == pytest.approx(1.0)
    assert p < 1e-6


def test_same_distribution_usually_survives():
    rng = random.Random(5)
    a = [rng.gauss(0.5, 0.15) for _ in range(150)]
    b = [rng.gauss(0.5, 0.15) for _ in range(150)]
    _, p = ks_2sample(a, b)
    assert p > 0.01


def test_empty_samples_are_rejected():
    with pytest.raises(CalibrationError, match="non-empty"):
        ks_2sample([], [0.1])


# ===========================================================================
# detect_drift / Gate.check_drift
# ===========================================================================


def test_drift_refuses_a_verdict_on_thin_history():
    with pytest.raises(CalibrationError, match="at least 20"):
        detect_drift([0.5] * 5, [0.5] * 50)


def test_no_drift_when_traffic_matches_the_dev_set():
    rng = random.Random(6)
    dev = [rng.gauss(0.4, 0.1) for _ in range(80)]
    live = [rng.gauss(0.4, 0.1) for _ in range(80)]
    report = detect_drift(live, dev)
    assert report.drifted is False
    assert "still describes current traffic" in report.explain()


def test_drift_detected_when_traffic_shifts():
    rng = random.Random(7)
    dev = [rng.gauss(0.3, 0.08) for _ in range(80)]
    live = [rng.gauss(0.7, 0.08) for _ in range(80)]
    report = detect_drift(live, dev)
    assert report.drifted is True
    text = report.explain()
    assert "DRIFT DETECTED" in text
    assert "not automatically wrong" in text  # the claim stays honest
    assert "sem-gate label" in text  # and the remedy is concrete
    assert "HIGHER" in text


def test_lower_drift_points_at_the_sampler():
    rng = random.Random(8)
    dev = [rng.gauss(0.7, 0.08) for _ in range(80)]
    live = [rng.gauss(0.2, 0.08) for _ in range(80)]
    report = detect_drift(live, dev)
    assert report.drifted is True
    assert "degraded sampler" in report.explain()


def test_drift_report_serialises():
    rng = random.Random(9)
    report = detect_drift([rng.random() for _ in range(30)], [rng.random() for _ in range(30)])
    data = report.to_dict()
    assert set(data) >= {"statistic", "p_value", "drifted", "n_live", "n_dev"}


def test_gate_check_drift_uses_its_own_history(confabulating_samples, confident_samples):
    # Calibrate on a mixed dev set, then serve mixed traffic: no drift.
    backend = LexicalEntailment()
    from semantic_entropy_gate import score_samples

    dev_results, dev_labels = [], []
    for i in range(15):
        dev_results.append(score_samples(f"k{i}", confident_samples, entailment=backend))
        dev_labels.append(0)
        dev_results.append(score_samples(f"g{i}", confabulating_samples, entailment=backend))
        dev_labels.append(1)
    calibration = calibrate(dev_results, dev_labels)
    assert calibration.dev_scores  # the reference distribution travelled

    gate = Gate(None, threshold=max(0.05, calibration.threshold), entailment=backend)
    for i in range(15):
        gate.check_samples(f"live-k{i}", confident_samples)
        gate.check_samples(f"live-g{i}", confabulating_samples)

    report = gate.check_drift(calibration)
    assert report.n_live == 30
    assert report.drifted is False


def test_gate_check_drift_detects_a_shifted_workload(confabulating_samples, confident_samples):
    backend = LexicalEntailment()
    from semantic_entropy_gate import score_samples

    # Dev set: overwhelmingly confident traffic.
    dev_results = [
        score_samples(f"k{i}", confident_samples, entailment=backend) for i in range(40)
    ] + [score_samples("g", confabulating_samples, entailment=backend)]
    calibration = calibrate(dev_results, [0] * 40 + [1])

    # Live: nothing but guessing.
    gate = Gate(None, threshold=0.55, entailment=backend)
    for i in range(25):
        gate.check_samples(f"live-{i}", confabulating_samples)

    report = gate.check_drift(calibration)
    assert report.drifted is True
    assert report.live_mean > report.dev_mean


def test_gate_drift_excludes_failed_measurements(confident_samples):
    backend = LexicalEntailment()
    gate = Gate(None, threshold=0.55, entailment=backend)
    for i in range(25):
        gate.check_samples(f"ok-{i}", confident_samples)
    # A degenerate call lands in history as unreliable with a synthetic score...
    gate.check_samples("broken", ["same", "same", "same"])
    report = gate.check_drift([0.0] * 40)
    # ...and must not be counted as traffic evidence.
    assert report.n_live == 25


def test_drift_survives_the_report_round_trip(tmp_path, confident_samples, confabulating_samples):
    """The whole point of storing dev_scores: drift is checkable months later."""
    from semantic_entropy_gate import score_samples

    backend = LexicalEntailment()
    results = [score_samples(f"k{i}", confident_samples, entailment=backend) for i in range(10)]
    results += [
        score_samples(f"g{i}", confabulating_samples, entailment=backend) for i in range(10)
    ]
    calibration = calibrate(results, [0] * 10 + [1] * 10)

    path = str(tmp_path / "report.json")
    build_report(results, calibration=calibration).to_json(path)
    restored = Report.load(path).calibration
    assert restored.dev_scores == pytest.approx(calibration.dev_scores, abs=1e-5)

    rng = random.Random(10)
    live = [rng.random() for _ in range(30)]
    assert detect_drift(live, restored.dev_scores).n_dev == 20
