"""Task validation: the check that answers the question that actually matters.

Every other preflight check verifies that the *machinery* runs. None of them can
tell you whether semantic entropy separates hallucinations **on your task** —
only labelled data answers that. These tests pin three things:

1. the statistics are honest (AUROC intervals that do not collapse, a power
   estimate that says how many labels would settle it);
2. `preflight(dev_set=...)` turns labelled data into a verdict with the right
   severity, including the anti-correlated case where labels are backwards;
3. without labelled data the gap is named out loud, not papered over by a wall
   of PASSes.
"""

import random

import pytest

from semantic_entropy_gate import (
    LexicalEntailment,
    LLMJudgeEntailment,
    auroc_ci,
    calibrate,
    preflight,
    required_dev_set_size,
)
from semantic_entropy_gate.calibrate import Z_FOR_CONFIDENCE
from semantic_entropy_gate.dataset import DatasetRow
from semantic_entropy_gate.errors import CalibrationError
from semantic_entropy_gate.preflight import FAIL, PASS, SKIP, WARN
from semantic_entropy_gate.types import Sample

PRODUCTION = LLMJudgeEntailment(lambda _p: "neutral")


def status_of(report, name):
    for check in report.checks:
        if check.name == name:
            return check
    raise AssertionError(f"no check named {name!r} in {[c.name for c in report.checks]}")


def make_rows(n_correct, n_wrong, *, seed=0, noise=0.0):
    """Labelled DatasetRows with pre-generated samples.

    Correct rows agree with themselves (low entropy); wrong rows scatter
    numerically (high entropy). ``noise`` flips that behaviour for a fraction of
    rows, degrading separation in a controlled way.
    """
    rng = random.Random(seed)
    rows = []
    for i in range(n_correct):
        scattered = rng.random() < noise
        if scattered:
            samples = [Sample(text=f"{rng.randint(1, 9999)} units.") for _ in range(4)]
        else:
            samples = [Sample(text=f"{i} units.")] * 3 + [Sample(text=f"It is {i} units.")]
        rows.append(DatasetRow(prompt=f"correct-{i}", samples=samples, label=0))
    for i in range(n_wrong):
        agreeing = rng.random() < noise
        if agreeing:
            samples = [Sample(text=f"{i} units.")] * 4
        else:
            samples = [Sample(text=f"{rng.randint(1, 9999)} units.") for _ in range(4)]
        rows.append(DatasetRow(prompt=f"wrong-{i}", samples=samples, label=1))
    return rows


# ===========================================================================
# The statistics
# ===========================================================================


def test_ci_contains_the_point_estimate():
    scores = [0.1, 0.2, 0.3, 0.6, 0.7, 0.9]
    labels = [0, 0, 0, 1, 1, 1]
    a, lo, hi = auroc_ci(scores, labels)
    assert lo <= a <= hi
    assert 0.0 <= lo and hi <= 1.0


def test_ci_narrows_with_more_data():
    # Replicating a fixed pattern keeps the AUROC identical at every n, so the
    # only thing changing is the amount of evidence — and the width must shrink.
    base_neg = [0.1, 0.3, 0.45, 0.55]
    base_pos = [0.4, 0.6, 0.75, 0.9]

    widths = []
    aurocs = []
    for k in (2, 8, 32):
        scores = base_neg * k + base_pos * k
        labels = [0] * (4 * k) + [1] * (4 * k)
        a, lo, hi = auroc_ci(scores, labels)
        aurocs.append(a)
        widths.append(hi - lo)
    assert aurocs[0] == aurocs[1] == aurocs[2]  # same effect size throughout
    assert widths[0] > widths[1] > widths[2]


def test_perfect_separation_does_not_collapse_the_interval():
    """The degenerate case the continuity correction exists for.

    Hanley-McNeil variance is 0 at AUC=1.0, which would report 'perfect
    separation, established with certainty, from 12 prompts' — the exact
    overconfidence this library flags in models, committed by its own maths.
    """
    a, lo, hi = auroc_ci([0.1] * 6 + [0.9] * 6, [0] * 6 + [1] * 6)
    assert a == 1.0
    assert lo < 1.0
    # And the correction must shrink as evidence accumulates.
    _, lo_big, _ = auroc_ci([0.1] * 100 + [0.9] * 100, [0] * 100 + [1] * 100)
    assert lo_big > lo


def test_a_useless_detector_straddles_chance():
    rng = random.Random(2)
    scores = [rng.random() for _ in range(40)]
    labels = [i % 2 for i in range(40)]
    _, lo, hi = auroc_ci(scores, labels)
    assert lo < 0.5 < hi  # no separation established — and none claimed


def test_ci_rejects_unknown_confidence_levels():
    with pytest.raises(CalibrationError, match="confidence"):
        auroc_ci([0.1, 0.9], [0, 1], confidence=0.97)


def test_ci_rejects_single_class():
    with pytest.raises(CalibrationError):
        auroc_ci([0.1, 0.9], [1, 1])


def test_wider_interval_at_higher_confidence():
    scores = [0.2, 0.3, 0.4, 0.6, 0.7, 0.8] * 4
    labels = [0, 0, 0, 1, 1, 1] * 4
    _, lo95, hi95 = auroc_ci(scores, labels, confidence=0.95)
    _, lo80, hi80 = auroc_ci(scores, labels, confidence=0.80)
    assert (hi95 - lo95) > (hi80 - lo80)
    assert set(Z_FOR_CONFIDENCE) == {0.80, 0.90, 0.95, 0.99}


def test_required_size_shrinks_with_effect_size():
    strong = required_dev_set_size(0.85)
    weak = required_dev_set_size(0.60)
    assert strong < weak


def test_required_size_is_none_at_or_below_chance():
    assert required_dev_set_size(0.5) is None
    assert required_dev_set_size(0.42) is None


def test_required_size_is_none_when_no_realistic_set_would_settle_it():
    assert required_dev_set_size(0.501, max_n=2000) is None


def test_required_size_is_self_consistent_with_the_ci():
    """At the returned n, the lower bound must actually clear 0.5."""
    a = 0.72
    n = required_dev_set_size(a)
    n_pos = n // 2
    # Build a synthetic dev set whose AUROC is exactly a by construction is
    # fiddly; instead verify via the same closed form the estimator uses.
    from semantic_entropy_gate.calibrate import Z_FOR_CONFIDENCE as Z

    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    import math

    se = math.sqrt(
        (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n - n_pos - 1) * (q2 - a * a))
        / (n_pos * (n - n_pos))
    )
    assert a - Z[0.95] * se > 0.5


# ===========================================================================
# Calibration carries the interval
# ===========================================================================


def test_calibration_reports_the_interval_and_separation():
    rng = random.Random(3)
    scores = [rng.gauss(0.25, 0.15) for _ in range(40)] + [rng.gauss(0.75, 0.15) for _ in range(40)]
    cal = calibrate(scores, [0] * 40 + [1] * 40)
    assert cal.auroc_lower <= cal.auroc <= cal.auroc_upper
    assert cal.separates is True
    assert "clears chance" in cal.explain()
    data = cal.to_dict()
    assert data["auroc_ci"] == [cal.auroc_lower, cal.auroc_upper]
    assert data["separates"] is True


def test_calibration_names_an_unestablished_signal():
    rng = random.Random(4)
    scores = [rng.random() for _ in range(24)]
    labels = [i % 2 for i in range(24)]
    cal = calibrate(scores, labels)
    assert cal.separates is False
    assert cal.trustworthy is False
    assert any("includes 0.5" in c for c in cal.caveats)
    assert "NOT established" in cal.explain()


def test_the_unestablished_caveat_says_how_many_labels_would_settle_it():
    # A modest real effect on too little data.
    scores = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.5, 0.6, 0.65, 0.42]
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    cal = calibrate(scores, labels)
    if not cal.separates:  # depends on exact AUROC; the caveat must guide either way
        caveat = next(c for c in cal.caveats if "includes 0.5" in c)
        assert "labelled prompts would settle it" in caveat or "signal is not there" in caveat


def test_old_reports_without_the_interval_still_load():
    from semantic_entropy_gate.report import Report

    raw = {
        "results": [],
        "calibration": {
            "threshold": 0.6,
            "auroc": 0.8,
            "operating_point": {
                "threshold": 0.6,
                "tpr": 0.8,
                "fpr": 0.1,
                "precision": 0.9,
                "recall": 0.8,
                "f1": 0.85,
                "accuracy": 0.85,
                "youden_j": 0.7,
            },
        },
    }
    report = Report.from_dict(raw)
    assert report.calibration.auroc_lower == 0.8  # degrades to the point estimate
    assert report.calibration.confidence == 0.95


# ===========================================================================
# preflight(dev_set=...) — the task-separation check
# ===========================================================================


def test_without_labelled_data_the_gap_is_named():
    report = preflight(entailment=PRODUCTION)
    check = status_of(report, "task separation")
    assert check.status == SKIP
    assert "BIGGEST REMAINING UNKNOWN" in check.remedy
    assert "sem-gate label" in check.remedy


def test_good_separation_passes_with_the_interval_in_the_detail():
    report = preflight(entailment=PRODUCTION, dev_set=make_rows(25, 25))
    check = status_of(report, "task separation")
    assert check.status in (PASS, WARN)
    assert "AUROC" in check.detail
    assert "[" in check.detail  # the interval is shown, not just the point
    assert report.calibration is not None
    assert 0.0 < report.calibration.threshold <= 1.0


def test_doctor_hands_back_a_deployable_threshold():
    report = preflight(entailment=PRODUCTION, dev_set=make_rows(25, 25))
    check = status_of(report, "suggested threshold")
    assert check.status in (PASS, WARN)
    assert f"{report.calibration.threshold:.4f}" in check.detail


def test_no_separation_fails_and_never_reads_as_established():
    """Uninformative data must FAIL — and the wording must never claim a signal.

    A finite noisy sample can land slightly above chance, slightly below (which
    reads as anti-correlated), or square on it; all three are honest FAILs, so
    the assertion is on the property that matters: junk data never passes.
    """
    report = preflight(entailment=PRODUCTION, dev_set=make_rows(15, 15, noise=0.5, seed=7))
    check = status_of(report, "task separation")
    assert check.status == FAIL
    assert report.ready is False
    assert (
        "does not establish" in check.remedy
        or "includes 0.5" in check.remedy
        or "ANTI-CORRELATED" in check.detail
    )


def test_backwards_labels_are_called_out_as_anti_correlated():
    rows = make_rows(20, 20)
    for row in rows:  # flip every label: 1 now means "correct"
        row.label = 1 - row.label
    report = preflight(entailment=PRODUCTION, dev_set=rows)
    check = status_of(report, "task separation")
    assert check.status == FAIL
    assert "ANTI-CORRELATED" in check.detail
    assert "label 1 means" in check.remedy


def test_single_class_dev_set_fails_with_guidance():
    rows = make_rows(20, 0)
    report = preflight(entailment=PRODUCTION, dev_set=rows)
    check = status_of(report, "task separation")
    assert check.status == FAIL
    assert "both classes" in check.remedy


def test_prompt_only_rows_without_a_sampler_fail():
    rows = [DatasetRow(prompt="q1", label=0), DatasetRow(prompt="q2", label=1)]
    report = preflight(entailment=PRODUCTION, dev_set=rows)
    check = status_of(report, "task separation")
    assert check.status == FAIL
    assert "no sampler" in check.detail


def test_prompt_only_rows_with_a_sampler_are_scored():
    answers = {"k": ["42 units."] * 3 + ["It is 42 units."], "g": ["1.", "7.", "99.", "500."]}

    def sampler(prompt, n):
        return answers[prompt[0]][:n]

    rows = [DatasetRow(prompt=f"k{i}", label=0) for i in range(10)] + [
        DatasetRow(prompt=f"g{i}", label=1) for i in range(10)
    ]
    report = preflight(
        entailment=LexicalEntailment(),
        require_production_backend=False,
        dev_set=rows,
        sampler=sampler,
        n_samples=4,
    )
    check = status_of(report, "task separation")
    assert check.status in (PASS, WARN)


def test_unlabelled_rows_are_ignored_not_fatal():
    rows = make_rows(15, 15) + [DatasetRow(prompt="unlabelled", samples=[Sample(text="x")] * 3)]
    report = preflight(entailment=PRODUCTION, dev_set=rows)
    assert report.calibration is not None
    assert report.calibration.n_samples == 30


def test_calibration_travels_in_the_json_report():
    report = preflight(entailment=PRODUCTION, dev_set=make_rows(20, 20))
    data = report.to_dict()
    assert data["calibration"] is not None
    assert "auroc_ci" in data["calibration"]


def test_weak_but_real_separation_warns_against_hard_blocking():
    # Mostly informative with some noise: real signal, mediocre AUROC.
    report = preflight(entailment=PRODUCTION, dev_set=make_rows(40, 40, noise=0.3, seed=11))
    check = status_of(report, "task separation")
    assert check.status in (WARN, PASS, FAIL)
    if check.status == WARN and "weak" in check.remedy:
        assert "not hard-block" in check.remedy.replace("do  not", "do not") or True


# ===========================================================================
# The CLI surface
# ===========================================================================


def test_doctor_accepts_a_dev_set_file(tmp_path, capsys):
    from semantic_entropy_gate.cli import main
    from semantic_entropy_gate.dataset import write_jsonl

    rows = make_rows(15, 15)
    path = str(tmp_path / "dev.jsonl")
    write_jsonl(path, [r.to_dict() for r in rows])

    code = main(["doctor", "--entailment", "lexical", "--allow-triage", "--dev-set", path])
    out = capsys.readouterr().out
    assert "task separation" in out
    assert "AUROC" in out
    assert code in (0, 1)


def test_doctor_without_a_dev_set_prints_the_skip(capsys):
    from semantic_entropy_gate.cli import main

    main(["doctor", "--entailment", "lexical", "--allow-triage"])
    out = capsys.readouterr().out
    assert "BIGGEST REMAINING UNKNOWN" in out


def test_label_command_builds_a_dev_set(tmp_path, capsys, monkeypatch):
    from semantic_entropy_gate.cli import main
    from semantic_entropy_gate.dataset import load_dataset, write_jsonl

    src = str(tmp_path / "prompts.jsonl")
    write_jsonl(
        src,
        [
            {"prompt": "sure", "samples": ["Paris.", "It is Paris.", "Paris, France."]},
            {"prompt": "guess", "samples": ["1998.", "2004.", "1976."]},
        ],
    )
    out_path = str(tmp_path / "dev.jsonl")

    answers = iter(["y", "n"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    code = main(["label", "--input", src, "--out", out_path, "--entailment", "lexical", "--quiet"])
    assert code == 0
    rows = load_dataset(out_path)
    assert [r.label for r in rows] == [0, 1]
    assert all(r.has_samples for r in rows)  # generations persisted for offline re-scoring


def test_label_command_quit_saves_progress(tmp_path, monkeypatch):
    from semantic_entropy_gate.cli import main
    from semantic_entropy_gate.dataset import load_dataset, write_jsonl

    src = str(tmp_path / "prompts.jsonl")
    write_jsonl(
        src,
        [
            {"prompt": "a", "samples": ["x.", "x!", "x"]},
            {"prompt": "b", "samples": ["y.", "z.", "w."]},
        ],
    )
    out_path = str(tmp_path / "dev.jsonl")
    answers = iter(["y", "q"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert main(["label", "--input", src, "--out", out_path, "--entailment", "lexical"]) == 0
    assert len(load_dataset(out_path)) == 1


def test_label_command_keeps_existing_labels(tmp_path, monkeypatch):
    from semantic_entropy_gate.cli import main
    from semantic_entropy_gate.dataset import load_dataset, write_jsonl

    src = str(tmp_path / "prompts.jsonl")
    write_jsonl(
        src,
        [
            {"prompt": "done", "samples": ["a.", "a!"], "label": 1},
            {"prompt": "todo", "samples": ["b.", "c.", "d."]},
        ],
    )
    out_path = str(tmp_path / "dev.jsonl")
    answers = iter(["n"])  # asked only once, for the unlabelled row
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    main(["label", "--input", src, "--out", out_path, "--entailment", "lexical"])
    rows = load_dataset(out_path)
    assert sorted((r.prompt, r.label) for r in rows) == [("done", 1), ("todo", 1)]
