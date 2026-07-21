"""Reports: the artefacts a reviewer actually reads."""

import json

import pytest

from semantic_entropy_gate import Gate, LexicalEntailment, build_report, calibrate, score_samples
from semantic_entropy_gate.report import Report, write_json, write_markdown


@pytest.fixture
def results(confident_samples, confabulating_samples, split_samples):
    backend = LexicalEntailment()
    return [
        score_samples("Where is the Eiffel Tower?", confident_samples, entailment=backend),
        score_samples("Boiling point of astatine?", confabulating_samples, entailment=backend),
        score_samples("Who discovered penicillin?", split_samples, entailment=backend),
    ]


def test_summary_counts_and_ranges(results):
    summary = build_report(results, threshold=0.55).summary()
    assert summary["n_prompts"] == 3
    assert summary["unanimous_prompts"] == 1
    assert summary["flagged"] == 1
    assert summary["min_normalized_entropy"] == 0.0
    assert summary["max_normalized_entropy"] > 0.8
    assert summary["entailment_backends"] == ["lexical-heuristic"]
    assert summary["total_samples"] == 18


def test_empty_report_summary_is_safe():
    assert build_report([]).summary() == {"n_prompts": 0}
    assert "No prompts scored" in build_report([]).to_markdown()


def test_riskiest_is_ordered_by_entropy(results):
    riskiest = build_report(results).riskiest(3)
    entropies = [r.normalized_entropy for r in riskiest]
    assert entropies == sorted(entropies, reverse=True)


def test_json_round_trip_preserves_results(results):
    report = build_report(results, threshold=0.55, title="t")
    restored = Report.from_dict(json.loads(report.to_json()))
    assert restored.title == "t"
    assert restored.threshold == 0.55
    assert len(restored.results) == 3
    assert restored.results[1].n_clusters == results[1].n_clusters


def test_json_round_trip_preserves_calibration(results):
    cal = calibrate(results, [0, 1, 1])
    report = build_report(results, calibration=cal)
    restored = Report.from_dict(json.loads(report.to_json()))
    assert restored.calibration.auroc == pytest.approx(cal.auroc)
    assert restored.calibration.threshold == pytest.approx(cal.threshold)
    assert len(restored.calibration.curve) == len(cal.curve)


def test_judgements_are_excluded_by_default_and_included_on_request(results):
    report = build_report(results)
    assert "judgements" not in json.loads(report.to_json())["results"][0]
    with_judgements = json.loads(report.to_json(include_judgements=True))
    assert "judgements" in with_judgements["results"][0]


def test_calibration_threshold_is_used_when_none_is_given(results):
    cal = calibrate(results, [0, 1, 1])
    assert build_report(results, calibration=cal).threshold == pytest.approx(cal.threshold)


def test_markdown_contains_the_evidence(results):
    markdown = build_report(results, threshold=0.55).to_markdown()
    assert "# Semantic Entropy Report" in markdown
    assert "## Summary" in markdown
    assert "## Riskiest prompts" in markdown
    assert "## Evidence" in markdown
    assert "Boiling point of astatine?" in markdown
    assert "| Cluster | p | n | Representative |" in markdown
    assert "<details><summary>All generations by cluster</summary>" in markdown


def test_markdown_includes_the_calibration_section_and_roc(results):
    cal = calibrate(results, [0, 1, 1])
    markdown = build_report(results, calibration=cal).to_markdown()
    assert "## Calibration" in markdown
    assert "AUROC" in markdown
    assert "### ROC curve" in markdown
    assert "TPR 1.0" in markdown
    assert "O = chosen" in markdown


def test_markdown_verdict_language_tracks_auroc():
    from semantic_entropy_gate.report import _auroc_verdict

    assert "strong" in _auroc_verdict(0.95)
    assert "moderate" in _auroc_verdict(0.7)
    assert "weak" in _auroc_verdict(0.5)
    assert "Do not deploy" in _auroc_verdict(0.5)


def test_markdown_escapes_pipes_in_prompts():
    backend = LexicalEntailment()
    result = score_samples("a | b | c", ["x", "y"], entailment=backend)
    markdown = build_report([result], threshold=0.5).to_markdown()
    assert "a \\| b \\| c" in markdown


def test_markdown_truncates_very_long_cells():
    backend = LexicalEntailment()
    result = score_samples("q" * 400, ["x", "y"], entailment=backend)
    markdown = build_report([result]).to_markdown()
    assert "…" in markdown
    assert "q" * 400 not in markdown


def test_gate_decisions_appear_in_the_report(confabulating_samples):
    gate = Gate(None, threshold=0.55, entailment=LexicalEntailment())
    decision = gate.check_samples("q", confabulating_samples)
    markdown = build_report([decision.result], threshold=0.55, decisions=[decision]).to_markdown()
    assert "## Gate decisions" in markdown
    assert "`defer`" in markdown


def test_files_are_written_to_disk(results, tmp_path):
    report = build_report(results, threshold=0.55)
    json_path = write_json(report, str(tmp_path / "out" / "report.json"))
    md_path = write_markdown(report, str(tmp_path / "out" / "report.md"))
    assert json.loads(open(json_path, encoding="utf-8").read())["summary"]["n_prompts"] == 3
    assert open(md_path, encoding="utf-8").read().startswith("# Semantic Entropy Report")


def test_report_loads_from_a_saved_file(results, tmp_path):
    path = tmp_path / "report.json"
    build_report(results, threshold=0.55).to_json(str(path))
    assert len(Report.load(str(path)).results) == 3


def test_schema_version_is_recorded(results):
    assert build_report(results).to_dict()["schema_version"] == 1
