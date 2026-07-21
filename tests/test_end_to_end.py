"""End-to-end: the workflow a developer actually runs, start to finish.

Score a labelled dev set -> calibrate a threshold -> deploy a gate -> stop the
irreversible action -> write the reports. All offline, all deterministic.
"""

import json

import pytest

from semantic_entropy_gate import (
    Gate,
    GateAction,
    LexicalEntailment,
    Report,
    build_report,
    calibrate,
    score,
    score_samples,
)
from semantic_entropy_gate.dataset import load_dataset
from semantic_entropy_gate.pytest_plugin import (
    assert_confident,
    assert_uncertain,
    entropy_of,
)
from semantic_entropy_gate.sampling import from_texts

DEV_SET = [
    (["Paris.", "In Paris.", "Paris, France.", "It is Paris."], 0),
    (["Jane Austen.", "It was Jane Austen.", "Austen.", "Jane Austen wrote it."], 0),
    (["100 degrees Celsius.", "100 C.", "It boils at 100 C.", "100."], 0),
    (["Guido van Rossum.", "It was Guido van Rossum.", "Van Rossum.", "Guido van Rossum"], 0),
    (["610 Kelvin.", "503 Kelvin.", "337 Kelvin.", "575 Kelvin."], 1),
    (["1998.", "2004.", "1976.", "2011."], 1),
    (["250 employees.", "1200 employees.", "40 staff.", "3000 people."], 1),
    (["4 million.", "19 million.", "72 million.", "6 million."], 1),
]


@pytest.fixture
def scored():
    backend = LexicalEntailment()
    results = [
        score_samples(f"q{i}", samples, entailment=backend, metadata={"label": label})
        for i, (samples, label) in enumerate(DEV_SET)
    ]
    return results, [label for _, label in DEV_SET]


def test_full_workflow(scored, tmp_path):
    results, labels = scored

    # 1. The signal separates the two classes.
    calibration = calibrate(results, labels, criterion="youden")
    assert calibration.auroc == pytest.approx(1.0)
    assert calibration.operating_point.fpr == 0.0
    assert calibration.operating_point.tpr == 1.0

    # 2. The calibrated threshold, deployed.
    gate = Gate(None, threshold=calibration.threshold, entailment=LexicalEntailment())
    actions = [gate.decide(result).action for result in results]
    assert [a.allowed for a in actions] == [not bool(label) for label in labels]

    # 3. The reports a reviewer reads.
    report = build_report(results, calibration=calibration, title="e2e")
    json_path = str(tmp_path / "report.json")
    md_path = str(tmp_path / "report.md")
    report.to_json(json_path)
    report.to_markdown(md_path)

    data = json.loads(open(json_path, encoding="utf-8").read())
    assert data["summary"]["n_prompts"] == 8
    assert data["summary"]["flagged"] == 4
    assert "AUROC" in open(md_path, encoding="utf-8").read()

    # 4. The decision is replayable months later.
    restored = Report.load(json_path)
    assert restored.results[4].n_clusters == results[4].n_clusters
    assert "SEMANTIC CLUSTERS" in restored.results[4].explain(threshold=restored.threshold)


def test_the_gate_actually_stops_the_side_effect():
    """The point of the library: money must not move on a guess."""
    refunds = []

    def issue_refund(order_id):
        refunds.append(order_id)
        return "refunded"

    confident = Gate(
        from_texts(["30 days.", "It is 30 days.", "Thirty days.", "30 days"]),
        threshold=0.55,
        n_samples=4,
        entailment=LexicalEntailment(),
    )
    guessing = Gate(
        from_texts(["14 days.", "60 days.", "7 days.", "90 days."]),
        threshold=0.55,
        n_samples=4,
        entailment=LexicalEntailment(),
    )

    assert confident.run("q", issue_refund, order_id=1).executed is True
    assert guessing.run("q", issue_refund, order_id=2).executed is False
    assert refunds == [1]


def test_bundled_demo_dataset_flows_through_the_pipeline():
    import os

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "data", "demo.jsonl")
    if not os.path.exists(path):  # pragma: no cover
        pytest.skip("bundled dataset not present")

    rows = load_dataset(path)
    backend = LexicalEntailment()
    results = [score_samples(r.prompt, r.samples, entailment=backend) for r in rows]
    calibration = calibrate(results, [r.label for r in rows])
    # The lexical backend is triage grade; it must still clear chance comfortably.
    assert calibration.auroc >= 0.9


def test_pytest_helpers_pass_and_fail_for_the_right_reasons():
    confident = entropy_of("q", samples=["Paris.", "In Paris.", "Paris, France."])
    uncertain = entropy_of("q", samples=["610 K.", "337 K.", "503 K."])

    assert_confident(confident, threshold=0.4)
    assert_uncertain(uncertain, threshold=0.5)

    with pytest.raises(AssertionError, match="SEMANTIC CLUSTERS"):
        assert_confident(uncertain, threshold=0.4)
    with pytest.raises(AssertionError, match="expected uncertainty"):
        assert_uncertain(confident, threshold=0.5)


def test_entropy_of_accepts_a_sampler():
    result = entropy_of("q", from_texts(["a", "a", "a"]), n_samples=3)
    assert result.n_clusters == 1


def test_entropy_of_needs_samples_or_a_sampler():
    with pytest.raises(ValueError, match="sampler or samples"):
        entropy_of("q")


def test_public_api_surface_is_importable():
    import semantic_entropy_gate as package

    for name in package.__all__:
        assert hasattr(package, name), f"{name} is exported but missing"


def test_version_is_exposed():
    import semantic_entropy_gate as package

    assert package.__version__ == "0.3.0"


def test_scoring_is_deterministic_for_fixed_samples(confabulating_samples):
    backend = LexicalEntailment()
    first = score("q", from_texts(confabulating_samples), n_samples=6, entailment=backend)
    second = score("q", from_texts(confabulating_samples), n_samples=6, entailment=backend)
    assert first.entropy == second.entropy
    assert first.cluster_assignments == second.cluster_assignments


def test_a_greedy_sampler_reports_zero_entropy_the_documented_failure_mode():
    # Documented trap: identical generations always look certain, however wrong.
    result = score(
        "q",
        lambda p, n: ["a wrong but confident answer"] * n,
        n_samples=8,
        entailment=LexicalEntailment(),
    )
    assert result.normalized_entropy == 0.0
    assert result.n_clusters == 1


def test_gate_action_ladder_covers_the_full_range(
    split_samples, confident_samples, confabulating_samples
):
    gate = Gate(
        None,
        threshold=0.5,
        warn_threshold=0.2,
        block_threshold=0.8,
        entailment=LexicalEntailment(),
    )
    assert gate.check_samples("q", confident_samples).action is GateAction.ALLOW
    assert gate.check_samples("q", split_samples).action is GateAction.WARN
    assert gate.check_samples("q", confabulating_samples).action is GateAction.BLOCK
