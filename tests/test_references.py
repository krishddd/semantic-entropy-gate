"""Derivable labels, exact staleness, and sampling as protocol.

The 0.6.0 honest note said label truth and re-sample honesty were irreducible.
This suite pins how far that shrank:

- a label with a **reference** is derived, not asserted — re-derivable by
  anyone from the same two strings and oracle;
- a label with **provenance** (the answer it judged) goes stale *exactly* when
  the model stops giving that answer;
- the drift re-label sample is **seeded and drawn from gate history** — there
  is no step where a human picks flattering prompts.

What remains is pinned too: the oracle can abstain (and must, rather than
laundering its ignorance into ground truth), and reference *truth* is still
the user's.
"""

import random

from semantic_entropy_gate import (
    Gate,
    LexicalEntailment,
    LLMJudgeEntailment,
    preflight,
    score_samples,
)
from semantic_entropy_gate.dataset import DatasetRow, load_dataset, write_jsonl
from semantic_entropy_gate.preflight import PASS, WARN
from semantic_entropy_gate.types import Sample
from semantic_entropy_gate.validation import (
    cross_check_references,
    derive_label,
    find_stale_labels,
)

LX = LexicalEntailment()
PRODUCTION = LLMJudgeEntailment(lambda _p: "neutral")


def check_named(report, name):
    matches = [c for c in report.checks if c.name == name]
    return matches[0] if matches else None


# ===========================================================================
# derive_label
# ===========================================================================


def test_equivalent_consensus_derives_correct():
    label, why = derive_label("Paris.", "Paris", LX)
    assert label == 0
    assert "equivalent" in why


def test_contradicting_consensus_derives_hallucinated():
    label, why = derive_label("It is 60 days.", "30 days", LX)
    assert label == 1
    assert "contradicts" in why


def test_an_undecidable_pair_abstains_instead_of_guessing():
    """The load-bearing property: deriving an uncertain verdict and recording
    it as ground truth would launder the oracle's ignorance into a fact."""
    label, why = derive_label("The mitochondria is the powerhouse.", "Paris", LX)
    assert label is None
    assert "human must judge" in why


def test_no_consensus_is_a_hallucination_by_definition():
    label, why = derive_label("", "Paris", LX)
    assert label == 1
    assert "no consensus" in why


def test_derivation_is_reproducible():
    """The point of derived labels: same strings + same oracle = same label."""
    first = derive_label("Paris.", "Paris", LX)
    second = derive_label("Paris.", "Paris", LX)
    assert first == second


# ===========================================================================
# cross_check_references
# ===========================================================================


def result_for(prompt, samples):
    return score_samples(prompt, samples, entailment=LX)


def test_agreeing_label_and_reference_produce_no_mismatch():
    results = [result_for("q1", ["Paris.", "It is Paris.", "Paris"])]
    assert cross_check_references(results, [0], ["Paris"], LX) == []


def test_a_label_contradicting_its_reference_is_caught():
    # Model consistently says 60 days; reference says 30; asserted label says correct.
    results = [result_for("q1", ["60 days.", "It is 60 days.", "60 days"])]
    mismatches = cross_check_references(results, [0], ["30 days"], LX)
    assert len(mismatches) == 1
    assert mismatches[0]["derived"] == 1
    assert mismatches[0]["label"] == 0
    assert "contradicts" in mismatches[0]["why"]


def test_rows_without_references_are_skipped():
    results = [result_for("q", ["Paris.", "Paris"])]
    assert cross_check_references(results, [1], [None], LX) == []


def test_oracle_abstention_produces_no_mismatch():
    # Lexically unrelatable pair: the oracle abstains, so nothing contradicts.
    results = [result_for("q", ["The powerhouse.", "The powerhouse!"])]
    assert cross_check_references(results, [0], ["Paris"], LX) == []


# ===========================================================================
# find_stale_labels
# ===========================================================================


def test_a_label_about_the_current_answer_is_not_stale():
    results = [result_for("q", ["30 days.", "It is 30 days.", "30 days"])]
    assert find_stale_labels(results, ["30 days."], LX) == []


def test_a_label_about_a_vanished_answer_is_stale():
    # Labelled when the model said 30 days; it now says 90.
    results = [result_for("q", ["90 days.", "It is 90 days.", "90 days"])]
    stale = find_stale_labels(results, ["30 days."], LX)
    assert len(stale) == 1
    assert stale[0]["labeled_answer"] == "30 days."
    assert "90" in stale[0]["current_consensus"]


def test_a_reworded_but_equivalent_answer_is_not_stale():
    # Same meaning, different phrasing: the label still applies.
    results = [result_for("q", ["It is 30 days.", "30 days, yes.", "It is 30 days."])]
    assert find_stale_labels(results, ["30 days."], LX) == []


def test_rows_without_provenance_are_skipped():
    results = [result_for("q", ["anything.", "anything"])]
    assert find_stale_labels(results, [None], LX) == []


# ===========================================================================
# doctor wiring
# ===========================================================================


def make_row(i, samples, label, **kwargs):
    return DatasetRow(
        prompt=f"q{i}", samples=[Sample(text=t) for t in samples], label=label, **kwargs
    )


def base_rows():
    rows = []
    rng = random.Random(0)
    for i in range(12):
        rows.append(make_row(i, [f"{i} units."] * 3 + [f"It is {i} units."], 0))
    for i in range(12, 24):
        rows.append(make_row(i, [f"{rng.randint(1, 9999)} units." for _ in range(4)], 1))
    return rows


def test_doctor_skips_reference_checks_when_none_exist():
    report = preflight(entailment=PRODUCTION, dev_set=base_rows())
    assert check_named(report, "labels vs references") is None
    assert check_named(report, "label staleness") is None


def test_doctor_passes_agreeing_references():
    # The cross-check leans on the entailment oracle, so it needs one that can
    # actually judge the pairs (the always-neutral stub abstains on everything).
    rows = base_rows()
    rows[0].reference = "0 units"
    report = preflight(entailment=LX, require_production_backend=False, dev_set=rows)
    check = check_named(report, "labels vs references")
    assert check.status == PASS


def test_an_abstaining_oracle_cannot_flag_anything():
    """With an oracle that answers neutral to every pair, the cross-check
    abstains wholesale - a PASS by silence, never a WARN by guesswork."""
    rows = base_rows()
    rows[0].reference = "999 units"  # would be a mismatch under a real oracle
    report = preflight(entailment=PRODUCTION, dev_set=rows)
    assert check_named(report, "labels vs references").status == PASS


def test_doctor_warns_on_a_label_its_reference_contradicts():
    rows = base_rows()
    rows[0].reference = "999 units"  # model consistently says 0; label asserts correct
    report = preflight(entailment=LX, require_production_backend=False, dev_set=rows)
    check = check_named(report, "labels vs references")
    assert check.status == WARN
    # The wording keeps all three explanations alive - label, reference, or oracle:
    assert "the oracle misread the pair" in check.remedy
    assert "AUROC" in check.remedy


def test_doctor_passes_fresh_provenance():
    rows = base_rows()
    rows[0].labeled_answer = "0 units."
    report = preflight(entailment=LX, require_production_backend=False, dev_set=rows)
    assert check_named(report, "label staleness").status == PASS


def test_doctor_warns_on_stale_provenance():
    rows = base_rows()
    rows[0].labeled_answer = "an answer the model never gives now"
    report = preflight(entailment=LX, require_production_backend=False, dev_set=rows)
    check = check_named(report, "label staleness")
    assert check.status == WARN
    assert "no longer gives" in check.detail
    assert "sem-gate label --relabel" in check.remedy


# ===========================================================================
# Gate.relabel_sample: protocol, not trust
# ===========================================================================


def serve(gate, n=30):
    confident = ["Paris.", "It is Paris.", "Paris, France.", "In Paris."]
    for i in range(n):
        gate.check_samples(f"prompt-{i}", confident)


def test_relabel_sample_is_deterministic():
    gate = Gate(None, threshold=0.55, entailment=LX)
    serve(gate)
    first = gate.relabel_sample(10, seed=42)
    second = gate.relabel_sample(10, seed=42)
    assert first == second
    assert len(first) == 10


def test_different_seeds_draw_different_samples():
    gate = Gate(None, threshold=0.55, entailment=LX)
    serve(gate)
    a = {r["prompt"] for r in gate.relabel_sample(10, seed=1)}
    b = {r["prompt"] for r in gate.relabel_sample(10, seed=2)}
    assert a != b  # 10-of-30 twice colliding exactly is ~impossible


def test_relabel_rows_are_label_ready(tmp_path):
    """The sample round-trips straight into `sem-gate label`."""
    gate = Gate(None, threshold=0.55, entailment=LX)
    serve(gate, 25)
    rows = gate.relabel_sample(5)
    path = str(tmp_path / "relabel.jsonl")
    write_jsonl(path, rows)
    loaded = load_dataset(path)
    assert len(loaded) == 5
    assert all(r.has_samples for r in loaded)
    assert all(r.labeled_answer for r in loaded)  # provenance travels from birth


def test_relabel_sample_excludes_broken_measurements():
    gate = Gate(None, threshold=0.55, entailment=LX)
    serve(gate, 22)
    gate.check_samples("degenerate", ["same", "same", "same"])  # unreliable
    rows = gate.relabel_sample(100)
    assert len(rows) == 22
    assert all(r["prompt"] != "degenerate" for r in rows)


def test_small_history_returns_everything():
    gate = Gate(None, threshold=0.55, entailment=LX)
    serve(gate, 3)
    assert len(gate.relabel_sample(30)) == 3


# ===========================================================================
# label --auto end to end
# ===========================================================================


def test_auto_labelling_derives_without_a_human(tmp_path, monkeypatch):
    from semantic_entropy_gate.cli import main

    src = str(tmp_path / "p.jsonl")
    write_jsonl(
        src,
        [
            {
                "prompt": "capital?",
                "samples": ["Paris.", "It is Paris.", "Paris"],
                "reference": "Paris",
            },
            {
                "prompt": "window?",
                "samples": ["60 days.", "It is 60 days.", "60 days"],
                "reference": "30 days",
            },
        ],
    )
    out = str(tmp_path / "dev.jsonl")

    def no_input(_prompt=""):  # pragma: no cover - reaching it is the failure
        raise AssertionError("auto mode must not ask a human for decidable rows")

    monkeypatch.setattr("builtins.input", no_input)
    assert (
        main(
            ["label", "--input", src, "--out", out, "--entailment", "lexical", "--auto", "--quiet"]
        )
        == 0
    )

    rows = load_dataset(out)
    assert [r.label for r in rows] == [0, 1]
    assert all(r.labeled_answer for r in rows)
    assert all("label_derivation" in r.extra for r in rows)  # the audit trail


def test_auto_falls_back_to_the_human_when_the_oracle_abstains(tmp_path, monkeypatch):
    from semantic_entropy_gate.cli import main

    src = str(tmp_path / "p.jsonl")
    write_jsonl(
        src,
        [
            {
                "prompt": "q",
                "samples": ["The powerhouse.", "The powerhouse!", "The powerhouse"],
                "reference": "Paris",  # lexically unrelatable -> abstention
            }
        ],
    )
    out = str(tmp_path / "dev.jsonl")
    answers = iter(["n"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert (
        main(
            ["label", "--input", src, "--out", out, "--entailment", "lexical", "--auto", "--quiet"]
        )
        == 0
    )
    rows = load_dataset(out)
    assert rows[0].label == 1  # the human's verdict, not a laundered guess


def test_derived_labels_flow_into_doctor(tmp_path):
    """The full loop: references -> derived labels -> validated separation."""
    from semantic_entropy_gate.cli import main

    rng = random.Random(1)
    src_rows = []
    for i in range(15):
        src_rows.append(
            {
                "prompt": f"known-{i}",
                "samples": [f"{i} units."] * 3 + [f"It is {i} units."],
                "reference": f"{i} units",
            }
        )
    for i in range(15):
        wrong = [f"{rng.randint(100, 9999)} units." for _ in range(4)]
        src_rows.append({"prompt": f"guess-{i}", "samples": wrong, "reference": "7 units"})

    src = str(tmp_path / "p.jsonl")
    out = str(tmp_path / "dev.jsonl")
    write_jsonl(src, src_rows)
    assert (
        main(
            ["label", "--input", src, "--out", out, "--entailment", "lexical", "--auto", "--quiet"]
        )
        == 0
    )

    rows = load_dataset(out)
    report = preflight(entailment=PRODUCTION, dev_set=rows)
    separation = check_named(report, "task separation")
    assert separation.status in (PASS, WARN)
    assert "AUROC" in separation.detail
    assert check_named(report, "labels vs references").status == PASS
    assert check_named(report, "label staleness").status == PASS
