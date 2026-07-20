"""The CLI, end to end, entirely offline."""

import json

import pytest

from semantic_entropy_gate.cli import _load_entrypoint, main
from semantic_entropy_gate.dataset import write_jsonl
from semantic_entropy_gate.errors import SemanticEntropyError

DEMO_ROWS = [
    {
        "id": "eiffel",
        "prompt": "In which city is the Eiffel Tower located?",
        "samples": ["Paris.", "It is in Paris.", "Paris, France.", "In Paris."],
        "label": 0,
    },
    {
        "id": "astatine",
        "prompt": "What is the boiling point of astatine?",
        "samples": ["610 Kelvin.", "503 Kelvin.", "337 Kelvin.", "575 Kelvin."],
        "label": 1,
    },
    {
        "id": "austen",
        "prompt": "Who wrote Pride and Prejudice?",
        "samples": ["Jane Austen.", "It was Jane Austen.", "Jane Austen wrote it.", "Austen."],
        "label": 0,
    },
    {
        "id": "zorbex",
        "prompt": "In what year was the Zorbex Protocol ratified?",
        "samples": ["1998.", "2004.", "1976.", "2011."],
        "label": 1,
    },
]


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "dev.jsonl"
    write_jsonl(str(path), DEMO_ROWS)
    return str(path)


# ---------------------------------------------------------------- entrypoints


def test_entrypoint_loader_imports_a_function():
    fn = _load_entrypoint("semantic_entropy_gate.entropy:shannon_entropy")
    assert fn([1.0]) == 0.0


def test_entrypoint_loader_rejects_a_malformed_spec():
    with pytest.raises(SemanticEntropyError, match="invalid entrypoint"):
        _load_entrypoint("no_colon_here")


def test_entrypoint_loader_reports_a_missing_module():
    with pytest.raises(SemanticEntropyError, match="cannot import"):
        _load_entrypoint("definitely_not_a_module:fn")


def test_entrypoint_loader_reports_a_missing_attribute():
    with pytest.raises(SemanticEntropyError, match="no attribute"):
        _load_entrypoint("semantic_entropy_gate.entropy:not_a_function")


# ---------------------------------------------------------------------- score


def test_score_writes_both_reports(dataset, tmp_path, capsys):
    out = str(tmp_path / "report")
    assert (
        main(["score", "--input", dataset, "--entailment", "lexical", "--out", out, "--quiet"]) == 0
    )

    data = json.loads(open(f"{out}.json", encoding="utf-8").read())
    assert data["summary"]["n_prompts"] == 4
    assert data["calibration"]["auroc"] == pytest.approx(1.0)

    markdown = open(f"{out}.md", encoding="utf-8").read()
    assert "## Calibration" in markdown

    captured = capsys.readouterr().out
    assert "scored 4 prompts" in captured
    assert "AUROC" in captured


def test_score_auto_calibrates_from_labels(dataset, tmp_path):
    out = str(tmp_path / "r")
    main(["score", "--input", dataset, "--entailment", "lexical", "--out", out, "--quiet"])
    data = json.loads(open(f"{out}.json", encoding="utf-8").read())
    assert data["threshold"] == pytest.approx(data["calibration"]["threshold"])


def test_score_can_skip_calibration(dataset, tmp_path):
    out = str(tmp_path / "r")
    main(
        [
            "score",
            "--input",
            dataset,
            "--entailment",
            "lexical",
            "--out",
            out,
            "--quiet",
            "--no-calibrate",
        ]
    )
    assert json.loads(open(f"{out}.json", encoding="utf-8").read())["calibration"] is None


def test_score_threshold_override_wins(dataset, tmp_path):
    out = str(tmp_path / "r")
    main(
        [
            "score",
            "--input",
            dataset,
            "--entailment",
            "lexical",
            "--out",
            out,
            "--quiet",
            "--threshold",
            "0.9",
        ]
    )
    assert json.loads(open(f"{out}.json", encoding="utf-8").read())["threshold"] == 0.9


def test_score_uses_a_sampler_for_prompt_only_rows(tmp_path):
    path = tmp_path / "prompts.jsonl"
    write_jsonl(str(path), [{"prompt": "q1"}, {"prompt": "q2"}])
    out = str(tmp_path / "r")
    code = main(
        [
            "score",
            "--input",
            str(path),
            "--entailment",
            "lexical",
            "--sampler",
            "tests.test_cli:stub_sampler",
            "--n-samples",
            "4",
            "--out",
            out,
            "--quiet",
        ]
    )
    assert code == 0
    data = json.loads(open(f"{out}.json", encoding="utf-8").read())
    assert data["results"][0]["n_samples"] == 4


def stub_sampler(prompt, n):
    """Deterministic offline sampler used by the CLI test above."""
    return [f"answer {i}" for i in range(n)]


def test_score_without_samples_or_sampler_fails(tmp_path, capsys):
    path = tmp_path / "prompts.jsonl"
    write_jsonl(str(path), [{"prompt": "q1"}])
    assert (
        main(
            ["score", "--input", str(path), "--entailment", "lexical", "--out", str(tmp_path / "r")]
        )
        == 1
    )
    assert "no 'samples'" in capsys.readouterr().err


def test_score_includes_judgements_on_request(dataset, tmp_path):
    out = str(tmp_path / "r")
    main(
        [
            "score",
            "--input",
            dataset,
            "--entailment",
            "lexical",
            "--out",
            out,
            "--quiet",
            "--include-judgements",
        ]
    )
    data = json.loads(open(f"{out}.json", encoding="utf-8").read())
    assert data["results"][1]["judgements"]


def test_score_progress_goes_to_stderr(dataset, tmp_path, capsys):
    main(["score", "--input", dataset, "--entailment", "lexical", "--out", str(tmp_path / "r")])
    assert "[1/4]" in capsys.readouterr().err


# ------------------------------------------------------------------ calibrate


def test_calibrate_from_a_labelled_dataset(dataset, tmp_path, capsys):
    out = str(tmp_path / "cal.json")
    assert (
        main(["calibrate", "--input", dataset, "--entailment", "lexical", "--out", out, "--quiet"])
        == 0
    )
    data = json.loads(open(out, encoding="utf-8").read())
    assert data["auroc"] == pytest.approx(1.0)
    assert data["criterion"] == "youden"
    assert "SEMANTIC ENTROPY CALIBRATION" in capsys.readouterr().out


def test_calibrate_from_a_scored_report(dataset, tmp_path):
    report_out = str(tmp_path / "report")
    main(["score", "--input", dataset, "--entailment", "lexical", "--out", report_out, "--quiet"])
    cal_out = str(tmp_path / "cal.json")
    assert main(["calibrate", "--input", f"{report_out}.json", "--out", cal_out]) == 0
    assert json.loads(open(cal_out, encoding="utf-8").read())["n_samples"] == 4


def test_calibrate_respects_the_criterion(dataset, tmp_path):
    out = str(tmp_path / "cal.json")
    main(
        [
            "calibrate",
            "--input",
            dataset,
            "--entailment",
            "lexical",
            "--out",
            out,
            "--criterion",
            "target_fpr",
            "--target-fpr",
            "0.0",
            "--quiet",
        ]
    )
    data = json.loads(open(out, encoding="utf-8").read())
    assert data["criterion"] == "target_fpr"
    assert data["operating_point"]["fpr"] == 0.0


def test_calibrate_requires_labels(tmp_path, capsys):
    path = tmp_path / "unlabelled.jsonl"
    write_jsonl(str(path), [{"prompt": "q", "samples": ["a", "b"]}])
    assert (
        main(
            [
                "calibrate",
                "--input",
                str(path),
                "--entailment",
                "lexical",
                "--out",
                str(tmp_path / "c.json"),
            ]
        )
        == 1
    )
    assert "needs a 'label'" in capsys.readouterr().err


# -------------------------------------------------------------- report/explain


def test_report_rerenders_markdown(dataset, tmp_path):
    out = str(tmp_path / "report")
    main(["score", "--input", dataset, "--entailment", "lexical", "--out", out, "--quiet"])
    md = str(tmp_path / "again.md")
    assert main(["report", "--json", f"{out}.json", "--out", md]) == 0
    assert "# Semantic Entropy Report" in open(md, encoding="utf-8").read()


def test_explain_prints_one_result(dataset, tmp_path, capsys):
    out = str(tmp_path / "report")
    main(["score", "--input", dataset, "--entailment", "lexical", "--out", out, "--quiet"])
    capsys.readouterr()
    assert main(["explain", "--json", f"{out}.json", "--index", "1"]) == 0
    printed = capsys.readouterr().out
    assert "SEMANTIC ENTROPY REPORT" in printed
    assert "SEMANTIC CLUSTERS" in printed


def test_explain_rejects_an_out_of_range_index(dataset, tmp_path, capsys):
    out = str(tmp_path / "report")
    main(["score", "--input", dataset, "--entailment", "lexical", "--out", out, "--quiet"])
    assert main(["explain", "--json", f"{out}.json", "--index", "99"]) == 1
    assert "out of range" in capsys.readouterr().err


# ------------------------------------------------------------------ gate/demo


def test_gate_allows_a_confident_prompt(capsys):
    code = main(
        [
            "gate",
            "Where is the Eiffel Tower?",
            "--entailment",
            "lexical",
            "--samples",
            "Paris.",
            "--samples",
            "It is in Paris.",
            "--samples",
            "Paris, France.",
        ]
    )
    assert code == 0
    assert "GATE DECISION: ALLOW" in capsys.readouterr().out


def test_gate_exits_nonzero_when_it_would_not_allow(capsys):
    code = main(
        [
            "gate",
            "Boiling point?",
            "--entailment",
            "lexical",
            "--threshold",
            "0.3",
            "--samples",
            "610 K",
            "--samples",
            "337 K",
            "--samples",
            "503 K",
        ]
    )
    assert code == 2
    assert "GATE DECISION: DEFER" in capsys.readouterr().out


def test_gate_without_samples_or_sampler_fails(capsys):
    assert main(["gate", "q", "--entailment", "lexical"]) == 1
    assert "--sampler" in capsys.readouterr().err


def test_demo_runs_offline(capsys):
    assert main(["demo"]) == 0
    printed = capsys.readouterr().out
    assert "SEMANTIC ENTROPY GATE - DEMO" in printed
    assert "SEMANTIC ENTROPY CALIBRATION" in printed
    assert "GATE DECISION" in printed


def test_demo_can_write_reports(tmp_path):
    from semantic_entropy_gate.cli import DEMO_ROWS

    out = str(tmp_path / "demo")
    assert main(["demo", "--out", out]) == 0
    data = json.loads(open(f"{out}.json", encoding="utf-8").read())
    assert data["summary"]["n_prompts"] == len(DEMO_ROWS)
    assert data["calibration"]["auroc"] == pytest.approx(1.0)
    assert open(f"{out}.md", encoding="utf-8").read().startswith("#")


# ----------------------------------------------------------------- misc/errors


def test_missing_input_file_is_reported(capsys, tmp_path):
    assert (
        main(["score", "--input", str(tmp_path / "nope.jsonl"), "--out", str(tmp_path / "r")]) == 1
    )
    assert "file not found" in capsys.readouterr().err


def test_judge_entailment_requires_a_judge(dataset, tmp_path, capsys):
    assert (
        main(["score", "--input", dataset, "--entailment", "judge", "--out", str(tmp_path / "r")])
        == 1
    )
    assert "--judge" in capsys.readouterr().err


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "semantic-entropy-gate" in capsys.readouterr().out


def test_no_subcommand_exits_with_usage(capsys):
    with pytest.raises(SystemExit):
        main([])
