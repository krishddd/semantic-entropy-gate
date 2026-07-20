"""JSONL dataset parsing — errors must name the line, never truncate silently."""

import pytest

from semantic_entropy_gate.dataset import load_dataset, read_jsonl, write_jsonl
from semantic_entropy_gate.errors import SemanticEntropyError


def write(tmp_path, text, name="data.jsonl"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_round_trip(tmp_path):
    rows = [{"prompt": "q", "samples": ["a", "b"], "label": 1}]
    path = write_jsonl(str(tmp_path / "d.jsonl"), rows)
    loaded = load_dataset(path)
    assert loaded[0].prompt == "q"
    assert [s.text for s in loaded[0].samples] == ["a", "b"]
    assert loaded[0].label == 1


def test_blank_lines_are_skipped(tmp_path):
    path = write(tmp_path, '{"prompt": "a"}\n\n\n{"prompt": "b"}\n')
    assert len(load_dataset(path)) == 2


def test_invalid_json_names_the_line(tmp_path):
    path = write(tmp_path, '{"prompt": "ok"}\n{not json}\n')
    with pytest.raises(SemanticEntropyError, match=r":2: invalid JSON"):
        list(read_jsonl(path))


def test_missing_prompt_field_is_reported(tmp_path):
    path = write(tmp_path, '{"question": "oops"}\n')
    with pytest.raises(SemanticEntropyError, match="has no 'prompt'"):
        load_dataset(path)


def test_logprob_length_mismatch_is_rejected(tmp_path):
    path = write(tmp_path, '{"prompt": "q", "samples": ["a", "b"], "logprobs": [-0.1]}\n')
    with pytest.raises(SemanticEntropyError, match="2 samples but 1 logprobs"):
        load_dataset(path)


def test_logprobs_are_attached(tmp_path):
    path = write(tmp_path, '{"prompt": "q", "samples": ["a", "b"], "logprobs": [-0.1, -0.2]}\n')
    rows = load_dataset(path)
    assert [s.logprob for s in rows[0].samples] == [-0.1, -0.2]


def test_a_string_samples_field_is_wrapped(tmp_path):
    path = write(tmp_path, '{"prompt": "q", "samples": "only one"}\n')
    assert [s.text for s in load_dataset(path)[0].samples] == ["only one"]


def test_custom_field_names(tmp_path):
    path = write(tmp_path, '{"question": "q", "answers": ["a"], "is_wrong": 1}\n')
    rows = load_dataset(path, prompt_key="question", samples_key="answers", label_key="is_wrong")
    assert rows[0].prompt == "q"
    assert rows[0].label == 1


def test_unknown_fields_are_preserved_in_extra(tmp_path):
    path = write(tmp_path, '{"prompt": "q", "domain": "medical", "difficulty": 3}\n')
    assert load_dataset(path)[0].extra == {"domain": "medical", "difficulty": 3}


def test_labels_are_coerced_to_zero_or_one(tmp_path):
    path = write(tmp_path, '{"prompt": "a", "label": true}\n{"prompt": "b", "label": 0}\n')
    assert [r.label for r in load_dataset(path)] == [1, 0]


def test_missing_label_stays_none(tmp_path):
    path = write(tmp_path, '{"prompt": "q"}\n')
    assert load_dataset(path)[0].label is None


def test_empty_file_is_rejected(tmp_path):
    with pytest.raises(SemanticEntropyError, match="no rows"):
        load_dataset(write(tmp_path, "\n\n"))


def test_row_serialisation_omits_absent_fields(tmp_path):
    path = write(tmp_path, '{"prompt": "q"}\n')
    assert load_dataset(path)[0].to_dict() == {"prompt": "q"}


def test_row_serialisation_includes_logprobs_when_present(tmp_path):
    path = write(
        tmp_path, '{"id": "x", "prompt": "q", "samples": ["a"], "logprobs": [-1.0], "label": 1}\n'
    )
    data = load_dataset(path)[0].to_dict()
    assert data == {"id": "x", "prompt": "q", "samples": ["a"], "logprobs": [-1.0], "label": 1}


def test_has_samples_flag(tmp_path):
    path = write(tmp_path, '{"prompt": "a", "samples": ["x"]}\n{"prompt": "b"}\n')
    rows = load_dataset(path)
    assert rows[0].has_samples is True
    assert rows[1].has_samples is False


def test_unicode_survives_the_round_trip(tmp_path):
    path = write_jsonl(
        str(tmp_path / "u.jsonl"), [{"prompt": "¿Dónde está París?", "samples": ["París"]}]
    )
    assert load_dataset(path)[0].prompt == "¿Dónde está París?"


def test_bundled_demo_dataset_is_valid():
    import os

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "data", "demo.jsonl")
    if not os.path.exists(path):  # pragma: no cover - sdist without data/
        pytest.skip("bundled dataset not present")
    rows = load_dataset(path)
    assert len(rows) >= 10
    assert all(row.has_samples for row in rows)
    assert {row.label for row in rows} == {0, 1}
