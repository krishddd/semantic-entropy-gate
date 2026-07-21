"""JSONL dataset I/O for batch scoring.

One JSON object per line. Two shapes are accepted, and the difference decides
whether the CLI needs a model at all:

**Pre-generated (offline, no network, no API key)** — the row already carries the
samples, so scoring is pure computation::

    {"prompt": "Who discovered penicillin?",
     "samples": ["Alexander Fleming", "Fleming discovered it", "Howard Florey"],
     "logprobs": [-0.21, -0.35, -0.98],     # optional -> Rao-Blackwell estimator
     "label": 0}                            # optional -> enables calibration

**Prompt-only** — the row is just a question and ``sem-gate score`` will call the
sampler you point it at with ``--sampler mypkg.mod:fn``::

    {"prompt": "What is the boiling point of astatine?", "label": 1}

Field names are configurable so you can point the CLI at a dataset you already
have rather than reshaping it first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

from .errors import SemanticEntropyError
from .types import Sample

__all__ = ["DatasetRow", "read_jsonl", "write_jsonl", "load_dataset"]


@dataclass
class DatasetRow:
    """One prompt from a batch-scoring dataset."""

    prompt: str
    samples: List[Sample] = field(default_factory=list)
    label: Optional[int] = None
    id: Optional[str] = None
    reference: Optional[str] = None
    """The known-correct answer, when one exists. Turns the label from an
    opinion into a *derivable* artifact: correct iff the model's consensus
    entails the reference (and vice versa) — re-derivable by anyone, auditable
    against one document instead of per-row judgement."""

    labeled_answer: Optional[str] = None
    """The consensus answer the label was judged against, recorded at label
    time. A label is a claim about a *specific answer*; if the model no longer
    gives that answer, the label is stale — an exact, checkable condition."""

    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_samples(self) -> bool:
        return bool(self.samples)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"prompt": self.prompt}
        if self.id is not None:
            data["id"] = self.id
        if self.samples:
            data["samples"] = [s.text for s in self.samples]
            if any(s.logprob is not None for s in self.samples):
                data["logprobs"] = [s.logprob for s in self.samples]
        if self.label is not None:
            data["label"] = self.label
        if self.reference is not None:
            data["reference"] = self.reference
        if self.labeled_answer is not None:
            data["labeled_answer"] = self.labeled_answer
        data.update(self.extra)
        return data


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """Yield parsed rows, reporting the offending line number on bad JSON."""
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SemanticEntropyError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def load_dataset(
    path: str,
    *,
    prompt_key: str = "prompt",
    samples_key: str = "samples",
    logprobs_key: str = "logprobs",
    label_key: str = "label",
    id_key: str = "id",
    reference_key: str = "reference",
) -> List[DatasetRow]:
    """Parse a JSONL file into :class:`DatasetRow` objects.

    Raises :class:`SemanticEntropyError` with the line number for any row missing
    the prompt field or whose ``logprobs`` length disagrees with ``samples`` —
    a silent zip-truncation there would corrupt every downstream probability.
    """
    rows: List[DatasetRow] = []
    for index, raw in enumerate(read_jsonl(path), start=1):
        if prompt_key not in raw:
            raise SemanticEntropyError(
                f"{path}: row {index} has no {prompt_key!r} field (keys: {sorted(raw)})"
            )
        texts = raw.get(samples_key) or []
        if isinstance(texts, str):
            texts = [texts]
        logprobs = raw.get(logprobs_key)
        if logprobs is not None and len(logprobs) != len(texts):
            raise SemanticEntropyError(
                f"{path}: row {index} has {len(texts)} samples but {len(logprobs)} logprobs"
            )
        samples = [
            Sample(text=str(text), logprob=None if logprobs is None else float(logprobs[i]))
            for i, text in enumerate(texts)
        ]
        label = raw.get(label_key)
        reference = raw.get(reference_key)
        labeled_answer = raw.get("labeled_answer")
        known = {
            prompt_key,
            samples_key,
            logprobs_key,
            label_key,
            id_key,
            reference_key,
            "labeled_answer",
        }
        rows.append(
            DatasetRow(
                prompt=str(raw[prompt_key]),
                samples=samples,
                label=None if label is None else int(bool(label)),
                id=raw.get(id_key),
                reference=None if reference is None else str(reference),
                labeled_answer=None if labeled_answer is None else str(labeled_answer),
                extra={k: v for k, v in raw.items() if k not in known},
            )
        )
    if not rows:
        raise SemanticEntropyError(f"{path}: no rows found")
    return rows
