"""Reports: machine-readable JSON + a human-readable markdown audit trail.

The markdown report is written for the person who has to *decide something* —
whether the gate is trustworthy, which prompts are risky, where the threshold
should sit. It always shows the evidence (the semantic clusters, the
disagreements, the backend that judged them), never just the score.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .types import CalibrationResult, EntropyResult, GateDecision

__all__ = ["Report", "build_report", "write_json", "write_markdown"]

_SCHEMA_VERSION = 1


@dataclass
class Report:
    """A batch of scored prompts, plus optional calibration and gate decisions."""

    results: List[EntropyResult] = field(default_factory=list)
    calibration: Optional[CalibrationResult] = None
    decisions: List[GateDecision] = field(default_factory=list)
    threshold: Optional[float] = None
    title: str = "Semantic Entropy Report"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- summaries

    def summary(self) -> Dict[str, Any]:
        n = len(self.results)
        if n == 0:
            return {"n_prompts": 0}
        entropies = [r.normalized_entropy for r in self.results]
        threshold = self.threshold if self.threshold is not None else _calibrated(self.calibration)
        flagged = (
            [r for r in self.results if r.is_confabulation(threshold)]
            if threshold is not None
            else []
        )
        ordered = sorted(entropies)
        return {
            "n_prompts": n,
            "mean_normalized_entropy": sum(entropies) / n,
            "median_normalized_entropy": ordered[n // 2],
            "min_normalized_entropy": ordered[0],
            "max_normalized_entropy": ordered[-1],
            "mean_clusters": sum(r.n_clusters for r in self.results) / n,
            "unanimous_prompts": sum(1 for r in self.results if r.n_clusters == 1),
            "threshold": threshold,
            "flagged": len(flagged),
            "flag_rate": (len(flagged) / n) if threshold is not None else None,
            "entailment_backends": sorted({r.entailment_backend for r in self.results}),
            "estimators": sorted({r.estimator.value for r in self.results}),
            "total_samples": sum(r.n_samples for r in self.results),
            "total_entailment_calls": sum(
                int(r.metadata.get("entailment_calls", 0)) for r in self.results
            ),
        }

    def riskiest(self, k: int = 10) -> List[EntropyResult]:
        return sorted(self.results, key=lambda r: -r.normalized_entropy)[:k]

    # ------------------------------------------------------------ serialising

    def to_dict(self, *, include_judgements: bool = False) -> Dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "title": self.title,
            "summary": self.summary(),
            "threshold": self.threshold,
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "results": [r.to_dict(include_judgements=include_judgements) for r in self.results],
            "decisions": [d.to_dict() for d in self.decisions],
            "metadata": dict(self.metadata),
        }

    def to_json(self, path: Optional[str] = None, *, include_judgements: bool = False) -> str:
        payload = json.dumps(self.to_dict(include_judgements=include_judgements), indent=2)
        if path:
            _write(path, payload)
        return payload

    def to_markdown(self, path: Optional[str] = None, *, top_k: int = 10) -> str:
        text = _render_markdown(self, top_k=top_k)
        if path:
            _write(path, text)
        return text

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Report":
        calibration = None
        raw_cal = data.get("calibration")
        if raw_cal:
            from .types import ThresholdPoint

            calibration = CalibrationResult(
                threshold=raw_cal["threshold"],
                auroc=raw_cal["auroc"],
                auprc=raw_cal.get("auprc", 0.0),
                criterion=raw_cal.get("criterion", "youden"),
                n_samples=raw_cal.get("n_samples", 0),
                n_positive=raw_cal.get("n_positive", 0),
                n_negative=raw_cal.get("n_negative", 0),
                operating_point=ThresholdPoint(**raw_cal["operating_point"]),
                curve=[ThresholdPoint(**p) for p in raw_cal.get("curve", [])],
                base_rate=raw_cal.get("base_rate", 0.0),
                metadata=raw_cal.get("metadata", {}),
            )
        return cls(
            results=[EntropyResult.from_dict(r) for r in data.get("results", [])],
            calibration=calibration,
            threshold=data.get("threshold"),
            title=data.get("title", "Semantic Entropy Report"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def load(cls, path: str) -> "Report":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def build_report(
    results: Sequence[EntropyResult],
    *,
    calibration: Optional[CalibrationResult] = None,
    threshold: Optional[float] = None,
    decisions: Sequence[GateDecision] = (),
    title: str = "Semantic Entropy Report",
    metadata: Optional[Dict[str, Any]] = None,
) -> Report:
    """Assemble a :class:`Report` from scored results."""
    return Report(
        results=list(results),
        calibration=calibration,
        decisions=list(decisions),
        threshold=threshold if threshold is not None else _calibrated(calibration),
        title=title,
        metadata=dict(metadata or {}),
    )


def write_json(report: Report, path: str, *, include_judgements: bool = False) -> str:
    report.to_json(path, include_judgements=include_judgements)
    return path


def write_markdown(report: Report, path: str, *, top_k: int = 10) -> str:
    report.to_markdown(path, top_k=top_k)
    return path


# ---------------------------------------------------------------- rendering


def _render_markdown(report: Report, *, top_k: int) -> str:
    summary = report.summary()
    lines: List[str] = [f"# {report.title}", ""]

    if summary.get("n_prompts", 0) == 0:
        lines.append("_No prompts scored._")
        return "\n".join(lines) + "\n"

    threshold = summary.get("threshold")
    lines += [
        "Semantic entropy measures how many *distinct meanings* a model produces when",
        "asked the same question several times. Low entropy means the model committed to",
        "one answer; high entropy means it was guessing (confabulating).",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Prompts scored | {summary['n_prompts']} |",
        f"| Total generations | {summary['total_samples']} |",
        f"| Entailment calls | {summary['total_entailment_calls']} |",
        f"| Mean normalized entropy | {summary['mean_normalized_entropy']:.4f} |",
        f"| Median normalized entropy | {summary['median_normalized_entropy']:.4f} |",
        f"| Range | {summary['min_normalized_entropy']:.4f} – {summary['max_normalized_entropy']:.4f} |",
        f"| Mean semantic clusters | {summary['mean_clusters']:.2f} |",
        f"| Unanimous prompts (1 cluster) | {summary['unanimous_prompts']} |",
        f"| Entailment backend(s) | {', '.join(summary['entailment_backends']) or 'n/a'} |",
        f"| Estimator(s) | {', '.join(summary['estimators'])} |",
    ]
    if threshold is not None:
        lines.append(f"| Threshold | {threshold:.4f} |")
        lines.append(
            f"| Flagged as confabulation | {summary['flagged']} ({summary['flag_rate']:.1%}) |"
        )
    lines.append("")

    if report.calibration:
        cal = report.calibration
        op = cal.operating_point
        lines += [
            "## Calibration",
            "",
            f"Fitted on **{cal.n_samples}** labelled prompts "
            f"({cal.n_positive} hallucinated, {cal.n_negative} correct; "
            f"base rate {cal.base_rate:.1%}).",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| AUROC | **{cal.auroc:.4f}** |",
            f"| AUPRC | {cal.auprc:.4f} |",
            f"| Criterion | `{cal.criterion}` |",
            f"| Chosen threshold | **{cal.threshold:.4f}** |",
            f"| TPR (hallucinations caught) | {op.tpr:.3f} |",
            f"| FPR (false alarms) | {op.fpr:.3f} |",
            f"| Precision | {op.precision:.3f} |",
            f"| F1 | {op.f1:.3f} |",
            f"| Accuracy | {op.accuracy:.3f} |",
            "",
            _auroc_verdict(cal.auroc),
            "",
            "### ROC curve",
            "",
            "```",
            _ascii_roc(cal),
            "```",
            "",
        ]

    lines += [
        "## Riskiest prompts",
        "",
        "| # | Norm. entropy | Clusters | Agreement | Prompt | Consensus answer |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for rank, result in enumerate(report.riskiest(top_k), start=1):
        flag = " ⚠️" if threshold is not None and result.is_confabulation(threshold) else ""
        lines.append(
            f"| {rank} | {result.normalized_entropy:.4f}{flag} | {result.n_clusters} | "
            f"{result.agreement:.0%} | {_cell(result.prompt)} | {_cell(result.consensus_answer or '')} |"
        )
    lines.append("")

    lines += [
        "## Evidence",
        "",
        "Each block below is the full audit trail for one prompt: the semantic clusters the",
        "model produced, their probability mass, and every generation inside them.",
        "",
    ]
    for result in report.riskiest(top_k):
        lines += _render_result(result, threshold)

    if report.decisions:
        lines += [
            "## Gate decisions",
            "",
            "| Action | Score | Executed | Reason |",
            "| --- | --- | --- | --- |",
        ]
        for decision in report.decisions:
            lines.append(
                f"| `{decision.action.value}` | {decision.score:.4f} | "
                f"{'yes' if decision.executed else 'no'} | {_cell(decision.reason)} |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "Generated by [semantic-entropy-gate](https://github.com/krishddd/semantic-entropy-gate) —",
        "semantic entropy for hallucination detection (Farquhar et al., *Nature* 2024).",
        "",
    ]
    return "\n".join(lines)


def _render_result(result: EntropyResult, threshold: Optional[float]) -> List[str]:
    verdict = ""
    if threshold is not None:
        verdict = " — **flagged**" if result.is_confabulation(threshold) else " — within budget"
    lines = [
        f"### `{_cell(result.prompt)}`{verdict}",
        "",
        f"- semantic entropy **{result.entropy:.4f} nats** "
        f"(normalized **{result.normalized_entropy:.4f}**, max {result.max_entropy:.4f})",
        f"- naive string entropy {result.naive_entropy:.4f} nats "
        f"→ lexical-only component {result.lexical_entropy:.4f} "
        "(the part a token-level detector would have mistaken for hallucination)",
        f"- estimator `{result.estimator.value}`, entailment `{result.entailment_backend}`, "
        f"{result.n_samples} samples → {result.n_clusters} meanings",
        "",
        "| Cluster | p | n | Representative |",
        "| --- | --- | --- | --- |",
    ]
    for cid, size, prob, rep in result.cluster_table():
        lines.append(f"| {cid} | {prob:.3f} | {size} | {_cell(rep)} |")
    lines.append("")
    if result.n_clusters > 1:
        lines.append("<details><summary>All generations by cluster</summary>")
        lines.append("")
        for semantic_cluster in sorted(result.clusters, key=lambda c: -c.probability):
            lines.append(
                f"**Cluster {semantic_cluster.id}** (p={semantic_cluster.probability:.3f})"
            )
            lines.append("")
            for member in semantic_cluster.members:
                lines.append(f"- {_cell(member)}")
            lines.append("")
        lines.append("</details>")
        lines.append("")
    return lines


def _auroc_verdict(value: float) -> str:
    if value >= 0.8:
        return (
            f"> **AUROC {value:.3f} — strong.** Semantic entropy separates hallucinations from "
            "correct answers well on this task; the threshold above is safe to deploy."
        )
    if value >= 0.65:
        return (
            f"> **AUROC {value:.3f} — moderate.** Usable as a soft signal (warn / defer), but "
            "do not hard-block on it without a larger dev set."
        )
    return (
        f"> **AUROC {value:.3f} — weak.** Semantic entropy carries little signal for this task. "
        "Do not deploy this gate: check that sampling temperature is non-zero, that the "
        "entailment backend is a real NLI model, and that the labels mean what you think."
    )


def _ascii_roc(calibration: CalibrationResult, width: int = 46, height: int = 16) -> str:
    """Terminal-friendly ROC plot, so the curve survives a plain-text pipeline."""
    grid = [[" "] * (width + 1) for _ in range(height + 1)]
    for row in range(height + 1):
        grid[row][0] = "|"
    for col in range(width + 1):
        grid[height][col] = "-"
    grid[height][0] = "+"
    for i in range(min(width, height) + 1):  # chance diagonal
        col = int(round(i * width / max(1, min(width, height))))
        row = height - int(round(i * height / max(1, min(width, height))))
        if 0 <= row <= height and 0 <= col <= width and grid[row][col] == " ":
            grid[row][col] = "."
    for point in calibration.curve:
        col = int(round(point.fpr * width))
        row = height - int(round(point.tpr * height))
        col = max(0, min(width, col))
        row = max(0, min(height, row))
        grid[row][col] = "*"
    chosen = calibration.operating_point
    col = max(0, min(width, int(round(chosen.fpr * width))))
    row = max(0, min(height, height - int(round(chosen.tpr * height))))
    grid[row][col] = "O"
    body = "\n".join("".join(row) for row in grid)
    return (
        "TPR 1.0\n"
        + body
        + "\n    0.0"
        + " " * (width - 12)
        + "FPR 1.0\n"
        + f"  * = operating point   O = chosen ({chosen.fpr:.2f}, {chosen.tpr:.2f})   . = chance"
    )


def _cell(text: str, limit: int = 90) -> str:
    """Markdown-table-safe one-liner."""
    flat = " ".join(str(text).split())
    flat = flat.replace("|", "\\|")
    if len(flat) > limit:
        flat = flat[: limit - 3] + "..."
    return flat


def _calibrated(calibration: Optional[CalibrationResult]) -> Optional[float]:
    return calibration.threshold if calibration else None


def _write(path: str, text: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
