"""``sem-gate`` — the batch/CLI surface.

    sem-gate demo                              # 10 seconds, no network, no API key
    sem-gate score   --input data.jsonl --out report
    sem-gate calibrate --input data.jsonl --criterion youden --out calibration.json
    sem-gate report  --json report.json --out report.md
    sem-gate explain --json report.json --index 0

The CLI never bundles a model. It either scores generations you already have (the
fully offline path) or calls a sampler you point it at with
``--sampler mypkg.module:function``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any, Callable, List, Optional, Sequence

from . import __version__
from .calibrate import calibrate
from .dataset import DatasetRow, load_dataset
from .entailment import (
    SMALL_CROSS_ENCODER,
    CachedEntailment,
    CrossEncoderEntailment,
    EntailmentModel,
    LexicalEntailment,
    LLMJudgeEntailment,
    auto_entailment,
)
from .errors import SemanticEntropyError
from .gate import DEFAULT_THRESHOLD, Gate
from .report import Report, build_report
from .sampling import from_texts, resolve_sampler
from .score import DEFAULT_N_SAMPLES, score, score_samples
from .types import CalibrationResult, EntropyResult

__all__ = ["main"]


# --------------------------------------------------------------------- helpers


def _load_entrypoint(spec: str) -> Callable[..., Any]:
    """Import ``package.module:function``."""
    if ":" not in spec:
        raise SemanticEntropyError(
            f"invalid entrypoint {spec!r}; expected 'package.module:function'"
        )
    module_name, attribute = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SemanticEntropyError(f"cannot import module {module_name!r}: {exc}") from exc
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise SemanticEntropyError(
            f"module {module_name!r} has no attribute {attribute!r}"
        ) from exc


def _build_entailment(args: argparse.Namespace) -> EntailmentModel:
    kind = getattr(args, "entailment", "auto")
    judge = None
    if getattr(args, "judge", None):
        judge = _load_entrypoint(args.judge)
    if kind == "lexical":
        return CachedEntailment(LexicalEntailment())
    if kind == "hf":
        return CachedEntailment(
            CrossEncoderEntailment(getattr(args, "nli_model", SMALL_CROSS_ENCODER))
        )
    if kind == "judge":
        if judge is None:
            raise SemanticEntropyError("--entailment judge requires --judge module:function")
        return CachedEntailment(LLMJudgeEntailment(judge))
    return auto_entailment(model_name=getattr(args, "nli_model", SMALL_CROSS_ENCODER), judge=judge)


def _score_rows(
    rows: Sequence[DatasetRow],
    args: argparse.Namespace,
    entailment: EntailmentModel,
    *,
    quiet: bool = False,
) -> List[EntropyResult]:
    sampler = _load_entrypoint(args.sampler) if getattr(args, "sampler", None) else None
    results: List[EntropyResult] = []
    for index, row in enumerate(rows, start=1):
        if row.has_samples:
            result = score_samples(
                row.prompt,
                row.samples,
                entailment=entailment,
                strict=not args.relaxed,
                metadata={"row_id": row.id, "label": row.label},
            )
        elif sampler is not None:
            result = score(
                row.prompt,
                sampler,
                n_samples=args.n_samples,
                entailment=entailment,
                strict=not args.relaxed,
                metadata={"row_id": row.id, "label": row.label},
            )
        else:
            raise SemanticEntropyError(
                f"row {index} ({row.prompt[:40]!r}) has no 'samples' and no --sampler "
                "was given; supply one or the other"
            )
        results.append(result)
        if not quiet:
            print(
                f"[{index}/{len(rows)}] H={result.normalized_entropy:.3f} "
                f"clusters={result.n_clusters:<2d} {result.prompt[:56]}",
                file=sys.stderr,
            )
    return results


def _maybe_calibrate(
    rows: Sequence[DatasetRow], results: Sequence[EntropyResult], args: argparse.Namespace
) -> Optional[CalibrationResult]:
    labels = [row.label for row in rows]
    if any(label is None for label in labels):
        return None
    if len(set(labels)) < 2:
        print(
            "note: dataset labels are single-class; skipping calibration",
            file=sys.stderr,
        )
        return None
    return calibrate(
        list(results),
        [int(label) for label in labels],  # type: ignore[arg-type]
        criterion=args.criterion,
        target_fpr=args.target_fpr,
        target_recall=args.target_recall,
    )


# -------------------------------------------------------------------- commands


def cmd_score(args: argparse.Namespace) -> int:
    rows = load_dataset(
        args.input,
        prompt_key=args.prompt_key,
        samples_key=args.samples_key,
        label_key=args.label_key,
    )
    entailment = _build_entailment(args)
    results = _score_rows(rows, args, entailment, quiet=args.quiet)

    calibration = None if args.no_calibrate else _maybe_calibrate(rows, results, args)
    threshold = args.threshold
    if threshold is None and calibration is not None:
        threshold = calibration.threshold
    if threshold is None:
        threshold = DEFAULT_THRESHOLD

    report = build_report(
        results,
        calibration=calibration,
        threshold=threshold,
        title=args.title,
        metadata={
            "input": args.input,
            "entailment": entailment.name,
            "n_samples": args.n_samples,
            "strict": not args.relaxed,
            "version": __version__,
        },
    )
    json_path = f"{args.out}.json"
    md_path = f"{args.out}.md"
    report.to_json(json_path, include_judgements=args.include_judgements)
    report.to_markdown(md_path, top_k=args.top_k)

    summary = report.summary()
    print(f"scored {summary['n_prompts']} prompts with {entailment.name}")
    print(
        f"  mean normalized entropy {summary['mean_normalized_entropy']:.4f} | "
        f"mean clusters {summary['mean_clusters']:.2f} | "
        f"unanimous {summary['unanimous_prompts']}"
    )
    if calibration is not None:
        print(
            f"  AUROC {calibration.auroc:.4f} | threshold {calibration.threshold:.4f} "
            f"({calibration.criterion})"
        )
    print(f"  flagged {summary['flagged']}/{summary['n_prompts']} at threshold {threshold:.4f}")
    print(f"wrote {json_path} and {md_path}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    if args.input.endswith(".json") and not args.input.endswith(".jsonl"):
        report = Report.load(args.input)
        results = report.results
        labels = [r.metadata.get("label") for r in results]
        if any(label is None for label in labels):
            raise SemanticEntropyError(
                f"{args.input}: results carry no 'label' metadata; re-run "
                "`sem-gate score` on a dataset that has a label field"
            )
    else:
        rows = load_dataset(args.input, label_key=args.label_key)
        if any(row.label is None for row in rows):
            raise SemanticEntropyError(f"{args.input}: every row needs a {args.label_key!r} field")
        entailment = _build_entailment(args)
        results = _score_rows(rows, args, entailment, quiet=args.quiet)
        labels = [row.label for row in rows]

    calibration = calibrate(
        results,
        [int(label) for label in labels],  # type: ignore[arg-type]
        criterion=args.criterion,
        target_fpr=args.target_fpr,
        target_recall=args.target_recall,
    )
    print(calibration.explain())
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(calibration.to_dict(), handle, indent=2)
    print(f"wrote {args.out}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    report = Report.load(args.json)
    report.to_markdown(args.out, top_k=args.top_k)
    print(f"wrote {args.out}")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    report = Report.load(args.json)
    if not report.results:
        raise SemanticEntropyError(f"{args.json}: report contains no results")
    if args.index >= len(report.results):
        raise SemanticEntropyError(
            f"index {args.index} out of range (report has {len(report.results)} results)"
        )
    result = report.results[args.index]
    print(result.explain(threshold=report.threshold))
    return 0


DEMO_ROWS = [
    {
        "prompt": "In which city is the Eiffel Tower located?",
        "samples": [
            "Paris.",
            "It is Paris.",
            "Paris, France.",
            "In Paris.",
            "Paris",
            "The city is Paris.",
        ],
        "label": 0,
    },
    {
        "prompt": "What is the boiling point of the element astatine, in Kelvin?",
        "samples": [
            "About 610 Kelvin.",
            "Roughly 503 Kelvin.",
            "It boils at approximately 337 Kelvin.",
            "Around 610 K.",
            "Approximately 575 Kelvin.",
            "It is about 400 Kelvin.",
        ],
        "label": 1,
    },
    {
        "prompt": "Who wrote the novel 'Pride and Prejudice'?",
        "samples": [
            "Jane Austen.",
            "It was written by Jane Austen.",
            "Jane Austen wrote it in 1813.",
            "Jane Austen.",
            "The author is Jane Austen.",
            "Austen, Jane.",
        ],
        "label": 0,
    },
    {
        "prompt": "How many employees did the company Arclight Dynamics have in 2019?",
        "samples": [
            "Around 250 employees.",
            "Approximately 1200 employees.",
            "It had about 40 staff.",
            "Roughly 3000 people worked there.",
            "About 250.",
            "Nearly 800 employees.",
        ],
        "label": 1,
    },
    {
        "prompt": "At what temperature does water boil at sea level, in Celsius?",
        "samples": [
            "100 degrees Celsius.",
            "It is 100 degrees Celsius.",
            "100 degrees Celsius",
            "Water boils at 100 degrees Celsius.",
            "At 100 degrees Celsius.",
            "100 degrees Celsius, at sea level.",
        ],
        "label": 0,
    },
    {
        "prompt": "In what year was the Zorbex Protocol ratified?",
        "samples": [
            "In 1998.",
            "It was ratified in 2004.",
            "1976.",
            "Around 2011.",
            "In 1998, I believe.",
            "It was 1987.",
        ],
        "label": 1,
    },
    {
        "prompt": "Who created the Python programming language?",
        "samples": [
            "Guido van Rossum.",
            "Python was created by Guido van Rossum.",
            "Guido van Rossum created it in 1991.",
            "Guido van Rossum",
            "It was Guido van Rossum.",
            "Van Rossum, Guido.",
        ],
        "label": 0,
    },
    {
        "prompt": "Who was the CEO of Helix Cartography Ltd in 2015?",
        "samples": [
            "Margaret Ellis.",
            "I think it was David Chen.",
            "Sarah Whitfield led the company.",
            "It was Margaret Ellis.",
            "Robert Nakamura was CEO.",
            "Probably Alan Pierce.",
        ],
        "label": 1,
    },
    {
        "prompt": "What is the height of Mount Everest in metres?",
        "samples": [
            "8849 metres.",
            "It is 8849 metres.",
            "About 8849 metres.",
            "8849 metres",
            "Mount Everest is 8849 metres.",
            "Roughly 8849 metres.",
        ],
        "label": 0,
    },
    {
        "prompt": "What is the capital of Australia?",
        "samples": [
            "Canberra.",
            "The capital is Canberra.",
            "Canberra, in the ACT.",
            "Canberra",
            "It is Canberra.",
            "Canberra.",
        ],
        "label": 0,
    },
]


def cmd_demo(args: argparse.Namespace) -> int:
    """Run the whole pipeline on canned data: score, calibrate, gate, report."""
    entailment = CachedEntailment(LexicalEntailment())
    results = [
        score_samples(
            row["prompt"],
            row["samples"],
            entailment=entailment,
            metadata={"label": row["label"]},
        )
        for row in DEMO_ROWS
    ]
    labels = [row["label"] for row in DEMO_ROWS]

    print("SEMANTIC ENTROPY GATE - DEMO (offline: no network, no GPU, no API key)")
    print(f"{len(results)} canned prompts, scored with the stdlib lexical backend.\n")

    # Two full traces: one prompt the model knows, one it is guessing at.
    print(results[0].explain(threshold=DEFAULT_THRESHOLD))
    print()
    print(results[1].explain(threshold=DEFAULT_THRESHOLD))
    print()

    print("=" * 78)
    print("ALL PROMPTS")
    print("=" * 78)
    print(f"{'H(norm)':>8}  {'clusters':>8}  {'truth':<13} prompt")
    for result, label in zip(results, labels):
        truth = "hallucinated" if label else "correct"
        print(
            f"{result.normalized_entropy:>8.3f}  {result.n_clusters:>8d}  "
            f"{truth:<13} {result.prompt[:44]}"
        )
    print()

    calibration = calibrate(results, labels, criterion="youden")
    print(calibration.explain())
    print()

    gate = Gate(
        from_texts(DEMO_ROWS[1]["samples"]),
        threshold=calibration.threshold,
        entailment=entailment,
    )
    decision = gate.run(DEMO_ROWS[1]["prompt"], lambda: "…executes the risky tool call…")
    print(decision.explain())
    print()
    print(f"action taken: {decision.action.value}  executed={decision.executed}")

    if args.out:
        report = build_report(
            results,
            calibration=calibration,
            title="semantic-entropy-gate demo",
            metadata={"demo": True, "version": __version__},
        )
        report.to_json(f"{args.out}.json")
        report.to_markdown(f"{args.out}.md")
        print(f"\nwrote {args.out}.json and {args.out}.md")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    """Gate a single prompt from the command line."""
    entailment = _build_entailment(args)
    n_samples = args.n_samples
    if args.samples:
        sampler = from_texts(args.samples)
        # The generations were supplied directly, so asking for more than were
        # given would (correctly) be reported as a short sampler return and mark
        # the measurement unreliable. Ask for exactly what is here.
        n_samples = len(args.samples)
    elif args.sampler:
        sampler = resolve_sampler(_load_entrypoint(args.sampler))
    else:
        raise SemanticEntropyError("provide --sampler module:function or repeated --samples")
    gate = Gate(
        sampler,
        threshold=args.threshold if args.threshold is not None else DEFAULT_THRESHOLD,
        block_threshold=args.block_threshold,
        n_samples=n_samples,
        entailment=entailment,
        strict=not args.relaxed,
    )
    decision = gate.check(args.prompt)
    print(decision.explain())
    return 0 if decision.allowed else 2


# ---------------------------------------------------------------------- parser


def _add_scoring_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--entailment",
        choices=["auto", "lexical", "hf", "judge"],
        default="auto",
        help="equivalence oracle: auto (best available), lexical (stdlib), "
        "hf (NLI cross-encoder), judge (LLM)",
    )
    parser.add_argument("--nli-model", default=SMALL_CROSS_ENCODER, help="HF NLI checkpoint")
    parser.add_argument("--judge", help="LLM judge entrypoint, module:function")
    parser.add_argument("--sampler", help="sampler entrypoint, module:function")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument(
        "--relaxed",
        action="store_true",
        help="relaxed clustering (no contradiction and not mutually neutral)",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress per-row progress")


def _add_calibration_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--criterion",
        choices=["youden", "f1", "accuracy", "target_fpr", "target_recall"],
        default="youden",
    )
    parser.add_argument("--target-fpr", type=float, default=0.1)
    parser.add_argument("--target-recall", type=float, default=0.8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sem-gate",
        description="Semantic entropy for hallucination detection (Farquhar et al., Nature 2024).",
    )
    parser.add_argument(
        "--version", action="version", version=f"semantic-entropy-gate {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score", help="batch-score a JSONL dataset")
    p_score.add_argument("--input", required=True, help="JSONL with prompt/samples[/label]")
    p_score.add_argument("--out", default="report", help="output prefix (writes .json and .md)")
    p_score.add_argument("--threshold", type=float, help="override the flagging threshold")
    p_score.add_argument("--title", default="Semantic Entropy Report")
    p_score.add_argument("--top-k", type=int, default=10, help="prompts detailed in the markdown")
    p_score.add_argument("--prompt-key", default="prompt")
    p_score.add_argument("--samples-key", default="samples")
    p_score.add_argument("--label-key", default="label")
    p_score.add_argument(
        "--include-judgements", action="store_true", help="embed every NLI verdict"
    )
    p_score.add_argument(
        "--no-calibrate", action="store_true", help="skip auto-calibration on labels"
    )
    _add_scoring_args(p_score)
    _add_calibration_args(p_score)
    p_score.set_defaults(func=cmd_score)

    p_cal = sub.add_parser("calibrate", help="pick a threshold from a labelled dev set")
    p_cal.add_argument("--input", required=True, help="labelled JSONL, or a scored report .json")
    p_cal.add_argument("--out", default="calibration.json")
    p_cal.add_argument("--label-key", default="label")
    _add_scoring_args(p_cal)
    _add_calibration_args(p_cal)
    p_cal.set_defaults(func=cmd_calibrate)

    p_report = sub.add_parser("report", help="re-render markdown from a saved report")
    p_report.add_argument("--json", required=True)
    p_report.add_argument("--out", default="report.md")
    p_report.add_argument("--top-k", type=int, default=10)
    p_report.set_defaults(func=cmd_report)

    p_explain = sub.add_parser("explain", help="print the audit trail for one scored prompt")
    p_explain.add_argument("--json", required=True)
    p_explain.add_argument("--index", type=int, default=0)
    p_explain.set_defaults(func=cmd_explain)

    p_gate = sub.add_parser("gate", help="gate a single prompt")
    p_gate.add_argument("prompt")
    p_gate.add_argument("--samples", action="append", help="pre-generated answer (repeatable)")
    p_gate.add_argument("--threshold", type=float)
    p_gate.add_argument("--block-threshold", type=float)
    _add_scoring_args(p_gate)
    p_gate.set_defaults(func=cmd_gate)

    p_demo = sub.add_parser("demo", help="run the full pipeline on canned data (offline)")
    p_demo.add_argument("--out", help="also write <out>.json and <out>.md")
    p_demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except SemanticEntropyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"error: file not found: {exc.filename}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
