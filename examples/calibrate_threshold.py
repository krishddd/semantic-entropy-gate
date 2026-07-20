"""Pick a deployment threshold from a labelled dev set, and report the AUROC.

Never ship the default threshold. Label ~100 prompts from your own task, run
this, and read the AUROC first: if it is near 0.5 the signal is not there and no
threshold will save you.

    python examples/calibrate_threshold.py
"""

import os

from semantic_entropy_gate import Gate, LexicalEntailment, build_report, calibrate, score_samples
from semantic_entropy_gate.dataset import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "demo.jsonl")


def main() -> None:
    rows = load_dataset(DATA)
    entailment = LexicalEntailment()

    # 1. Score the dev set. Generations are already in the file, so this needs
    #    no model, no network and no API key.
    results = [
        score_samples(row.prompt, row.samples, entailment=entailment, metadata={"label": row.label})
        for row in rows
    ]
    labels = [row.label for row in rows]

    # 2. Calibrate. Two criteria, so you can see the trade-off explicitly.
    balanced = calibrate(results, labels, criterion="youden")
    conservative = calibrate(results, labels, criterion="target_fpr", target_fpr=0.0)

    print(balanced.explain())
    print()
    print(conservative.explain())
    print()
    print(
        "The conservative threshold tolerates zero false alarms; the balanced one\n"
        "catches more hallucinations but will occasionally defer on a correct answer.\n"
        "Which is right depends on whether a false alarm or a missed hallucination\n"
        "costs you more."
    )
    print()

    # 3. Deploy the calibrated number.
    gate = Gate(None, threshold=balanced.threshold, entailment=entailment)
    for row, result in zip(rows, results):
        decision = gate.decide(result)
        truth = "hallucinated" if row.label else "correct"
        print(
            f"  {decision.action.value:<6} H={decision.score:.3f}  "
            f"(truth: {truth:<12}) {row.prompt[:52]}"
        )

    # 4. Write the reports a reviewer will actually read.
    report = build_report(
        results,
        calibration=balanced,
        title="semantic-entropy-gate calibration example",
    )
    out = os.path.join(HERE, "..", "calibration_example")
    report.to_json(f"{out}.json")
    report.to_markdown(f"{out}.md")
    print(f"\nwrote {out}.json and {out}.md")


if __name__ == "__main__":
    main()
