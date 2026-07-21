"""The shape of a real deployment, start to finish.

Not a demo — this is the wiring you would actually ship, with the parts that
matter for a live agent made explicit:

1. a **production-tier** entailment backend, enforced rather than hoped for;
2. a **preflight** check that fails the process before it protects anything;
3. a **calibrated** threshold, not the default;
4. a gate that fails closed, refuses non-answers, and hands DEFER somewhere
   useful;
5. an audit record for every decision.

Run it as-is (it uses stand-ins so it works offline), then replace the three
marked functions with your own.

    python examples/production_agent.py
"""

import json
import sys

from semantic_entropy_gate import (
    DatasetRow,
    Gate,
    LexicalEntailment,
    LLMJudgeEntailment,
    calibrate,
    preflight,
    score_samples,
)
from semantic_entropy_gate.types import Sample

# ===========================================================================
# 1. YOUR MODEL. Replace these three.
# ===========================================================================


def sampler(prompt: str, n: int):
    """Draw n INDEPENDENT generations at temperature ~1.0.

    Real version:

        def sampler(prompt, n):
            r = client.chat.completions.create(
                model="gpt-4o-mini", n=n, temperature=1.0, logprobs=True,
                messages=[{"role": "user", "content": prompt}],
            )
            return [c.message.content for c in r.choices]

    Or use the bundled adapter: `openai_sampler(client, "gpt-4o-mini")`.
    """
    return _CANNED.get(prompt, [f"stand-in answer {i}" for i in range(n)])[:n]


def entailment_judge(prompt: str) -> str:
    """A chat model acting as the entailment oracle (no GPU needed).

    Real version:

        def entailment_judge(prompt):
            r = client.chat.completions.create(
                model="gpt-4o-mini", temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return r.choices[0].message.content
    """
    return _canned_judge(prompt)


def retrieve_and_answer(prompt: str, result) -> str:
    """Where DEFER goes: resolve the uncertainty instead of acting on it.

    Real version: a vector-store lookup, a database query, or a clarifying
    question put back to the user.
    """
    return f"[escalated to retrieval] {prompt}"


# ===========================================================================
# 2. PREFLIGHT. Fail the process before it protects anything.
# ===========================================================================


def build_gate(threshold: float, *, calibrated: bool) -> Gate:
    # A production-tier oracle. LLMJudgeEntailment needs no local model;
    # CrossEncoderEntailment (pip install "semantic-entropy-gate[hf]") is the
    # paper's approach and cheaper per call once loaded.
    backend = LLMJudgeEntailment(entailment_judge)

    report = preflight(
        sampler=sampler,
        entailment=backend,
        threshold=threshold,
        calibrated=calibrated,
        n_samples=6,
        probe_prompt="What is the refund window for order 41?",
        # The labelled dev set unlocks the check none of the others can
        # substitute for: does semantic entropy separate hallucinations on THIS
        # task? Without it, doctor reports the gap as the biggest remaining
        # unknown rather than letting a wall of PASSes imply otherwise.
        dev_set=[
            DatasetRow(prompt=prompt, samples=[Sample(text=t) for t in samples], label=label)
            for prompt, (samples, label) in _DEV_SET.items()
        ],
    )
    print(report.render())
    print()
    if not report.ready:
        print("refusing to start: preflight failed", file=sys.stderr)
        raise SystemExit(1)

    # Derive the hard-refusal boundary from the calibrated threshold rather than
    # hard-coding it: calibration can legitimately return a threshold above any
    # constant you picked in advance, and Gate rejects block < defer.
    block_threshold = min(1.0, max(threshold, threshold + 0.15))

    return Gate(
        sampler,
        threshold=threshold,
        block_threshold=block_threshold,
        n_samples=6,
        entailment=backend,
        require_production_backend=True,  # refuse to start on a triage backend
        fail_closed=True,  # a broken sampler defers, never allows
        require_reliable=True,  # a broken measurement is not confidence
        refusal_policy="defer",  # "I don't know" is not authorisation
        on_defer=retrieve_and_answer,
    )


# ===========================================================================
# 3. CALIBRATE. The threshold is a claim about your task; measure it.
# ===========================================================================


def calibrate_threshold():
    """In production this runs offline over a labelled dev set, once."""
    backend = LLMJudgeEntailment(entailment_judge)
    results, labels = [], []
    for prompt, (samples, label) in _DEV_SET.items():
        results.append(score_samples(prompt, samples, entailment=backend))
        labels.append(label)

    calibration = calibrate(results, labels, criterion="target_fpr", target_fpr=0.05)
    print(calibration.explain())
    print()
    if calibration.auroc < 0.65:
        print(
            "WARNING: AUROC below 0.65 - semantic entropy barely separates "
            "hallucinations on this task. Do not hard-block on it.",
            file=sys.stderr,
        )
    if not calibration.trustworthy:
        print(
            "WARNING: this calibration carries caveats (see above). Treat the "
            "threshold as provisional and re-fit on more labelled prompts.",
            file=sys.stderr,
        )
    return calibration


# ===========================================================================
# 4. SERVE. Every decision is gated and recorded.
# ===========================================================================


def issue_refund(order_id: int) -> str:
    """The irreversible action."""
    return f"refund issued for order {order_id}"


def handle(gate: Gate, prompt: str, order_id: int, audit) -> None:
    decision = gate.run(prompt, issue_refund, order_id=order_id)

    audit.append(
        {
            "prompt": prompt,
            "action": decision.action.value,
            "executed": decision.executed,
            "score": round(decision.score, 4),
            "reliable": decision.result.reliable,
            "abstained": decision.result.abstained,
            "clusters": decision.result.n_clusters,
            "backend": decision.result.entailment_backend,
            "reason": decision.reason,
        }
    )

    print(f"  {decision.action.value.upper():6s} order {order_id}: {decision.reason}")
    if decision.warning:
        print(f"         warning: {decision.warning[:100]}")


def main() -> None:
    calibration = calibrate_threshold()
    gate = build_gate(calibration.threshold, calibrated=True)

    audit = []
    print("=" * 78)
    print("SERVING")
    print("=" * 78)
    for order_id, prompt in [
        (41, "What is the refund window for order 41?"),  # knows
        (9931, "What is the refund window for order 9931?"),  # declines
        (5502, "What is the refund window for order 5502?"),  # guesses
    ]:
        handle(gate, prompt, order_id, audit)

    print()
    print("=" * 78)
    print("AUDIT RECORD (ship this to your log pipeline)")
    print("=" * 78)
    print(json.dumps(audit, indent=2))
    print()
    print("gate stats:", json.dumps(gate.stats(), indent=2, default=str))

    executed = [row for row in audit if row["executed"]]
    assert len(executed) == 1, "only the confidently-answered order should refund"

    # =======================================================================
    # 5. MONITOR. The one assumption doctor could not verify at deploy time -
    #    that the dev set represents live traffic - is measurable now that the
    #    gate has history. Run this periodically (a cron, a metrics hook).
    # =======================================================================
    print()
    print("=" * 78)
    print("DRIFT CHECK (run periodically once history accumulates)")
    print("=" * 78)
    try:
        drift = gate.check_drift(calibration)
        print(drift.explain())
    except Exception as exc:  # noqa: BLE001 - not enough history yet is normal
        print(f"not enough live history yet ({exc}) - keep serving and re-check")


# ===========================================================================
# Offline stand-ins so this file runs without a network.
# ===========================================================================

_CANNED = {
    "What is the refund window for order 41?": [
        "30 days.",
        "It is 30 days.",
        "Thirty days.",
        "30 days from delivery.",
        "You have 30 days.",
        "The window is 30 days.",
    ],
    "What is the refund window for order 9931?": [
        "I don't know.",
        "Unknown.",
        "I'm not sure.",
        "I cannot answer that.",
        "I do not know.",
        "I have no information about this.",
    ],
    "What is the refund window for order 5502?": [
        "14 days.",
        "It is 60 days.",
        "About 30 days.",
        "7 days, I think.",
        "The window is 90 days.",
        "Around 21 days.",
    ],
}


def _dev_set():
    """A stand-in labelled dev set with GRADED difficulty.

    Real calibration needs ~100 prompts from your own task. Four clean ones give
    a threshold of 1.0 and an AUROC of 1.0 that mean nothing — which is why
    `calibration.caveats` exists and this set is deliberately messier.
    """
    rows = {}
    for i in range(6):  # confident, correct
        rows[f"known-{i}"] = ([f"{30 + i} days."] * 3 + [f"It is {30 + i} days."], 0)
    for i in range(3):  # mostly agreeing, correct
        rows[f"mostly-{i}"] = (
            [f"{50 + i} units.", f"It is {50 + i} units.", f"{50 + i} units", "About 99 units."],
            0,
        )
    for i in range(6):  # guessing, wrong
        rows[f"guess-{i}"] = (
            [f"{7 + i} days.", f"{60 + i} days.", f"{90 + i} days.", f"{21 + i} days."],
            1,
        )
    for i in range(3):  # two-way split, wrong
        rows[f"split-{i}"] = (
            [f"{200 + i} units.", f"{200 + i} units.", f"{800 + i} units.", f"{800 + i} units."],
            1,
        )
    return rows


_DEV_SET = _dev_set()


def _canned_judge(prompt: str) -> str:
    """Stands in for a real chat model, using the same lexical rules offline."""
    lexical = LexicalEntailment()
    try:
        premise = prompt.split("Statement A (premise): <<<", 1)[1].split(">>>", 1)[0]
        hypothesis = prompt.split("Statement B (hypothesis): <<<", 1)[1].split(">>>", 1)[0]
    except IndexError:  # pragma: no cover - prompt template changed
        return "neutral"
    return lexical.classify(premise, hypothesis)[0].value


if __name__ == "__main__":
    main()
