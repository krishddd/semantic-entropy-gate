"""Gate a real agent action behind the model's own uncertainty.

The agent is asked to issue a refund. When it is confident about the order, the
tool call runs. When it is guessing, the gate defers and forages for information
instead of executing an irreversible action.

    python examples/gate_an_agent.py
"""

from semantic_entropy_gate import Gate, LexicalEntailment
from semantic_entropy_gate.active_inference import (
    default_policies,
    explain_ranking,
    rank_policies,
)

# ---------------------------------------------------------------- fake world

KNOWLEDGE = {
    "What is the refund window for order 41?": [
        "30 days.",
        "The refund window is 30 days.",
        "It is 30 days from delivery.",
        "30 days.",
        "Thirty days.",
        "You have 30 days.",
    ],
    "What is the refund window for order 9931?": [
        "14 days.",
        "It is 60 days.",
        "About 30 days.",
        "7 days, I think.",
        "The window is 90 days.",
        "Around 14 days.",
    ],
}

REFUNDS_ISSUED = []


def sampler(prompt: str, n: int):
    return KNOWLEDGE[prompt][:n]


def issue_refund(order_id: int) -> str:
    """The irreversible action. Money actually moves here."""
    REFUNDS_ISSUED.append(order_id)
    return f"refund issued for order {order_id}"


def forage(prompt: str, result) -> str:
    """What to do instead of acting: resolve the uncertainty.

    In a real system this is a retrieval call, a database lookup, or a
    clarifying question put back to the user.
    """
    return (
        f"DEFERRED: the model produced {result.n_clusters} incompatible answers "
        f"({result.agreement:.0%} agreement). Escalating to a human agent instead "
        "of moving money."
    )


def main() -> None:
    gate = Gate(
        sampler,
        threshold=0.55,
        block_threshold=0.9,
        n_samples=6,
        entailment=LexicalEntailment(),
        on_defer=forage,
    )

    for order_id, prompt in [
        (41, "What is the refund window for order 41?"),
        (9931, "What is the refund window for order 9931?"),
    ]:
        print("=" * 78)
        print(f"ORDER {order_id}")
        decision = gate.run(prompt, issue_refund, order_id=order_id)
        print(decision.explain())
        print()
        print(f"executed: {decision.executed}")
        print(f"returned: {decision.answer}")
        print()

        # The same decision, expressed as an expected-free-energy policy choice.
        ranked = rank_policies(default_policies(), decision.result)
        print(explain_ranking(ranked))
        print()

    print("=" * 78)
    print(f"refunds actually issued: {REFUNDS_ISSUED}")
    print("gate stats:", gate.stats())


if __name__ == "__main__":
    main()
