"""A non-answer is not authorisation to act.

Three prompts, three outcomes. The interesting one is the middle case: the model
is perfectly *consistent* — it says "I don't know" every single time — so
semantic entropy is legitimately low. A gate that only reads the score would
treat that as permission and issue the refund.

    python examples/refusal_aware_gate.py
"""

from semantic_entropy_gate import Gate, LexicalEntailment, score_samples

KNOWLEDGE = {
    # Knows the answer: many phrasings, one meaning -> low entropy, real answer.
    "What is the refund window for order 41?": [
        "30 days.",
        "It is 30 days.",
        "Thirty days.",
        "30 days from delivery.",
        "You have 30 days.",
        "The window is 30 days.",
    ],
    # Knows that it does not know: consistent refusal -> low entropy, NO answer.
    "What is the refund window for order 9931?": [
        "I don't know.",
        "I do not know.",
        "I don't know",
        "I don't know!",
        "I don't know.",
        "I do not know.",
    ],
    # Guessing: mutually exclusive answers -> high entropy.
    "What is the refund window for order 5502?": [
        "14 days.",
        "It is 60 days.",
        "About 30 days.",
        "7 days, I think.",
        "The window is 90 days.",
        "Around 21 days.",
    ],
}

REFUNDS_ISSUED = []


def issue_refund(order_id: int) -> str:
    """The irreversible action. Money actually moves here."""
    REFUNDS_ISSUED.append(order_id)
    return f"refund issued for order {order_id}"


def escalate(prompt: str, result) -> str:
    """Where DEFER goes. Two different reasons land here, and the message says which."""
    if result.abstained:
        return "ESCALATED: the model said it does not know. Routing to a human."
    return (
        f"ESCALATED: the model gave {result.n_clusters} incompatible answers "
        f"({result.agreement:.0%} agreement). Routing to a human."
    )


def main() -> None:
    gate = Gate(
        None,
        threshold=0.55,
        entailment=LexicalEntailment(),
        on_defer=escalate,
        refusal_policy="defer",  # the default; shown here to make it explicit
    )

    for order_id, prompt in [
        (41, "What is the refund window for order 41?"),
        (9931, "What is the refund window for order 9931?"),
        (5502, "What is the refund window for order 5502?"),
    ]:
        samples = KNOWLEDGE[prompt]
        print("=" * 78)
        print(f"ORDER {order_id}")
        print("=" * 78)

        result = score_samples(prompt, samples, entailment=LexicalEntailment())
        print(
            f"normalized entropy {result.normalized_entropy:.3f} | "
            f"clusters {result.n_clusters} | "
            f"reliable {result.reliable} | "
            f"abstained {result.abstained} (refusal rate {result.refusal_rate:.0%})"
        )

        decision = gate.run(prompt, issue_refund, order_id=order_id, samples=samples)
        print(f"action:   {decision.action.value}")
        print(f"reason:   {decision.reason}")
        print(f"executed: {decision.executed}")
        print(f"returned: {decision.answer}")
        print()

    print("=" * 78)
    print(f"refunds actually issued: {REFUNDS_ISSUED}")
    print()
    print(
        "Order 9931 is the point of this example. Its entropy (low) is a correct\n"
        "measurement - the model was consistent. It was consistently declining to\n"
        "answer, and 'I don't know' repeated six times is not permission to move\n"
        "money. `reliable` stays True (nothing broke); `abstained` is what stops it."
    )
    assert REFUNDS_ISSUED == [41], "only the genuinely-answered order should refund"


if __name__ == "__main__":
    main()
