"""Quickstart: score a confident model and a confabulating one, side by side.

Runs offline with a fake sampler so you can see the shape of the output before
wiring in a real model. Swap ``fake_llm`` for your own client and you are done.

    python examples/quickstart.py
"""

from semantic_entropy_gate import LexicalEntailment, score

# ---------------------------------------------------------------- fake models

CONFIDENT = [
    "Paris.",
    "It is Paris.",
    "Paris, France.",
    "In Paris.",
    "Paris",
    "The city is Paris.",
]

CONFABULATING = [
    "About 610 Kelvin.",
    "Roughly 503 Kelvin.",
    "It boils at approximately 337 Kelvin.",
    "Around 610 K.",
    "Approximately 575 Kelvin.",
    "It is about 400 Kelvin.",
]


def make_sampler(pool):
    """A stand-in for your real model.

    A real sampler looks like:

        def sampler(prompt, n):
            return [client.complete(prompt, temperature=1.0) for _ in range(n)]

    Temperature MUST be > 0 — greedy decoding gives identical samples and an
    entropy of 0 no matter how wrong the answer is.
    """

    def sampler(prompt, n):
        return pool[:n]

    return sampler


def main() -> None:
    # LexicalEntailment keeps this example dependency-free. In production use
    # the NLI cross-encoder (`pip install "semantic-entropy-gate[hf]"`) or an
    # LLM judge; `score()` picks the best available one automatically.
    entailment = LexicalEntailment()

    confident = score(
        "In which city is the Eiffel Tower located?",
        make_sampler(CONFIDENT),
        n_samples=6,
        entailment=entailment,
    )
    uncertain = score(
        "What is the boiling point of the element astatine, in Kelvin?",
        make_sampler(CONFABULATING),
        n_samples=6,
        entailment=entailment,
    )

    for result in (confident, uncertain):
        print(result.explain(threshold=0.55))
        print()

    print("side by side")
    print("-" * 64)
    for label, result in (("confident", confident), ("confabulating", uncertain)):
        print(
            f"{label:<15} normalized entropy {result.normalized_entropy:.3f}  "
            f"clusters {result.n_clusters}  "
            f"agreement {result.agreement:.0%}  "
            f"flagged={result.is_confabulation(0.55)}"
        )
    print()
    print(
        "Note the confident case: six different strings, one meaning. A token-level\n"
        "detector would have called that maximally uncertain. Semantic clustering\n"
        f"attributes {confident.lexical_entropy:.3f} nats of it to phrasing alone."
    )


if __name__ == "__main__":
    main()
