"""Wiring semantic-entropy-gate to a real model — every integration path.

Nothing in this file runs a network call by default; each block is a small,
copy-pasteable recipe. Run it to see which paths are available in your
environment.

    python examples/connect_your_llm.py
"""

from semantic_entropy_gate import (
    Gate,
    LexicalEntailment,
    LLMJudgeEntailment,
    auto_entailment,
    openai_sampler,
    score,
)

# ---------------------------------------------------------------------------
# 1. The sampler. Any of these shapes works — the library normalises them.
# ---------------------------------------------------------------------------


def sampler_batch(prompt, n):
    """Preferred: one call, n generations. Temperature MUST be > 0."""
    # return [c.text for c in client.generate(prompt, n=n, temperature=1.0)]
    return ["stub"] * n


def sampler_single(prompt):
    """Single-shot: the library calls it n times for you."""
    # return client.complete(prompt, temperature=1.0)
    return "stub"


def sampler_with_logprobs(prompt, n):
    """White-box: return (text, mean_token_logprob) pairs.

    Supplying log-probabilities switches the estimator from the discrete
    (count-based) formulation to the Rao-Blackwellised one, which is more
    precise at the same sample count. The value must be *length-normalised* —
    the mean per-token log-probability, not the sequence sum.
    """
    return [("stub", -0.5)] * n


# ---------------------------------------------------------------------------
# 2. OpenAI-compatible endpoints (OpenAI, Azure, vLLM, Together, Ollama...)
# ---------------------------------------------------------------------------

OPENAI_RECIPE = """
from openai import OpenAI
from semantic_entropy_gate import openai_sampler, score

client  = OpenAI()
sampler = openai_sampler(client, "gpt-4o-mini", temperature=1.0, logprobs=True)

result = score("Who won the 2032 election?", sampler, n_samples=10)
print(result.explain())
"""

# ---------------------------------------------------------------------------
# 3. The Anthropic Messages API
# ---------------------------------------------------------------------------

ANTHROPIC_RECIPE = """
import anthropic
from semantic_entropy_gate import score

client = anthropic.Anthropic()

def sampler(prompt, n):
    # The Messages API returns one completion per call, so draw n of them.
    # Temperature 1.0 is what exposes confabulation.
    out = []
    for _ in range(n):
        msg = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=256,
            temperature=1.0,
            messages=[{"role": "user", "content": prompt}],
        )
        out.append(msg.content[0].text)
    return out

result = score("What is our Q4 2031 revenue?", sampler, n_samples=10)
"""

# ---------------------------------------------------------------------------
# 4. The entailment oracle
# ---------------------------------------------------------------------------

ENTAILMENT_RECIPE = """
# (a) Local NLI cross-encoder — best fidelity.
#     pip install "semantic-entropy-gate[hf]"
from semantic_entropy_gate import CrossEncoderEntailment
entailment = CrossEncoderEntailment("microsoft/deberta-large-mnli")   # or the xsmall default

# (b) No GPU? Use any chat model as the entailment judge.
from openai import OpenAI
from semantic_entropy_gate import LLMJudgeEntailment
client = OpenAI()

def judge(prompt: str) -> str:
    r = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content

entailment = LLMJudgeEntailment(judge)

# (c) Let the library choose the best available, and warn if it falls back.
from semantic_entropy_gate import auto_entailment
entailment = auto_entailment(judge=judge)
"""

# ---------------------------------------------------------------------------
# 5. Gating a LangChain / LlamaIndex / custom agent step
# ---------------------------------------------------------------------------

AGENT_RECIPE = '''
from semantic_entropy_gate import Gate

def forage(prompt, result):
    """DEFER handler: resolve the uncertainty instead of acting on it."""
    docs = retriever.get_relevant_documents(prompt)
    return llm.invoke(f"Answer using ONLY these sources:\\n{docs}\\n\\n{prompt}")

gate = Gate(sampler, threshold=0.55, block_threshold=0.9, on_defer=forage)

decision = gate.run(user_question, agent_executor.invoke, {"input": user_question})
if decision.executed:
    respond(decision.answer)
else:
    log.warning("gate deferred: %s", decision.reason)
    log.info(decision.explain())          # the full evidence, for the audit trail
    respond(decision.answer)              # whatever forage() produced
'''


def main() -> None:
    print("Integration recipes (not executed — copy the one you need):\n")
    for title, recipe in [
        ("OpenAI-compatible", OPENAI_RECIPE),
        ("Anthropic Messages API", ANTHROPIC_RECIPE),
        ("Entailment backends", ENTAILMENT_RECIPE),
        ("Gating an agent", AGENT_RECIPE),
    ]:
        print("=" * 78)
        print(title)
        print("=" * 78)
        print(recipe.strip())
        print()

    print("=" * 78)
    print("What is available in THIS environment")
    print("=" * 78)
    backend = auto_entailment(quiet=True)
    print(f"auto_entailment() selected: {backend.name}")
    if "lexical" in backend.name:
        print(
            '  -> no NLI model installed. `pip install "semantic-entropy-gate[hf]"`\n'
            "     or pass judge=<callable> for production-grade scoring."
        )

    # The stub samplers above still exercise the full pipeline end to end.
    result = score("stub prompt", sampler_batch, n_samples=4, entailment=LexicalEntailment())
    print(
        f"\npipeline check: {result.n_samples} samples -> {result.n_clusters} cluster(s), "
        f"normalized entropy {result.normalized_entropy:.3f} "
        f"(identical stubs, so 0.000 is correct)"
    )

    gate = Gate(sampler_with_logprobs, threshold=0.55, entailment=LexicalEntailment())
    decision = gate.check("stub prompt", n_samples=4)
    print(f"gate check: {decision.action.value} - {decision.reason}")

    # Referenced so the imports above are meaningful to a reader skimming the file.
    assert callable(sampler_single) and callable(openai_sampler) and LLMJudgeEntailment


if __name__ == "__main__":
    main()
