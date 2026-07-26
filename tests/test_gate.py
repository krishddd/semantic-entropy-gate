"""The middleware: does the irreversible action actually get stopped?"""

import pytest

from semantic_entropy_gate import Gate, GateAction, LexicalEntailment
from semantic_entropy_gate import gate as make_gate
from semantic_entropy_gate.errors import GateBlockedError
from semantic_entropy_gate.sampling import from_texts


def confident_gate(samples, **kwargs):
    kwargs.setdefault("entailment", LexicalEntailment())
    kwargs.setdefault("n_samples", 6)
    return Gate(from_texts(samples), **kwargs)


def test_confident_prompt_allows_and_executes(confident_samples):
    gate = confident_gate(confident_samples, threshold=0.55)
    executed = []
    decision = gate.run("q", lambda: executed.append(True) or "done")
    assert decision.action is GateAction.ALLOW
    assert decision.executed is True
    assert decision.answer == "done"
    assert executed == [True]


def test_confabulating_prompt_defers_and_does_not_execute(confabulating_samples):
    gate = confident_gate(confabulating_samples, threshold=0.55)
    executed = []
    decision = gate.run("q", lambda: executed.append(True))
    assert decision.action is GateAction.DEFER
    assert decision.executed is False
    assert executed == []


def test_defer_hook_supplies_the_fallback_answer(confabulating_samples):
    gate = confident_gate(
        confabulating_samples,
        threshold=0.55,
        on_defer=lambda prompt, result: f"escalated: {result.n_clusters} meanings",
    )
    decision = gate.run("q", lambda: "should not run")
    assert decision.answer.startswith("escalated:")
    assert decision.executed is False


def test_block_threshold_refuses_outright(confabulating_samples):
    gate = confident_gate(confabulating_samples, threshold=0.4, block_threshold=0.6)
    decision = gate.run("q", lambda: "nope")
    assert decision.action is GateAction.BLOCK
    assert decision.allowed is False


def test_block_hook_is_called(confabulating_samples):
    gate = confident_gate(
        confabulating_samples,
        threshold=0.4,
        block_threshold=0.6,
        on_block=lambda p, r: "refused",
    )
    assert gate.run("q", lambda: "nope").answer == "refused"


def test_raise_on_block_raises_with_the_decision_attached(confabulating_samples):
    gate = confident_gate(
        confabulating_samples, threshold=0.4, block_threshold=0.6, raise_on_block=True
    )
    with pytest.raises(GateBlockedError) as excinfo:
        gate.run("q", lambda: "nope")
    assert excinfo.value.decision.action is GateAction.BLOCK
    assert excinfo.value.decision.result.n_clusters >= 4


def test_warn_band_executes_but_annotates(split_samples):
    # Two meanings split 4/2 -> normalized entropy ~0.35.
    gate = confident_gate(split_samples, threshold=0.5, warn_threshold=0.2)
    decision = gate.run("Who discovered penicillin?", lambda: "answered")
    assert decision.action is GateAction.WARN
    assert decision.executed is True
    assert decision.warning is not None
    assert "uncertainty" in decision.warning.lower()


def test_allow_and_warn_are_the_allowed_actions():
    assert GateAction.ALLOW.allowed and GateAction.WARN.allowed
    assert not GateAction.DEFER.allowed and not GateAction.BLOCK.allowed


def test_action_receives_the_entropy_result_when_it_asks(confident_samples):
    seen = {}

    def action(entropy_result=None):
        seen["clusters"] = entropy_result.n_clusters
        return "ok"

    confident_gate(confident_samples, threshold=0.55).run("q", action)
    assert seen["clusters"] == 1


def test_action_without_the_keyword_is_called_plainly(confident_samples):
    def action(a, b):
        return a + b

    decision = confident_gate(confident_samples, threshold=0.55).run("q", action, 2, b=3)
    assert decision.answer == 5


def test_action_with_kwargs_receives_the_result(confident_samples):
    def action(**kwargs):
        return kwargs["entropy_result"].n_clusters

    assert confident_gate(confident_samples, threshold=0.55).run("q", action).answer == 1


def test_no_action_returns_the_consensus_answer(confident_samples):
    decision = confident_gate(confident_samples, threshold=0.55).run("q")
    assert decision.executed is True
    assert decision.answer in confident_samples


def test_check_samples_needs_no_sampler(confabulating_samples):
    gate = Gate(None, threshold=0.55, entailment=LexicalEntailment())
    decision = gate.check_samples("q", confabulating_samples)
    assert decision.action is GateAction.DEFER


def test_measure_without_a_sampler_raises():
    with pytest.raises(ValueError, match="no sampler"):
        Gate(None, entailment=LexicalEntailment()).measure("q")


def test_run_with_inline_samples(confabulating_samples):
    gate = Gate(None, threshold=0.55, entailment=LexicalEntailment())
    decision = gate.run("q", lambda: "x", samples=confabulating_samples)
    assert decision.executed is False


def test_wrap_gates_a_function(confabulating_samples):
    gate = confident_gate(confabulating_samples, threshold=0.55)
    calls = []

    @gate.guard
    def answer(prompt):
        calls.append(prompt)
        return "answered"

    decision = answer("q")
    assert decision.action is GateAction.DEFER
    assert calls == []
    assert answer.__name__ == "answer"
    assert answer.gate is gate


def test_wrap_accepts_the_prompt_as_a_keyword(confident_samples):
    gate = confident_gate(confident_samples, threshold=0.55)
    wrapped = gate.wrap(lambda prompt: f"echo {prompt}")
    assert wrapped(prompt="hello").answer == "echo hello"


def test_wrap_without_a_prompt_raises(confident_samples):
    wrapped = confident_gate(confident_samples, threshold=0.55).wrap(lambda: "x")
    with pytest.raises(ValueError, match="needs a prompt"):
        wrapped()


def test_thresholds_are_validated():
    with pytest.raises(ValueError, match="warn_threshold"):
        Gate(None, threshold=0.5, warn_threshold=0.9, entailment=LexicalEntailment())
    with pytest.raises(ValueError, match="block_threshold"):
        Gate(None, threshold=0.5, block_threshold=0.1, entailment=LexicalEntailment())


def test_zero_width_warn_band_is_flagged_not_silent():
    with pytest.warns(UserWarning, match="WARN tier is zero-width"):
        gate = Gate(None, threshold=0.5, warn_threshold=0.5, entailment=LexicalEntailment())
    # The config is legal (WARN deliberately disabled), just announced.
    assert any("zero-width" in w for w in gate.config_warnings)


def test_zero_width_defer_band_is_flagged_not_silent():
    with pytest.warns(UserWarning, match="DEFER tier is zero-width"):
        Gate(None, threshold=0.5, block_threshold=0.5, entailment=LexicalEntailment())


# --------------------------------------------------------------- bounded resolve


def _conditioned_gate():
    """A gate whose entropy drops only once the prompt carries 'RESOLVED'."""
    from semantic_entropy_gate.types import Sample

    # Semantically one answer but textually distinct, so the measurement is
    # reliable (not the byte-identical case the failsafe flags) yet low-entropy.
    _agree = [
        "the answer is 42",
        "the answer is 42.",
        "answer is 42",
        "yes the answer is 42",
        "the answer is 42!",
        "the answer is 42 indeed",
    ]

    def sampler(prompt, n):
        if "RESOLVED" in prompt:
            return [Sample(_agree[i % len(_agree)]) for i in range(n)]
        return [Sample(f"the answer is {i}") for i in range(n)]  # all disagree

    return Gate(sampler, threshold=0.5, n_samples=6, entailment=LexicalEntailment())


def test_resolve_stops_when_foraging_resolves_the_uncertainty():
    gate = _conditioned_gate()
    calls = []

    def forage(prompt, decision):
        calls.append(prompt)
        return prompt + " RESOLVED"

    decision = gate.resolve("q", forage)
    assert decision.action is GateAction.ALLOW
    assert len(calls) == 1  # one forage, then it cleared
    assert decision.metadata["defer_attempts"] == 1
    assert "max_retries_exceeded" not in decision.metadata


def test_resolve_terminates_at_the_cap_when_foraging_never_helps():
    gate = _conditioned_gate()
    attempts = []

    def forage(prompt, decision):
        attempts.append(prompt)
        return prompt + " still-vague"  # never introduces RESOLVED

    decision = gate.resolve("q", forage, max_retries=3)
    assert decision.action is GateAction.DEFER
    assert len(attempts) == 3  # hard cap, not infinite
    assert decision.metadata["defer_attempts"] == 3
    assert decision.metadata["max_retries_exceeded"] is True
    assert "escalate to a human" in decision.reason


def test_resolve_stops_early_when_forager_gives_up():
    gate = _conditioned_gate()

    def forage(prompt, decision):
        return None  # nothing more to try

    decision = gate.resolve("q", forage, max_retries=5)
    assert decision.action is GateAction.DEFER
    assert decision.metadata["defer_attempts"] == 1
    assert "max_retries_exceeded" not in decision.metadata


def test_resolve_with_zero_retries_is_single_shot():
    gate = _conditioned_gate()
    called = []

    def forage(prompt, decision):
        called.append(prompt)
        return prompt + " RESOLVED"

    decision = gate.resolve("q", forage, max_retries=0)
    assert decision.action is GateAction.DEFER
    assert called == []  # never foraged
    assert "max_retries_exceeded" not in decision.metadata


def test_resolve_rejects_negative_retries():
    gate = _conditioned_gate()
    with pytest.raises(ValueError, match="max_retries"):
        gate.resolve("q", lambda p, d: None, max_retries=-1)


def test_default_warn_threshold_sits_below_the_defer_threshold():
    gate = Gate(None, threshold=0.5, entailment=LexicalEntailment())
    assert gate.warn_threshold == pytest.approx(0.3)


def test_reason_names_the_threshold_that_fired(confabulating_samples):
    gate = confident_gate(confabulating_samples, threshold=0.55)
    decision = gate.check("q")
    assert "defer threshold" in decision.reason
    assert "0.550" in decision.reason


def test_history_and_stats_accumulate(confident_samples, confabulating_samples):
    gate = Gate(None, threshold=0.55, entailment=LexicalEntailment())
    gate.check_samples("q1", confident_samples)
    gate.check_samples("q2", confabulating_samples)
    stats = gate.stats()
    assert stats["total"] == 2
    assert stats["counts"]["allow"] == 1
    assert stats["counts"]["defer"] == 1
    assert stats["deferral_rate"] == pytest.approx(0.5)
    assert stats["entailment_backend"] == "lexical-heuristic"


def test_history_can_be_disabled(confident_samples):
    gate = Gate(None, threshold=0.55, entailment=LexicalEntailment(), record_history=False)
    gate.check_samples("q", confident_samples)
    assert gate.history == []


def test_history_is_bounded(confident_samples):
    gate = Gate(None, threshold=0.55, entailment=LexicalEntailment(), max_history=2)
    for _ in range(5):
        gate.check_samples("q", confident_samples)
    assert len(gate.history) == 2


def test_decision_explain_leads_with_the_action(confabulating_samples):
    gate = Gate(None, threshold=0.55, entailment=LexicalEntailment())
    text = gate.check_samples("q", confabulating_samples).explain()
    assert text.startswith("GATE DECISION: DEFER")
    assert "SEMANTIC CLUSTERS" in text


def test_decision_serialises(confabulating_samples):
    gate = Gate(None, threshold=0.55, entailment=LexicalEntailment())
    data = gate.check_samples("q", confabulating_samples).to_dict()
    assert data["action"] == "defer"
    assert data["result"]["n_clusters"] >= 4
    assert data["metadata"]["score"] > 0.55


def test_gate_helper_constructor(confident_samples):
    gate = make_gate(from_texts(confident_samples), threshold=0.6, entailment=LexicalEntailment())
    assert isinstance(gate, Gate)
    assert gate.threshold == 0.6


def test_raw_nats_thresholding_is_supported(confabulating_samples):
    gate = Gate(None, threshold=1.5, normalized=False, entailment=LexicalEntailment())
    decision = gate.check_samples("q", confabulating_samples)
    assert decision.metadata["normalized"] is False
    assert decision.action is GateAction.DEFER
