# Architecture

## Design constraints

The whole design follows from four rules:

1. **No score without its evidence.** Every stage records what it decided.
   A guardrail nobody can interrogate gets switched off the first time it is
   wrong.
2. **Zero runtime dependencies.** `pip install semantic-entropy-gate` pulls
   nothing. Torch, transformers and API clients are strictly optional, behind
   lazy imports, and never touched unless selected.
3. **Bring your own model.** The library never instantiates an LLM. You pass a
   sampler; it returns strings.
4. **Offline by default in tests and demos.** CI, `pytest`, and `sem-gate demo`
   must run on an air-gapped machine with no downloads.

## Data flow

```
prompt ─┬─► sampler ──► [Sample, ...] ─┐
        │                              │
        └──────── context ────┐        │
                              ▼        ▼
                    ┌──────────────────────────┐
                    │  clustering.cluster()    │
                    │  greedy bidirectional    │◄── entailment.EntailmentModel
                    │  entailment assignment   │     (cross-encoder | judge | lexical)
                    └────────────┬─────────────┘
                                 │ clusters + assignments + every judgement
                                 ▼
                    ┌──────────────────────────┐
                    │  entropy.semantic_entropy│
                    │  Rao-Blackwell | discrete│
                    │  + Chao1 + Miller-Madow  │
                    └────────────┬─────────────┘
                                 ▼
                         ┌───────────────┐
                         │ EntropyResult │  ← the audit artefact; JSON round-trips
                         └───────┬───────┘
             ┌───────────────────┼───────────────────┬──────────────────┐
             ▼                   ▼                   ▼                  ▼
      ┌─────────────┐   ┌────────────────┐   ┌──────────────┐   ┌──────────────┐
      │ Gate.decide │   │  calibrate()   │   │ Report .md   │   │ active_      │
      │ ALLOW/WARN/ │   │ AUROC + sweep  │   │ Report .json │   │ inference    │
      │ DEFER/BLOCK │   │ + criterion    │   │ + ASCII ROC  │   │ EFE ranking  │
      └──────┬──────┘   └────────────────┘   └──────────────┘   └──────────────┘
             ▼
      GateDecision ──► your action runs, or does not
```

## Modules

| Module | Responsibility | Depends on |
| --- | --- | --- |
| `types.py` | every dataclass + `explain()` renderers; the serialisation contract | — |
| `errors.py` | one exception root so a host app can wrap the gate in a single `except` | — |
| `sampling.py` | normalises any sampler shape to `(prompt, n) -> [Sample]`; OpenAI-compatible adapter | `types` |
| `entailment.py` | the equivalence oracle: cross-encoder / LLM judge / lexical / canned / cached | `types` |
| `clustering.py` | greedy bidirectional-entailment partition, recording every verdict | `entailment`, `types` |
| `entropy.py` | Rao-Blackwell + discrete estimators, naive & predictive entropy, bias diagnostics | `types` |
| `score.py` | the pipeline; `score()` / `score_samples()` / `score_batch()` | all of the above |
| `gate.py` | threshold ladder, hooks, decorators, decision history | `score` |
| `calibrate.py` | AUROC (mid-rank), AUPRC, threshold sweep, five criteria | `types` |
| `probes.py` | Semantic Entropy Probes: pure-Python logistic regression on hidden states | `calibrate`, `types` |
| `active_inference.py` | expected-free-energy policy ranking; act vs forage | `types` |
| `report.py` | JSON + markdown reports, ASCII ROC plot | `types` |
| `dataset.py` | JSONL parsing with line-numbered errors | `types` |
| `cli.py` | `sem-gate` subcommands | everything |
| `pytest_plugin.py` | `assert_confident` / `assert_uncertain`, fixtures | `score` |

Import direction is strictly one-way (`types` ← everything else); there are no
cycles, so any module can be vendored on its own.

## Why the seams are where they are

**`EntailmentModel` is an ABC with three implementations.** The equivalence
oracle is the one component whose fidelity/cost trade-off genuinely differs per
deployment: a GPU box wants DeBERTa, a laptop wants a 70 MB cross-encoder, a
serverless function wants an LLM judge, and CI wants something deterministic with
no downloads. Everything above the oracle is identical in all four cases, and the
chosen backend name travels on the result so scores are never silently compared
across oracles.

**Estimator selection is automatic but recorded.** Whether you get the white-box
or black-box formulation depends on data you may not control (does your provider
return log-probs?). Auto-selecting keeps the API uniform; stamping
`result.estimator` keeps it honest.

**`Gate` returns a `GateDecision` rather than raising.** Deferral is a normal
control-flow outcome in an agent, not an error. Raising is available
(`raise_on_block=True`) for pipelines that prefer it.

**Reports are two files.** `report.json` is the replayable artefact —
`EntropyResult.from_dict()` reconstructs the object, so a production decision can
be re-explained months later. `report.md` is for the person who has to decide
whether to trust the gate.

**Calibration is a separate step, not a constructor default.** The library ships
`DEFAULT_THRESHOLD = 0.55` and says loudly that it is a starting point. Baking a
number into the gate would imply a claim about your task that nobody measured.

## Performance notes

Per gated prompt: `N` generations (your cost) + up to `N(N−1)` directional NLI
calls (halved in practice by the exact-match shortcut, the greedy nucleus
comparison, and `CachedEntailment`).

- Confident prompts collapse to a single cluster after `N−1` comparisons.
- The cache is keyed on `(context, premise, hypothesis)` and typically removes
  30–60% of calls in batch scoring.
- `score_batch()` and the CLI share one backend instance across the run so the
  cache survives between prompts.
- For latency-critical paths, train a `SemanticEntropyProbe` and pay one forward
  pass instead.
