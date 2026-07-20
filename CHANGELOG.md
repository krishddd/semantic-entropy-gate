# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-20

First public release. Implements semantic entropy for hallucination detection
(Farquhar et al., *Nature* 630, 2024) with three usable surfaces.

### Added

- **Scoring API** — `score(prompt, sampler)` and `score_samples(prompt, samples)`
  returning a fully auditable `EntropyResult` (samples, every NLI verdict,
  cluster assignments, cluster masses, bias diagnostics, `explain()`).
- **Entailment backends** — `CrossEncoderEntailment` (HuggingFace NLI, optional
  `[hf]` extra), `LLMJudgeEntailment` (no-GPU fallback via any chat callable),
  `LexicalEntailment` (stdlib heuristic for offline triage and tests),
  `CannedEntailment` (fixtures), `CachedEntailment` (memoisation), and
  `auto_entailment()` with an explicit warning when it falls back.
- **Clustering** — greedy bidirectional-entailment equivalence classes with
  strict and relaxed modes and an exact-match shortcut.
- **Estimators** — Rao-Blackwellised (length-normalised sequence likelihoods)
  and discrete (black-box) semantic entropy, auto-selected; plus naive string
  entropy, predictive entropy, Chao1 alphabet size and the Miller–Madow bias
  correction as small-sample diagnostics.
- **Middleware** — `Gate` with an ALLOW / WARN / DEFER / BLOCK ladder,
  `on_defer` / `on_block` hooks, `wrap()`/`guard()` decorators, decision history
  and `stats()`.
- **Calibration** — `calibrate()` with exact Mann–Whitney AUROC (mid-rank tie
  handling), step-wise AUPRC, a full threshold sweep and five selection criteria
  (`youden`, `f1`, `accuracy`, `target_fpr`, `target_recall`).
- **Semantic Entropy Probes** — pure-Python L2-regularised logistic regression
  over hidden states (Kossen et al., 2024), with TBG/SLT positions, save/load,
  and self-reported training AUROC.
- **Active inference bridge** — expected-free-energy policy ranking that turns a
  DEFER into a principled information-seeking phase transition.
- **Reports** — JSON (replayable, round-trips through `from_dict`) and markdown
  (summary, calibration table, ASCII ROC plot, per-prompt cluster evidence).
- **CLI** — `sem-gate score | calibrate | gate | report | explain | demo`,
  including a fully offline demo requiring no network, GPU or API key.
- **pytest plugin** — `assert_confident` / `assert_uncertain` and the
  `semantic_entropy` fixture, auto-registered on install.
- Zero runtime dependencies; MIT licensed; CI across Python 3.9–3.12 with lint,
  tests, an offline CLI smoke test and a clean-env wheel install check.

[0.1.0]: https://github.com/krishddd/semantic-entropy-gate/releases/tag/v0.1.0
