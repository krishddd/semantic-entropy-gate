# Research notes

Where each design decision came from, and which claims are load-bearing.

## Primary sources

| Source | What was taken from it |
| --- | --- |
| Farquhar, Kossen, Kuhn & Gal, *Detecting hallucinations in large language models using semantic entropy*, **Nature 630**, 625–630 (2024). [doi:10.1038/s41586-024-07421-0](https://www.nature.com/articles/s41586-024-07421-0) | The method itself: sample N, cluster by bidirectional entailment, take entropy over clusters. Both the Rao-Blackwellised and the *discrete* (no-log-probs) variants. The framing of the target failure mode as **confabulation** rather than hallucination generally. |
| Kossen, Han, Razzak, Schut, Malik & Gal, *Semantic Entropy Probes*, [arXiv:2406.15927](https://arxiv.org/abs/2406.15927) (2024) | `probes.py`. The claim being implemented: hidden states already encode semantic entropy, so a linear probe over a single forward pass approximates it and "reduces the overhead of semantic uncertainty quantification to almost zero"; SEPs generalise out-of-distribution better than probes trained directly on accuracy. TBG/SLT token positions and mid-to-late layers come from the paper's ablations. |
| Kuhn, Gal & Farquhar, *Semantic Uncertainty*, ICLR (2023) | The linguistic-invariance argument: uncertainty must be measured over meanings, not strings. |
| Reference implementation, [jlko/semantic_uncertainty](https://github.com/jlko/semantic_uncertainty) | Greedy nucleus-based cluster assignment; prepending the question to both sides of the NLI call; strict vs relaxed entailment rules. |
| *Estimating Semantic Alphabet Size for LLM Uncertainty Quantification*, [arXiv:2509.14478](https://arxiv.org/html/2509.14478v1) | The discrete plugin estimator's downward bias at small N, and alphabet-size estimation as the correction. Motivates the Chao1 / Miller–Madow diagnostics in `entropy.py`. |
| Local research note: *Active Inference and Epistemic Agency: Integrating Semantic Entropy for Uncertainty Surfacing* | The expected-free-energy framing in `active_inference.py`: semantic entropy as the **ambiguity** term of `G(u)`, and the act→forage phase transition that the DEFER branch implements. |

## Claims this library relies on, and how far

**Load-bearing (the library is pointless without them):**

- Confabulation is resampling-unstable; correct recall is resampling-stable.
  This is the entire signal.
- Meaning-level clustering removes the paraphrase variance that makes
  token-level uncertainty a poor hallucination detector. Verifiable locally:
  compare `result.naive_entropy` against `result.entropy` on any confident
  prompt.

**Assumed but user-verifiable (which is why `calibrate()` reports AUROC before it
reports a threshold):**

- That semantic entropy separates hallucinated from correct answers *on your
  task*. The paper reports strong AUROC on free-form QA; your task may differ,
  and the library refuses to assert otherwise.

**Explicitly not claimed:**

- That low entropy means true. It means internally consistent. Systematic errors
  score low and are invisible to this method.
- That the lexical backend approximates a real NLI model. It is triage grade and
  says so in its own docstring, in the README, and in every result via
  `entailment_backend`.
- Any specific AUROC number for this implementation. No benchmark reproduction
  has been run here; quoting the paper's numbers as if they were this library's
  would be dishonest.

## Decisions that departed from the sources

**Default cross-encoder is `cross-encoder/nli-deberta-v3-xsmall`, not
`microsoft/deberta-large-mnli`.** The paper uses DeBERTa-large. The default here
optimises for "works on a laptop in CI without a 1.6 GB download"; the large
checkpoint is one constructor argument away and is named in the README as the
full-fidelity option.

**A stdlib entailment heuristic exists at all.** Not in any source. It exists so
that the test suite, the demo and CI are deterministic and network-free, and so
that a developer evaluating the library gets a working result in ten seconds. It
is exact on the failure mode confabulation usually takes (mutually exclusive
numeric or named facts) and coarse everywhere else. Every result records that it
was used.

**The gate thresholds the *uncorrected* entropy.** Miller–Madow gives a better
point estimate, but the calibrated threshold and the runtime score must be the
same quantity or calibration is meaningless. The corrected value is reported for
interpretation only.

**Mid-rank tie handling in AUROC.** Not discussed in the sources, but with N=10
samples the normalised entropy takes few distinct values, so ties dominate and
naive handling measurably inflates AUROC.

**Four gate actions, not two.** The papers produce a score; turning it into
ALLOW/WARN/DEFER/BLOCK is this library's addition, motivated by the active
inference note: an agent that can only "proceed" or "fail" has no way to express
*go and find out*, which is the entire point of surfacing uncertainty.

## Open threads

Deliberately out of scope for 0.1.0, in rough priority order:

1. **Claim-level decomposition for long-form answers.** Whole-response entailment
   is blunt for paragraphs. Split into atomic claims, cluster per claim,
   aggregate.
2. **Kernel Language Entropy (KLE).** Von Neumann entropy over a semantic
   similarity kernel instead of hard clusters — captures partial overlap that
   binary entailment discards.
3. **Semantic Reformulation Entropy (SRE).** Paraphrase the *input* as well, to
   separate genuine epistemic uncertainty from prompt-phrasing artefacts.
4. **Benchmark reproduction.** TriviaQA / SQuAD / BioASQ AUROC numbers produced
   by *this* implementation, published in the repo so the claims are checkable.
5. **A hidden-state extraction helper** for `probes.py`, so users of local HF
   models do not have to wire `output_hidden_states=True` themselves.
