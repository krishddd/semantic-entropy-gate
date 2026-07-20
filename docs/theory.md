# Theory: semantic entropy, end to end

This is the long-form version of the README's theory section — the maths, the
estimator choices, and the places where the method quietly breaks.

## 1. The target: confabulation, not "hallucination"

"Hallucination" is an umbrella term covering training-data bias, context
misalignment, sycophancy, distribution shift and outright fabrication. Semantic
entropy targets one specific member of that family: **confabulation** —
arbitrary, incorrect generations produced because the model has no stable fact
to draw on.

The diagnostic signature is **instability under resampling**. Ask a confabulating
model the same question ten times at temperature 1.0 and it returns ten different
meanings, because each one is an independent guess. Ask a model that learned a
fact and it returns one meaning in ten phrasings.

The corollary is the method's main limitation: a **systematic** error — a wrong
fact the model actually internalised — is reproduced identically every time and
scores *low* entropy. Semantic entropy cannot see it. This is a detector for
"the model does not know", not for "the model is wrong".

## 2. Why token-level uncertainty is the wrong instrument

Given a query `x`, an autoregressive model defines a distribution over token
sequences. Its sequence log-likelihood is easy to get and tempting to use:

```
log p(s | x) = Σ_t log p(t_t | t_<t, x)
```

The problem is that this distribution is over **strings**, and the thing we care
about is defined over **meanings**. Those differ by the entire combinatorial mass
of paraphrase:

```
"It is located in Paris"   |
"Paris, France"            |  one meaning
"The city is Paris"        |  spread across many high-entropy strings
"Paris."                   |
```

Sequence-level entropy is high here, and nothing is wrong. Conversely a fluent
fabrication that follows a highly probable syntactic template can have *low*
token entropy while being entirely invented.

So token entropy conflates two independent quantities:

- **lexical/syntactic uncertainty** — undecided about wording;
- **semantic uncertainty** — undecided about the fact.

Only the second is predictive of error. Every `EntropyResult` reports both, and
their difference:

```python
result.naive_entropy    # entropy over raw strings (the token-level view)
result.entropy          # entropy over meanings   (what we actually want)
result.lexical_entropy  # the difference — the false alarms you just avoided
```

## 3. The estimator

### 3.1 Sampling

Draw `N` generations `{s_1 … s_N}` from `p(s | x)` at non-zero temperature
(≈1.0). Independence matters: caching, greedy decoding, or a sampler that returns
the same completion N times all yield entropy 0 regardless of truth. This is the
single most common misuse of the method, and it fails *silently* — the score just
looks reassuring.

`N = 10` is the default here. `N = 5` is a workable budget cut. Below 5 the
plugin estimator's downward bias (§3.4) dominates the signal.

### 3.2 Semantic equivalence via bidirectional entailment

Two answers belong to the same meaning class iff each entails the other:

```
s_a ≡ s_b   ⟺   (s_a ⊨ s_b)  ∧  (s_b ⊨ s_a)
```

judged by an NLI model, with the **question prepended to both sides**. Without
that conditioning, bare short answers are not comparable propositions: "Paris"
and "France" are unrelated strings, but "Q: Where is the Eiffel Tower? A: Paris"
and "Q: Where is the Eiffel Tower? A: France" are contradictory claims.

Single-directional entailment is deliberately insufficient. "Paris" is entailed
by "the capital of France, founded in antiquity" but not vice versa; merging on
one direction collapses specific claims into vague ones and systematically
*under*-reports uncertainty.

**Strict vs relaxed.** Strict mode (the default, and the paper's) requires
`entailment` in both directions. Relaxed mode groups answers as long as neither
direction contradicts and the pair is not mutually neutral — useful with small
NLI checkpoints that timidly label obvious paraphrases "neutral". Relaxed mode
merges more, so it reports *less* entropy; recalibrate when you switch.

**Greedy assignment.** Each unassigned generation opens a cluster and is compared
against the remaining unassigned ones — O(N²) worst case, O(N) when the model is
confident (the common case). Because comparison is against the cluster *nucleus*
rather than every member, the partition is order-sensitive. `clustering.py`
therefore records every judgement it made, so any grouping can be re-derived.

### 3.3 Entropy over the clusters

Given semantic classes `C_1 … C_K`:

```
SE = -Σ_k p(C_k) log p(C_k)          [nats]
```

**Rao-Blackwellised (white-box).** With token log-probabilities, each generation
contributes its **length-normalised** mean token log-probability

```
L_i = (1/T_i) Σ_t log p(t_t | t_<t, x)
```

and cluster mass is the log-sum-exp of its members, renormalised over clusters:

```
log p̃(C_k) = logsumexp_{i ∈ C_k} L_i
p(C_k)     = exp(log p̃(C_k)) / Σ_j exp(log p̃(C_j))
```

Length normalisation is not cosmetic: raw sequence likelihood decays
exponentially with length, so without it a detailed correct answer is punished
for being detailed.

**Discrete (black-box).** Text-only APIs hide log-probs. Then

```
p(C_k) = |C_k| / N
```

`cluster_probabilities()` selects automatically: Rao-Blackwell iff **every**
sample carries a log-probability, discrete otherwise. Mixing is refused — a
partially scored sample set would silently weight scored generations against
unscored ones.

### 3.4 Small-sample bias, and being honest about it

The discrete plugin estimator **under-estimates** true semantic entropy at small
`N`. Text generation is heavy-tailed: rare-but-valid meanings exist and simply are
not drawn in ten samples, so the observed alphabet is smaller than the real one
and the entropy comes out low. Bias is toward *false confidence*, which is the
dangerous direction for a guardrail.

Two diagnostics are computed on every result and stored in `result.metadata`:

- **Chao1 alphabet size** — `S_obs + f₁²/(2f₂)` (with the `f₂ = 0` correction),
  a lower bound on the number of distinct meanings the model can emit. If this
  greatly exceeds the observed cluster count, your `N` is too small.
- **Miller–Madow correction** — `H + (K−1)/(2N)`, a first-order de-biasing.

The gate thresholds the **uncorrected** value, deliberately: the calibrated
threshold and the runtime score must be the same quantity, or the calibration
means nothing. The corrected number is reported for interpretation, not used for
decisions.

### 3.5 Normalisation

`normalized_entropy = entropy / log(N)` ∈ `[0, 1]`. Threshold this rather than
raw nats, so that changing `N` from 10 to 5 does not silently move your operating
point.

## 4. Calibration

A threshold is a claim about your task, so it has to be measured on your task.

**AUROC first.** `calibrate()` reports it before it reports a threshold. AUROC
≈ 0.5 means semantic entropy does not separate your hallucinations from your
correct answers at all, and no threshold fixes that — check that sampling
temperature is non-zero, that the entailment oracle is a real NLI model, and that
the labels mean what you think they mean.

AUROC here is the exact Mann–Whitney U identity with **mid-rank tie handling**.
At `N = 10` samples the normalised entropy takes only a handful of distinct
values, so ties are the norm rather than the exception, and naive tie handling
inflates AUROC noticeably.

**Then the criterion.** The threshold sweep enumerates every distinct operating
point; the criterion picks one:

| Criterion | Objective | Choose it when |
| --- | --- | --- |
| `youden` | max `TPR − FPR` | no strong asymmetry |
| `f1` | max F1 on hallucinations | positives are rare, precision matters |
| `accuracy` | max accuracy | balanced classes |
| `target_fpr` | max TPR s.t. `FPR ≤ b` | false alarms are expensive — a gate that defers constantly gets switched off, which is a 0% catch rate |
| `target_recall` | min FPR s.t. `TPR ≥ r` | a missed hallucination is expensive |

When a constraint is infeasible the library returns the least-bad point rather
than silently pretending it was satisfied — check `operating_point` against your
target.

## 5. Cost, and the single-pass approximation

Canonical semantic entropy costs `N` generations plus up to `O(N²)` NLI calls:
a 5–10× latency multiplier that a real-time agent cannot always pay.

Shipped mitigations: exact-match shortcut, `CachedEntailment` memoisation
(typically 30–60% of pairs recur), and greedy clustering that collapses to O(N)
when the model is confident.

The structural fix is **Semantic Entropy Probes** (Kossen et al., 2024): the
model's hidden states already encode its semantic uncertainty, so a linear probe
maps a single forward pass to a high/low semantic-entropy prediction.
`probes.py` implements this in pure Python:

1. Score a calibration corpus the expensive way.
2. Cache hidden states at **TBG** (token before generating — predicts uncertainty
   *before* any output token exists) or **SLT** (second-last generated token —
   after the trajectory has collapsed, slightly more accurate but you have
   already paid for the generation), from a mid-to-late layer.
3. Fit an L2-regularised logistic regression on the binarised entropy label.
4. Serve at ~1/10th the cost.

The probe reports its own training AUROC and offers `.evaluate()` for held-out
data. It is an approximation of an approximation; validate it before letting it
gate anything irreversible.

## 6. From a number to a decision

Thresholding a score is a policy choice dressed up as a measurement. The
`active_inference` module makes the choice explicit using the expected-free-energy
decomposition:

```
G(u) = pragmatic_cost(u) + ambiguity(u) − information_gain(u)
```

Semantic entropy is the **ambiguity** term for a language model: the diffuseness
of the mapping between the underlying fact and the observed generation. When
ambiguity dominates the pragmatic term, no acting policy minimises `G`, and the
free-energy-minimising choice is an epistemic one — retrieve, search, ask.

That is the DEFER branch of the gate, derived rather than asserted. Irreversible
policies weight ambiguity harder (`irreversible_penalty`), because the same
uncertainty should stop a database write long before it stops drafting a
sentence.

## 7. Known limitations

1. **Confabulation only.** Confident, systematic errors are invisible to it.
2. **Consistency ≠ truth.** This measures internal agreement. Pair it with
   retrieval grounding for factuality.
3. **Oracle-dependent.** Scores from different entailment backends are not
   comparable. Recalibrate on a backend change; the backend name is recorded on
   every result so you can detect one.
4. **Long-form answers.** A paragraph asserts many propositions, and
   whole-response entailment is a blunt instrument for it. Decompose into claims
   and score them separately.
5. **Sampling discipline.** Non-zero temperature, independent draws, no caching.
   Violating this fails silently and looks like confidence.
6. **Cost.** 5–10× per gated call unless you use a probe.

## References

- Farquhar, S., Kossen, J., Kuhn, L. & Gal, Y. *Detecting hallucinations in large
  language models using semantic entropy*. **Nature 630**, 625–630 (2024).
  <https://www.nature.com/articles/s41586-024-07421-0>
- Kossen, J., Han, J., Razzak, M., Schut, L., Malik, S. & Gal, Y. *Semantic
  Entropy Probes: Robust and Cheap Hallucination Detection in LLMs*. arXiv
  2406.15927 (2024). <https://arxiv.org/abs/2406.15927>
- Kuhn, L., Gal, Y. & Farquhar, S. *Semantic Uncertainty: Linguistic Invariances
  for Uncertainty Estimation in Natural Language Generation*. ICLR (2023).
- Reference implementation: <https://github.com/jlko/semantic_uncertainty>
