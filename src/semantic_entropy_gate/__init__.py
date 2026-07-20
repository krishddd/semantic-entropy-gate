"""semantic-entropy-gate: catch LLM confabulation before it acts.

Sample a model N times, cluster the answers by *meaning* (bidirectional NLI
entailment), and measure the entropy over those meaning-clusters. Low entropy =
the model committed to one answer. High entropy = it was guessing.

    from semantic_entropy_gate import score, Gate

    result = score("Who invented the telephone?", my_sampler)
    print(result.normalized_entropy)   # 0.0 (certain) .. 1.0 (pure guesswork)
    print(result.explain())            # the full audit trail

Three surfaces
--------------
- **API**        :func:`score` / :func:`score_samples` -> :class:`EntropyResult`
- **Middleware** :class:`Gate` — allow / warn / defer / block any agent call
- **CLI**        ``sem-gate score|calibrate|gate|report|explain|demo``

Method: Farquhar, Kossen, Kuhn & Gal, *Detecting hallucinations in large language
models using semantic entropy*, Nature 630 (2024).
"""

from __future__ import annotations

__version__ = "0.1.0"

from .active_inference import (
    Policy,
    PolicyEvaluation,
    ambiguity_from_entropy,
    expected_free_energy,
    rank_policies,
    should_forage,
)
from .calibrate import auprc, auroc, calibrate, roc_curve, threshold_sweep
from .clustering import cluster
from .dataset import DatasetRow, load_dataset, write_jsonl
from .entailment import (
    CachedEntailment,
    CannedEntailment,
    CrossEncoderEntailment,
    EntailmentModel,
    LexicalEntailment,
    LLMJudgeEntailment,
    auto_entailment,
)
from .entropy import (
    chao1_alphabet_size,
    naive_string_entropy,
    predictive_entropy,
    semantic_entropy,
    shannon_entropy,
)
from .errors import (
    CalibrationError,
    EntailmentBackendError,
    GateBlockedError,
    SamplingError,
    SemanticEntropyError,
)
from .gate import DEFAULT_THRESHOLD, Gate, gate
from .probes import SemanticEntropyProbe, TokenPosition
from .report import Report, build_report
from .sampling import from_texts, openai_sampler, resolve_sampler
from .score import DEFAULT_N_SAMPLES, score, score_batch, score_samples
from .types import (
    CalibrationResult,
    EntailmentJudgement,
    EntailmentLabel,
    EntropyResult,
    Estimator,
    GateAction,
    GateDecision,
    Sample,
    SemanticCluster,
    ThresholdPoint,
)

__all__ = [
    "__version__",
    # scoring
    "score",
    "score_samples",
    "score_batch",
    "DEFAULT_N_SAMPLES",
    "cluster",
    # gating
    "Gate",
    "gate",
    "DEFAULT_THRESHOLD",
    # entailment backends
    "EntailmentModel",
    "CrossEncoderEntailment",
    "LLMJudgeEntailment",
    "LexicalEntailment",
    "CannedEntailment",
    "CachedEntailment",
    "auto_entailment",
    # samplers
    "resolve_sampler",
    "from_texts",
    "openai_sampler",
    # entropy maths
    "semantic_entropy",
    "shannon_entropy",
    "naive_string_entropy",
    "predictive_entropy",
    "chao1_alphabet_size",
    # calibration
    "calibrate",
    "auroc",
    "auprc",
    "roc_curve",
    "threshold_sweep",
    # probes
    "SemanticEntropyProbe",
    "TokenPosition",
    # active inference
    "Policy",
    "PolicyEvaluation",
    "expected_free_energy",
    "rank_policies",
    "ambiguity_from_entropy",
    "should_forage",
    # reporting / data
    "Report",
    "build_report",
    "DatasetRow",
    "load_dataset",
    "write_jsonl",
    # types
    "EntropyResult",
    "Sample",
    "SemanticCluster",
    "EntailmentJudgement",
    "EntailmentLabel",
    "Estimator",
    "GateAction",
    "GateDecision",
    "CalibrationResult",
    "ThresholdPoint",
    # errors
    "SemanticEntropyError",
    "SamplingError",
    "EntailmentBackendError",
    "CalibrationError",
    "GateBlockedError",
]
