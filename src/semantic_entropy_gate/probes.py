"""Semantic Entropy Probes: one forward pass instead of ten.

Canonical semantic entropy costs N generations plus O(N^2) NLI calls — a 5-10x
latency multiplier that a real-time agent cannot always pay. Kossen et al. (2024)
showed that a *linear probe* on the model's hidden states predicts the semantic
entropy label directly from a single forward pass, and generalises out of
distribution better than probes trained on correctness labels.

This module implements that probe end to end, in pure Python:

1. Score a calibration corpus the expensive way (:func:`~semantic_entropy_gate.score.score`).
2. Cache each prompt's hidden state at the **TBG** (token before generating) or
   **SLT** (second-last token) position from a mid-to-late layer.
3. :meth:`SemanticEntropyProbe.fit` an L2-regularised logistic regression mapping
   hidden state -> ``P(high semantic entropy)``.
4. Serve it with :meth:`predict_proba` at a fraction of the cost.

The probe is an *approximation of an approximation*, so it reports its own
training AUROC and refuses to pretend otherwise: check
:attr:`SemanticEntropyProbe.train_auroc` before trusting it, and re-validate on
held-out data with :meth:`evaluate`.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .calibrate import auroc
from .errors import CalibrationError
from .types import EntropyResult

__all__ = ["SemanticEntropyProbe", "TokenPosition", "binarize_entropy"]


class TokenPosition:
    """The two hidden-state extraction points validated in the SEP paper."""

    TBG = "tbg"
    """Token Before Generating — the last prompt token. Predicts uncertainty
    *before* a single output token exists, so it can gate a call pre-emptively."""

    SLT = "slt"
    """Second Last Token — the penultimate token of the generated answer, after
    the model has collapsed its trajectory. Slightly more accurate, but you have
    already paid for the generation."""


def binarize_entropy(
    results: Sequence[EntropyResult],
    *,
    threshold: Optional[float] = None,
    normalized: bool = True,
) -> Tuple[List[int], float]:
    """Turn continuous semantic entropy into the binary SEP training target.

    ``threshold=None`` uses the median, which guarantees a balanced training set
    — the setup the paper uses when no task-specific threshold is known yet.
    Returns ``(labels, threshold_used)``.
    """
    scores = [r.score_for(normalized=normalized) for r in results]
    if not scores:
        raise CalibrationError("no results to binarize")
    if threshold is None:
        ordered = sorted(scores)
        mid = len(ordered) // 2
        threshold = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
    return [1 if s >= threshold else 0 for s in scores], float(threshold)


@dataclass
class SemanticEntropyProbe:
    """L2-regularised logistic regression over hidden states.

    Trained with full-batch gradient descent on standardised features — no
    numpy, no torch, so the probe ships and runs anywhere the library does. A
    hidden state of a few thousand dimensions with a few hundred examples trains
    in well under a second.

    Attributes
    ----------
    weights / bias:
        The learned linear model.
    mean / scale:
        Per-feature standardisation fitted at training time and applied at
        inference; without it a raw residual-stream vector's wildly different
        per-dimension magnitudes make gradient descent crawl.
    """

    position: str = TokenPosition.SLT
    layer: int = -1
    l2: float = 1.0
    learning_rate: float = 0.5
    epochs: int = 400
    weights: List[float] = field(default_factory=list)
    bias: float = 0.0
    mean: List[float] = field(default_factory=list)
    scale: List[float] = field(default_factory=list)
    entropy_threshold: Optional[float] = None
    train_auroc: Optional[float] = None
    n_train: int = 0
    model_family: Optional[str] = None
    """Which model's hidden states this probe was trained on (e.g.
    ``"meta-llama/Llama-3-8B"``). A probe learns the *representation geometry* of
    one model; feeding it another model's activations produces confident nonsense.
    Recorded here so a gate can refuse a mismatched probe with a hard error rather
    than score silently against the wrong geometry. ``None`` = unrecorded, which a
    gate treats as unverifiable rather than safe."""

    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ train

    def fit(
        self,
        hidden_states: Sequence[Sequence[float]],
        labels: Sequence[int],
        *,
        seed: int = 0,
        verbose: bool = False,
    ) -> "SemanticEntropyProbe":
        """Fit the probe. ``labels`` are 1 = high semantic entropy."""
        if len(hidden_states) != len(labels):
            raise CalibrationError(
                f"hidden_states/labels mismatch: {len(hidden_states)} vs {len(labels)}"
            )
        if not hidden_states:
            raise CalibrationError("no training examples")
        dimension = len(hidden_states[0])
        if any(len(h) != dimension for h in hidden_states):
            raise CalibrationError("all hidden states must have the same dimensionality")
        if len(set(int(bool(y)) for y in labels)) < 2:
            raise CalibrationError("training labels contain a single class")

        self.mean, self.scale = _fit_standardizer(hidden_states)
        features = [_standardize(h, self.mean, self.scale) for h in hidden_states]
        ys = [1.0 if y else 0.0 for y in labels]

        rng = random.Random(seed)
        self.weights = [rng.uniform(-0.01, 0.01) for _ in range(dimension)]
        self.bias = 0.0
        n = len(features)

        for epoch in range(self.epochs):
            grad_w = [0.0] * dimension
            grad_b = 0.0
            loss = 0.0
            for x, y in zip(features, ys):
                p = _sigmoid(self.bias + sum(w * v for w, v in zip(self.weights, x)))
                error = p - y
                grad_b += error
                for k, v in enumerate(x):
                    grad_w[k] += error * v
                loss -= y * math.log(max(p, 1e-12)) + (1 - y) * math.log(max(1 - p, 1e-12))
            inv_n = 1.0 / n
            for k in range(dimension):
                gradient = grad_w[k] * inv_n + self.l2 * self.weights[k] / n
                self.weights[k] -= self.learning_rate * gradient
            self.bias -= self.learning_rate * grad_b * inv_n
            if verbose and epoch % max(1, self.epochs // 10) == 0:  # pragma: no cover
                print(f"epoch {epoch:4d}  loss {loss / n:.5f}")

        self.n_train = n
        predictions = [self._raw_predict(x) for x in features]
        try:
            self.train_auroc = auroc(predictions, [int(y) for y in ys])
        except CalibrationError:  # pragma: no cover - guarded above
            self.train_auroc = None
        return self

    @classmethod
    def train_from_results(
        cls,
        results: Sequence[EntropyResult],
        hidden_states: Sequence[Sequence[float]],
        *,
        threshold: Optional[float] = None,
        position: str = TokenPosition.SLT,
        layer: int = -1,
        model_family: Optional[str] = None,
        **kwargs: Any,
    ) -> "SemanticEntropyProbe":
        """Convenience path: canonical scores in, trained probe out.

        Pass ``model_family`` to stamp which model produced ``hidden_states`` — a
        gate will later refuse to run this probe against a different model's
        activations.
        """
        labels, used = binarize_entropy(results, threshold=threshold)
        probe = cls(position=position, layer=layer, model_family=model_family, **kwargs)
        probe.entropy_threshold = used
        return probe.fit(hidden_states, labels)

    # -------------------------------------------------------------- inference

    @property
    def is_fitted(self) -> bool:
        """Whether the probe has learned weights it can predict from."""
        return bool(self.weights)

    def predict_proba(self, hidden_state: Sequence[float]) -> float:
        """``P(high semantic entropy)`` for a single hidden state."""
        if not self.weights:
            raise CalibrationError("probe is not fitted")
        if len(hidden_state) != len(self.weights):
            raise CalibrationError(
                f"expected {len(self.weights)}-dim hidden state, got {len(hidden_state)}"
            )
        return self._raw_predict(_standardize(hidden_state, self.mean, self.scale))

    def predict(self, hidden_state: Sequence[float], *, cutoff: float = 0.5) -> bool:
        return self.predict_proba(hidden_state) >= cutoff

    def evaluate(
        self, hidden_states: Sequence[Sequence[float]], labels: Sequence[int]
    ) -> Dict[str, float]:
        """Held-out AUROC and accuracy — always run this before deploying."""
        probabilities = [self.predict_proba(h) for h in hidden_states]
        ys = [int(bool(y)) for y in labels]
        correct = sum(1 for p, y in zip(probabilities, ys) if (p >= 0.5) == bool(y))
        return {
            "auroc": auroc(probabilities, ys),
            "accuracy": correct / len(ys),
            "n": len(ys),
        }

    def _raw_predict(self, standardized: Sequence[float]) -> float:
        return _sigmoid(self.bias + sum(w * v for w, v in zip(self.weights, standardized)))

    # ----------------------------------------------------------- persistence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position,
            "layer": self.layer,
            "l2": self.l2,
            "weights": list(self.weights),
            "bias": self.bias,
            "mean": list(self.mean),
            "scale": list(self.scale),
            "entropy_threshold": self.entropy_threshold,
            "train_auroc": self.train_auroc,
            "n_train": self.n_train,
            "model_family": self.model_family,
            "metadata": dict(self.metadata),
        }

    def save(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return path

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticEntropyProbe":
        probe = cls(
            position=data.get("position", TokenPosition.SLT),
            layer=data.get("layer", -1),
            l2=data.get("l2", 1.0),
        )
        probe.weights = list(data.get("weights", []))
        probe.bias = data.get("bias", 0.0)
        probe.mean = list(data.get("mean", []))
        probe.scale = list(data.get("scale", []))
        probe.entropy_threshold = data.get("entropy_threshold")
        probe.train_auroc = data.get("train_auroc")
        probe.n_train = data.get("n_train", 0)
        probe.model_family = data.get("model_family")
        probe.metadata = dict(data.get("metadata", {}))
        return probe

    @classmethod
    def load(cls, path: str) -> "SemanticEntropyProbe":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def explain(self, width: int = 78) -> str:
        bar = "=" * width
        quality = (
            "not fitted"
            if self.train_auroc is None
            else "strong"
            if self.train_auroc >= 0.8
            else "moderate"
            if self.train_auroc >= 0.65
            else "weak"
        )
        return "\n".join(
            [
                bar,
                "SEMANTIC ENTROPY PROBE",
                bar,
                f"Hidden-state position: {self.position.upper()}   layer: {self.layer}",
                f"Dimensionality:        {len(self.weights)}",
                f"Training examples:     {self.n_train}",
                f"Entropy threshold:     "
                f"{'n/a' if self.entropy_threshold is None else f'{self.entropy_threshold:.4f}'}",
                f"Training AUROC:        "
                f"{'n/a' if self.train_auroc is None else f'{self.train_auroc:.4f}'} ({quality})",
                "",
                "A probe is a cheap approximation of canonical semantic entropy. Validate on",
                "held-out data with .evaluate() before letting it gate anything irreversible.",
                bar,
            ]
        )


# ------------------------------------------------------------------ internals


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)  # avoids overflow for large negative z
    return exp_z / (1.0 + exp_z)


def _fit_standardizer(rows: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
    n = len(rows)
    dimension = len(rows[0])
    mean = [0.0] * dimension
    for row in rows:
        for k, v in enumerate(row):
            mean[k] += v
    mean = [m / n for m in mean]
    variance = [0.0] * dimension
    for row in rows:
        for k, v in enumerate(row):
            variance[k] += (v - mean[k]) ** 2
    scale = [math.sqrt(v / n) or 1.0 for v in variance]
    return mean, [s if s > 1e-12 else 1.0 for s in scale]


def _standardize(
    row: Sequence[float], mean: Sequence[float], scale: Sequence[float]
) -> List[float]:
    if not mean:
        return list(row)
    return [(v - m) / s for v, m, s in zip(row, mean, scale)]
