"""Samplers: the only thing you have to supply.

A *sampler* draws N independent generations for one prompt. The library accepts
almost any shape of callable and normalises it, because the point of the
integration is that you keep your own model client:

* ``f(prompt, n) -> list[str]``            - batch sampler (preferred)
* ``f(prompt) -> str``                     - single-shot; called ``n`` times
* ``f(prompt, n) -> list[(str, logprob)]`` - white-box, enables Rao-Blackwell
* ``f(prompt, n) -> list[Sample]``         - full control

Confabulation is only exposed by *independent* draws at non-zero temperature —
Farquhar et al. use ~1.0. A greedy or cached sampler will always report zero
entropy, so :func:`resolve_sampler` refuses to score a set where every generation
is byte-identical *and* the caller asked for a temperature warning.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, List, Optional, Sequence

from .errors import SamplingError
from .types import Sample, normalize_texts

__all__ = [
    "Sampler",
    "resolve_sampler",
    "from_texts",
    "openai_sampler",
    "FixedSampler",
]

Sampler = Callable[..., Sequence[Any]]


def resolve_sampler(sampler: Sampler) -> Callable[[str, int], List[Sample]]:
    """Wrap any accepted sampler shape into ``(prompt, n) -> list[Sample]``.

    Arity is detected by signature inspection, falling back to a call-and-retry
    if introspection fails (builtins, C callables, ``functools.partial`` chains).
    """
    if not callable(sampler):
        raise SamplingError(f"sampler must be callable, got {type(sampler).__name__}")

    takes_n = _accepts_two_positional(sampler)

    def call(prompt: str, n: int) -> List[Sample]:
        if takes_n:
            raw = sampler(prompt, n)
            if raw is None:
                raise SamplingError("sampler returned None")
            if isinstance(raw, (str, bytes)):
                raise SamplingError(
                    "sampler(prompt, n) returned a single string; it must return a "
                    "sequence of n generations"
                )
            samples = normalize_texts(list(raw))
        else:
            samples = normalize_texts([sampler(prompt) for _ in range(n)])
        if not samples:
            raise SamplingError("sampler produced no generations")
        return samples

    return call


def _accepts_two_positional(fn: Callable[..., Any]) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return True
    positional = 0
    for param in sig.parameters.values():
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            positional += 1
        elif param.kind is param.VAR_POSITIONAL:
            return True
    return positional >= 2


class FixedSampler:
    """Replay a canned list of generations. Deterministic, offline, no network.

    Used by the test suite, ``sem-gate demo`` and by batch scoring of a JSONL file
    whose rows already contain the generations.
    """

    def __init__(self, texts: Sequence[Any]) -> None:
        self.samples = normalize_texts(list(texts))

    def __call__(self, prompt: str, n: int) -> List[Sample]:
        if n >= len(self.samples):
            return list(self.samples)
        return list(self.samples[:n])


def from_texts(texts: Sequence[Any]) -> FixedSampler:
    """Build a :class:`FixedSampler` from pre-generated outputs."""
    return FixedSampler(texts)


def openai_sampler(
    client: Any,
    model: str,
    *,
    temperature: float = 1.0,
    max_tokens: int = 256,
    system: Optional[str] = None,
    logprobs: bool = True,
    extra: Optional[dict] = None,
) -> Callable[[str, int], List[Sample]]:
    """Adapter for an OpenAI-compatible chat client (OpenAI, Azure, vLLM, Together...).

    Requests ``n`` choices in a single call and, when the endpoint supports it,
    harvests token log-probabilities so the estimator can use the
    Rao-Blackwellised (white-box) formulation. If log-probabilities come back
    empty the samples simply carry ``logprob=None`` and the discrete estimator
    takes over — no error, just a less precise score, which is recorded in the
    result's ``estimator`` field.

    ``client`` is anything exposing ``client.chat.completions.create(...)``; the
    library never imports ``openai`` itself.
    """

    def sample(prompt: str, n: int) -> List[Sample]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": n,
        }
        if logprobs:
            kwargs["logprobs"] = True
        if extra:
            kwargs.update(extra)
        try:
            response = client.chat.completions.create(**kwargs)
        except TypeError:
            kwargs.pop("logprobs", None)
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise SamplingError(f"OpenAI-compatible sampler failed: {exc}") from exc

        out: List[Sample] = []
        for choice in response.choices:
            text = (getattr(choice.message, "content", None) or "").strip()
            mean_lp, n_tokens = _extract_logprob(choice)
            out.append(Sample(text=text, logprob=mean_lp, tokens=n_tokens, raw=choice))
        return out

    return sample


def _extract_logprob(choice: Any) -> "tuple[Optional[float], Optional[int]]":
    """Mean token log-probability of a choice, or ``(None, None)``.

    Length normalisation happens right here so that no downstream code ever sees
    an un-normalised sequence likelihood.
    """
    container = getattr(choice, "logprobs", None)
    tokens = getattr(container, "content", None) if container is not None else None
    if not tokens:
        return None, None
    values = [
        getattr(t, "logprob", None) for t in tokens if getattr(t, "logprob", None) is not None
    ]
    if not values:
        return None, None
    return sum(values) / len(values), len(values)
