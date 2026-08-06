"""Reward functions and lazy objective scorers for TTS post-training.

The scalar mappings in this module follow recent TTS alignment work while
remaining independent of any particular trainer.  Pure reward functions have
no third-party dependencies.  Optional ASR, speaker, and quality models are
loaded only when their scorer's :meth:`load` method (or a scoring method) is
called.

Primary references:

* Inworld TTS-1 (WER and speaker rewards): https://arxiv.org/abs/2507.21138
* GRPO for TTS (CER/NLL harmonic reward): https://arxiv.org/abs/2509.18798
* Align2Speak (piecewise reward normalization): https://arxiv.org/abs/2509.21718
* RL for audio LLMs (duration/F0 rewards): https://arxiv.org/abs/2509.18569
* Koel-TTS (Pareto preference filtering):
  https://aclanthology.org/2025.emnlp-main.1076/
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from numbers import Integral
from typing import Any, TypeVar

PRIMARY_SOURCES = {
    "inworld_tts1": "https://arxiv.org/abs/2507.21138",
    "grpo_tts": "https://arxiv.org/abs/2509.18798",
    "align2speak": "https://arxiv.org/abs/2509.21718",
    "audio_llm_rl": "https://arxiv.org/abs/2509.18569",
    "koel_tts": "https://aclanthology.org/2025.emnlp-main.1076/",
    "multidimensional_preference_optimization": "https://arxiv.org/abs/2509.00685",
}

_Token = TypeVar("_Token")
TextNormalizer = Callable[[str], str]


def basic_text_normalize(text: str) -> str:
    """Apply Unicode-aware case, punctuation, and whitespace normalization."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = "".join(
        " " if unicodedata.category(character).startswith(("P", "Z")) else character
        for character in normalized
    )
    return " ".join(normalized.split())


def resolve_text_normalizer(value: TextNormalizer | str | None) -> TextNormalizer | None:
    """Resolve a callable or the ``basic``/``none`` configuration presets."""

    if value is None or (isinstance(value, str) and value.lower() in {"none", "raw"}):
        return None
    if callable(value):
        return value
    if isinstance(value, str) and value.lower() in {"basic", "multilingual"}:
        return basic_text_normalize
    raise ValueError("text normalizer must be callable, 'basic', 'multilingual', or 'none'")


class RewardScorerError(RuntimeError):
    """Raised when an optional objective scorer cannot produce a score."""


class OptionalDependencyError(ImportError):
    """Raised when a lazily requested scoring dependency is unavailable."""


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _nonnegative(value: float, name: str) -> float:
    value = _finite(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return value


def _positive(value: float, name: str) -> float:
    value = _finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value


def levenshtein_distance(reference: Sequence[_Token], hypothesis: Sequence[_Token]) -> int:
    """Return insertion, deletion, and substitution edit distance.

    The implementation retains only one dynamic-programming row, making its
    memory use linear in the shorter sequence.

    Args:
        reference: Ground-truth token sequence.
        hypothesis: Predicted token sequence.

    Returns:
        The non-negative Levenshtein distance.
    """

    if len(reference) < len(hypothesis):
        short, long = reference, hypothesis
    else:
        short, long = hypothesis, reference

    previous = list(range(len(short) + 1))
    for long_index, long_token in enumerate(long, start=1):
        current = [long_index]
        for short_index, short_token in enumerate(short, start=1):
            insertion = current[-1] + 1
            deletion = previous[short_index] + 1
            substitution = previous[short_index - 1] + (short_token != long_token)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def word_error_rate(
    reference: str,
    hypothesis: str,
    *,
    normalizer: TextNormalizer | None = None,
) -> float:
    """Compute WER using whitespace-delimited words.

    No case, punctuation, or language normalization is applied implicitly.
    Pass the same ``normalizer`` used by the evaluation pipeline when needed.
    For an empty reference, the denominator is one so insertions remain
    penalized; two inserted words therefore produce a score of ``2.0``.
    """

    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise TypeError("reference and hypothesis must be strings")
    if normalizer is not None:
        reference = normalizer(reference)
        hypothesis = normalizer(hypothesis)
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    distance = levenshtein_distance(reference_words, hypothesis_words)
    return distance / max(1, len(reference_words))


def character_error_rate(
    reference: str,
    hypothesis: str,
    *,
    normalizer: TextNormalizer | None = None,
    ignore_whitespace: bool = True,
) -> float:
    """Compute Unicode character error rate.

    Whitespace is ignored by default, matching common multilingual TTS CER
    evaluation.  Disable ``ignore_whitespace`` when spacing is part of the
    target orthography.
    """

    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise TypeError("reference and hypothesis must be strings")
    if normalizer is not None:
        reference = normalizer(reference)
        hypothesis = normalizer(hypothesis)
    if ignore_whitespace:
        reference = "".join(character for character in reference if not character.isspace())
        hypothesis = "".join(character for character in hypothesis if not character.isspace())
    distance = levenshtein_distance(reference, hypothesis)
    return distance / max(1, len(reference))


# Conventional metric aliases for trainer integrations.
compute_wer = word_error_rate
compute_cer = character_error_rate


def wer_exponential_reward(wer: float, k: float = 2.5) -> float:
    """Map WER to ``exp(-k * WER)`` (Inworld TTS-1, default ``k=2.5``)."""

    wer = _nonnegative(wer, "wer")
    k = _positive(k, "k")
    return math.exp(-k * wer)


def cer_tanh_reward(cer: float, alpha: float = 3.0) -> float:
    """Map CER to ``1 - tanh(alpha * CER)`` (GRPO for TTS)."""

    cer = _nonnegative(cer, "cer")
    alpha = _positive(alpha, "alpha")
    return 1.0 - math.tanh(alpha * cer)


def asr_nll_reward(nll: float, alpha: float = 3.0) -> float:
    """Map teacher-forced ASR NLL to ``exp(-NLL / alpha)``.

    The paper writes NLL as a token sum but does not disclose the reduction
    used in its reward implementation.  Prefer mean token NLL with the default
    ``alpha=3`` so ordinary transcript lengths do not collapse the reward.
    """

    nll = _nonnegative(nll, "nll")
    alpha = _positive(alpha, "alpha")
    return math.exp(-nll / alpha)


# Short aliases keep reward configuration code readable.
wer_reward = wer_exponential_reward
cer_reward = cer_tanh_reward
nll_reward = asr_nll_reward


def weighted_harmonic_mean(
    values: Sequence[float],
    weights: Sequence[float] | None = None,
) -> float:
    """Return the weighted harmonic mean of non-negative rewards.

    A positively weighted zero value makes the result zero.  Zero-weighted
    values are ignored.
    """

    if not values:
        raise ValueError("values must not be empty")
    if weights is None:
        weights = [1.0] * len(values)
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")

    denominator = 0.0
    total_weight = 0.0
    for index, (value, weight) in enumerate(zip(values, weights)):
        value = _nonnegative(value, f"values[{index}]")
        weight = _nonnegative(weight, f"weights[{index}]")
        if weight == 0.0:
            continue
        total_weight += weight
        if value == 0.0:
            return 0.0
        denominator += weight / value
    if total_weight == 0.0:
        raise ValueError("at least one weight must be positive")
    return total_weight / denominator


def weighted_harmonic_reward(
    cer_reward: float,
    nll_reward: float,
    *,
    cer_weight: float = 0.6,
    nll_weight: float = 0.4,
) -> float:
    """Combine CER and ASR-NLL rewards with the published harmonic formula."""

    return weighted_harmonic_mean(
        [cer_reward, nll_reward],
        [cer_weight, nll_weight],
    )


def asr_intelligibility_reward(
    cer: float,
    nll: float,
    *,
    cer_alpha: float = 3.0,
    nll_alpha: float = 3.0,
    cer_weight: float = 0.6,
    nll_weight: float = 0.4,
) -> float:
    """Map raw CER/NLL and combine them as proposed for TTS GRPO."""

    return weighted_harmonic_reward(
        cer_tanh_reward(cer, alpha=cer_alpha),
        asr_nll_reward(nll, alpha=nll_alpha),
        cer_weight=cer_weight,
        nll_weight=nll_weight,
    )


def piecewise_baseline_normalize(
    value: float,
    *,
    worst: float,
    baseline: float,
    best: float,
    higher_is_better: bool = True,
    clamp: bool = True,
) -> float:
    """Map ``worst -> 0``, ``baseline -> 0.5``, and ``best -> 1``.

    Align2Speak uses this piecewise-linear transformation to place metrics with
    different scales into a common reward range.  Set ``higher_is_better=False``
    for error and distance metrics.
    """

    value = _finite(value, "value")
    worst = _finite(worst, "worst")
    baseline = _finite(baseline, "baseline")
    best = _finite(best, "best")
    direction = 1.0 if higher_is_better else -1.0
    directed_value = direction * value
    directed_worst = direction * worst
    directed_baseline = direction * baseline
    directed_best = direction * best
    if not directed_worst < directed_baseline < directed_best:
        relation = "worst < baseline < best" if higher_is_better else "worst > baseline > best"
        raise ValueError(f"expected {relation}")

    if directed_value <= directed_baseline:
        normalized = 0.5 * (
            (directed_value - directed_worst) / (directed_baseline - directed_worst)
        )
    else:
        normalized = 0.5 + 0.5 * (
            (directed_value - directed_baseline) / (directed_best - directed_baseline)
        )
    if clamp:
        return min(1.0, max(0.0, normalized))
    return normalized


def speaker_cosine_reward(similarity: float, *, clamp: bool = True) -> float:
    """Normalize speaker cosine similarity with ``(similarity + 1) / 2``."""

    similarity = _finite(similarity, "similarity")
    if clamp:
        similarity = min(1.0, max(-1.0, similarity))
    elif not -1.0 <= similarity <= 1.0:
        raise ValueError("similarity must be in [-1, 1] when clamp=False")
    return (similarity + 1.0) / 2.0


def duration_reward(duration: float, target_duration: float) -> float:
    """Return the group-relative duration reward ``-|duration-target|/target``.

    The maximum is zero.  This is the exact duration term used in the 2025
    audio-LLM RL study when ``target_duration`` is the rollout group's median.
    """

    duration = _nonnegative(duration, "duration")
    target_duration = _positive(target_duration, "target_duration")
    return -abs(duration - target_duration) / target_duration


def duration_similarity_reward(
    duration: float,
    target_duration: float,
    *,
    alpha: float = 1.0,
) -> float:
    """Return a bounded ``exp(-alpha * relative_duration_error)`` reward."""

    alpha = _positive(alpha, "alpha")
    return math.exp(alpha * duration_reward(duration, target_duration))


def duration_tolerance_reward(
    duration: float,
    target_duration: float,
    *,
    minimum_ratio: float = 0.8,
    maximum_ratio: float = 1.2,
) -> float:
    """Return one when duration lies in a target ratio interval, else zero."""

    duration = _nonnegative(duration, "duration")
    target_duration = _positive(target_duration, "target_duration")
    minimum_ratio = _nonnegative(minimum_ratio, "minimum_ratio")
    maximum_ratio = _positive(maximum_ratio, "maximum_ratio")
    if minimum_ratio > maximum_ratio:
        raise ValueError("minimum_ratio must not exceed maximum_ratio")
    ratio = duration / target_duration
    return float(minimum_ratio <= ratio <= maximum_ratio)


def codec_validity_reward(
    token_ids: Sequence[int],
    *,
    codes_per_group: int,
    codebook_size: int,
    token_offset: int = 0,
    allow_partial_group: bool = False,
    strict: bool = False,
) -> float:
    """Score an interleaved codec sequence's structural and range validity.

    Token position ``i`` is expected in codebook ``i % codes_per_group`` and
    therefore in ``[offset + k*size, offset + (k+1)*size)``.  By default,
    trailing tokens that do not complete a frame group count as invalid.
    ``strict=True`` converts the fractional score into an all-or-nothing gate.
    Special BOS/EOS tokens must be removed before calling this function.
    """

    if not isinstance(codes_per_group, Integral) or codes_per_group <= 0:
        raise ValueError("codes_per_group must be a positive integer")
    if not isinstance(codebook_size, Integral) or codebook_size <= 0:
        raise ValueError("codebook_size must be a positive integer")
    if not isinstance(token_offset, Integral):
        raise TypeError("token_offset must be an integer")
    if not token_ids:
        return 0.0

    complete_length = len(token_ids) - (len(token_ids) % int(codes_per_group))
    valid = 0
    for position, token_id in enumerate(token_ids):
        if not isinstance(token_id, Integral):
            raise TypeError(f"token_ids[{position}] must be an integer")
        if position >= complete_length and not allow_partial_group:
            continue
        codebook = position % int(codes_per_group)
        lower = int(token_offset) + codebook * int(codebook_size)
        upper = lower + int(codebook_size)
        valid += int(lower <= int(token_id) < upper)

    score = valid / len(token_ids)
    if strict:
        return float(score == 1.0)
    return score


def coefficient_of_variation(
    values: Sequence[float],
    *,
    positive_only: bool = False,
) -> float:
    """Return population standard deviation divided by the absolute mean."""

    cleaned: list[float] = []
    for index, value in enumerate(values):
        value = _finite(value, f"values[{index}]")
        if positive_only and value <= 0.0:
            continue
        cleaned.append(value)
    if not cleaned:
        raise ValueError("values must contain at least one usable sample")
    mean = sum(cleaned) / len(cleaned)
    variance = sum((value - mean) ** 2 for value in cleaned) / len(cleaned)
    standard_deviation = math.sqrt(variance)
    if abs(mean) <= 1e-12:
        if standard_deviation <= 1e-12:
            return 0.0
        raise ValueError("coefficient of variation is undefined for a zero mean")
    return standard_deviation / abs(mean)


def cv_similarity_reward(
    values: Sequence[float],
    target_cv: float,
    *,
    alpha: float = 1.0,
    positive_only: bool = False,
) -> float:
    """Compare observed variation to a target CV with exponential decay."""

    observed_cv = coefficient_of_variation(values, positive_only=positive_only)
    target_cv = _nonnegative(target_cv, "target_cv")
    alpha = _positive(alpha, "alpha")
    if target_cv == 0.0:
        return 1.0 if observed_cv == 0.0 else 0.0
    relative_error = abs(observed_cv - target_cv) / target_cv
    return math.exp(-alpha * relative_error)


def f0_cv_reward(
    f0_values: Sequence[float],
    target_cv: float,
    *,
    alpha: float = 1.0,
) -> float:
    """Return CV similarity for F0, ignoring unvoiced (non-positive) frames."""

    return cv_similarity_reward(
        f0_values,
        target_cv,
        alpha=alpha,
        positive_only=True,
    )


def energy_cv_reward(
    energy_values: Sequence[float],
    target_cv: float,
    *,
    alpha: float = 1.0,
) -> float:
    """Return CV similarity for non-negative frame-energy values."""

    if any(_finite(value, f"energy_values[{index}]") < 0.0 for index, value in enumerate(energy_values)):
        raise ValueError("energy_values must be non-negative")
    return cv_similarity_reward(energy_values, target_cv, alpha=alpha)


def prosody_cv_reward(
    *,
    f0_values: Sequence[float] | None = None,
    energy_values: Sequence[float] | None = None,
    target_f0_cv: float | None = None,
    target_energy_cv: float | None = None,
    f0_weight: float = 0.5,
    energy_weight: float = 0.5,
    alpha: float = 1.0,
) -> float:
    """Combine any available F0 and energy CV similarity rewards."""

    values: list[float] = []
    weights: list[float] = []
    if f0_values is not None or target_f0_cv is not None:
        if f0_values is None or target_f0_cv is None:
            raise ValueError("f0_values and target_f0_cv must be provided together")
        values.append(f0_cv_reward(f0_values, target_f0_cv, alpha=alpha))
        weights.append(_nonnegative(f0_weight, "f0_weight"))
    if energy_values is not None or target_energy_cv is not None:
        if energy_values is None or target_energy_cv is None:
            raise ValueError("energy_values and target_energy_cv must be provided together")
        values.append(energy_cv_reward(energy_values, target_energy_cv, alpha=alpha))
        weights.append(_nonnegative(energy_weight, "energy_weight"))
    if not values:
        raise ValueError("at least one prosody component must be provided")
    total_weight = sum(weights)
    if total_weight == 0.0:
        raise ValueError("at least one active prosody weight must be positive")
    return sum(weight * value for weight, value in zip(weights, values)) / total_weight


Direction = bool | str


class CompositeReward:
    """Weighted reward with protected-metric and Pareto preference checks.

    Args:
        weights: Component weights in the linear reward.  Weights must be
            non-negative and at least one must be positive.
        maximize: Optional metric directions.  Values may be booleans or
            ``"max"``/``"min"`` strings; unspecified metrics are maximized.
        pareto_metrics: Metrics that a preferred candidate may not regress.
        protected_metrics: Metrics protected against regression relative to
            an incumbent, even when they are not part of the weighted score.
        protected_thresholds: Absolute minimum (maximize) or maximum
            (minimize) accepted values.
        protected_tolerances: Allowed regression per protected metric.
        normalize_weights: Divide the weighted sum by active weight mass.
        allow_missing: Ignore absent weighted components rather than raising.
    """

    def __init__(
        self,
        weights: Mapping[str, float],
        *,
        maximize: Mapping[str, Direction] | None = None,
        pareto_metrics: Iterable[str] = (),
        protected_metrics: Iterable[str] = (),
        protected_thresholds: Mapping[str, float] | None = None,
        protected_tolerances: Mapping[str, float] | None = None,
        normalize_weights: bool = False,
        allow_missing: bool = False,
        asr_scorer: Any = None,
        speaker_scorer: Any = None,
        quality_scorer: Any = None,
    ) -> None:
        if not weights:
            raise ValueError("weights must not be empty")
        self.weights = {
            str(name): _nonnegative(weight, f"weights[{name!r}]")
            for name, weight in weights.items()
        }
        if sum(self.weights.values()) == 0.0:
            raise ValueError("at least one component weight must be positive")
        self.maximize = {
            str(name): self._parse_direction(direction, str(name))
            for name, direction in (maximize or {}).items()
        }
        self.pareto_metrics = tuple(dict.fromkeys(str(name) for name in pareto_metrics))
        self.protected_metrics = tuple(dict.fromkeys(str(name) for name in protected_metrics))
        self.protected_thresholds = {
            str(name): _finite(value, f"protected_thresholds[{name!r}]")
            for name, value in (protected_thresholds or {}).items()
        }
        tolerances = protected_tolerances or {}
        self.protected_tolerances = {
            name: _nonnegative(tolerances.get(name, 0.0), f"protected_tolerances[{name!r}]")
            for name in self.protected_metrics
        }
        unknown_tolerances = set(tolerances) - set(self.protected_metrics)
        if unknown_tolerances:
            raise ValueError(
                "protected_tolerances contains non-protected metrics: "
                + ", ".join(sorted(unknown_tolerances))
            )
        self.normalize_weights = bool(normalize_weights)
        self.allow_missing = bool(allow_missing)
        self.asr_scorer = asr_scorer
        self.speaker_scorer = speaker_scorer
        self.quality_scorer = quality_scorer

    @staticmethod
    def _parse_direction(value: Direction, name: str) -> bool:
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"max", "maximize", "higher"}:
            return True
        if normalized in {"min", "minimize", "lower"}:
            return False
        raise ValueError(f"invalid direction for {name!r}: {value!r}")

    def _is_maximized(self, name: str) -> bool:
        return self.maximize.get(name, True)

    @staticmethod
    def _metric(metrics: Mapping[str, float], name: str) -> float:
        if name not in metrics:
            raise KeyError(f"missing reward metric: {name!r}")
        return _finite(metrics[name], f"metrics[{name!r}]")

    def aggregate(self, metrics: Mapping[str, float]) -> float:
        """Return the configured linear reward for one candidate."""

        weighted_sum = 0.0
        active_weight = 0.0
        for name, weight in self.weights.items():
            if name not in metrics:
                if self.allow_missing:
                    continue
                raise KeyError(f"missing weighted reward component: {name!r}")
            value = _finite(metrics[name], f"metrics[{name!r}]")
            weighted_sum += weight * value
            active_weight += weight
        if active_weight == 0.0:
            raise ValueError("no positive-weight reward components are active")
        if self.normalize_weights:
            return weighted_sum / active_weight
        return weighted_sum

    def score(
        self,
        audio: Any,
        target_text: str | None = None,
        sample_rate: int | None = None,
        reference_audio: Any = None,
        *,
        extra_metrics: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """Score generated audio and return component values plus ``total``.

        Args:
            audio: Generated waveform.  A numeric metric mapping is also
                accepted for offline aggregation when ``target_text`` and
                ``sample_rate`` are omitted.
            target_text: Text that the generated speech should realize.
            sample_rate: Waveform sample rate in Hz.
            reference_audio: Optional prompt/reference audio for speaker
                similarity.
            extra_metrics: Precomputed rewards such as duration, codec
                validity, or prosody.  These override backend values with the
                same name.

        Returns:
            A numeric reward breakdown containing a ``total`` key.
        """

        metrics: dict[str, float] = {}
        if isinstance(audio, Mapping) and target_text is None and sample_rate is None:
            for name, value in audio.items():
                metrics[str(name)] = _finite(value, f"metrics[{name!r}]")
        else:
            if target_text is None:
                raise ValueError("target_text is required when scoring audio")
            if sample_rate is None or sample_rate <= 0:
                raise ValueError("sample_rate must be positive when scoring audio")

            if self.asr_scorer is not None:
                try:
                    asr_scores = self.asr_scorer.score(
                        audio,
                        target_text,
                        sampling_rate=sample_rate,
                    )
                except Exception as error:
                    if isinstance(error, (RewardScorerError, OptionalDependencyError)):
                        raise
                    raise RewardScorerError("ASR reward scoring failed") from error
                self._merge_numeric(metrics, asr_scores, "ASR")

            if self.speaker_scorer is not None and reference_audio is not None:
                try:
                    speaker_scores = self.speaker_scorer.score(
                        audio,
                        reference_audio,
                        sampling_rate=sample_rate,
                    )
                except Exception as error:
                    if isinstance(error, (RewardScorerError, OptionalDependencyError)):
                        raise
                    raise RewardScorerError("speaker reward scoring failed") from error
                self._merge_numeric(metrics, speaker_scores, "speaker")

            if self.quality_scorer is not None:
                try:
                    quality_scores = self.quality_scorer.score(
                        audio,
                        sampling_rate=sample_rate,
                    )
                except Exception as error:
                    if isinstance(error, (RewardScorerError, OptionalDependencyError)):
                        raise
                    raise RewardScorerError("quality reward scoring failed") from error
                self._merge_numeric(metrics, quality_scores, "quality")

        if extra_metrics is not None:
            for name, value in extra_metrics.items():
                if name == "total":
                    raise ValueError("extra_metrics must not contain the reserved key 'total'")
                metrics[str(name)] = _finite(value, f"extra_metrics[{name!r}]")

        missing = [name for name in self.weights if name not in metrics and self.weights[name] > 0.0]
        if missing and not self.allow_missing:
            hints = []
            if any(name.startswith("speaker") for name in missing) and reference_audio is None:
                hints.append("reference_audio is required for speaker rewards")
            if self.asr_scorer is None and any(
                name in {"wer", "cer", "nll", "wer_reward", "cer_reward", "nll_reward", "intelligibility_reward"}
                for name in missing
            ):
                hints.append("configure an ASR scorer")
            if self.quality_scorer is None and any(
                name in {"stoi", "pesq", "si_sdr", "quality_reward"} for name in missing
            ):
                hints.append("configure a quality scorer")
            suffix = f" ({'; '.join(hints)})" if hints else ""
            raise RewardScorerError(
                "missing required reward components: " + ", ".join(missing) + suffix
            )

        metrics["total"] = self.aggregate(metrics)
        return metrics

    __call__ = score

    @staticmethod
    def _merge_numeric(
        destination: dict[str, float],
        source: Mapping[str, Any],
        source_name: str,
    ) -> None:
        if not isinstance(source, Mapping):
            raise RewardScorerError(f"{source_name} scorer must return a mapping")
        for name, value in source.items():
            if name == "total" or isinstance(value, (str, bytes)):
                continue
            try:
                destination[str(name)] = _finite(value, f"{source_name}[{name!r}]")
            except (TypeError, ValueError) as error:
                raise RewardScorerError(
                    f"{source_name} scorer returned a non-numeric value for {name!r}"
                ) from error

    def passes_protected_thresholds(self, metrics: Mapping[str, float]) -> bool:
        """Return whether all configured absolute protection gates pass."""

        for name, threshold in self.protected_thresholds.items():
            value = self._metric(metrics, name)
            if self._is_maximized(name):
                if value < threshold:
                    return False
            elif value > threshold:
                return False
        return True

    def preserves(
        self,
        candidate: Mapping[str, float],
        incumbent: Mapping[str, float],
    ) -> bool:
        """Return whether a candidate avoids protected-metric regressions."""

        for name in self.protected_metrics:
            candidate_value = self._metric(candidate, name)
            incumbent_value = self._metric(incumbent, name)
            tolerance = self.protected_tolerances[name]
            if self._is_maximized(name):
                if candidate_value < incumbent_value - tolerance:
                    return False
            elif candidate_value > incumbent_value + tolerance:
                return False
        return True

    def pareto_not_worse(
        self,
        candidate: Mapping[str, float],
        incumbent: Mapping[str, float],
        *,
        tolerance: float = 0.0,
    ) -> bool:
        """Return whether a candidate is no worse on every Pareto metric."""

        tolerance = _nonnegative(tolerance, "tolerance")
        for name in self.pareto_metrics:
            candidate_value = self._metric(candidate, name)
            incumbent_value = self._metric(incumbent, name)
            if self._is_maximized(name):
                if candidate_value < incumbent_value - tolerance:
                    return False
            elif candidate_value > incumbent_value + tolerance:
                return False
        return True

    def pareto_dominates(
        self,
        candidate: Mapping[str, float],
        incumbent: Mapping[str, float],
        *,
        tolerance: float = 0.0,
    ) -> bool:
        """Return true for no regression and one strict Pareto improvement."""

        if not self.pareto_metrics:
            raise ValueError("pareto_metrics must be configured")
        tolerance = _nonnegative(tolerance, "tolerance")
        if not self.pareto_not_worse(candidate, incumbent, tolerance=tolerance):
            return False
        for name in self.pareto_metrics:
            candidate_value = self._metric(candidate, name)
            incumbent_value = self._metric(incumbent, name)
            if self._is_maximized(name):
                if candidate_value > incumbent_value + tolerance:
                    return True
            elif candidate_value < incumbent_value - tolerance:
                return True
        return False

    def is_preferred(
        self,
        candidate: Mapping[str, float],
        incumbent: Mapping[str, float],
        *,
        score_tolerance: float = 0.0,
    ) -> bool:
        """Apply gates, Pareto constraints, then compare weighted rewards."""

        score_tolerance = _nonnegative(score_tolerance, "score_tolerance")
        if not self.passes_protected_thresholds(candidate):
            return False
        if not self.preserves(candidate, incumbent):
            return False
        if self.pareto_metrics and not self.pareto_not_worse(candidate, incumbent):
            return False
        return self.aggregate(candidate) > self.aggregate(incumbent) + score_tolerance

    def compare(
        self,
        left: Mapping[str, float],
        right: Mapping[str, float],
        *,
        score_tolerance: float = 0.0,
    ) -> int:
        """Return ``1`` for left, ``-1`` for right, or ``0`` for no safe winner."""

        left_preferred = self.is_preferred(left, right, score_tolerance=score_tolerance)
        right_preferred = self.is_preferred(right, left, score_tolerance=score_tolerance)
        if left_preferred == right_preferred:
            return 0
        return 1 if left_preferred else -1


def _require_torch() -> Any:
    try:
        import torch
    except (ImportError, OSError) as error:
        raise OptionalDependencyError(
            "objective scorers require PyTorch; install the project dependencies"
        ) from error
    return torch


def _resample_waveform(waveform: Any, source_rate: int, target_rate: int) -> Any:
    """Convert one waveform to mono and resample for a frozen reward model."""

    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("source and target sampling rates must be positive")
    torch = _require_torch()
    try:
        tensor = waveform if isinstance(waveform, torch.Tensor) else torch.as_tensor(waveform)
        tensor = tensor.detach().float().cpu()
        while tensor.ndim > 2 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim == 2:
            if tensor.shape[0] <= 8:
                tensor = tensor.mean(dim=0)
            elif tensor.shape[-1] <= 8:
                tensor = tensor.mean(dim=-1)
        if tensor.ndim != 1:
            raise RewardScorerError("reward waveform must contain one audio example")
        if source_rate == target_rate:
            return tensor
        import torchaudio.functional as audio_functional

        return audio_functional.resample(tensor, source_rate, target_rate)
    except RewardScorerError:
        raise
    except (ImportError, OSError) as error:
        raise OptionalDependencyError(
            "reward-model resampling requires a working torchaudio installation"
        ) from error
    except Exception as error:
        raise RewardScorerError("failed to resample reward waveform") from error


def _move_to_device(batch: Any, device: str) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
    raise RewardScorerError("processor output must be a mapping or expose .to(device)")


def _as_mapping(batch: Any) -> dict[str, Any]:
    if isinstance(batch, Mapping):
        return dict(batch)
    if hasattr(batch, "items"):
        return dict(batch.items())
    raise RewardScorerError("processor output is not a mapping")


class WhisperASRScorer:
    """Lazy Whisper transcription and teacher-forced NLL scorer.

    ``processor_loader`` and ``model_loader`` receive ``model_name`` and make
    the class mockable without downloads.  Supplying ready-made ``processor``
    and ``model`` objects is also supported.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        *,
        device: str = "cpu",
        processor: Any = None,
        model: Any = None,
        processor_loader: Callable[[str], Any] | None = None,
        model_loader: Callable[[str], Any] | None = None,
        generation_kwargs: Mapping[str, Any] | None = None,
        normalizer: TextNormalizer | str | None = "basic",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._processor = processor
        self._model = model
        self._processor_loader = processor_loader
        self._model_loader = model_loader
        self.generation_kwargs = dict(generation_kwargs or {})
        self.normalizer = resolve_text_normalizer(normalizer)

    @property
    def loaded(self) -> bool:
        """Whether both processor and model are available in memory."""

        return self._processor is not None and self._model is not None

    def load(self) -> WhisperASRScorer:
        """Load missing backend objects and return ``self``."""

        if self.loaded:
            return self
        if self._processor is None:
            loader = self._processor_loader
            if loader is None:
                try:
                    from transformers import AutoProcessor
                except (ImportError, OSError) as error:
                    raise OptionalDependencyError(
                        "Whisper scoring requires transformers"
                    ) from error
                loader = AutoProcessor.from_pretrained
            try:
                self._processor = loader(self.model_name)
            except Exception as error:
                raise RewardScorerError(
                    f"failed to load Whisper processor {self.model_name!r}"
                ) from error
        if self._model is None:
            loader = self._model_loader
            if loader is None:
                try:
                    from transformers import WhisperForConditionalGeneration
                except (ImportError, OSError) as error:
                    raise OptionalDependencyError(
                        "Whisper scoring requires transformers with Whisper support"
                    ) from error
                loader = WhisperForConditionalGeneration.from_pretrained
            try:
                self._model = loader(self.model_name)
            except Exception as error:
                raise RewardScorerError(
                    f"failed to load Whisper model {self.model_name!r}"
                ) from error
        try:
            if hasattr(self._model, "to"):
                self._model = self._model.to(self.device)
            if hasattr(self._model, "eval"):
                self._model.eval()
        except Exception as error:
            raise RewardScorerError("failed to prepare Whisper model") from error
        return self

    def _audio_inputs(self, waveform: Any, sampling_rate: int) -> dict[str, Any]:
        self.load()
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        try:
            feature_extractor = getattr(self._processor, "feature_extractor", self._processor)
            target_rate = int(getattr(feature_extractor, "sampling_rate", sampling_rate))
            waveform = _resample_waveform(waveform, sampling_rate, target_rate)
            batch = self._processor(
                waveform,
                sampling_rate=target_rate,
                return_tensors="pt",
            )
            batch = _move_to_device(batch, self.device)
            return _as_mapping(batch)
        except RewardScorerError:
            raise
        except Exception as error:
            raise RewardScorerError("Whisper processor failed on waveform") from error

    def transcribe(
        self,
        waveform: Any,
        *,
        sampling_rate: int,
        generation_kwargs: Mapping[str, Any] | None = None,
    ) -> str:
        """Transcribe one waveform with Whisper generation."""

        torch = _require_torch()
        inputs = self._audio_inputs(waveform, sampling_rate)
        kwargs = dict(self.generation_kwargs)
        kwargs.update(generation_kwargs or {})
        try:
            with torch.inference_mode():
                token_ids = self._model.generate(**inputs, **kwargs)
            decoded = self._processor.batch_decode(token_ids, skip_special_tokens=True)
        except Exception as error:
            raise RewardScorerError("Whisper transcription failed") from error
        if not decoded:
            raise RewardScorerError("Whisper returned no transcription")
        return str(decoded[0]).strip()

    def teacher_forced_nll(
        self,
        waveform: Any,
        reference_text: str,
        *,
        sampling_rate: int,
        reduction: str = "mean",
    ) -> float:
        """Return Whisper loss on the ground-truth text.

        ``mean`` is the production default because the paper does not disclose
        its implementation reduction and ``exp(-sum_NLL/3)`` rapidly collapses
        with transcript length.  ``sum`` is available to reproduce the sequence
        NLL written in the paper's equation.
        """

        if not reference_text:
            raise ValueError("reference_text must not be empty")
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        torch = _require_torch()
        inputs = self._audio_inputs(waveform, sampling_rate)
        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is None:
            raise RewardScorerError("Whisper processor does not expose a tokenizer")
        try:
            target = tokenizer(reference_text, return_tensors="pt")
            target = _move_to_device(target, self.device)
            labels = _as_mapping(target).get("input_ids")
            if labels is None:
                raise RewardScorerError("Whisper tokenizer returned no input_ids")
            with torch.inference_mode():
                output = self._model(**inputs, labels=labels)
            if not hasattr(output, "loss"):
                raise RewardScorerError("Whisper output does not contain loss")
            loss = float(output.loss.detach().float().item())
            if reduction == "sum":
                pad_token_id = getattr(tokenizer, "pad_token_id", None)
                if pad_token_id is None:
                    token_count = int(labels.numel())
                else:
                    token_count = int((labels != pad_token_id).sum().item())
                loss *= token_count
        except RewardScorerError:
            raise
        except Exception as error:
            raise RewardScorerError("Whisper teacher-forced NLL failed") from error
        return _nonnegative(loss, "Whisper NLL")

    def score(
        self,
        waveform: Any,
        reference_text: str,
        *,
        sampling_rate: int,
        normalizer: TextNormalizer | None = None,
    ) -> dict[str, str | float]:
        """Return transcript, edit metrics, NLL, and mapped rewards."""

        transcript = self.transcribe(waveform, sampling_rate=sampling_rate)
        active_normalizer = self.normalizer if normalizer is None else normalizer
        wer = word_error_rate(reference_text, transcript, normalizer=active_normalizer)
        cer = character_error_rate(reference_text, transcript, normalizer=active_normalizer)
        nll = self.teacher_forced_nll(
            waveform,
            reference_text,
            sampling_rate=sampling_rate,
        )
        cer_reward = cer_tanh_reward(cer)
        nll_reward = asr_nll_reward(nll)
        return {
            "transcript": transcript,
            "wer": wer,
            "cer": cer,
            "nll": nll,
            "wer_reward": wer_exponential_reward(wer),
            "cer_reward": cer_reward,
            "nll_reward": nll_reward,
            "intelligibility_reward": weighted_harmonic_reward(cer_reward, nll_reward),
        }


class WavLMSpeakerScorer:
    """Lazy WavLM x-vector speaker similarity scorer."""

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus-sv",
        *,
        device: str = "cpu",
        feature_extractor: Any = None,
        model: Any = None,
        feature_extractor_loader: Callable[[str], Any] | None = None,
        model_loader: Callable[[str], Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._feature_extractor = feature_extractor
        self._model = model
        self._feature_extractor_loader = feature_extractor_loader
        self._model_loader = model_loader

    @property
    def loaded(self) -> bool:
        """Whether both feature extractor and model are in memory."""

        return self._feature_extractor is not None and self._model is not None

    def load(self) -> WavLMSpeakerScorer:
        """Load missing backend objects and return ``self``."""

        if self.loaded:
            return self
        if self._feature_extractor is None:
            loader = self._feature_extractor_loader
            if loader is None:
                try:
                    from transformers import AutoFeatureExtractor
                except (ImportError, OSError) as error:
                    raise OptionalDependencyError(
                        "WavLM speaker scoring requires transformers"
                    ) from error
                loader = AutoFeatureExtractor.from_pretrained
            try:
                self._feature_extractor = loader(self.model_name)
            except Exception as error:
                raise RewardScorerError(
                    f"failed to load WavLM feature extractor {self.model_name!r}"
                ) from error
        if self._model is None:
            loader = self._model_loader
            if loader is None:
                try:
                    from transformers import WavLMForXVector
                except (ImportError, OSError) as error:
                    raise OptionalDependencyError(
                        "WavLM speaker scoring requires WavLMForXVector"
                    ) from error
                loader = WavLMForXVector.from_pretrained
            try:
                self._model = loader(self.model_name)
            except Exception as error:
                raise RewardScorerError(
                    f"failed to load WavLM model {self.model_name!r}"
                ) from error
        try:
            if hasattr(self._model, "to"):
                self._model = self._model.to(self.device)
            if hasattr(self._model, "eval"):
                self._model.eval()
        except Exception as error:
            raise RewardScorerError("failed to prepare WavLM model") from error
        return self

    def embedding(self, waveform: Any, *, sampling_rate: int) -> Any:
        """Return a unit-normalized speaker embedding for one waveform."""

        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        torch = _require_torch()
        self.load()
        try:
            target_rate = int(
                getattr(self._feature_extractor, "sampling_rate", sampling_rate)
            )
            waveform = _resample_waveform(waveform, sampling_rate, target_rate)
            inputs = self._feature_extractor(
                waveform,
                sampling_rate=target_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = _move_to_device(inputs, self.device)
            with torch.inference_mode():
                output = self._model(**_as_mapping(inputs))
            embedding = getattr(output, "embeddings", None)
            if embedding is None and isinstance(output, (tuple, list)) and output:
                embedding = output[0]
            if embedding is None:
                raise RewardScorerError("WavLM output does not contain embeddings")
            return torch.nn.functional.normalize(embedding, dim=-1)
        except RewardScorerError:
            raise
        except Exception as error:
            raise RewardScorerError("WavLM embedding extraction failed") from error

    def similarity(
        self,
        candidate_waveform: Any,
        reference_waveform: Any,
        *,
        sampling_rate: int,
    ) -> float:
        """Return raw cosine similarity between candidate and reference."""

        torch = _require_torch()
        candidate = self.embedding(candidate_waveform, sampling_rate=sampling_rate)
        reference = self.embedding(reference_waveform, sampling_rate=sampling_rate)
        if candidate.shape != reference.shape:
            raise RewardScorerError(
                f"speaker embedding shapes differ: {candidate.shape} != {reference.shape}"
            )
        cosine = torch.nn.functional.cosine_similarity(candidate, reference, dim=-1)
        return _finite(float(cosine.mean().item()), "speaker cosine similarity")

    def score(
        self,
        candidate_waveform: Any,
        reference_waveform: Any,
        *,
        sampling_rate: int,
    ) -> dict[str, float]:
        """Return raw cosine similarity and its normalized reward."""

        similarity = self.similarity(
            candidate_waveform,
            reference_waveform,
            sampling_rate=sampling_rate,
        )
        return {
            "speaker_cosine": similarity,
            "speaker_reward": speaker_cosine_reward(similarity),
        }


class SQUIMObjectiveScorer:
    """Lazy torchaudio SQUIM objective quality scorer.

    Torchaudio's objective bundle estimates STOI, wide-band PESQ, and SI-SDR.
    The default scalar reward is clipped ``PESQ / 4.5``, as used for the
    reference-free quality component in Align2Speak.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        target_sampling_rate: int = 16000,
        model: Any = None,
        model_loader: Callable[[], Any] | None = None,
    ) -> None:
        if target_sampling_rate <= 0:
            raise ValueError("target_sampling_rate must be positive")
        self.device = device
        self.target_sampling_rate = int(target_sampling_rate)
        self._model = model
        self._model_loader = model_loader
        self._torchaudio = None

    @property
    def loaded(self) -> bool:
        """Whether the SQUIM model is in memory."""

        return self._model is not None

    def load(self) -> SQUIMObjectiveScorer:
        """Load the SQUIM objective model and return ``self``."""

        if self.loaded:
            return self
        loader = self._model_loader
        if loader is None:
            try:
                import torchaudio
            except (ImportError, OSError) as error:
                raise OptionalDependencyError(
                    "SQUIM quality scoring requires a working torchaudio installation"
                ) from error
            self._torchaudio = torchaudio
            try:
                loader = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model
            except AttributeError as error:
                raise OptionalDependencyError(
                    "installed torchaudio does not provide SQUIM_OBJECTIVE"
                ) from error
        try:
            self._model = loader()
            if hasattr(self._model, "to"):
                self._model = self._model.to(self.device)
            if hasattr(self._model, "eval"):
                self._model.eval()
        except Exception as error:
            raise RewardScorerError("failed to load torchaudio SQUIM_OBJECTIVE") from error
        return self

    def _prepare_waveform(self, waveform: Any, sampling_rate: int) -> Any:
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        torch = _require_torch()
        try:
            tensor = waveform if isinstance(waveform, torch.Tensor) else torch.as_tensor(waveform)
            tensor = tensor.detach().float()
            while tensor.ndim > 2 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 2:
                raise RewardScorerError(
                    "waveform must have shape [time], [channel, time], or singleton batches"
                )
            if tensor.shape[0] > 1:
                tensor = tensor.mean(dim=0, keepdim=True)
            if sampling_rate != self.target_sampling_rate:
                if self._torchaudio is None:
                    try:
                        import torchaudio
                    except (ImportError, OSError) as error:
                        raise OptionalDependencyError(
                            "resampling for SQUIM requires torchaudio"
                        ) from error
                    self._torchaudio = torchaudio
                tensor = self._torchaudio.functional.resample(
                    tensor,
                    sampling_rate,
                    self.target_sampling_rate,
                )
            return tensor.to(self.device)
        except (RewardScorerError, OptionalDependencyError):
            raise
        except Exception as error:
            raise RewardScorerError("failed to prepare waveform for SQUIM") from error

    @staticmethod
    def _scalar(value: Any, name: str) -> float:
        if hasattr(value, "detach"):
            value = value.detach().float().mean().item()
        return _finite(value, name)

    def objective_scores(self, waveform: Any, *, sampling_rate: int) -> dict[str, float]:
        """Return raw ``stoi``, ``pesq``, and ``si_sdr`` estimates."""

        torch = _require_torch()
        self.load()
        tensor = self._prepare_waveform(waveform, sampling_rate)
        try:
            with torch.inference_mode():
                output = self._model(tensor)
            if not isinstance(output, (tuple, list)) or len(output) != 3:
                raise RewardScorerError(
                    "SQUIM objective model must return (stoi, pesq, si_sdr)"
                )
            stoi, pesq, si_sdr = output
            return {
                "stoi": self._scalar(stoi, "SQUIM STOI"),
                "pesq": self._scalar(pesq, "SQUIM PESQ"),
                "si_sdr": self._scalar(si_sdr, "SQUIM SI-SDR"),
            }
        except RewardScorerError:
            raise
        except Exception as error:
            raise RewardScorerError("SQUIM objective scoring failed") from error

    def score(self, waveform: Any, *, sampling_rate: int) -> dict[str, float]:
        """Return raw objective estimates and a normalized PESQ reward."""

        scores = self.objective_scores(waveform, sampling_rate=sampling_rate)
        scores["quality_reward"] = min(1.0, max(0.0, scores["pesq"] / 4.5))
        return scores


# Readable alias for callers that prefer conventional class-name casing.
SquimObjectiveScorer = SQUIMObjectiveScorer


def _build_lazy_scorer(
    specification: Any,
    scorer_class: Any,
    *,
    device: str,
    label: str,
) -> Any:
    if specification is None or specification is False:
        return None
    if specification is True:
        kwargs: dict[str, Any] = {}
    elif isinstance(specification, Mapping):
        kwargs = dict(specification)
        if not bool(kwargs.pop("enabled", True)):
            return None
    elif hasattr(specification, "score"):
        return specification
    else:
        raise TypeError(
            f"{label} scorer config must be a mapping, boolean, or scorer object"
        )
    kwargs.setdefault("device", device)
    try:
        return scorer_class(**kwargs)
    except Exception as error:
        raise ValueError(f"invalid {label} scorer configuration") from error


def build_composite_reward(config: Mapping[str, Any]) -> CompositeReward:
    """Build an online :class:`CompositeReward` from a plain configuration.

    Models are instantiated as lazy scorer wrappers; this function never
    downloads or loads a checkpoint.  Scorers are auto-enabled only when their
    output appears in ``weights``.  They can be disabled explicitly with
    ``{"enabled": False}`` or replaced by a mock/object exposing ``score``.

    Example::

        reward = build_composite_reward({
            "weights": {
                "intelligibility_reward": 0.45,
                "speaker_reward": 0.45,
                "quality_reward": 0.10,
            },
            "device": "cuda",
            "pareto_metrics": ["wer_reward", "speaker_reward"],
            "protected_metrics": ["speaker_reward"],
            "asr": {"model_name": "openai/whisper-large-v3"},
            "speaker": {"model_name": "microsoft/wavlm-base-plus-sv"},
            "quality": True,
        })
    """

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if "weights" not in config:
        raise ValueError("config must contain reward component 'weights'")
    weights = config["weights"]
    if not isinstance(weights, Mapping):
        raise TypeError("config['weights'] must be a mapping")
    unsafe_raw_errors = {"wer", "cer", "nll"}.intersection(weights)
    if unsafe_raw_errors:
        raise ValueError(
            "raw error metrics cannot have positive reward weights; use mapped "
            + ", ".join(f"{name}_reward" for name in sorted(unsafe_raw_errors))
        )

    device = str(config.get("device", "cpu"))
    weight_names = {str(name) for name in weights}
    asr_names = {
        "wer_reward",
        "cer_reward",
        "nll_reward",
        "intelligibility_reward",
    }
    speaker_names = {"speaker_cosine", "speaker_reward"}
    quality_names = {"stoi", "pesq", "si_sdr", "quality_reward"}

    asr_specification = config.get(
        "asr_scorer",
        config.get("asr", bool(weight_names.intersection(asr_names))),
    )
    speaker_specification = config.get(
        "speaker_scorer",
        config.get("speaker", bool(weight_names.intersection(speaker_names))),
    )
    quality_specification = config.get(
        "quality_scorer",
        config.get("quality", bool(weight_names.intersection(quality_names))),
    )

    default_directions: dict[str, Direction] = {
        "wer": "min",
        "cer": "min",
        "nll": "min",
        "duration_error": "min",
    }
    configured_directions = config.get("maximize") or {}
    if not isinstance(configured_directions, Mapping):
        raise TypeError("config['maximize'] must be a mapping")
    default_directions.update(configured_directions)

    return CompositeReward(
        weights,
        maximize=default_directions,
        pareto_metrics=config.get("pareto_metrics", ()),
        protected_metrics=config.get("protected_metrics", ()),
        protected_thresholds=config.get("protected_thresholds"),
        protected_tolerances=config.get("protected_tolerances"),
        normalize_weights=bool(config.get("normalize_weights", False)),
        allow_missing=bool(config.get("allow_missing", False)),
        asr_scorer=_build_lazy_scorer(
            asr_specification,
            WhisperASRScorer,
            device=device,
            label="ASR",
        ),
        speaker_scorer=_build_lazy_scorer(
            speaker_specification,
            WavLMSpeakerScorer,
            device=device,
            label="speaker",
        ),
        quality_scorer=_build_lazy_scorer(
            quality_specification,
            SQUIMObjectiveScorer,
            device=device,
            label="quality",
        ),
    )


__all__ = [
    "PRIMARY_SOURCES",
    "CompositeReward",
    "OptionalDependencyError",
    "RewardScorerError",
    "SQUIMObjectiveScorer",
    "SquimObjectiveScorer",
    "WavLMSpeakerScorer",
    "WhisperASRScorer",
    "asr_intelligibility_reward",
    "asr_nll_reward",
    "basic_text_normalize",
    "build_composite_reward",
    "cer_reward",
    "cer_tanh_reward",
    "character_error_rate",
    "codec_validity_reward",
    "coefficient_of_variation",
    "compute_cer",
    "compute_wer",
    "cv_similarity_reward",
    "duration_reward",
    "duration_similarity_reward",
    "duration_tolerance_reward",
    "energy_cv_reward",
    "f0_cv_reward",
    "levenshtein_distance",
    "nll_reward",
    "piecewise_baseline_normalize",
    "prosody_cv_reward",
    "resolve_text_normalizer",
    "speaker_cosine_reward",
    "weighted_harmonic_mean",
    "weighted_harmonic_reward",
    "wer_exponential_reward",
    "wer_reward",
    "word_error_rate",
]
