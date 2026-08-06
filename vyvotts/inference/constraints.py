"""Token grammar helpers for autoregressive audio generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from transformers import LogitsProcessor


class AudioTokenSequenceError(ValueError):
    """Raised when generated audio tokens do not follow the codec grammar."""


def extract_audio_codes(
    sequence: Sequence[int] | torch.Tensor,
    start_of_speech: int,
    end_of_speech: int,
    audio_tokens_start: int,
    codes_per_group: int,
    codebook_size: int,
    strict: bool = True,
) -> list[int]:
    """Extract and validate one row of interleaved codec tokens.

    The returned codes retain their per-codebook offsets but no longer include
    ``audio_tokens_start``. Invalid or incomplete frames are rejected instead
    of being coerced into a decodable range.
    """
    if codes_per_group <= 0 or codebook_size <= 0:
        raise ValueError("codec dimensions must be positive")

    if isinstance(sequence, torch.Tensor):
        values = [int(token) for token in sequence.detach().cpu().tolist()]
    else:
        values = [int(token) for token in sequence]

    speech_starts = [index for index, token in enumerate(values) if token == start_of_speech]
    if not speech_starts:
        if strict:
            raise AudioTokenSequenceError("missing START_OF_SPEECH token")
        return []

    speech_tokens = values[speech_starts[-1] + 1 :]
    try:
        speech_end = speech_tokens.index(end_of_speech)
    except ValueError as exc:
        if strict:
            raise AudioTokenSequenceError("missing END_OF_SPEECH token") from exc
        speech_end = len(speech_tokens)
    audio_tokens = speech_tokens[:speech_end]

    if not audio_tokens:
        if strict:
            raise AudioTokenSequenceError("audio sequence contains no codec frames")
        return []
    if len(audio_tokens) % codes_per_group:
        if strict:
            raise AudioTokenSequenceError(
                f"audio sequence has {len(audio_tokens)} tokens; expected a multiple "
                f"of {codes_per_group}"
            )
        audio_tokens = audio_tokens[: len(audio_tokens) // codes_per_group * codes_per_group]
        if not audio_tokens:
            return []

    codes = []
    for position, token in enumerate(audio_tokens):
        phase = position % codes_per_group
        lower = audio_tokens_start + phase * codebook_size
        upper = lower + codebook_size
        if not lower <= token < upper:
            if strict:
                raise AudioTokenSequenceError(
                    f"token {token} at audio position {position} is outside "
                    f"phase-{phase} range [{lower}, {upper})"
                )
            return []
        codes.append(token - audio_tokens_start)

    return codes


def include_end_token_for_stop(
    token_ids: Sequence[int],
    end_of_speech: int,
    finish_reason: Any,
) -> list[int]:
    """Restore a backend-excluded stop token without masking length truncation."""

    values = [int(token) for token in token_ids]
    if end_of_speech in values:
        return values
    reason_type = finish_reason
    matched = None
    if isinstance(finish_reason, Mapping):
        reason_type = finish_reason.get("type", finish_reason.get("reason"))
        matched = finish_reason.get("matched", finish_reason.get("stop_reason"))
    stopped = str(reason_type).lower() in {"stop", "eos", "eos_token"}
    if not stopped:
        return values
    if isinstance(matched, int) and matched != end_of_speech:
        return values
    return [*values, int(end_of_speech)]


class AudioTokenLogitsProcessor(LogitsProcessor):
    """Enforce the assistant/speech header and interleaved codec grammar.

    Generation is constrained to ``START_OF_AI``, then ``START_OF_SPEECH``,
    followed by the token range belonging to each codebook phase. Speech may
    end only after at least ``min_audio_frames`` complete frame groups.
    """

    def __init__(
        self,
        *,
        prompt_length: int,
        start_of_ai: int,
        start_of_speech: int,
        end_of_speech: int,
        audio_tokens_start: int,
        codes_per_group: int,
        codebook_size: int,
        pad_token_id: int | None = None,
        min_audio_frames: int = 1,
    ) -> None:
        if prompt_length < 0:
            raise ValueError("prompt_length must be non-negative")
        if codes_per_group <= 0 or codebook_size <= 0:
            raise ValueError("codec dimensions must be positive")
        if min_audio_frames <= 0:
            raise ValueError("min_audio_frames must be positive")
        if len({start_of_ai, start_of_speech, end_of_speech}) != 3:
            raise ValueError("assistant and speech boundary token IDs must be distinct")

        audio_tokens_end = audio_tokens_start + codes_per_group * codebook_size
        if any(
            audio_tokens_start <= token_id < audio_tokens_end
            for token_id in (start_of_ai, start_of_speech, end_of_speech)
        ):
            raise ValueError("speech boundary token IDs must not overlap codec token ranges")

        self.prompt_length = prompt_length
        self.start_of_ai = start_of_ai
        self.start_of_speech = start_of_speech
        self.end_of_speech = end_of_speech
        self.audio_tokens_start = audio_tokens_start
        self.codes_per_group = codes_per_group
        self.codebook_size = codebook_size
        self.pad_token_id = pad_token_id
        self.min_audio_tokens = min_audio_frames * codes_per_group

    def _validate_vocab(self, vocab_size: int) -> None:
        last_audio_token = (
            self.audio_tokens_start
            + self.codes_per_group * self.codebook_size
            - 1
        )
        required_ids = [
            self.start_of_ai,
            self.start_of_speech,
            self.end_of_speech,
            last_audio_token,
        ]
        if self.pad_token_id is not None:
            required_ids.append(self.pad_token_id)
        if min(required_ids) < 0 or max(required_ids) >= vocab_size:
            raise ValueError(
                f"audio token grammar requires token ID {max(required_ids)}, "
                f"but model vocabulary size is {vocab_size}"
            )

    @staticmethod
    def _copy_token_score(
        constrained: torch.Tensor,
        scores: torch.Tensor,
        row_index: int,
        token_id: int,
    ) -> None:
        constrained[row_index, token_id] = scores[row_index, token_id]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.ndim != 2 or scores.ndim != 2:
            raise ValueError("input_ids and scores must both be rank-2 tensors")
        if input_ids.shape[0] != scores.shape[0]:
            raise ValueError("input_ids and scores batch sizes do not match")
        if input_ids.shape[1] < self.prompt_length:
            raise ValueError("input_ids are shorter than the configured prompt")

        self._validate_vocab(scores.shape[-1])
        constrained = torch.full_like(scores, -torch.inf)

        for row_index, row in enumerate(input_ids):
            suffix = [int(token) for token in row[self.prompt_length :].detach().cpu().tolist()]

            if suffix and suffix[0] != self.start_of_ai:
                raise AudioTokenSequenceError("generation does not start with START_OF_AI")
            if len(suffix) >= 2 and suffix[1] != self.start_of_speech:
                raise AudioTokenSequenceError("START_OF_AI is not followed by START_OF_SPEECH")

            if self.end_of_speech in suffix:
                end_index = suffix.index(self.end_of_speech)
                audio_count = end_index - 2
                if audio_count < self.min_audio_tokens or audio_count % self.codes_per_group:
                    raise AudioTokenSequenceError("END_OF_SPEECH occurs before a complete audio frame")
                terminal_id = self.pad_token_id if self.pad_token_id is not None else self.end_of_speech
                self._copy_token_score(constrained, scores, row_index, terminal_id)
                continue

            if not suffix:
                self._copy_token_score(constrained, scores, row_index, self.start_of_ai)
                continue
            if len(suffix) == 1:
                self._copy_token_score(constrained, scores, row_index, self.start_of_speech)
                continue

            audio_tokens = suffix[2:]
            for position, token in enumerate(audio_tokens):
                phase = position % self.codes_per_group
                lower = self.audio_tokens_start + phase * self.codebook_size
                upper = lower + self.codebook_size
                if not lower <= token < upper:
                    raise AudioTokenSequenceError(
                        f"existing token {token} violates phase-{phase} range [{lower}, {upper})"
                    )

            phase = len(audio_tokens) % self.codes_per_group
            lower = self.audio_tokens_start + phase * self.codebook_size
            upper = lower + self.codebook_size
            constrained[row_index, lower:upper] = scores[row_index, lower:upper]

            if len(audio_tokens) >= self.min_audio_tokens and phase == 0:
                self._copy_token_score(constrained, scores, row_index, self.end_of_speech)

        return constrained
