import pytest
import torch

from vyvotts.inference.base import BaseVyvoTTSInference
from vyvotts.inference.constraints import (
    AudioTokenLogitsProcessor,
    AudioTokenSequenceError,
    extract_audio_codes,
    include_end_token_for_stop,
)
from vyvotts.voice_clone import VyvoTTSVoiceClone

PROMPT_LENGTH = 2
START_OF_AI = 6
START_OF_SPEECH = 7
END_OF_SPEECH = 8
AUDIO_TOKENS_START = 20
CODES_PER_GROUP = 2
CODEBOOK_SIZE = 3
PAD_TOKEN = 5
VOCAB_SIZE = 40


def _processor():
    return AudioTokenLogitsProcessor(
        prompt_length=PROMPT_LENGTH,
        start_of_ai=START_OF_AI,
        start_of_speech=START_OF_SPEECH,
        end_of_speech=END_OF_SPEECH,
        audio_tokens_start=AUDIO_TOKENS_START,
        codes_per_group=CODES_PER_GROUP,
        codebook_size=CODEBOOK_SIZE,
        pad_token_id=PAD_TOKEN,
    )


def _allowed_tokens(processor, input_ids):
    scores = torch.zeros((input_ids.shape[0], VOCAB_SIZE))
    constrained = processor(input_ids, scores)
    return torch.isfinite(constrained[0]).nonzero(as_tuple=True)[0].tolist()


def test_logits_processor_forces_headers_and_codebook_phases():
    processor = _processor()

    prompt = torch.tensor([[1, 2]])
    assert _allowed_tokens(processor, prompt) == [START_OF_AI]

    with_assistant = torch.tensor([[1, 2, START_OF_AI]])
    assert _allowed_tokens(processor, with_assistant) == [START_OF_SPEECH]

    phase_zero = torch.tensor([[1, 2, START_OF_AI, START_OF_SPEECH]])
    assert _allowed_tokens(processor, phase_zero) == [20, 21, 22]

    phase_one = torch.tensor([[1, 2, START_OF_AI, START_OF_SPEECH, 21]])
    assert _allowed_tokens(processor, phase_one) == [23, 24, 25]


def test_logits_processor_allows_eos_only_on_complete_frame_boundary():
    processor = _processor()
    complete_frame = torch.tensor(
        [[1, 2, START_OF_AI, START_OF_SPEECH, 21, 24]]
    )
    assert _allowed_tokens(processor, complete_frame) == [
        END_OF_SPEECH,
        20,
        21,
        22,
    ]

    early_eos = torch.tensor(
        [[1, 2, START_OF_AI, START_OF_SPEECH, 21, END_OF_SPEECH]]
    )
    with pytest.raises(AudioTokenSequenceError):
        _allowed_tokens(processor, early_eos)


def test_logits_processor_rejects_existing_wrong_phase_token():
    wrong_phase = torch.tensor(
        [[1, 2, START_OF_AI, START_OF_SPEECH, 24]]
    )
    with pytest.raises(AudioTokenSequenceError):
        _allowed_tokens(_processor(), wrong_phase)


@pytest.mark.parametrize(
    ("codes_per_group", "codebook_size"),
    [(7, 4096), (8, 2048)],
)
def test_logits_processor_supports_snac_and_mimi_layouts(
    codes_per_group, codebook_size
):
    audio_start = 100
    processor = AudioTokenLogitsProcessor(
        prompt_length=1,
        start_of_ai=2,
        start_of_speech=3,
        end_of_speech=4,
        audio_tokens_start=audio_start,
        codes_per_group=codes_per_group,
        codebook_size=codebook_size,
        min_audio_frames=1,
    )
    frame = [audio_start + phase * codebook_size for phase in range(codes_per_group)]
    input_ids = torch.tensor([[1, 2, 3, *frame]])
    vocab_size = audio_start + codes_per_group * codebook_size + 1
    constrained = processor(input_ids, torch.zeros((1, vocab_size)))
    allowed = torch.isfinite(constrained[0]).nonzero(as_tuple=True)[0].tolist()

    assert allowed[0] == 4
    assert allowed[1:] == list(range(audio_start, audio_start + codebook_size))


def test_extract_audio_codes_is_strict_and_preserves_phase_offsets():
    sequence = [99, START_OF_SPEECH, 21, 24, END_OF_SPEECH, 10]
    assert extract_audio_codes(
        sequence,
        START_OF_SPEECH,
        END_OF_SPEECH,
        AUDIO_TOKENS_START,
        CODES_PER_GROUP,
        CODEBOOK_SIZE,
    ) == [1, 4]

    with pytest.raises(AudioTokenSequenceError):
        extract_audio_codes(
            [START_OF_SPEECH, 21, 21, END_OF_SPEECH],
            START_OF_SPEECH,
            END_OF_SPEECH,
            AUDIO_TOKENS_START,
            CODES_PER_GROUP,
            CODEBOOK_SIZE,
        )


def test_backend_stop_token_is_restored_but_length_truncation_stays_invalid():
    tokens = [START_OF_SPEECH, 21, 24]
    assert include_end_token_for_stop(
        tokens, END_OF_SPEECH, {"type": "stop", "matched": END_OF_SPEECH}
    ) == [*tokens, END_OF_SPEECH]
    assert include_end_token_for_stop(tokens, END_OF_SPEECH, "length") == tokens

    with pytest.raises(AudioTokenSequenceError, match="END_OF_SPEECH"):
        extract_audio_codes(
            [START_OF_SPEECH, 21, 24],
            START_OF_SPEECH,
            END_OF_SPEECH,
            AUDIO_TOKENS_START,
            CODES_PER_GROUP,
            CODEBOOK_SIZE,
        )


class _FakeCodec:
    codes_per_group = CODES_PER_GROUP
    codebook_size = CODEBOOK_SIZE
    sample_rate = 24000

    def __init__(self):
        self.decoded = []

    def decode(self, codes, device="cpu"):
        self.decoded.append((codes, device))
        return torch.ones((1, 1, 4))


class _Tokenized(dict):
    def __init__(self, input_ids):
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids


class _FakeTokenizer:
    def __init__(self):
        self.prompts = []

    def __call__(self, text, return_tensors="pt"):
        self.prompts.append(text)
        return _Tokenized(torch.tensor([[30]], dtype=torch.long))


def _config():
    return {
        "TOKENIZER_LENGTH": 100,
        "START_OF_TEXT": 1,
        "END_OF_TEXT": 2,
        "START_OF_SPEECH": START_OF_SPEECH,
        "END_OF_SPEECH": END_OF_SPEECH,
        "START_OF_HUMAN": 3,
        "END_OF_HUMAN": 4,
        "START_OF_AI": START_OF_AI,
        "END_OF_AI": 9,
        "PAD_TOKEN": PAD_TOKEN,
        "AUDIO_TOKENS_START": AUDIO_TOKENS_START,
    }


def test_audio_extraction_is_strict_and_preserves_batch_positions():
    engine = BaseVyvoTTSInference(config=_config())
    engine.codec = _FakeCodec()
    generated = torch.tensor(
        [
            [START_OF_SPEECH, 21, 24, END_OF_SPEECH],
            [START_OF_SPEECH, 21, 21, END_OF_SPEECH],
        ]
    )

    audio = engine._extract_audio_from_tokens(generated)

    assert len(audio) == 2
    assert isinstance(audio[0], torch.Tensor)
    assert audio[1] is None
    assert engine.codec.decoded == [([1, 4], "cpu")]


def test_voice_none_matches_single_speaker_training_prompt():
    engine = BaseVyvoTTSInference(config=_config())
    engine.tokenizer = _FakeTokenizer()

    engine._build_prompt_tokens("hello")
    engine._build_prompt_tokens("hello", voice="Ada")
    engine.DEFAULT_SPEAKERS = ["Known"]
    engine._build_prompt_tokens("hello", use_random_voice=True)

    assert engine.tokenizer.prompts == ["hello", "Ada: hello", "Known: hello"]


def test_voice_clone_reference_and_target_use_training_frame_markers():
    cloner = VyvoTTSVoiceClone.__new__(VyvoTTSVoiceClone)
    cloner.device = "cpu"
    cloner.tokenizer = _FakeTokenizer()
    cloner.START_OF_HUMAN = 3
    cloner.END_OF_TEXT = 2
    cloner.END_OF_HUMAN = 4
    cloner.START_OF_AI = START_OF_AI
    cloner.START_OF_SPEECH = START_OF_SPEECH
    cloner.END_OF_SPEECH = END_OF_SPEECH
    cloner.END_OF_AI = 9
    cloner.PAD_TOKEN = PAD_TOKEN
    cloner.encode_reference_audio = lambda _: [21, 24]

    input_ids, attention_mask = cloner.prepare_voice_clone_inputs(
        "reference.wav", "reference", ["target"]
    )
    values = input_ids[0].tolist()

    reference_speech = [START_OF_AI, START_OF_SPEECH, 21, 24, END_OF_SPEECH, 9]
    assert any(
        values[index : index + len(reference_speech)] == reference_speech
        for index in range(len(values) - len(reference_speech) + 1)
    )
    assert values[-2:] == [2, 4]
    assert torch.equal(attention_mask, torch.ones_like(attention_mask))
