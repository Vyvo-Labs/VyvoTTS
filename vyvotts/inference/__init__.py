from vyvotts.inference.constraints import (
    AudioTokenLogitsProcessor,
    AudioTokenSequenceError,
    extract_audio_codes,
    include_end_token_for_stop,
)


def VyvoTTSTransformersInference(*args, **kwargs):
    from vyvotts.inference.transformers_inference import (
        VyvoTTSTransformersInference as _cls,
    )
    return _cls(*args, **kwargs)

def VyvoTTSUnslothInference(*args, **kwargs):
    from vyvotts.inference.unsloth_inference import VyvoTTSUnslothInference as _cls
    return _cls(*args, **kwargs)

def VyvoTTSvLLMInference(*args, **kwargs):
    from vyvotts.inference.vllm_inference import VyvoTTSInference as _cls
    return _cls(*args, **kwargs)

def VyvoTTSSGLangInference(*args, **kwargs):
    from vyvotts.inference.sglang_inference import VyvoTTSSGLangInference as _cls
    return _cls(*args, **kwargs)

__all__ = [
    "AudioTokenLogitsProcessor",
    "AudioTokenSequenceError",
    "VyvoTTSSGLangInference",
    "VyvoTTSTransformersInference",
    "VyvoTTSUnslothInference",
    "VyvoTTSvLLMInference",
    "extract_audio_codes",
    "include_end_token_for_stop",
]
