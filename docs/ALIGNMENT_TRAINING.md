# TTS Quality and WER Alignment

This guide maps the directly relevant 2025–2026 primary literature available
through 6 August 2026 onto VyvoTTS. It is a targeted, reproducible review—not a
claim that every paper using the phrase “text-to-speech” has been enumerated.
The current model is a decoder-only LM over interleaved SNAC or Mimi codes, so
post-training methods are implemented directly while codec/decoder replacements
are listed separately.

## What Changed

- Dataset tokenization no longer removes frames merely because the first
  codebook repeats. Exact full-frame deduplication is opt-in because it changes
  duration and prosody.
- SFT labels mask the text/user prompt. `post_training.sft` adds completion-only
  loss, heavier semantic/coarse-codebook weights, and speech-boundary weighting.
- Transformers and Unsloth inference now enforce `START_OF_AI`,
  `START_OF_SPEECH`, the correct range for every codebook phase, complete-frame
  EOS, and strict decoding. Invalid IDs are no longer hidden by codec clamping.
- Raw-token DPO, SPO, selective-token FPO, REINFORCE, and GRPO are available.
  Speech tokens never round-trip through `tokenizer.decode()`.
- Lazy Whisper, WavLM, and SQUIM scorers provide intelligibility, speaker, and
  objective-quality rewards. Duration, codec validity, F0/energy variation,
  Pareto gates, and protected-metric checks are also available.

## Recommended Multi-Stage Recipe

Do not blindly run every preference objective in one chain. DPO, SPO, and FPO
are competing offline objectives; compare them from the same SFT checkpoint,
then apply online GRPO to the best validation checkpoint.

1. **Pretrain:** retain diverse, normalized text/audio pairs. Filter clipping,
   silence, bad transcripts, extreme duration, and low DNSMOS; preserve hard
   names, numbers, code-switching, tongue twisters, and long-form examples.
2. **Quality SFT:** use completion-only labels and slightly emphasize Mimi
   codebooks 0–1 or SNAC's coarse streams. Start with the supplied weights;
   excessive weighting can reduce acoustic detail.
3. **Generate preferences:** sample 4–8 grammar-constrained candidates per
   prompt at temperature 0.6–0.8. Score every waveform with the same normalized
   text used by evaluation. Keep a pair only when its aggregate gap is large
   enough and WER, speaker similarity, and quality pass Pareto/no-regression
   gates.
4. **Offline alignment:** run DPO as the stable baseline; run reference-free SPO
   when reference-model memory is limiting; run FPO when word timestamps or
   error spans identify mispronunciation, repetition, or truncation. Local
   errors select their codec frames; repetition/truncation select the causal
   tail from the first error.
5. **Online alignment:** use GRPO with 8 rollouts as the default. REINFORCE/`rl`
   is included as a lower-memory, higher-variance baseline. Begin with centered,
   unscaled group advantages and one policy update; add KL only if drift appears.
6. **Select by a metric vector:** require lower normalized WER/CER without a
   statistically meaningful loss in speaker similarity or quality. Keep an
   untouched human-listening set; reward models are optimization targets, not a
   substitute for MOS/ABX evaluation.

## Commands and Data Contracts

Install the training dependencies, tokenize without lossy deduplication, then
run SFT:

```bash
uv pip install -e ".[train]"
python -m vyvotts.train.post_training.sft \
  --config vyvotts/configs/train/sft_quality.yaml
```

Candidate JSONL keeps raw IDs and waveform-derived metrics:

```json
{"prompt_input_ids":[64403,100,7,64404],"candidates":[{"completion_input_ids":[64405,64401,64410,66458,68506,70554,72602,74650,76698,78746,64402,64406],"metrics":{"intelligibility_reward":0.91,"speaker_reward":0.84,"quality_reward":0.78}},{"completion_input_ids":[64405,64401,64411,66458,68506,70554,72602,74650,76698,78746,64402,64406],"metrics":{"intelligibility_reward":0.10,"speaker_reward":0.40,"quality_reward":0.40},"error_spans":[{"start_frame":0,"end_frame":1,"type":"pronunciation"}]}]}
```

Build Pareto-safe pairs and train an offline objective:

```bash
python -m vyvotts.train.post_training.data \
  --config vyvotts/configs/train/build_preferences.yaml
python -m vyvotts.train.post_training.preference \
  --config vyvotts/configs/train/preference_quality.yaml
```

Preference rows require `prompt_input_ids`, `chosen_input_ids`, and
`rejected_input_ids`. Optional completion-relative
`chosen_selection_mask`/`rejected_selection_mask` fields enable FPO;
`sample_weight` can reflect confidence. Set `objective` to `dpo`, `spo`, or
`fpo`. DPO/FPO require a frozen reference checkpoint; SPO uses the 2025 SiLU
self-regularized, reference-free loss.

Run online GRPO:

```bash
python -m accelerate.commands.launch \
  -m vyvotts.train.post_training.online \
  --config vyvotts/configs/train/grpo_quality.yaml
```

Use a single process or ordinary unsharded data parallelism (DDP) for online
rollouts. FSDP, DeepSpeed, and Megatron-LM are rejected because unwrapped
generation does not safely gather their sharded parameters.

Set `training.method: reinforce` (or `rl`) for the running-baseline policy
gradient. An online prompt dataset must retain `reference_text`; speaker reward
also needs `reference_audio_column`.

The staged runner can resolve branched SFT/DPO/SPO/FPO/REINFORCE/GRPO configs
and pass checkpoint artifacts between them. The example is a dry run by default:

```bash
python -m vyvotts.train.post_training.pipeline \
  --config vyvotts/configs/train/staged_alignment.yaml --dry-run
```

Inspect each generated `resolved.yaml`, disable objectives that are not part of
the selected ablation, then set `dry_run: false`. Running every branch is useful
for comparison but is not a recommended single linear training recipe.

## WER-Specific Techniques

- Use one language-aware normalization policy for dataset text, ASR reward, and
  evaluation. Report both WER and CER for languages without reliable word
  segmentation.
- Combine decoded CER with teacher-forced ASR confidence. The implemented
  mapping is `1-tanh(3*CER)` and `exp(-NLL/3)`, combined by a 0.6/0.4 harmonic
  mean. Mean NLL is the stable default; `reduction="sum"` reproduces the paper's
  written sequence equation.
- Mine hard prompts by baseline WER and error type, but reject examples where
  reference transcription or audio quality is suspect. Include insertion,
  deletion, repetition, early-stop, number, entity, heteronym, and code-switch
  buckets in every validation report.
- Prefer FPO over utterance-level DPO for timestamped localized errors. Use
  causal-tail masks for repetition/truncation, because all later codec tokens
  were generated from the corrupted history.
- Keep codec grammar constraints enabled during both candidate generation and
  deployment. They eliminate invalid phase IDs and partial frames but cannot
  fix linguistic errors by themselves.
- For multilingual checkpoints, evaluate a hybrid grapheme/phoneme or
  character/pinyin input ablation. It is a data/model-interface change and is
  intentionally not silently applied to existing checkpoints.

## Reward Design and Stability

The default reward is `0.45*intelligibility + 0.45*speaker + 0.10*quality`.
Alternatives implemented in `rewards.py` include `exp(-2.5*WER)`, normalized
speaker cosine `(sim+1)/2`, piecewise worst/baseline/best calibration, duration
similarity, codec validity, and F0/energy coefficient-of-variation similarity.
Calibrate metric ranges on the frozen SFT model before choosing weights. Log raw
WER, CER, NLL, PESQ, STOI, SI-SDR, speaker cosine, completion length, invalid
rollouts, KL, and reward components. Stop or reduce the learning rate if a
protected metric regresses even while total reward rises.

## Research Map

Directly implemented ideas came from [FPO](https://arxiv.org/abs/2502.02950),
[Inworld TTS-1](https://arxiv.org/abs/2507.21138),
[multidimensional preference optimization](https://arxiv.org/abs/2509.00685),
[GRPO for TTS](https://arxiv.org/abs/2509.18798),
[Align2Speak](https://arxiv.org/abs/2509.21718),
[Koel-TTS](https://aclanthology.org/2025.emnlp-main.1076/), and
[SPO](https://aclanthology.org/2025.findings-emnlp.300/). The 2026 evidence for
GRPO pronunciation post-training and style-aware rewards comes from
[IndexTTS 2.5](https://arxiv.org/abs/2601.03888) and
[VoiceTTA](https://arxiv.org/abs/2606.26534).

The broader scan also covered 2025 reports on
[MegaTTS 3](https://arxiv.org/abs/2502.18924),
[IndexTTS](https://arxiv.org/abs/2502.05512),
[Spark-TTS](https://arxiv.org/abs/2503.01710),
[CosyVoice 3](https://arxiv.org/abs/2505.17589),
[intelligibility preference data](https://arxiv.org/abs/2505.04113),
[MiniMax-Speech](https://arxiv.org/abs/2505.07916),
[IndexTTS 2](https://arxiv.org/abs/2506.21619),
[CLEAR](https://arxiv.org/abs/2508.19098),
[VibeVoice](https://arxiv.org/abs/2508.19205),
[LatinX TTS](https://arxiv.org/abs/2509.05863),
[ARDM-DPO](https://arxiv.org/abs/2509.18928), and
[GLM-TTS](https://arxiv.org/abs/2512.14291). Their recurring lessons are
semantic/acoustic disentanglement, stronger alignment, hard multilingual data,
and multidimensional rather than WER-only optimization.

The 2026 scan included [Qwen3-TTS](https://arxiv.org/abs/2601.15621),
[MOSS audio tokenizer](https://arxiv.org/abs/2602.10934),
[MOSS-TTS](https://arxiv.org/abs/2603.18090),
[Fish Audio S2](https://arxiv.org/abs/2603.08823),
[StepAudio 2.5](https://arxiv.org/abs/2605.23463),
[VoxCPM 2](https://arxiv.org/abs/2606.06928), and
[dots.tts](https://arxiv.org/abs/2606.07080). Their 12.5/25 Hz semantic codecs,
dual-track streaming, multi-task tokenizers, flow/AudioVAE decoders,
full-history conditioning, and self-corrective flow post-training require new
codec tokens or model heads. They are high-value next-generation architecture
experiments, but retrofitting them into an existing SNAC/Mimi checkpoint would
invalidate its vocabulary and learned embeddings.

CosyVoice 3's multi-task semantic tokenizer and differentiable reward path
(token-to-text scoring through soft token samples), MegaTTS 3's sparse
alignment/flow decoder, and the continuous-latent systems likewise require a
new tokenizer or differentiable acoustic head. They are not represented as
drop-in flags in this implementation because doing so would falsely imply
checkpoint compatibility.
