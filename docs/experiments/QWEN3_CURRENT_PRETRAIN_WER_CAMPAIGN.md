# Qwen3-1.7B current-pretrain WER campaign

Experiment completed on 2026-08-10.

## Summary

Twelve training recipes were evaluated on top of
`Vyvo/VyvoTTS-Qwen3-1.7-PT` at revision
`e7335d4c2e28138bc250798fcc0e9d72ea776e8f`. The best deployable result is the
two-stage **completion SFT + codebook-boundary SFT** adapter:

- Seed-TTS English macro WER: **1.463962%**
- corpus WER: **1.448539%**
- edits: 145 substitutions, 19 deletions, and 7 insertions over 11,805 words
- coverage: 1,088/1,088, with no failed or length-truncated generations
- voice-quality score: **0.949792** (PESQ 4.0983, STOI 0.9972, WavLM speaker
  cosine 0.9474)

This reduced the unfine-tuned pretrain macro WER from 110.969934% by 98.68%
relative. No offline preference or online policy-optimization method improved
the boundary-SFT checkpoint, and no standalone checkpoint reached the target
of less than 1% WER.

## Data and fixed evaluation protocol

Training used only `spk_3048`, the highest-ranked speaker selected from the
streamed `SynDataLab-EN/echo-clones-4m-en` corpus: 1,010 utterances and 2.281748
hours. The 1.5 TB source dataset was processed incrementally, so it was never
downloaded in full. No old low-quality dataset was used, and the approved
large-data fallback `Vyvo/en-dataset-3` was not needed.

Every checkpoint synthesized the same 1,088 Seed-TTS English texts with seed
3407, temperature 0.6, top-p 0.95, top-k 20, repetition penalty 1.1, and a
1,200-token limit. WER was measured with
`Qwen/Qwen3-ASR-1.7B-hf@bcd2b5b7f32b480ab5790554cfa8347f246a14f3` and Seed-TTS
English normalization. Macro utterance WER is the promotion metric; corpus WER
is reported as a secondary metric. Voice quality was measured on the same
fixed 64-utterance subset using PESQ, STOI, signal checks, and WavLM speaker
similarity. All stages used identical inference and scoring settings.

## Results

| Stage | Method | Macro WER (%) | Corpus WER (%) | Quality | Decision |
|---|---|---:|---:|---:|---|
| Pretrain | Reference | 110.969934 | 109.758577 | 0.614773 | Baseline |
| Completion SFT | SFT | 1.543153 | 1.507836 | 0.947781 | Promoted |
| **Codebook-boundary SFT** | **SFT** | **1.463962** | **1.448539** | **0.949792** | **Champion** |
| CFG SFT | SFT | 1.516603 | 1.490894 | 0.946193 | Not promoted |
| Qwen self-distillation | SFT | 1.671511 | 1.634900 | 0.947234 | Not promoted |
| DPO | Preference | 1.580081 | 1.550191 | 0.946183 | Not promoted |
| RPO | Preference | 1.477233 | 1.465481 | 0.949468 | Not promoted |
| SPO | Preference | 1.482728 | 1.473952 | 0.948978 | Not promoted |
| FPO | Preference | 1.549834 | 1.524778 | 0.946763 | Not promoted |
| TKTO | Preference | 1.494376 | 1.482423 | 0.947091 | Not promoted |
| Targeted unlikelihood | Preference | 1.603151 | 1.584075 | 0.948773 | Not promoted |
| REINFORCE | Online RL | 1.562307 | 1.524778 | 0.949305 | Not promoted |
| GRPO | Online RL | 1.499197 | 1.473952 | 0.944718 | Not promoted |

### DNSMOS

| Stage | DNSMOS ↑ |
|---|---:|
| Pretrain | 1.919922 |
| Completion SFT | 3.257910 |
| Codebook-boundary SFT | 3.223899 |
| CFG SFT | 3.215698 |
| Qwen self-distillation | 3.268373 |
| DPO | 3.239253 |
| RPO | 3.261122 |
| SPO | 3.235983 |
| FPO | 3.230782 |
| TKTO | 3.228371 |
| Targeted unlikelihood | 3.248273 |
| **REINFORCE** | **3.274036** |
| GRPO | 3.257930 |

Self-distillation generated 2,048 candidates from 512 prompts and retained 365
exact-ASR groups. The shared preference campaign generated 3,072 candidates
and retained 328 strict chosen/rejected pairs. REINFORCE and GRPO each ran 200
on-policy updates; GRPO used four rollouts, reference-policy KL regularization,
and joint intelligibility, codec-validity, PESQ/STOI, and speaker rewards.

RPO was the closest post-SFT result, but its macro WER was still 0.013271
percentage points worse than the champion. GRPO also preserved complete test
coverage, yet increased macro WER by 0.035235 points and reduced the aggregate
quality score. These negative results show that reward improvement on sampled
training groups did not transfer to the fixed Seed-TTS test set.

## Reproducibility and artifacts

The campaign is defined by
`vyvotts/configs/train/qwen3_repretrain_wer_pipeline.yaml`; local results are
stored under `outputs/qwen3_current_pretrain_wer_campaign/`. The champion LoRA
adapter is in `training/02-codebook_boundary_sft/final/`, while every rejected
adapter and its full evaluation are retained for audit. The complete campaign
occupies approximately 14 GB. Repository verification completed with 204/204
pytest tests passing, and all Python files changed for the campaign passing
Ruff.
