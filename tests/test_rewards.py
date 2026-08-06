import math
import unittest
from types import SimpleNamespace

import torch

from vyvotts.train.post_training.rewards import (
    CompositeReward,
    RewardScorerError,
    SQUIMObjectiveScorer,
    WavLMSpeakerScorer,
    WhisperASRScorer,
    asr_intelligibility_reward,
    asr_nll_reward,
    basic_text_normalize,
    build_composite_reward,
    cer_tanh_reward,
    character_error_rate,
    codec_validity_reward,
    coefficient_of_variation,
    duration_reward,
    duration_similarity_reward,
    energy_cv_reward,
    f0_cv_reward,
    levenshtein_distance,
    piecewise_baseline_normalize,
    prosody_cv_reward,
    speaker_cosine_reward,
    weighted_harmonic_mean,
    weighted_harmonic_reward,
    wer_exponential_reward,
    word_error_rate,
)


class EditMetricTests(unittest.TestCase):
    def test_levenshtein_and_word_error_rate(self):
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        self.assertAlmostEqual(word_error_rate("one two three", "one four"), 2 / 3)
        self.assertEqual(word_error_rate("", "two insertions"), 2.0)
        self.assertEqual(word_error_rate("", ""), 0.0)

    def test_character_error_rate_and_normalization(self):
        self.assertAlmostEqual(character_error_rate("a b c", "a x c"), 1 / 3)
        self.assertEqual(character_error_rate("A", "a"), 1.0)
        lowercase = lambda value: value.lower()
        self.assertEqual(character_error_rate("A", "a", normalizer=lowercase), 0.0)
        self.assertAlmostEqual(
            character_error_rate("a b", "ab", ignore_whitespace=False),
            1 / 3,
        )

    def test_basic_multilingual_normalizer_removes_case_and_punctuation_noise(self):
        self.assertEqual(basic_text_normalize("  Hello, WORLD！  "), "hello world")
        self.assertEqual(
            word_error_rate("Hello, WORLD!", "hello world", normalizer=basic_text_normalize),
            0.0,
        )


class RewardMappingTests(unittest.TestCase):
    def test_published_asr_reward_mappings(self):
        self.assertAlmostEqual(wer_exponential_reward(0.2), math.exp(-0.5))
        self.assertAlmostEqual(cer_tanh_reward(0.2), 1.0 - math.tanh(0.6))
        self.assertAlmostEqual(asr_nll_reward(1.5), math.exp(-0.5))
        self.assertEqual(cer_tanh_reward(0.0), 1.0)
        self.assertEqual(asr_nll_reward(0.0), 1.0)

    def test_weighted_harmonic_rewards(self):
        expected = 1.0 / (0.6 / 0.5 + 0.4 / 0.25)
        self.assertAlmostEqual(weighted_harmonic_reward(0.5, 0.25), expected)
        self.assertAlmostEqual(
            weighted_harmonic_mean([0.5, 0.25], [0.6, 0.4]),
            expected,
        )
        self.assertEqual(weighted_harmonic_mean([1.0, 0.0]), 0.0)
        self.assertAlmostEqual(
            asr_intelligibility_reward(0.2, 1.5),
            weighted_harmonic_reward(cer_tanh_reward(0.2), asr_nll_reward(1.5)),
        )

    def test_piecewise_baseline_normalization_in_both_directions(self):
        kwargs = {"worst": 0.0, "baseline": 5.0, "best": 10.0}
        self.assertEqual(piecewise_baseline_normalize(0.0, **kwargs), 0.0)
        self.assertEqual(piecewise_baseline_normalize(5.0, **kwargs), 0.5)
        self.assertEqual(piecewise_baseline_normalize(10.0, **kwargs), 1.0)
        self.assertEqual(piecewise_baseline_normalize(20.0, **kwargs), 1.0)

        lower_better = {
            "worst": 10.0,
            "baseline": 5.0,
            "best": 0.0,
            "higher_is_better": False,
        }
        self.assertEqual(piecewise_baseline_normalize(10.0, **lower_better), 0.0)
        self.assertEqual(piecewise_baseline_normalize(5.0, **lower_better), 0.5)
        self.assertEqual(piecewise_baseline_normalize(0.0, **lower_better), 1.0)
        with self.assertRaises(ValueError):
            piecewise_baseline_normalize(1.0, worst=0.0, baseline=0.0, best=2.0)

    def test_speaker_and_duration_rewards(self):
        self.assertEqual(speaker_cosine_reward(-1.0), 0.0)
        self.assertEqual(speaker_cosine_reward(0.0), 0.5)
        self.assertEqual(speaker_cosine_reward(1.0), 1.0)
        self.assertAlmostEqual(duration_reward(9.0, 10.0), -0.1)
        self.assertAlmostEqual(duration_similarity_reward(9.0, 10.0), math.exp(-0.1))
        with self.assertRaises(ValueError):
            duration_reward(1.0, 0.0)

    def test_codec_validity_reward(self):
        valid = [0, 10, 20, 9, 19, 29]
        self.assertEqual(
            codec_validity_reward(valid, codes_per_group=3, codebook_size=10),
            1.0,
        )
        one_bad = [0, 5, 20]
        self.assertAlmostEqual(
            codec_validity_reward(one_bad, codes_per_group=3, codebook_size=10),
            2 / 3,
        )
        self.assertEqual(
            codec_validity_reward(
                one_bad,
                codes_per_group=3,
                codebook_size=10,
                strict=True,
            ),
            0.0,
        )
        self.assertEqual(
            codec_validity_reward([0, 10], codes_per_group=3, codebook_size=10),
            0.0,
        )
        self.assertEqual(
            codec_validity_reward(
                [0, 10],
                codes_per_group=3,
                codebook_size=10,
                allow_partial_group=True,
            ),
            1.0,
        )

    def test_prosody_cv_rewards(self):
        self.assertAlmostEqual(coefficient_of_variation([1.0, 3.0]), 0.5)
        self.assertEqual(f0_cv_reward([0.0, 1.0, 3.0], 0.5), 1.0)
        self.assertEqual(energy_cv_reward([1.0, 3.0], 0.5), 1.0)
        self.assertEqual(
            prosody_cv_reward(
                f0_values=[0.0, 1.0, 3.0],
                target_f0_cv=0.5,
                energy_values=[1.0, 3.0],
                target_energy_cv=0.5,
            ),
            1.0,
        )
        with self.assertRaises(ValueError):
            energy_cv_reward([-1.0, 1.0], 0.5)

    def test_invalid_numeric_inputs_fail_explicitly(self):
        with self.assertRaises(ValueError):
            wer_exponential_reward(-0.1)
        with self.assertRaises(ValueError):
            asr_nll_reward(float("nan"))
        with self.assertRaises(ValueError):
            weighted_harmonic_mean([], [])


class CompositeRewardTests(unittest.TestCase):
    def test_offline_score_returns_breakdown_with_total(self):
        reward = CompositeReward(
            {"intelligibility_reward": 0.45, "speaker_reward": 0.45, "quality_reward": 0.1},
        )
        result = reward.score(
            {
                "intelligibility_reward": 0.8,
                "speaker_reward": 0.7,
                "quality_reward": 0.9,
            }
        )
        self.assertAlmostEqual(result["total"], 0.45 * 0.8 + 0.45 * 0.7 + 0.1 * 0.9)
        self.assertAlmostEqual(reward.aggregate(result), result["total"])

    def test_pareto_and_protected_metric_guards(self):
        reward = CompositeReward(
            {"quality": 0.5, "similarity": 0.5},
            pareto_metrics=["quality", "similarity"],
            protected_metrics=["similarity"],
            protected_thresholds={"similarity": 0.6},
        )
        incumbent = {"quality": 0.7, "similarity": 0.7}
        winner = {"quality": 0.8, "similarity": 0.8}
        tradeoff = {"quality": 0.9, "similarity": 0.65}
        self.assertTrue(reward.pareto_dominates(winner, incumbent))
        self.assertTrue(reward.is_preferred(winner, incumbent))
        self.assertFalse(reward.is_preferred(tradeoff, incumbent))
        self.assertEqual(reward.compare(winner, incumbent), 1)
        self.assertEqual(reward.compare(tradeoff, incumbent), 0)

    def test_minimized_protected_metric(self):
        reward = CompositeReward(
            {"quality_reward": 1.0},
            maximize={"wer": "min"},
            protected_metrics=["wer"],
            protected_thresholds={"wer": 0.2},
        )
        incumbent = {"quality_reward": 0.5, "wer": 0.1}
        self.assertTrue(reward.preserves({"quality_reward": 0.6, "wer": 0.08}, incumbent))
        self.assertFalse(reward.preserves({"quality_reward": 0.9, "wer": 0.15}, incumbent))
        self.assertFalse(reward.passes_protected_thresholds({"wer": 0.3}))

    def test_online_scorers_are_composed_and_non_numeric_metadata_is_ignored(self):
        class ASR:
            def score(self, audio, text, *, sampling_rate):
                return {"transcript": text, "intelligibility_reward": 0.8}

        class Speaker:
            def score(self, audio, reference, *, sampling_rate):
                return {"speaker_reward": 0.6}

        class Quality:
            def score(self, audio, *, sampling_rate):
                return {"quality_reward": 0.9}

        reward = CompositeReward(
            {"intelligibility_reward": 0.5, "speaker_reward": 0.3, "quality_reward": 0.2},
            asr_scorer=ASR(),
            speaker_scorer=Speaker(),
            quality_scorer=Quality(),
        )
        result = reward.score([0.0], "hello", 16000, reference_audio=[0.0])
        self.assertEqual(set(result), {
            "intelligibility_reward",
            "speaker_reward",
            "quality_reward",
            "total",
        })
        self.assertAlmostEqual(result["total"], 0.5 * 0.8 + 0.3 * 0.6 + 0.2 * 0.9)

    def test_required_reference_audio_failure_is_explicit(self):
        class Speaker:
            def score(self, audio, reference, *, sampling_rate):
                return {"speaker_reward": 1.0}

        reward = CompositeReward(
            {"speaker_reward": 1.0},
            speaker_scorer=Speaker(),
        )
        with self.assertRaisesRegex(RewardScorerError, "reference_audio"):
            reward.score([0.0], "hello", 16000)

    def test_factory_builds_unloaded_backends(self):
        reward = build_composite_reward({
            "weights": {
                "intelligibility_reward": 0.5,
                "speaker_reward": 0.4,
                "quality_reward": 0.1,
            }
        })
        self.assertIsInstance(reward.asr_scorer, WhisperASRScorer)
        self.assertIsInstance(reward.speaker_scorer, WavLMSpeakerScorer)
        self.assertIsInstance(reward.quality_scorer, SQUIMObjectiveScorer)
        self.assertFalse(reward.asr_scorer.loaded)
        self.assertFalse(reward.speaker_scorer.loaded)
        self.assertFalse(reward.quality_scorer.loaded)
        with self.assertRaisesRegex(ValueError, "raw error metrics"):
            build_composite_reward({"weights": {"wer": 1.0}})


class LazyScorerTests(unittest.TestCase):
    class FakeModel:
        def __init__(self):
            self.device = None
            self.is_eval = False

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.is_eval = True
            return self

    def test_whisper_loaders_are_not_called_at_construction(self):
        calls = []
        model = self.FakeModel()
        scorer = WhisperASRScorer(
            processor_loader=lambda name: calls.append(("processor", name)) or object(),
            model_loader=lambda name: calls.append(("model", name)) or model,
        )
        self.assertEqual(calls, [])
        self.assertFalse(scorer.loaded)
        scorer.load()
        self.assertEqual([kind for kind, _ in calls], ["processor", "model"])
        self.assertTrue(scorer.loaded)
        self.assertTrue(model.is_eval)

    def test_wavlm_and_squim_loaders_are_lazy(self):
        calls = []
        wavlm = WavLMSpeakerScorer(
            feature_extractor_loader=lambda name: calls.append("features") or object(),
            model_loader=lambda name: calls.append("wavlm") or self.FakeModel(),
        )
        squim = SQUIMObjectiveScorer(
            model_loader=lambda: calls.append("squim") or self.FakeModel(),
        )
        self.assertEqual(calls, [])
        wavlm.load()
        squim.load()
        self.assertEqual(calls, ["features", "wavlm", "squim"])

    def test_loader_failures_include_backend_context(self):
        def fail(name):
            raise OSError("offline")

        scorer = WhisperASRScorer(
            processor_loader=fail,
            model_loader=fail,
        )
        with self.assertRaisesRegex(RewardScorerError, "Whisper processor"):
            scorer.load()

    def test_whisper_and_wavlm_resample_codec_audio_to_backend_rate(self):
        class WhisperProcessor:
            feature_extractor = SimpleNamespace(sampling_rate=16000)

            def __init__(self):
                self.observed = None

            def __call__(self, waveform, *, sampling_rate, return_tensors, **kwargs):
                self.observed = (len(waveform), sampling_rate, return_tensors)
                return {"input_features": torch.zeros(1, 80, 2)}

        whisper_processor = WhisperProcessor()
        whisper = WhisperASRScorer(
            processor=whisper_processor,
            model=self.FakeModel(),
        )
        whisper._audio_inputs(torch.zeros(24000), 24000)
        self.assertEqual(whisper_processor.observed, (16000, 16000, "pt"))

        class FeatureExtractor:
            sampling_rate = 16000

            def __init__(self):
                self.observed = None

            def __call__(self, waveform, *, sampling_rate, return_tensors, padding):
                self.observed = (len(waveform), sampling_rate, return_tensors, padding)
                return {"input_values": torch.zeros(1, len(waveform))}

        class SpeakerModel(self.FakeModel):
            def __call__(self, **inputs):
                return SimpleNamespace(embeddings=torch.ones(1, 4))

        feature_extractor = FeatureExtractor()
        wavlm = WavLMSpeakerScorer(
            feature_extractor=feature_extractor,
            model=SpeakerModel(),
        )
        wavlm.embedding(torch.zeros(24000), sampling_rate=24000)
        self.assertEqual(feature_extractor.observed, (16000, 16000, "pt", True))

    def test_whisper_nll_defaults_to_mean_and_supports_paper_sum(self):
        import torch

        class Batch(dict):
            def to(self, device):
                return Batch({key: value.to(device) for key, value in self.items()})

        class Tokenizer:
            pad_token_id = 0

            def __call__(self, text, *, return_tensors):
                return Batch({"input_ids": torch.tensor([[4, 5, 0]])})

        class Processor:
            tokenizer = Tokenizer()

            def __call__(self, waveform, *, sampling_rate, return_tensors):
                return Batch({"input_features": torch.ones(1, 2, 3)})

        class Model:
            def __call__(self, **kwargs):
                return SimpleNamespace(loss=torch.tensor(2.0))

        scorer = WhisperASRScorer(processor=Processor(), model=Model())
        mean_nll = scorer.teacher_forced_nll(
            [0.0],
            "hello",
            sampling_rate=16000,
        )
        sum_nll = scorer.teacher_forced_nll(
            [0.0],
            "hello",
            sampling_rate=16000,
            reduction="sum",
        )
        self.assertEqual(mean_nll, 2.0)
        self.assertEqual(sum_nll, 4.0)


if __name__ == "__main__":
    unittest.main()
