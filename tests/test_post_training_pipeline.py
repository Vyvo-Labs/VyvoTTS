import yaml

from vyvotts.train.post_training.pipeline import resolve_stage_config, run_pipeline


def test_resolve_preference_stage_sets_objective_reference_and_output():
    resolved = resolve_stage_config(
        "fpo",
        {"preference": {"beta": 0.2}, "training": {"learning_rate": 1e-6}},
        input_model="sft/final",
        output_dir="alignment/fpo",
    )
    assert resolved["model"]["name_or_path"] == "sft/final"
    assert resolved["model"]["reference_name_or_path"] == "sft/final"
    assert resolved["model"]["tokenizer_name_or_path"] == "sft/final"
    assert resolved["preference"] == {"beta": 0.2, "objective": "fpo"}
    assert resolved["training"]["output_dir"] == "alignment/fpo"


def test_chained_stage_replaces_stale_tokenizer_path():
    resolved = resolve_stage_config(
        "grpo",
        {
            "model_name": "stale/model",
            "tokenizer_name": "stale/tokenizer",
            "training": {"kl_beta": 0.1},
        },
        input_model="selected/final",
        output_dir="alignment/grpo",
    )
    assert resolved["model_name"] == "selected/final"
    assert resolved["tokenizer_name"] == "selected/final"
    assert resolved["reference_model_name"] == "selected/final"


def test_dry_run_supports_branches_from_same_sft_checkpoint(tmp_path):
    sft_config = tmp_path / "sft.yaml"
    preference_config = tmp_path / "preference.yaml"
    sft_config.write_text(
        yaml.safe_dump({"model_name": "unused", "dataset": "data"}), encoding="utf-8"
    )
    preference_config.write_text(
        yaml.safe_dump({"data": {"dataset": "pairs"}}), encoding="utf-8"
    )
    config = {
        "initial_model": "pretrain/final",
        "output_root": str(tmp_path / "runs"),
        "stages": [
            {"name": "sft", "method": "sft", "config": str(sft_config)},
            {
                "name": "dpo",
                "method": "dpo",
                "input_from": "sft",
                "config": str(preference_config),
            },
            {
                "name": "spo",
                "method": "spo",
                "input_from": "sft",
                "config": str(preference_config),
            },
        ],
    }

    artifacts = run_pipeline(config, dry_run=True)
    assert set(artifacts) == {"sft", "dpo", "spo"}
    dpo = yaml.safe_load((tmp_path / "runs/02-dpo/resolved.yaml").read_text())
    spo = yaml.safe_load((tmp_path / "runs/03-spo/resolved.yaml").read_text())
    assert dpo["model"]["name_or_path"] == artifacts["sft"]
    assert dpo["model"]["reference_name_or_path"] == artifacts["sft"]
    assert spo["model"]["name_or_path"] == artifacts["sft"]
    assert "reference_name_or_path" not in spo["model"]
