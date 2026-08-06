"""Reproducible multi-stage SFT, preference, and online-RL orchestration."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

_MODULES = {
    "sft": "vyvotts.train.post_training.sft",
    "dpo": "vyvotts.train.post_training.preference",
    "spo": "vyvotts.train.post_training.preference",
    "fpo": "vyvotts.train.post_training.preference",
    "reinforce": "vyvotts.train.post_training.online",
    "rl": "vyvotts.train.post_training.online",
    "grpo": "vyvotts.train.post_training.online",
}


def _load_mapping(value: Any, *, base_dir: Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    if not isinstance(value, str):
        raise TypeError("stage config must be a YAML path or mapping")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, Mapping):
        raise TypeError(f"stage config must contain a mapping: {path}")
    return deepcopy(dict(loaded))


def resolve_stage_config(
    method: str,
    config: Mapping[str, Any],
    *,
    input_model: str | None,
    output_dir: str,
) -> dict[str, Any]:
    """Inject stage input/output artifacts into a method-specific config."""

    method = method.lower()
    if method not in _MODULES:
        raise ValueError(f"unsupported stage method {method!r}")
    resolved = deepcopy(dict(config))

    if method == "sft":
        if input_model is not None:
            resolved["model_name"] = input_model
            resolved["tokenizer_name"] = input_model
        elif not resolved.get("model_name"):
            raise ValueError("SFT needs an input model or model_name")
        resolved["output_dir"] = output_dir
    elif method in {"dpo", "spo", "fpo"}:
        model = dict(resolved.get("model") or {})
        if input_model is not None:
            model["name_or_path"] = input_model
            model["tokenizer_name_or_path"] = input_model
            if method in {"dpo", "fpo"}:
                model["reference_name_or_path"] = input_model
        elif not model.get("name_or_path") and not resolved.get("model_name_or_path"):
            raise ValueError(f"{method.upper()} needs an input model")
        resolved["model"] = model
        preference = dict(resolved.get("preference") or {})
        preference["objective"] = method
        resolved["preference"] = preference
        training = dict(resolved.get("training") or {})
        training["output_dir"] = output_dir
        resolved["training"] = training
    else:
        if input_model is not None:
            resolved["model_name"] = input_model
            resolved["tokenizer_name"] = input_model
            training = dict(resolved.get("training") or {})
            if float(training.get("kl_beta", 0.0)) > 0:
                resolved["reference_model_name"] = input_model
        elif not resolved.get("model_name"):
            raise ValueError(f"{method.upper()} needs an input model")
        training = dict(resolved.get("training") or {})
        training["method"] = method
        resolved["training"] = training
        resolved["output_dir"] = output_dir
    return resolved


def run_pipeline(
    config: Mapping[str, Any] | str | Path,
    *,
    dry_run: bool | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, str]:
    """Run enabled stages and return their final-checkpoint paths by name."""

    if isinstance(config, (str, Path)):
        config_path = Path(config).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            root = yaml.safe_load(handle)
        base_dir = config_path.parent
    else:
        root = deepcopy(dict(config))
        base_dir = Path.cwd()
    execution_dir = Path.cwd()
    if not isinstance(root, Mapping):
        raise TypeError("pipeline config must be a mapping")
    stages = root.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("pipeline config requires a non-empty stages list")

    output_root = Path(str(root.get("output_root", "outputs/alignment"))).expanduser()
    if not output_root.is_absolute():
        output_root = execution_dir / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    is_dry_run = bool(root.get("dry_run", False)) if dry_run is None else bool(dry_run)
    artifacts: dict[str, str] = {}
    initial_model = root.get("initial_model")
    previous_artifact = str(initial_model) if initial_model else None
    seen_names: set[str] = set()

    for index, stage in enumerate(stages, start=1):
        if not isinstance(stage, Mapping):
            raise TypeError(f"stages[{index - 1}] must be a mapping")
        if not bool(stage.get("enabled", True)):
            continue
        name = str(stage.get("name", f"stage-{index}"))
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", ".."}:
            raise ValueError(f"invalid or unsafe stage name {name!r}")
        if name in seen_names:
            raise ValueError(f"duplicate stage name {name!r}")
        seen_names.add(name)
        method = str(stage.get("method", "")).lower()
        if method not in _MODULES:
            raise ValueError(f"stage {name!r} has unsupported method {method!r}")

        input_from = stage.get("input_from")
        if input_from is None:
            input_model = previous_artifact
        elif input_from == "initial":
            input_model = str(initial_model) if initial_model else None
        else:
            try:
                input_model = artifacts[str(input_from)]
            except KeyError as exc:
                raise ValueError(
                    f"stage {name!r} references unfinished input_from={input_from!r}"
                ) from exc

        stage_dir = output_root / f"{index:02d}-{name}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_config = _load_mapping(stage.get("config", {}), base_dir=base_dir)
        resolved = resolve_stage_config(
            method,
            stage_config,
            input_model=input_model,
            output_dir=str(stage_dir),
        )
        resolved_path = stage_dir / "resolved.yaml"
        with resolved_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(resolved, handle, sort_keys=False)

        module = _MODULES[method]
        if bool(stage.get("accelerate", False)):
            command = [
                sys.executable,
                "-m",
                "accelerate.commands.launch",
                "-m",
                module,
                "--config",
                str(resolved_path),
            ]
        else:
            command = [sys.executable, "-m", module, "--config", str(resolved_path)]
        if not is_dry_run:
            runner(command, check=True, cwd=str(execution_dir))

        final_path = str(stage_dir / "final")
        if not is_dry_run and not Path(final_path).is_dir():
            raise RuntimeError(f"stage {name!r} completed without expected artifact {final_path}")
        artifacts[name] = final_path
        previous_artifact = final_path
        manifest_path = output_root / "pipeline_state.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump({"dry_run": is_dry_run, "artifacts": artifacts}, handle, indent=2)

    if not artifacts:
        raise ValueError("pipeline has no enabled stages")
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run staged VyvoTTS post-training")
    parser.add_argument("--config", required=True, help="Pipeline YAML")
    parser.add_argument("--dry-run", action="store_true", help="Resolve configs only")
    args = parser.parse_args()
    print(
        json.dumps(
            run_pipeline(args.config, dry_run=True if args.dry_run else None),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["resolve_stage_config", "run_pipeline"]
