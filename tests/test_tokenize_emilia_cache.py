import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

import vyvotts.tokenize_emilia as emilia


def _source_shards(tmp_path: Path) -> tuple[Path, Path]:
    first = tmp_path / "one" / "part.parquet"
    second = tmp_path / "two" / "part.parquet"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"first shard")
    second.write_bytes(b"second shard")
    return first, second


def _manifest(shards: list[Path], **overrides) -> dict:
    values = {
        "shard_type": "parquet",
        "dataset": "ylacombe/emilia-subset",
        "subsets": [],
        "model_type": "qwen3",
        "codec_type": "mimi",
        "codec_model_name": "kyutai/mimi",
        "audio_tokens_start": 151936,
        "target_sample_rate": 24000,
    }
    values.update(overrides)
    return emilia.make_cache_manifest([str(path) for path in shards], **values)


def test_manifest_is_order_independent_and_captures_source_identity(tmp_path: Path):
    first, second = _source_shards(tmp_path)

    forward = _manifest([first, second])
    reverse = _manifest([second, first])
    assert forward == reverse
    assert emilia.cache_fingerprint(forward) == emilia.cache_fingerprint(reverse)

    first.write_bytes(b"a replacement shard with a different size")
    replaced = _manifest([first, second])
    assert emilia.cache_fingerprint(replaced) != emilia.cache_fingerprint(forward)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset", "amphion/Emilia-Dataset"),
        ("subsets", ["emilia"]),
        ("model_type", "lfm2"),
        ("codec_type", "snac"),
        ("codec_model_name", "org/other-codec"),
        ("audio_tokens_start", 128256),
        ("target_sample_rate", 16000),
    ],
)
def test_manifest_fingerprint_changes_with_tokenization_inputs(
    tmp_path: Path,
    field: str,
    value,
):
    shards = list(_source_shards(tmp_path))
    baseline = _manifest(shards)
    changed = _manifest(shards, **{field: value})
    assert emilia.cache_fingerprint(changed) != emilia.cache_fingerprint(baseline)


def test_cache_namespace_does_not_depend_on_gpu_count(tmp_path: Path):
    shards = list(_source_shards(tmp_path))
    manifest = _manifest(shards)

    # GPU count is deliberately absent from the manifest and namespace API.
    first_run = emilia.prepare_cache_namespace(tmp_path / "cache", manifest)
    resumed_with_another_gpu_count = emilia.prepare_cache_namespace(
        tmp_path / "cache",
        deepcopy(manifest),
    )

    assert resumed_with_another_gpu_count == first_run
    stored = json.loads((first_run / "manifest.json").read_text())
    assert stored["fingerprint"] == emilia.cache_fingerprint(manifest)


def test_shard_names_are_rank_independent_and_same_stems_do_not_collide(
    tmp_path: Path,
):
    first, second = _source_shards(tmp_path)

    first_name = emilia.cache_shard_filename(first)
    assert first_name == emilia.cache_shard_filename(first)
    assert first_name != emilia.cache_shard_filename(second)
    assert first_name.startswith("part-")
    assert first_name.endswith(".pt")


def test_build_input_listing_is_scoped_to_selected_namespace(tmp_path: Path):
    shards = list(_source_shards(tmp_path))
    selected = emilia.prepare_cache_namespace(tmp_path / "cache", _manifest(shards))
    other = emilia.prepare_cache_namespace(
        tmp_path / "cache",
        _manifest(shards, codec_model_name="org/other-codec"),
    )

    selected_file = selected / "shards" / "selected.pt"
    selected_file.write_bytes(b"selected")
    (other / "shards" / "other.pt").write_bytes(b"other")
    (tmp_path / "cache" / "legacy.pt").write_bytes(b"legacy")

    assert emilia.cache_shard_files(selected) == [selected_file]


def test_prepare_cache_namespace_rejects_a_tampered_manifest(tmp_path: Path):
    shards = list(_source_shards(tmp_path))
    manifest = _manifest(shards)
    namespace = emilia.prepare_cache_namespace(tmp_path / "cache", manifest)
    manifest_path = namespace / "manifest.json"
    stored = json.loads(manifest_path.read_text())
    stored["model_inputs"]["target_sample_rate"] = 16000
    manifest_path.write_text(json.dumps(stored))

    with pytest.raises(RuntimeError, match="manifest mismatch"):
        emilia.prepare_cache_namespace(tmp_path / "cache", manifest)


def test_atomic_torch_save_never_replaces_good_output_with_partial_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "result.pt"
    emilia.atomic_torch_save(["complete"], output)

    def failing_save(_value, temporary_name):
        Path(temporary_name).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(emilia.torch, "save", failing_save)
    with pytest.raises(OSError, match="disk full"):
        emilia.atomic_torch_save(["new"], output)

    assert torch.load(output, weights_only=False) == ["complete"]
    assert not list(tmp_path.glob(".result.pt.*.tmp"))


def test_wait_for_workers_joins_all_processes_and_raises_on_failure():
    class Process:
        def __init__(self, exitcode):
            self.exitcode = exitcode
            self.joined = False

        def join(self):
            self.joined = True

    processes = [Process(0), Process(7), Process(0)]
    with pytest.raises(RuntimeError, match=r"GPU 1 \(exit code 7\)"):
        emilia.wait_for_workers(processes)

    assert all(process.joined for process in processes)
