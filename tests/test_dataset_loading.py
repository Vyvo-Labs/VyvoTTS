from datasets import Dataset, DatasetDict

from vyvotts.train.post_training.online import _load_dataset as load_online_dataset
from vyvotts.train.post_training.sft import _load_data as load_sft_dataset


def test_sft_and_online_select_requested_local_datasetdict_split(tmp_path):
    path = tmp_path / "dataset"
    DatasetDict(
        {
            "train": Dataset.from_dict({"value": [1]}),
            "validation": Dataset.from_dict({"value": [2]}),
        }
    ).save_to_disk(path)

    assert load_sft_dataset(str(path), "validation")[0]["value"] == 2
    assert load_online_dataset(str(path), "validation")[0]["value"] == 2
