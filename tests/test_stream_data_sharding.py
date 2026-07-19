from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

EAGLE_ROOT = Path(__file__).resolve().parents[1] / "training" / "Eagle" / "Embodied"
sys.path.insert(0, str(EAGLE_ROOT))

# The training host installs python-dotenv. The lightweight local evaluation
# environment does not need it, so keep this unit test independent of that
# optional process-level configuration dependency.
if "dotenv" not in sys.modules:
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: False
    sys.modules["dotenv"] = dotenv

from eaglevl.train.locany_finetune_magi_stream import (  # noqa: E402
    DeterministicIterator,
    StreamPackedDatasetMTP,
)

TRAIN_SOURCE = EAGLE_ROOT / "eaglevl" / "train" / "locany_finetune_magi_stream.py"


class _IndexDataset:
    ds_name = "indices"

    def __init__(self, length: int):
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        return index


class _FailingIndexDataset(_IndexDataset):
    def __getitem__(self, index: int) -> int:
        if index == 0:
            raise OSError("broken sample")
        return index


def test_data_fingerprint_does_not_depend_on_checkpoint_path() -> None:
    source = TRAIN_SOURCE.read_text(encoding="utf-8")

    assert "'tokenizer_path':" not in source
    assert "'tokenizer_vocab_size':" in source
    assert "'tokenizer_added_vocab':" in source


def _shard_pass(
    dataset: _IndexDataset,
    *,
    seed: int,
    shard_id: int,
    num_shards: int,
    pass_index: int,
) -> list[tuple[int, int]]:
    iterator = DeterministicIterator(
        dataset,
        seed=seed,
        shard_id=shard_id,
        num_shards=num_shards,
    )
    owned_count = len(range(shard_id, len(dataset), num_shards))
    for _ in range(pass_index * owned_count):
        next(iterator)
    return [next(iterator) for _ in range(owned_count)]


def test_rank_and_worker_partitions_are_globally_unique() -> None:
    stream = StreamPackedDatasetMTP.__new__(StreamPackedDatasetMTP)
    stream.data_world_size = 4

    partitions = []
    for rank in range(4):
        stream.data_rank = rank
        for worker_id in range(4):
            partitions.append(stream._worker_partition(worker_id, 4))

    assert partitions == [(worker_id, 16) for worker_id in range(16)]


def test_permanent_shards_require_enough_samples_for_every_worker() -> None:
    dataset = _IndexDataset(3)
    iterator = DeterministicIterator(
        dataset, seed=1, shard_id=3, num_shards=4
    )

    with pytest.raises(StopIteration):
        next(iterator)


def test_worker_shards_have_permanent_ownership_and_cover_every_sample() -> None:
    dataset = _IndexDataset(37)
    num_shards = 16

    for pass_index in range(3):
        pass_records = [
            record
            for shard_id in range(num_shards)
            for record in _shard_pass(
                dataset,
                seed=1234,
                shard_id=shard_id,
                num_shards=num_shards,
                pass_index=pass_index,
            )
        ]
        sample_indices = [sample_idx for sample_idx, _ in pass_records]
        assert sorted(sample_indices) == list(range(len(dataset)))


def test_uneven_worker_progress_never_crosses_sample_ownership() -> None:
    dataset = _IndexDataset(10)
    fast_worker = DeterministicIterator(
        dataset, seed=7, shard_id=0, num_shards=2
    )
    slow_worker = DeterministicIterator(
        dataset, seed=7, shard_id=1, num_shards=2
    )

    fast_samples = {next(fast_worker)[0] for _ in range(12)}
    slow_samples = {next(slow_worker)[0] for _ in range(2)}

    assert fast_samples.isdisjoint(slow_samples)
    assert fast_samples == {0, 2, 4, 6, 8}
    assert slow_samples <= {1, 3, 5, 7, 9}


def test_failed_sample_retry_stays_inside_worker_ownership() -> None:
    iterator = DeterministicIterator(
        _FailingIndexDataset(8), seed=3, shard_id=0, num_shards=2
    )
    returned = [next(iterator)[0] for _ in range(4)]

    assert all(index % 2 == 0 for index in returned)
    assert 1 not in returned


def test_worker_shards_are_reproducible_and_resume_exactly() -> None:
    dataset = _IndexDataset(23)
    first = DeterministicIterator(
        dataset, seed=99, shard_id=5, num_shards=8
    )
    repeated = DeterministicIterator(
        dataset, seed=99, shard_id=5, num_shards=8
    )

    assert [next(first) for _ in range(4)] == [
        next(repeated) for _ in range(4)
    ]

    state = first.state_dict()
    resumed = DeterministicIterator.from_state_dict(
        dataset, state, shard_id=5, num_shards=8
    )
    assert next(resumed) == next(first)

    with pytest.raises(ValueError, match="shard topology changed"):
        DeterministicIterator.from_state_dict(
            dataset, state, shard_id=5, num_shards=16
        )


def test_pre_fingerprint_resume_state_is_rejected() -> None:
    stream = StreamPackedDatasetMTP.__new__(StreamPackedDatasetMTP)
    stream._resume_states = {"stale": {}}
    stream.data_fingerprint = "current"
    stream.base_seed = 42

    with pytest.raises(ValueError, match="predates dataset fingerprints"):
        stream.load_state_dict(
            {"version": 5, "worker_states": {"worker_0": {"stale": True}}}
        )


def test_resume_rejects_changed_dataset_fingerprint() -> None:
    stream = StreamPackedDatasetMTP.__new__(StreamPackedDatasetMTP)
    stream._resume_states = {}
    stream.data_fingerprint = "current"
    stream.base_seed = 42

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        stream.load_state_dict({
            "version": 6,
            "base_seed": 42,
            "data_fingerprint": "different",
            "worker_states": {"worker_0": {}},
        })


def test_resume_rejects_missing_worker_snapshots() -> None:
    stream = StreamPackedDatasetMTP.__new__(StreamPackedDatasetMTP)
    stream._resume_states = {}
    stream.data_fingerprint = "current"
    stream.base_seed = 42

    with pytest.raises(ValueError, match="no worker snapshots"):
        stream.load_state_dict({
            "version": 6,
            "base_seed": 42,
            "data_fingerprint": "current",
        })
