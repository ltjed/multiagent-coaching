import math
from typing import Iterator, Optional, TypeVar

import torch
import torch.distributed as dist
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler

import random
import numpy as np


__all__ = ["DistributedSampler"]


_T_co = TypeVar("_T_co", covariant=True)


# Adapted from https://github.com/pytorch/pytorch/blob/5298acb5c76855bc5a99ae10016efc86b27949bd/torch/utils/data/distributed.py
class DistributedSampler(Sampler[_T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        consumed_samples=0,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.consumed_indicies = consumed_samples // self.num_replicas

    def __iter__(self) -> Iterator[_T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        # skip consumed_samples
        indices = indices[self.consumed_indicies :]
        assert len(indices) == self.num_samples - self.consumed_indicies

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.consumed_indicies

    def set_epoch(self, epoch: int, consumed_samples=0) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
        self.consumed_indicies = consumed_samples // self.num_replicas

class ResumableRandomSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size, shuffle=True, drop_last=True, seed=None, consumed_samples=0):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.consumed_samples = consumed_samples
        self.epoch = 0
        self.drop_last = drop_last
        self.batch_size = batch_size
        # if self.drop_last and len(self.data_source) % self.batch_size != 0:
        if self.drop_last:
            self.total_size = len(data_source) // batch_size * batch_size
        else:
            self.total_size = int(math.ceil(len(data_source) / batch_size) * batch_size)

    def get_stratification_key(self, idx):
        """Get stratification key for a sample. Override in subclass."""
        return None

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.data_source), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.data_source)))  # type: ignore[arg-type]

        # drop last
        if self.drop_last:
            indices = indices[:self.total_size]
        else:
            # TODO: padding
            indices = indices[:self.total_size]

        indices = indices[self.consumed_samples:]

        return iter(indices)

    def __len__(self):
        return len(self.data_source) - self.consumed_samples

    def set_epoch(self, epoch: int, consumed_samples=0):
        self.epoch = epoch
        self.consumed_samples = consumed_samples


class StratifiedBatchSampler(torch.utils.data.Sampler):
    """
    Stratified batch sampler that maintains similar proportions of data types
    (e.g., classification vs regression) within each batch.

    This is specifically designed for DSBench where tasks have different data_type
    (classification/regression) and we want balanced batches for stable training.

    MathChat is unaffected: if metadata doesn't have data_type, falls back to
    random sampling (no stratification).

    Args:
        data_source: Dataset with samples that have metadata containing data_type
        batch_size: Batch size for training
        shuffle: Whether to shuffle within each stratum
        drop_last: Drop last incomplete batch
        seed: Random seed for reproducibility
        consumed_samples: Number of samples already consumed (for resumption)
        stratify_key: Key path to stratification field in metadata (default: "data_type")
    """

    def __init__(
        self,
        data_source,
        batch_size,
        shuffle=True,
        drop_last=True,
        seed=None,
        consumed_samples=0,
        stratify_key="data_type"
    ):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed if seed is not None else 42
        self.consumed_samples = consumed_samples
        self.epoch = 0
        self.stratify_key = stratify_key

        # Build stratification groups
        self._build_strata()

    def _get_data_type(self, sample):
        """Extract data_type from sample metadata."""
        try:
            # Handle both dict and object-like samples
            if hasattr(sample, 'get'):
                metadata = sample.get('metadata', {})
            elif hasattr(sample, '__getitem__'):
                metadata = sample.get('metadata', {})
            else:
                return None

            # Parse metadata if it's a JSON string
            if isinstance(metadata, str):
                import json
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    return None

            # Get data_type from metadata
            if isinstance(metadata, dict):
                return metadata.get(self.stratify_key)
            return None
        except Exception:
            return None

    def _build_strata(self):
        """Build index groups based on data_type."""
        self.strata = {}  # data_type -> list of indices

        for idx in range(len(self.data_source)):
            sample = self.data_source[idx]
            data_type = self._get_data_type(sample)

            if data_type is None:
                data_type = "__unknown__"

            if data_type not in self.strata:
                self.strata[data_type] = []
            self.strata[data_type].append(idx)

        # Calculate proportions
        total = len(self.data_source)
        self.proportions = {k: len(v) / total for k, v in self.strata.items()}

        # If only one stratum or all unknown, disable stratification
        known_strata = [k for k in self.strata if k != "__unknown__"]
        self.stratification_enabled = len(known_strata) >= 2

        if self.stratification_enabled:
            # Calculate per-batch counts for each stratum
            self.batch_counts = {}
            remaining = self.batch_size
            for data_type in sorted(self.strata.keys()):
                if data_type == "__unknown__":
                    continue
                count = max(1, round(self.batch_size * self.proportions[data_type]))
                count = min(count, remaining, len(self.strata[data_type]))
                self.batch_counts[data_type] = count
                remaining -= count

            # Distribute remaining to largest stratum
            if remaining > 0 and known_strata:
                largest = max(known_strata, key=lambda k: len(self.strata[k]))
                self.batch_counts[largest] += remaining

        # Calculate total size
        if self.drop_last:
            self.total_size = len(self.data_source) // self.batch_size * self.batch_size
        else:
            self.total_size = int(math.ceil(len(self.data_source) / self.batch_size) * self.batch_size)

    def __iter__(self):
        # Create RNG for this epoch
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if not self.stratification_enabled:
            # Fall back to random sampling (MathChat case)
            if self.shuffle:
                indices = torch.randperm(len(self.data_source), generator=g).tolist()
            else:
                indices = list(range(len(self.data_source)))

            if self.drop_last:
                indices = indices[:self.total_size]

            indices = indices[self.consumed_samples:]
            return iter(indices)

        # Stratified sampling
        # Shuffle within each stratum
        shuffled_strata = {}
        for data_type, indices in self.strata.items():
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=g).tolist()
                shuffled_strata[data_type] = [indices[i] for i in perm]
            else:
                shuffled_strata[data_type] = indices.copy()

        # Build interleaved indices
        all_indices = []
        stratum_positions = {k: 0 for k in shuffled_strata}

        while True:
            batch_indices = []

            # Sample from each stratum according to batch_counts
            for data_type, count in self.batch_counts.items():
                stratum = shuffled_strata[data_type]
                pos = stratum_positions[data_type]

                for _ in range(count):
                    if pos < len(stratum):
                        batch_indices.append(stratum[pos])
                        pos += 1
                    else:
                        # Stratum exhausted - wrap around (oversample if needed)
                        if self.shuffle:
                            # Reshuffle exhausted stratum
                            perm = torch.randperm(len(stratum), generator=g).tolist()
                            shuffled_strata[data_type] = [self.strata[data_type][i] for i in perm]
                            pos = 0
                            batch_indices.append(shuffled_strata[data_type][pos])
                            pos += 1
                        else:
                            pos = 0
                            batch_indices.append(stratum[pos])
                            pos += 1

                stratum_positions[data_type] = pos

            # Check if we have enough for a batch
            if len(batch_indices) >= self.batch_size:
                # Shuffle within batch for randomness
                if self.shuffle:
                    perm = torch.randperm(len(batch_indices), generator=g).tolist()
                    batch_indices = [batch_indices[i] for i in perm]
                all_indices.extend(batch_indices[:self.batch_size])

            # Check termination
            total_remaining = sum(
                len(shuffled_strata[k]) - stratum_positions[k]
                for k in shuffled_strata
            )
            if total_remaining < self.batch_size:
                if not self.drop_last and total_remaining > 0:
                    # Add remaining samples
                    for data_type, stratum in shuffled_strata.items():
                        pos = stratum_positions[data_type]
                        all_indices.extend(stratum[pos:])
                break

            if len(all_indices) >= self.total_size:
                break

        # Trim to total_size and skip consumed
        all_indices = all_indices[:self.total_size]
        all_indices = all_indices[self.consumed_samples:]

        return iter(all_indices)

    def __len__(self):
        return self.total_size - self.consumed_samples

    def set_epoch(self, epoch: int, consumed_samples=0):
        self.epoch = epoch
        self.consumed_samples = consumed_samples