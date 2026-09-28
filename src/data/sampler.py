"""
Identity-balanced PK Sampler for Person Re-Identification.
Samples P identities and K instances per identity (Batch Size = P * K).
Guarantees positive pairs in every mini-batch for batch-hard triplet mining.
"""

import copy
import random
from collections import defaultdict
from typing import Dict, Iterator, List, Optional
import torch
from torch.utils.data.sampler import Sampler


class MultiDomainPKSampler(Sampler):
    """
    Random PK Identity Sampler.
    In each batch, samples P identities and K images per identity -> B = P * K.
    """
    def __init__(
        self,
        dataset,
        p: int = 16,
        k: int = 4,
        seed: Optional[int] = None
    ):
        super().__init__()
        self.dataset = dataset
        self.p = p
        self.k = k
        self.batch_size = p * k
        self.seed = seed

        # Build pid -> indices mapping
        if hasattr(dataset, 'pid_to_indices'):
            self.pid_to_indices = dataset.pid_to_indices
        else:
            self.pid_to_indices = defaultdict(list)
            for idx, item in enumerate(dataset.samples):
                pid = item[1]
                if pid != -1:
                    self.pid_to_indices[pid].append(idx)

        self.pids = list(self.pid_to_indices.keys())
        self.num_identities = len(self.pids)

        if self.num_identities < self.p:
            raise ValueError(
                f"Number of unique identities ({self.num_identities}) must be >= P ({self.p})"
            )

        # Estimate length: enough batches to cover roughly all samples
        self.length = len(dataset) // self.batch_size
        if self.length == 0:
            self.length = 1

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed) if self.seed is not None else random

        # Shallow copy index lists for each identity
        pid_indices = {pid: copy.copy(indices) for pid, indices in self.pid_to_indices.items()}

        # Shuffle indices for each identity
        for pid in self.pids:
            rng.shuffle(pid_indices[pid])

        batch = []
        pids_pool = copy.copy(self.pids)
        rng.shuffle(pids_pool)

        # Generate batches
        for _ in range(self.length):
            if len(pids_pool) < self.p:
                pids_pool = copy.copy(self.pids)
                rng.shuffle(pids_pool)

            selected_pids = [pids_pool.pop() for _ in range(self.p)]
            batch = []

            for pid in selected_pids:
                indices = pid_indices[pid]
                if len(indices) < self.k:
                    # Sample with replacement if fewer than K samples available
                    sampled = rng.choices(self.pid_to_indices[pid], k=self.k)
                else:
                    sampled = [indices.pop() for _ in range(self.k)]
                    # Refill if exhausted
                    if len(indices) < self.k:
                        pid_indices[pid] = copy.copy(self.pid_to_indices[pid])
                        rng.shuffle(pid_indices[pid])

                batch.extend(sampled)

            yield batch

    def __len__(self) -> int:
        return self.length
