"""Dataset loading and federated partitioning.

Sits between torchvision and the rest of the system: the server uses :func:`load_dataset` and
:func:`split_test_dval` (test set vs Libra's D_val, decisions.md D5); each client uses
:func:`dirichlet_partition` to find its own shard. All functions are deterministic in ``seed``
so server and clients agree on the split without exchanging it.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# Standard per-dataset normalisation statistics (channel mean / std of the training set).
_STATS = {
    "fmnist": ((0.2860,), (0.3530,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}
_MAX_RETRIES = 1000  # bound on Dirichlet resampling before we give up


def load_dataset(name: str, root: str) -> tuple[Dataset, Dataset]:
    """Load (train, test) splits of a torchvision dataset, downloading if needed.

    Args:
        name: ``"fmnist"`` or ``"cifar10"``.
        root: Directory used as torchvision's download/cache root.

    Returns:
        ``(train_set, test_set)`` yielding normalised ``(tensor, label)`` pairs.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    if name not in _STATS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {sorted(_STATS)}")
    mean, std = _STATS[name]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    cls = datasets.FashionMNIST if name == "fmnist" else datasets.CIFAR10
    return (
        cls(root, train=True, download=True, transform=tf),
        cls(root, train=False, download=True, transform=tf),
    )


def dataset_labels(ds: Dataset) -> np.ndarray:
    """Return all labels of a torchvision dataset as an int array.

    Args:
        ds: Dataset exposing ``.targets`` (a tensor for FMNIST, a list for CIFAR-10).

    Returns:
        Array of shape ``(len(ds),)``.
    """
    return np.asarray(ds.targets)


def dirichlet_partition(
    labels: np.ndarray, n_clients: int, beta: float, seed: int, min_size: int = 10
) -> list[np.ndarray]:
    """Split sample indices across clients with a per-class Dirichlet(beta) prior.

    For every class c, a vector q ~ Dir(beta * 1_N) says which share of class c each of the N
    clients receives. Small beta gives highly skewed (non-IID) clients, large beta near-IID.
    The draw is repeated until every client holds at least ``min_size`` samples.

    Args:
        labels: Integer class label of every sample.
        n_clients: Number of clients N.
        beta: Dirichlet concentration (beta in the papers).
        seed: RNG seed; same seed gives the same partition.
        min_size: Minimum samples per client.

    Returns:
        ``n_clients`` sorted index arrays that together cover every index exactly once.

    Raises:
        RuntimeError: If no valid partition was found within the retry bound.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    for _ in range(_MAX_RETRIES):
        shards: list[list[int]] = [[] for _ in range(n_clients)]
        for c in classes:
            idx = rng.permutation(np.flatnonzero(labels == c))
            q = rng.dirichlet(np.full(n_clients, beta))  # q_k: client k's share of class c
            # Cumulative proportions give the cut points that split idx into N chunks.
            cuts = (np.cumsum(q)[:-1] * len(idx)).astype(int)
            for k, part in enumerate(np.split(idx, cuts)):
                shards[k].extend(part.tolist())
        if min(len(s) for s in shards) >= min_size:
            return [np.sort(np.asarray(s, dtype=np.int64)) for s in shards]
    raise RuntimeError(
        f"no Dirichlet partition with min_size={min_size} after {_MAX_RETRIES} tries "
        f"(n={len(labels)}, clients={n_clients}, beta={beta}); raise beta or lower min_size"
    )


def split_test_dval(n_test: int, dval_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Carve Libra's server validation set D_val out of the test set (decisions.md D5).

    D_val is used by Libra for Sinkhorn distances, so it is excluded from the reported test
    metric to keep that metric untouched by algorithm internals.

    Args:
        n_test: Size of the test set.
        dval_size: |D_val|, number of samples reserved.
        seed: RNG seed.

    Returns:
        ``(test_idx, dval_idx)``: disjoint index arrays covering the whole test set.

    Raises:
        ValueError: If ``dval_size`` is not smaller than ``n_test``.
    """
    if not 0 <= dval_size < n_test:
        raise ValueError(f"dval_size={dval_size} must be in [0, n_test={n_test})")
    perm = np.random.default_rng(seed).permutation(n_test)
    return np.sort(perm[dval_size:]), np.sort(perm[:dval_size])
