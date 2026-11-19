import numpy as np
import pytest

from flsim.config import client_delays
from flsim.data import dirichlet_partition, split_test_dval


def _labels(n: int = 2000, classes: int = 10, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, classes, n)


def _mean_entropy(labels: np.ndarray, parts: list[np.ndarray]) -> float:
    ents = []
    for p in parts:
        pr = np.bincount(labels[p], minlength=10) / len(p)
        pr = pr[pr > 0]
        ents.append(-(pr * np.log(pr)).sum())
    return float(np.mean(ents))


def test_partition_covers_all_exactly_once():
    labels = _labels()
    parts = dirichlet_partition(labels, 20, 0.5, seed=1)
    assert len(parts) == 20
    allidx = np.concatenate(parts)
    assert len(allidx) == len(labels)
    assert np.array_equal(np.sort(allidx), np.arange(len(labels)))


def test_partition_deterministic_per_seed():
    labels = _labels()
    a = dirichlet_partition(labels, 10, 0.5, seed=3)
    b = dirichlet_partition(labels, 10, 0.5, seed=3)
    c = dirichlet_partition(labels, 10, 0.5, seed=4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_partition_min_size():
    parts = dirichlet_partition(_labels(), 20, 0.5, seed=0, min_size=25)
    assert min(len(p) for p in parts) >= 25


def test_partition_impossible_raises():
    with pytest.raises(RuntimeError):
        dirichlet_partition(_labels(100), 20, 0.5, seed=0, min_size=50)


def test_small_beta_is_more_skewed():
    labels = _labels(5000)
    skewed = _mean_entropy(labels, dirichlet_partition(labels, 20, 0.1, seed=0))
    uniform = _mean_entropy(labels, dirichlet_partition(labels, 20, 100.0, seed=0))
    assert skewed < uniform


def test_split_test_dval_disjoint_and_sized():
    test_idx, dval_idx = split_test_dval(10000, 500, seed=0)
    assert len(dval_idx) == 500 and len(test_idx) == 9500
    assert set(test_idx).isdisjoint(dval_idx)
    assert len(set(test_idx) | set(dval_idx)) == 10000


def test_client_delays_deterministic_and_bounded():
    a = client_delays(20, seed=0, base=2.0)
    b = client_delays(20, seed=0, base=2.0)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, client_delays(20, seed=1, base=2.0))
    assert a.shape == (20,)
    assert (a >= 2.0).all() and (a <= 2.0 * 20.0).all()
