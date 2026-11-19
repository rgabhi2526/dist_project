import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from flsim.model import evaluate, make_model, train_epoch


def _loader(shape: tuple[int, ...], n: int = 64) -> DataLoader:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, *shape, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=16)


@pytest.mark.parametrize("name,shape", [("fmnist", (1, 28, 28)), ("cifar10", (3, 32, 32))])
def test_forward_shape(name, shape):
    out = make_model(name)(torch.randn(4, *shape))
    assert out.shape == (4, 10)


def test_fmnist_param_count_sane():
    n = sum(p.numel() for p in make_model("fmnist").parameters())
    assert 50_000 <= n <= 200_000


def test_train_epoch_reduces_loss():
    torch.manual_seed(0)
    model, loader = make_model("fmnist"), _loader((1, 28, 28))
    first = train_epoch(model, loader, lr=0.05, device="cpu")
    for _ in range(10):
        last = train_epoch(model, loader, lr=0.05, device="cpu")
    assert math.isfinite(first) and math.isfinite(last)
    assert last < first


def test_evaluate_ranges():
    acc, loss = evaluate(make_model("fmnist"), _loader((1, 28, 28)), "cpu")
    assert 0.0 <= acc <= 1.0
    assert math.isfinite(loss) and loss > 0
