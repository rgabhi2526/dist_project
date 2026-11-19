"""Small CNN plus local-training and evaluation helpers.

Used by clients (:func:`train_epoch` on CPU) and by the server (:func:`evaluate` on GPU, plus
holding the global model). The architecture is fixed by decisions.md D5: 2 conv + 2 FC,
~100k parameters for FMNIST (~400 KB per message).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


class SmallCNN(nn.Module):
    """Two conv(+ReLU+maxpool) blocks followed by two fully connected layers.

    There is deliberately no BatchNorm: its running statistics are not trained by gradient and
    averaging them across non-IID clients is ill-defined, so omitting it keeps weight
    averaging (FedAvg / SEAFL / Libra aggregation) clean.

    Args:
        in_channels: Input channels (1 for FMNIST, 3 for CIFAR-10).
        n_classes: Number of output classes.
        width: Channels of the first conv; the second conv has ``2 * width``.
        image_size: Input height/width (28 for FMNIST, 32 for CIFAR-10).
    """

    def __init__(self, in_channels: int, n_classes: int, width: int, image_size: int = 28) -> None:
        """Build the layers."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(width, 2 * width, kernel_size=3, padding=1)
        side = image_size // 4  # two 2x2 max-pools each halve the spatial size
        self.fc1 = nn.Linear(2 * width * side * side, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits.

        Args:
            x: Batch of images, shape ``(B, in_channels, H, W)``.

        Returns:
            Logits of shape ``(B, n_classes)``.
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


def make_model(dataset: str) -> nn.Module:
    """Create the CNN sized for a dataset (decisions.md D5).

    Args:
        dataset: ``"fmnist"`` (1x28x28, ~100k params) or ``"cifar10"`` (3x32x32, wider).

    Returns:
        A freshly initialised :class:`SmallCNN`.

    Raises:
        ValueError: If ``dataset`` is unknown.
    """
    if dataset == "fmnist":
        return SmallCNN(in_channels=1, n_classes=10, width=16, image_size=28)
    if dataset == "cifar10":
        return SmallCNN(in_channels=3, n_classes=10, width=32, image_size=32)
    raise ValueError(f"unknown dataset {dataset!r}")


def train_epoch(model: nn.Module, loader: DataLoader, lr: float, device: str) -> float:
    """Run one epoch of plain SGD (decisions.md D6) on a client's local data.

    The optimizer is created inside on purpose: plain SGD has no momentum state, so nothing
    needs to persist between epochs and the client loop stays free of optimizer bookkeeping.

    Args:
        model: Model trained in place.
        loader: Local-data loader (batch size B from the config).
        lr: SGD learning rate (eta).
        device: Torch device string.

    Returns:
        Mean training loss over all samples of the epoch (reported to Libra as local loss).
    """
    model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        total += loss.item() * len(y)  # weight by batch size: last batch may be smaller
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    """Measure accuracy and mean cross-entropy loss without gradients.

    Args:
        model: Model to evaluate (switched to eval mode).
        loader: Evaluation data.
        device: Torch device string.

    Returns:
        ``(accuracy, mean_loss)`` with accuracy in ``[0, 1]``.
    """
    model.to(device).eval()
    correct, total, n = 0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return correct / max(n, 1), total / max(n, 1)
