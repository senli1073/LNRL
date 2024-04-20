import torch.nn as nn
import torch
from torch.nn import HuberLoss
from typing import Tuple



class CELoss(nn.Module):
    """
    Cross Entropy loss
    """

    _epsilon = 1e-6

    def __init__(self, weight=None) -> None:
        super().__init__()
        if weight is not None:
            print(f"[{self._get_name()}] Loss Weights:", weight)
            weight = torch.tensor(weight, dtype=torch.float32)
        else:
            weight = torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer("weight", weight)

    def forward(self, preds, targets):
        """Input shape: (N,C,L) or (N,Classes)"""
        loss = -targets * torch.log(preds + self._epsilon)
        loss *= self.weight
        loss = loss.sum(1).mean()
        return loss


class BCELoss(nn.Module):
    """
    Binary cross entropy loss for phase-picking and detection
    """

    _epsilon = 1e-6

    def __init__(self, weight=None) -> None:
        super().__init__()
        if weight is not None:
            print(f"[{self._get_name()}] Loss Weight:", weight)
            weight = torch.tensor(weight, dtype=torch.float32)
        else:
            weight = torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer("weight", weight)

    def forward(self, preds, targets):
        """Input shape: (N,C,L)"""
        loss = -(
            targets * torch.log(preds + self._epsilon)
            + (1 - targets) * torch.log(1 - preds + self._epsilon)
        )
        loss *= self.weight
        loss = loss.mean()
        return loss



class MSELoss(nn.Module):
    """
    MSE Loss.
    """

    def __init__(self, weight=None) -> None:
        super().__init__()
        if weight is not None:
            print(f"[{self._get_name()}] Loss Weights:", weight)
            weight = torch.tensor(weight, dtype=torch.float32)
        else:
            weight = torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer("weight", weight)

    def forward(self, preds, targets):
        """Input shape: (N,C,L)"""
        loss = (preds - targets) ** 2
        loss *= self.weight
        loss = loss.mean()
        return loss
