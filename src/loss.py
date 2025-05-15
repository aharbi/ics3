import torch
import torch.nn as nn

from torch import Tensor


class BCEDiceLoss(torch.nn.Module):
    def __init__(self, alpha: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.eps = eps

    def forward(self, prediction: Tensor, ground_truth: Tensor) -> float:
        intersection = torch.sum(prediction * ground_truth)
        dice_loss = 1 - (2.0 * intersection + self.eps) / (
            torch.sum(prediction) + torch.sum(ground_truth) + self.eps
        )
        bce_loss = self.bce(prediction, ground_truth)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss
