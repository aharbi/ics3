import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor


class DiceLoss(torch.nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, prediction: Tensor, ground_truth: Tensor) -> float:
        intersection = torch.sum(prediction * ground_truth)
        return 1 - (2.0 * intersection + self.eps) / (
            torch.sum(prediction) + torch.sum(ground_truth) + self.eps
        )


def prediction_figure(
    satellite_image: np.array, prediction: np.array, ground_truth: np.array = None
):

    satellite_image = satellite_image.transpose(1, 2, 0)
    prediction = prediction.transpose(1, 2, 0)

    if ground_truth is not None:
        ground_truth = ground_truth.transpose(1, 2, 0)

    satellite_image = satellite_image.astype(np.float32)
    prediction = prediction.astype(np.float32)
    if ground_truth is not None:
        ground_truth = ground_truth.astype(np.float32)

    fig, axes = plt.subplots(1, 3 if ground_truth is not None else 2, figsize=(15, 5))

    axes[0].imshow(satellite_image)
    axes[0].set_title("Satellite Image")
    axes[0].axis("off")

    axes[1].imshow(prediction)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    if ground_truth is not None:
        axes[2].imshow(ground_truth)
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

    return fig
