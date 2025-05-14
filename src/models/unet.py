import torch
import torch.nn as nn

from torch import Tensor
from diffusers import UNet2DModel

from src.model import BaseModel


class UNet(BaseModel):
    def __init__(
        self,
        sample_size: int = 128,
        in_channels: int = 3,
        out_channels: int = 1,
        layers_per_block: int = 2,
        block_out_channels: list[int] = (64, 128, 256, 512),
        down_block_types: list[str] = (
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types: list[str] = (
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        add_attention: bool = False,
        *args,
        **kwargs,
    ):
        super(UNet, self).__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            add_attention=add_attention,
        )

        self.output_activation = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:

        x = self.unet(x, timestep=0).sample

        x = self.output_activation(x)

        return x


if __name__ == "__main__":
    batch_size = 2
    in_channels = 3
    out_channels = 1
    img_size = 128
    layers_per_block = 2
    block_out_channels = [64, 128, 256, 512]

    input_tensor = torch.randn(batch_size, in_channels, img_size, img_size)

    model = UNet(
        sample_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=layers_per_block,
        block_out_channels=block_out_channels,
    )

    output_tensor = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
