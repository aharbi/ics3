import torch
import torch.nn as nn

from torch import Tensor
from einops import rearrange

from src.model import BaseModel
from src.models.transformer import Transformer
from transformers import ViTModel


class VisionTransformer(BaseModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: int,
        patch_size: int = 16,
        hidden_dim: int = 768,
        mlp_ratio: int = 2,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout: float = 0.1,
        pretrained: bool = False,
        pretrained_model: str = "google/vit-base-patch16-224-in21k",
        unfreeze_epoch: int = 0,
        *args,
        **kwargs,
    ):
        super(VisionTransformer, self).__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.patch_size = patch_size
        self.pretrained = pretrained
        self.unfreeze_epoch = unfreeze_epoch

        grid_size = img_size // patch_size
        num_patches = grid_size * grid_size

        self.patch_embed = nn.Conv2d(
            in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))

        if not pretrained:
            self.transformer = Transformer(
                dim=hidden_dim,
                depth=num_layers,
                heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
        else:
            vit = ViTModel.from_pretrained(
                pretrained_model,
                add_pooling_layer=False
            )
            self.transformer = vit.encoder
            self._freeze_transformer()

        self.output_projection = nn.Linear(
            hidden_dim, patch_size * patch_size * out_channels
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape

        # Shape: (B, hidden_dim, grid, grid)
        x = self.patch_embed(x)
        grid = x.shape[-1]

        # Shape: (B, num_patches, hidden_dim)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + self.pos_embed

        x = self.transformer(x)

        if self.pretrained:
            x = x["last_hidden_state"]

        # Shape: (B, num_patches, patch_size * patch_size * out_channels)
        x = self.output_projection(x)

        x = rearrange(
            x,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=grid,
            p1=self.patch_size,
            p2=self.patch_size,
        )

        return x


if __name__ == "__main__":
    batch_size = 2
    in_channels = 3
    out_channels = 1
    img_size = 128
    patch_size = 16
    hidden_dim = 768
    num_layers = 4
    num_heads = 8

    input_tensor = torch.randn(batch_size, in_channels, img_size, img_size)

    model = VisionTransformer(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        patch_size=patch_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        pretrained=True
    )

    output_tensor = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
