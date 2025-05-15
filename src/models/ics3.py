import torch
import torch.nn as nn

from torch import Tensor
from einops import rearrange
from diffusers import UNet2DConditionModel

from src.model import BaseModel
from src.models.transformer import Transformer


class ICS3(BaseModel):
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 16,
        hidden_dim: int = 768,
        in_channels: int = 3,
        out_channels: int = 1,
        layers_per_block: int = 2,
        block_out_channels: list[int] = (64, 128, 256, 512),
        down_block_types: list[str] = (
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "CrossAttnDownBlock2D",
        ),
        up_block_types: list[str] = (
            "CrossAttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        *args,
        **kwargs,
    ):
        super(ICS3, self).__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.sample_size = sample_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        grid_size = sample_size // patch_size
        self.num_patches = grid_size * grid_size

        # Projections
        self.image_embed = nn.Conv2d(
            in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        self.label_embed = nn.Conv2d(
            1, hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        # Learnable position embeddings
        self.patch_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_dim)
        )
        self.image_pos_embed = nn.Parameter(torch.zeros(1, hidden_dim))
        self.label_pos_embed = nn.Parameter(torch.zeros(1, hidden_dim))

        self.unet = UNet2DConditionModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            cross_attention_dim=hidden_dim,
        )

    def sinusoidal_positional_embedding(self, N: int) -> Tensor:
        position = torch.arange(N).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.hidden_dim, 2).float()
            * -(torch.log(torch.tensor(10000.0)) / self.hidden_dim)
        )
        pos_emb = torch.zeros(N, self.hidden_dim)
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        return pos_emb.unsqueeze(0)

    def tokenize_image(self, x: Tensor, context_pos_embed: Tensor = None) -> Tensor:
        x = self.image_embed(x)

        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + self.patch_pos_embed
        x = x + self.image_pos_embed
        x = x + context_pos_embed

        return x

    def tokenize_label(self, x: Tensor, context_pos_embed: Tensor = None) -> Tensor:
        x = self.label_embed(x)

        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + self.patch_pos_embed
        x = x + self.label_pos_embed
        x = x + context_pos_embed

        return x

    def forward(self, x: Tensor, context: Tensor = None) -> Tensor:
        
        if context is not None:
            x = self.unet(sample=x, timestep=0, encoder_hidden_states=context)
        else:
            x = self.unet(sample=x, timestep=0)

        x = x.sample

        return x

    def predict(self, x, context_set_images=None, context_set_labels=None):

        _, M, _, _, _ = context_set_images.shape

        context_pos_embeds = self.sinusoidal_positional_embedding(M)
        context_pos_embeds = context_pos_embeds.to(self.device)

        context = []

        for i in range(M):
            image = self.tokenize_image(
                context_set_images[:, i, :, :, :], context_pos_embeds[:, i, :]
            )
            label = self.tokenize_label(
                context_set_labels[:, i, :, :, :], context_pos_embeds[:, i, :]
            )

            context.append((image, label))

        context = [item for sublist in context for item in sublist]
        context = torch.cat(context, dim=1)

        y_hat = self(x, context=context)

        return y_hat


if __name__ == "__main__":
    batch_size = 6
    in_channels = 3
    out_channels = 1
    img_size = 128
    patch_size = 16
    hidden_dim = 768
    layers_per_block = 2
    block_out_channels = [64, 128, 256, 512]

    input_tensor = torch.randn(batch_size, in_channels, img_size, img_size)

    model = ICS3(
        sample_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        patch_size=patch_size,
        hidden_dim=hidden_dim,
        layers_per_block=layers_per_block,
        block_out_channels=block_out_channels,
    )

    context_set_images = torch.randn(batch_size, 2, in_channels, img_size, img_size)
    context_set_labels = torch.randn(batch_size, 2, out_channels, img_size, img_size)

    output_tensor = model.predict(input_tensor, context_set_images=context_set_images, context_set_labels=context_set_labels)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")