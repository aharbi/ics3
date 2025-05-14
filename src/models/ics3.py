import torch
import torch.nn as nn

from torch import Tensor
from einops import rearrange

from src.model import BaseModel
from src.models.transformer import Transformer


class ICS3(BaseModel):
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
        *args,
        **kwargs,
    ):
        super(ICS3, self).__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        grid_size = img_size // patch_size
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

        self.transformer = Transformer(
            dim=hidden_dim,
            depth=num_layers,
            heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.output_projection = nn.Linear(
            hidden_dim, patch_size * patch_size * out_channels
        )

        self.output_activation = nn.Sigmoid()

    def generate_attn_mask(self, num_blocks: int, num_patches) -> Tensor:
        blockwise_mask = torch.triu(torch.ones(num_blocks, num_blocks), diagonal=1)
        block = torch.ones((num_patches, num_patches), device=self.device)
        mask = torch.kron(blockwise_mask, block).bool()
        return mask

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

    def forward(self, x: Tensor, attn_mask: Tensor = None) -> Tensor:

        x = self.transformer(x, attn_mask=attn_mask)
        x = self.output_projection(x)
        x = self.output_activation(x)

        return x

    def process_output(self, x: Tensor, context_len: int) -> Tensor:

        y_hats = []

        for i in range(0, context_len + 1):
            if i % 2 != 0:
                # Ignore labels
                continue

            y_hat = x[:, i * self.num_patches : (i + 1) * self.num_patches, :]

            y_hat = rearrange(
                y_hat,
                "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                h=self.img_size // self.patch_size,
                p1=self.patch_size,
                p2=self.patch_size,
            )

            y_hats.append(y_hat)

        return y_hats

    def predict(self, x, context_set=None):

        context_pos_embeds = self.sinusoidal_positional_embedding(len(context_set) + 1)
        context_pos_embeds = context_pos_embeds.to(self.device)

        x = self.tokenize_image(x, context_pos_embeds[:, -1, :])
        attn_mask = self.generate_attn_mask(
            num_blocks=2 * len(context_set) + 1, num_patches=self.num_patches
        )

        if len(context_set) != 0:
            context = [
                (
                    self.tokenize_image(image, context_pos_embeds[:, index, :]),
                    self.tokenize_label(label, context_pos_embeds[:, index, :]),
                )
                for index, (image, label) in enumerate(context_set)
            ]

            context = [item for sublist in context for item in sublist]
            context = torch.cat(context, dim=1)

            x = torch.cat([context, x], dim=1)

        x = x.to(self.device)

        y_hat = self(x, attn_mask=attn_mask)
        y_hat = self.process_output(y_hat, 2 * len(context_set) + 1)

        return y_hat


if __name__ == "__main__":
    batch_size = 1
    in_channels = 3
    out_channels = 1
    img_size = 128
    patch_size = 16
    hidden_dim = 32
    num_layers = 2
    num_heads = 8
    mlp_ratio = 2

    input_tensor = torch.randn(batch_size, in_channels, img_size, img_size)

    model = ICS3(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        patch_size=patch_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
    )

    context_set = [
        (
            torch.randn(batch_size, in_channels, img_size, img_size),
            torch.randn(batch_size, 1, img_size, img_size),
        )
        for _ in range(3)
    ]

    context_set = []

    output_tensor = model.predict(input_tensor, context_set=context_set)

    print("Length of output tensor:", len(output_tensor))

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor[0].shape}")

    print(output_tensor[0])
