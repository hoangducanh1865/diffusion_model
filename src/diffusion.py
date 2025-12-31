import torch.nn as nn
from src.unet import UNET
from src.embedding import SinusoidalTimeEmbedding


class Diffusion(nn.Module):
    def __init__(
        self,
        in_channels=3,
        start_dim=64,
        dim_mults=(1, 2, 4, 4),
        residual_blocks_per_group=1,
        groupnorm_num_groups=16,
        time_embed_dim=128,
        time_embed_dim_ratio=2,
    ):

        super().__init__()
        self.in_channels = in_channels
        self.start_dim = start_dim
        self.dim_mults = dim_mults
        self.residual_blocks_per_group = residual_blocks_per_group
        self.groupnorm_num_groups = groupnorm_num_groups

        self.time_embed_dim = time_embed_dim
        self.scaled_time_embed_dim = int(time_embed_dim * time_embed_dim_ratio)

        self.sinusoid_time_embeddings = SinusoidalTimeEmbedding(
            time_embed_dim=self.time_embed_dim, scaled_time_embed_dim=self.scaled_time_embed_dim
        )

        self.unet = UNET(
            in_channels=in_channels,
            start_dim=start_dim,
            dim_mults=dim_mults,
            residual_blocks_per_group=residual_blocks_per_group,
            groupnorm_num_groups=groupnorm_num_groups,
            time_embed_dim=self.scaled_time_embed_dim,
        )

    def forward(self, noisy_inputs, timesteps):

        ### Embed the Timesteps ###
        timestep_embeddings = self.sinusoid_time_embeddings(timesteps)

        ### Pass Images + Time Embeddings through UNET ###
        noise_pred = self.unet(noisy_inputs, timestep_embeddings)

        return noise_pred
