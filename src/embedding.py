import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, time_embed_dim, scaled_time_embed_dim):
        super().__init__()

        # This one is untrainable
        self.inv_freq = nn.Parameter(
            1.0 / (10000 ** (torch.arange(0, time_embed_dim, 2).float() / time_embed_dim)),
            requires_grad=False,
        )

        # This one is trainable
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, scaled_time_embed_dim),
            nn.SiLU(),
            nn.Linear(scaled_time_embed_dim, scaled_time_embed_dim),
            nn.SiLU(),
        )

    def forward(self, timesteps: torch.Tensor):
        timestep_freqs = timesteps.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        embeddings = torch.cat([torch.sin(timestep_freqs), torch.cos(timestep_freqs)], dim=-1)
        embeddings = self.time_mlp(embeddings)
        return embeddings
