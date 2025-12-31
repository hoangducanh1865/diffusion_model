import torch
import torch.nn as nn
import torch.nn.functional as F
from src.transformer import TransformerBlock


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groupnorm_num_groups, time_embed_dim):
        super().__init__()
        self.time_expand = nn.Linear(time_embed_dim, out_channels)
        self.groupnorm_1 = nn.GroupNorm(groupnorm_num_groups, in_channels)
        self.conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding="same",
        )
        self.groupnorm_2 = nn.GroupNorm(groupnorm_num_groups, out_channels)
        self.conv_2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding="same",
        )
        self.resize_channels = (
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_embeddings):
        residual_connection = x
        time_embeddings = self.time_expand(time_embeddings)
        x = self.groupnorm_1(x)
        x = F.silu(x)  # @QUESTION: why do we use silu?
        x = self.conv_1(x)
        x = x + time_embeddings.unsqueeze(-1).unsqueeze(-1)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        x = x + self.resize_channels(residual_connection)
        return x


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
        )

    def forward(self, x):
        return self.upsample(x)


class UNET(nn.Module):
    def __init__(
        self,
        in_channels=3,
        start_dim=64,
        dim_mults=(1, 2, 4),
        residual_blocks_per_group=1,
        groupnorm_num_groups=16,
        time_embed_dim=128,
    ):
        super().__init__()

        #######################################
        ### COMPUTE ALL OF THE CONVOLUTIONS ###
        #######################################

        ### Store Number of Input channels from Original Image ###
        self.input_image_channels = in_channels

        ### Get Number of Channels at Each Block ###
        channel_sizes = [start_dim * i for i in dim_mults]
        starting_channel_size, ending_channel_size = channel_sizes[0], channel_sizes[-1]

        ### Compute the Input/Output Channel Sizes for Every Convolution of Encoder ###
        self.encoder_config = []

        for idx, d in enumerate(channel_sizes):
            ### For Every Channel Size add "residual_blocks_per_group" number of Residual Blocks that DONT Change the number of channels ###
            for _ in range(residual_blocks_per_group):
                self.encoder_config.append(
                    ((d, d), "residual")
                )  # Shape: (Batch x Channels x Height x Width) -> (Batch x Channels x Height x Width)

            ### After Residual Blocks include Downsampling (by factor of 2) but dont change number of channels ###
            self.encoder_config.append(
                ((d, d), "downsample")
            )  # Shape: (Batch x Channels x Height x Width) -> (Batch x Channels x Height/2 x Width/2)

            ### Compute Attention ###
            self.encoder_config.append((d, "attention"))

            ### If we are not at the last channel size, include a channel upsample (typically by factor of 2) ###
            if idx < len(channel_sizes) - 1:
                self.encoder_config.append(
                    ((d, channel_sizes[idx + 1]), "residual")
                )  # Shape: (Batch x Channels x Height x Width) -> (Batch x Channels*2 x Height x Width)

        ### The Bottleneck will have "residual_blocks_per_group" number of ResidualBlocks each with the input/output of our final channel size###
        self.bottleneck_config = []
        for _ in range(residual_blocks_per_group):
            self.bottleneck_config.append(((ending_channel_size, ending_channel_size), "residual"))

        ### Store a variable of the final Output Shape of our Encoder + Bottleneck so we can compute Decoder Shapes ###
        out_dim = ending_channel_size

        ### Reverse our Encoder config to compute the Decoder ###
        reversed_encoder_config = self.encoder_config[::-1]

        ### The output of our reversed encoder will be the number of channels added for residual connections ###
        self.decoder_config = []
        for idx, (metadata, type) in enumerate(reversed_encoder_config):
            ### Flip in_channels, out_channels with the previous out_dim added on ###
            if type != "attention":
                enc_in_channels, enc_out_channels = metadata

                self.decoder_config.append(
                    ((out_dim + enc_out_channels, enc_in_channels), "residual")
                )

                if type == "downsample":
                    ### If we did a downsample in our encoder, we need to upsample in our decoder ###
                    self.decoder_config.append(((enc_in_channels, enc_in_channels), "upsample"))

                ### The new out_dim will be the number of output channels from our block (or the cooresponding encoder input channels) ###
                out_dim = enc_in_channels
            else:
                in_channels = metadata
                self.decoder_config.append((in_channels, "attention"))

        ### Add Extra Residual Block for residual from input convolution ###
        # hint: We know that the initial convolution will have starting_channel_size
        # and the output of our decoder will also have starting_channel_size, so the
        # final ResidualBlock we need will need to go from starting_channel_size*2 to starting_channel_size

        self.decoder_config.append(((starting_channel_size * 2, starting_channel_size), "residual"))

        #######################################
        ### ACTUALLY BUILD THE CONVOLUTIONS ###
        #######################################

        ### Intial Convolution Block ###
        self.conv_in_proj = nn.Conv2d(
            self.input_image_channels, starting_channel_size, kernel_size=3, padding="same"
        )

        self.encoder = nn.ModuleList()
        for metadata, type in self.encoder_config:
            if type == "residual":
                in_channels, out_channels = metadata
                self.encoder.append(
                    ResidualBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        groupnorm_num_groups=groupnorm_num_groups,
                        time_embed_dim=time_embed_dim,
                    )
                )
            elif type == "downsample":
                in_channels, out_channels = metadata
                self.encoder.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
                )
            elif type == "attention":
                in_channels = metadata
                self.encoder.append(TransformerBlock(in_channels))

        ### Build Encoder Blocks ###
        self.bottleneck = nn.ModuleList()

        for (in_channels, out_channels), _ in self.bottleneck_config:
            self.bottleneck.append(
                ResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    groupnorm_num_groups=groupnorm_num_groups,
                    time_embed_dim=time_embed_dim,
                )
            )

        ### Build Decoder Blocks ###
        self.decoder = nn.ModuleList()
        for metadata, type in self.decoder_config:
            if type == "residual":
                in_channels, out_channels = metadata
                self.decoder.append(
                    ResidualBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        groupnorm_num_groups=groupnorm_num_groups,
                        time_embed_dim=time_embed_dim,
                    )
                )
            elif type == "upsample":
                in_channels, out_channels = metadata
                self.decoder.append(
                    UpSampleBlock(in_channels=in_channels, out_channels=out_channels)
                )

            elif type == "attention":
                in_channels = metadata
                self.decoder.append(TransformerBlock(in_channels))

        ### Output Convolution ###
        self.conv_out_proj = nn.Conv2d(
            in_channels=starting_channel_size,
            out_channels=self.input_image_channels,
            kernel_size=3,
            padding="same",
        )

    def forward(self, x, time_embeddings):
        residuals = []

        ### Pass Through Projection and Store Residual ###
        x = self.conv_in_proj(x)
        residuals.append(x)

        ### Pass through encoder and store residuals ##
        for module in self.encoder:
            if isinstance(module, (ResidualBlock)):
                x = module(x, time_embeddings)
                residuals.append(x)
            elif isinstance(module, nn.Conv2d):
                x = module(x)
                residuals.append(x)
            else:
                x = module(x)

        ### Pass Through BottleNeck ###
        for module in self.bottleneck:
            x = module(x, time_embeddings)

        ### Pass through Decoder while Concatenating Residuals ###
        for module in self.decoder:
            if isinstance(module, ResidualBlock):
                residual_tensor = residuals.pop()
                x = torch.cat([x, residual_tensor], axis=1)
                x = module(x, time_embeddings)
            else:
                x = module(x)

        ### Map back to num_channels for final output ###
        x = self.conv_out_proj(x)

        return x
