import os
os.environ['MPLBACKEND'] = 'Agg'
import torch
import numpy as np
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms


class Utils:
    @staticmethod
    @torch.no_grad()
    def sample_plot_image(
        step_idx,
        total_timesteps,
        sampler,
        image_size,
        num_channels,
        plot_freq,
        model,
        num_gens,
        path_to_generated_dir,
        device,
    ):
        ### Conver Tensor back to Image (From Huggingface Annotated Diffusion) ###
        if num_channels == 1:
            # Grayscale
            tensor2image_transform = transforms.Compose(
                [
                    transforms.Lambda(lambda t: t.squeeze(0)),  # (1, H, W) -> (H, W)
                    transforms.Lambda(lambda t: ((t + 1) / 2).clamp(0, 1)),
                    transforms.Lambda(lambda t: t * 255.0),
                    transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
                    transforms.ToPILImage(),
                ]
            )
        else:
            # RGB
            tensor2image_transform = transforms.Compose(
                [
                    transforms.Lambda(lambda t: t.squeeze(0)),  # (C, H, W) -> (C, H, W) if C>1
                    transforms.Lambda(lambda t: ((t + 1) / 2).clamp(0, 1)),
                    transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # (C, H, W) -> (H, W, C)
                    transforms.Lambda(lambda t: t * 255.0),
                    transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
                    transforms.ToPILImage(),
                ]
            )

        images = torch.randn((num_gens, num_channels, image_size, image_size))
        num_images_per_gen = total_timesteps // plot_freq

        images_to_vis = [[] for _ in range(num_gens)]
        for t in np.arange(total_timesteps)[::-1]:
            ts = torch.full((num_gens,), t)
            noise_pred = model(images.to(device), ts.to(device)).detach().cpu()
            images = sampler.remove_noise(images, ts, noise_pred)
            if t % plot_freq == 0:
                for idx, image in enumerate(images):
                    images_to_vis[idx].append(tensor2image_transform(image))

        images_to_vis = list(itertools.chain(*images_to_vis))

        fig, axes = plt.subplots(
            nrows=num_gens, ncols=num_images_per_gen, figsize=(num_images_per_gen, num_gens)
        )
        plt.tight_layout()
        for ax, image in zip(axes.ravel(), images_to_vis):
            ax.imshow(image)
            ax.axis("off")
        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.savefig(os.path.join(path_to_generated_dir, f"step_{step_idx}.png"))
        plt.show()
        plt.close()
