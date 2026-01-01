import os
import glob
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from src.config import Config
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, MNIST, CIFAR10
from torchvision import transforms
from transformers import get_cosine_schedule_with_warmup
from src.diffusion import Diffusion
from src.sampler import Sampler
from src.utils import Utils


class Trainer:
    def __init__(
        self,
        args=None,
        image_size=64,
        batch_size=64,
        total_timesteps=500,
        plot_freq_interval=50,
        num_generations=5,
        num_training_steps=50000,
        evaluation_interval=1000,
    ):
        self.device = args.device
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.dataset_name = args.dataset
        self.path_to_dataset = args.path_to_dataset
        self.path_to_checkpoints = args.path_to_checkpoints
        self.path_to_generated = args.path_to_generated
        self.num_checkpoints = args.num_checkpoints

        if self.dataset_name == "CIFAR10":
            self.image_size = 32
        else:
            self.image_size = image_size  # default 64 for model consistency, even though true image_size of MNIST dataset must be 28
        self.btach_size = batch_size
        self.total_timesteps = total_timesteps
        self.plot_freq_interval = plot_freq_interval
        self.num_generations = num_generations
        self.num_training_steps = num_training_steps
        self.evaluation_interval = evaluation_interval

        # Set num_input_channels based on dataset
        if self.dataset_name == "MNIST":
            self.num_input_channels = 1
        elif self.dataset_name == "CIFAR10" or self.dataset_name == "CelebA":
            self.num_input_channels = 3

        self.image2tensor = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                (
                    transforms.RandomHorizontalFlip()
                    if self.dataset_name == "CelebA"
                    else transforms.Lambda(lambda x: x)
                ),  # No flip for MNIST
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1),
            ]
        )

        if self.dataset_name == "MNIST":
            self.dataset = MNIST(
                root="data", train=True, download=True, transform=self.image2tensor
            )
        elif self.dataset_name == "CelebA":
            self.dataset = ImageFolder(root=self.path_to_dataset, transform=self.image2tensor)
        elif self.dataset_name == "CIFAR10":
            self.dataset = CIFAR10(
                root="data", train=True, download=True, transform=self.image2tensor
            )
        else:
            raise ValueError("Invalid dataset. Choose 'MNIST', 'CelebA', or 'CIFAR10'.")

        self.trainloader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,  # Just use when we are training on GPU
        )
        # Calculate total training steps based on epochs
        self.num_training_steps = self.num_epochs * len(self.trainloader)
        print(f"Total training steps: {self.num_training_steps}")

        self.model = Diffusion(in_channels=self.num_input_channels).to(self.device)

        self.model_params = filter(lambda p: p.requires_grad, self.model.parameters())  # @QUESTION
        self.params = sum([np.prod(p.size()) for p in self.model_params])  # @QUESTION
        print(f"Number of Parameters: {self.params}")

        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.lr)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer, num_warmup_steps=2500, num_training_steps=num_training_steps
        )  # @QUESTION
        self.ddpm_sampler = Sampler(num_timesteps=total_timesteps)
        self.loss_fn = nn.MSELoss()

        self.current_epoch = 0

    def train(self):
        progress_bar = tqdm(range(self.num_training_steps))
        completed_steps = 0

        train = True
        while train:
            training_losses = []

            for images, labels in self.trainloader:
                batch_size = images.shape[0]
                images = images.to(self.device)

                timesteps = torch.randint(
                    low=0, high=self.total_timesteps, device=self.device, size=(batch_size,)
                ).to(self.device)

                noisy_images, noise = self.ddpm_sampler.add_noise(
                    images=images, timesteps=timesteps
                )

                noise_pred = self.model(noisy_images, timesteps).to(self.device)

                loss = self.loss_fn(noise_pred, noise.to(self.device))

                training_losses.append(loss.cpu().item())  # @QUESTION

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)  # @QUESTION

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)  # @QUESTION

                progress_bar.update(1)
                completed_steps += 1

                if completed_steps % self.evaluation_interval == 0:
                    loss_mean = np.mean(training_losses)
                    print(f"\nTraining Loss: {loss_mean}")
                    lr = self.optimizer.param_groups[-1]["lr"]  # @QUESTION
                    print(f"Learning Rate: {lr}")

                    training_losses = []

                    print("Saving Image Generation")
                    Utils.sample_plot_image(
                        step_idx=completed_steps,
                        total_timesteps=self.total_timesteps,
                        sampler=self.ddpm_sampler,
                        image_size=self.image_size,
                        num_channels=self.num_input_channels,
                        plot_freq=self.plot_freq_interval,
                        model=self.model,
                        num_gens=self.num_generations,
                        path_to_generated_dir=self.path_to_generated,
                        device=self.device,
                    )

            self.current_epoch += 1
            if self.current_epoch % 5 == 0:
                self.save_checkpoint()

            if completed_steps >= self.num_training_steps:
                print("Training Completed")
                self.save_checkpoint()
                train = False
                break

    def save_checkpoint(self):
        os.makedirs(self.path_to_checkpoints, exist_ok=True)
        checkpoint_path = os.path.join(
            self.path_to_checkpoints, f"model_epoch_{self.current_epoch}.pth"
        )
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.lr_scheduler.state_dict(),
                "loss": self.loss_fn,
            },
            checkpoint_path,
        )
        print(f"\nCheckpoint saved to {checkpoint_path}")
        
        # Manage checkpoint count: keep only the top self.num_checkpoints newest
        checkpoint_files = glob.glob(os.path.join(self.path_to_checkpoints, "*.pth"))
        if len(checkpoint_files) > self.num_checkpoints:
            
            # Sort by modification time, newest first
            checkpoint_files.sort(key=os.path.getmtime, reverse=True)
            
            # Keep the newest num_checkpoints, delete the rest
            files_to_delete = checkpoint_files[self.num_checkpoints:]
            for file in files_to_delete:
                os.remove(file)
                print(f"Deleted old checkpoint: {file}")

def test():
    trainer = Trainer()


if __name__ == "__main__":
    test()
    pass
