import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from src.config import Config
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, MNIST
from torchvision import transforms
from transformers import get_cosine_schedule_with_warmup
from src.diffusion import Diffusion
from src.sampler import Sampler
from src.utils import Utils


class Trainer:
    def __init__(
        self,
        image_size=64,
        num_input_channels=None,  # Will be set based on dataset
        batch_size=64,
        total_timesteps=500,
        plot_freq_interval=50,
        num_generations=5,
        num_training_steps=50000,
        evaluation_interval=1000,
        path_to_generated="generated",
    ):
        self.image_size = image_size
        self.btach_size = batch_size
        self.total_timesteps = total_timesteps
        self.plot_freq_interval = plot_freq_interval
        self.num_generations = num_generations
        self.num_training_steps = num_training_steps
        self.evaluation_interval = evaluation_interval
        self.path_to_generated = path_to_generated

        self.device = Config.device
        self.dataset_name = Config.dataset
        self.path_to_dataset = Config.path_to_dataset
        self.lr = Config.lr
        self.num_epochs = Config.num_epochs

        # Set num_input_channels based on dataset
        if self.dataset_name == "MNIST":
            self.num_input_channels = 1
        elif self.dataset_name == "CelebA":
            self.num_input_channels = 3

        self.image2tensor = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip() if self.dataset_name == "CelebA" else transforms.Lambda(lambda x: x),  # No flip for MNIST
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1),
            ]
        )

        if self.dataset_name == "MNIST":
            self.dataset = MNIST(root="data", train=True, download=True, transform=self.image2tensor)
        elif self.dataset_name == "CelebA":
            self.dataset = ImageFolder(root=self.path_to_dataset, transform=self.image2tensor)
        else:
            raise ValueError("Invalid dataset. Choose 'MNIST' or 'CelebA'.")

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

    def train(self):
        progress_bar = tqdm(range(self.num_training_steps))
        completed_steps = 0

        train = True
        while train:
            training_losses = []

            for images, labels in self.trainloader:
                batch_size = images.shape[0]

                timesteps = torch.randint(
                    low=0, high=self.total_timesteps, device=self.device, size=(batch_size,)
                ).to(self.device)

                noisy_images, noise = self.ddpm_sampler.add_noise(
                    images=images, timesteps=timesteps
                )

                noise_pred = self.model(noisy_images, timesteps).to(self.device)
                
                loss = self.loss_fn(noise_pred, noise.to(self.device))
                
                training_losses.append(loss.cpu().item()) # @QUESTION
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # @QUESTION
                
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True) # @QUESTION
                
                progress_bar.update(1)
                completed_steps += 1
                
                if completed_steps % self.evaluation_interval == 0:
                    loss_mean = np.mean(training_losses)
                    print(f"Training Loss: {loss_mean}")
                    lr = self.optimizer.param_groups[-1]['lr']
                    print(f"Learning Rate: {lr}") # @QUESTION
                    
                    training_losses=[]
                    
                    print("Saving Image Generation")
                    Utils.sample_plot_image(step_idx=completed_steps, 
                                  total_timesteps=self.total_timesteps, 
                                  sampler=self.ddpm_sampler, 
                                  image_size=self.image_size,
                                  num_channels=self.num_input_channels,
                                  plot_freq=self.plot_freq_interval, 
                                  model=self.model,
                                  num_gens=self.num_generations,
                                  path_to_generated_dir=self.path_to_generated,
                                  device=self.device)

                if completed_steps >= self.num_training_steps:
                    print("Training Completed")
                    self.save_checkpoint()
                    train = False
                    break

    def save_checkpoint(self):
        import os
        os.makedirs("models/checkpoints", exist_ok=True)
        checkpoint_path = f"models/checkpoints/model_epoch_{self.num_epochs}.pth"
        torch.save({
            'epoch': self.num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'loss': self.loss_fn,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")


def test():
    trainer = Trainer()


if __name__ == "__main__":
    test()
    pass
