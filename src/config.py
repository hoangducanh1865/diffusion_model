import os
import torch


class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = "MNIST"  # ["MNIST", "CelebA"]
    path_to_dataset = (
        os.path("data/CelebA") if dataset == "CelebA" else "data"
    )  # MNIST downloads to data
    lr = 0.005
    num_epochs = 5
    path_to_checkpoints = os.path("models/checkpoints")
