import torch


class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = "MNIST"  # Options: "MNIST" or "CelebA"
    path_to_dataset = "data/CelebA" if dataset == "CelebA" else "data"  # MNIST downloads to data
    lr = 0.005
    num_epochs = 5
